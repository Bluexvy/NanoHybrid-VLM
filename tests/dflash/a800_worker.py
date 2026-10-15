"""One isolated model process for the A800 validation/benchmark suite."""
import argparse
import atexit
import json
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark_utils import (counters_delta, read_workloads, run_requests, summarize,
                             token_digest, write_json)


def chat_ids(tokenizer, text, thinking):
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
        add_generation_prompt=True, enable_thinking=thinking)


def run(args, spec):
    import torch
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    options = {}
    if spec["method"] == "dflash":
        options = dict(speculative_method="dflash", dflash_model=args.draft,
                       dflash_num_speculative_tokens=spec["drafts"])
    batch = spec["batch"]
    start = perf_counter()
    llm = LLM(args.model, tensor_parallel_size=args.tp, max_num_seqs=batch,
        max_num_batched_tokens=args.token_budget, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.memory_fraction, enforce_eager=not spec["graph"],
        scheduler_policy="decode_first" if spec["kind"] == "state_checks" else "prefill_first",
        max_prefill_wait_ms=60000,
        hybrid_cuda_graph_batch_sizes=tuple(i for i in (1, 2, 4, 8, 16, 32, 64, 128) if i <= batch),
        hybrid_prefix_cache_mode="opportunistic" if spec["kind"] == "state_checks" and options else "disabled",
        prefix_checkpoint_interval_blocks=1, prefix_recurrent_snapshot_dtype="float32",
        mtp_gdn_verification_backend=args.backend, mtp_gdn_replay_backend=args.backend,
        **options)
    startup = perf_counter() - start
    try:
        report = {"case": spec, "passed": False, "startup_seconds_not_timed": startup,
            "draft_architecture": llm.model_runner.dflash.config.architectures[0] if options else None}
        if spec["kind"] == "state_checks":
            shared = tokenizer.encode("A technical document about GPU memory, caching, and inference scheduling. ",
                                      add_special_tokens=False)
            shared = (shared * (512 // len(shared) + 1))[:384]
            trials = []
            for suffixes in ((" Explain shared memory.", " Explain global memory."),
                             (" Explain prefix caching.", " Explain tensor parallelism.")):
                prompts = [shared + tokenizer.encode(text, add_special_tokens=False) for text in suffixes]
                params = [SamplingParams(temperature=0, max_tokens=16 + 4 * i, ignore_eos=True) for i in range(2)]
                trials.append(run_requests(llm, prompts, params, torch.cuda.synchronize))
            hits = llm.scheduler.num_prefix_hit_tokens
            if options and hits <= 0:
                raise AssertionError("Prefix check did not exercise a real cache hit")
            initial = [chat_ids(tokenizer, "Count from one to ten.", args.thinking)] * 2
            more = [chat_ids(tokenizer, "Explain a Python list in simple terms.", args.thinking)] * 2
            initial_params = [SamplingParams(temperature=0, max_tokens=i, ignore_eos=True) for i in (12, 20)]
            more_params = [SamplingParams(temperature=0, max_tokens=i, ignore_eos=True) for i in (16, 24)]
            dynamic = run_requests(llm, initial, initial_params, torch.cuda.synchronize,
                                   injected=(more, more_params))
            if llm.model_runner.dflash_caches or llm.model_runner.mtp_caches:
                raise AssertionError("Finished requests leaked private draft caches")
            report.update(prefix_hit_tokens=hits, check_outputs=[t["outputs"] for t in trials] + [dynamic["outputs"]],
                          dynamic_injected_requests=dynamic["injected_requests"])
        else:
            rows = read_workloads(args.prompts_jsonl)
            trials = []
            for repeat in range(args.repeats):
                selection = [rows[(repeat * batch + i) % len(rows)] for i in range(batch)]
                prompts = [chat_ids(tokenizer, row["prompt"], args.thinking) for row in selection]
                if any(len(prompt) + args.output_tokens + 16 > args.max_model_len for prompt in prompts):
                    raise ValueError("Prompt/output exceeds max_model_len")
                params = [SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True)] * batch
                # Warm the SAME prompts and output budget, including the clipped
                # verification widths near the output boundary.
                for _ in range(args.warmups):
                    run_requests(llm, prompts, params, torch.cuda.synchronize)
                llm.model_runner.call("get_distributed_memory_stats", True)
                before = dict(llm.model_runner.mtp_stats)
                baseline_graph_before = llm.model_runner.num_hybrid_graph_replays
                baseline_fallback_before = llm.model_runner.num_hybrid_graph_eager_fallbacks
                trial = run_requests(llm, prompts, params, torch.cuda.synchronize)
                after = dict(llm.model_runner.mtp_stats)
                trial["counter_delta"] = counters_delta(after, before)
                trial["baseline_graph_replays_delta"] = llm.model_runner.num_hybrid_graph_replays - baseline_graph_before
                trial["baseline_eager_fallbacks_delta"] = llm.model_runner.num_hybrid_graph_eager_fallbacks - baseline_fallback_before
                trial["memory_per_rank"] = llm.model_runner.call("get_distributed_memory_stats")
                trial["workloads"] = [{"id": row["id"], "task": row["task"], "prompt_tokens": len(ids),
                    "input_sha256": token_digest(ids)} for row, ids in zip(selection, prompts)]
                if any(len(tokens) != args.output_tokens for tokens in trial["outputs"]):
                    raise AssertionError("A request did not produce its fixed output budget")
                trial["measurement_contains_graph_capture"] = trial["counter_delta"].get("verification_graph_captures", 0) > 0
                trials.append(trial)
            report.update(trials=trials, summary=summarize(trials))
        counters = dict(llm.model_runner.mtp_stats)
        if spec["graph"] and options and counters["verification_graph_replays"] == 0:
            raise AssertionError("Graph case never replayed a verification graph")
        if spec["graph"] and not options and llm.model_runner.num_hybrid_graph_replays == 0:
            raise AssertionError("Baseline Graph case never replayed its decode graph")
        report.update(passed=True, lifetime_counters=counters,
                      baseline_graph_replays=llm.model_runner.num_hybrid_graph_replays,
                      memory_per_rank=llm.model_runner.call("get_distributed_memory_stats"))
        return report
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--backend", choices=("auto", "reference", "cuda"), default="auto")
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--memory-fraction", type=float, default=.82)
    parser.add_argument("--prompts-jsonl")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    try:
        report = run(args, spec)
    except Exception:
        write_json(args.report, {"case": spec, "passed": False, "error": traceback.format_exc()})
        raise
    write_json(args.report, report)


if __name__ == "__main__":
    main()
