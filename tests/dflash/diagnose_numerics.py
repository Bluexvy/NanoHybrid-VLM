"""Fixed-history path comparisons, without downloading weights or using a drafter."""
import argparse
import atexit
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark_utils import chat_ids, read_workloads, token_digest, write_json


def select_trial(report, batch, trial):
    baseline = next(r for r in report["cases"] if r["case"]["name"] == f"b{batch}_baseline_eager")
    selected = baseline["trials"][trial]
    rows = {row["id"]: row for row in read_workloads()}
    prompts = [rows[w["id"]]["prompt"] for w in selected["workloads"]]
    return prompts, selected["outputs"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results", required=True, help="Previous cases/summary.json")
    parser.add_argument("--batch", type=int, choices=(1, 4, 16), default=1)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--drafts", type=int, choices=(3, 7), default=7)
    parser.add_argument("--probe-at", nargs="+", type=int, default=[0, 5, 11, 26, 37, 98, 100, 107])
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=.70)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.results).resolve() == Path(args.output).resolve():
        parser.error("Output must not overwrite the original benchmark report")
    source = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if source["manifest"]["arguments"].get("prompts_jsonl"):
        parser.error("This diagnostic currently replays built-in workloads only")
    prompts, oracle = select_trial(source, args.batch, args.trial)
    thinking = source["manifest"]["arguments"].get("thinking", False)
    limit = min(len(tokens) for tokens in oracle) - args.drafts - 1
    if not args.probe_at or min(args.probe_at) < 0 or max(args.probe_at) > limit:
        parser.error(f"probe-at must be in [0,{limit}] to preserve a full lookahead")
    import torch
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = [chat_ids(tokenizer, prompt, thinking) for prompt in prompts]
    baseline = next(r for r in source["cases"] if r["case"]["name"] == f"b{args.batch}_baseline_eager")
    expected_workloads = baseline["trials"][args.trial]["workloads"]
    if [token_digest(value) for value in ids] != [row["input_sha256"] for row in expected_workloads]:
        raise ValueError("Reconstructed prompt IDs differ from the saved benchmark")
    llm = LLM(args.model, tensor_parallel_size=args.tp, enforce_eager=False,
        max_num_seqs=args.batch, max_num_batched_tokens=256, max_model_len=1024,
        gpu_memory_utilization=args.memory_fraction, scheduler_policy="prefill_first",
        hybrid_cuda_graph_batch_sizes=tuple(b for b in (1, 2, 4, 8, 16) if b <= args.batch),
        gdn_decode_backend="state_aware_cuda", mtp_gdn_verification_backend="cuda",
        mtp_gdn_replay_backend="cuda", hybrid_prefix_cache_mode="disabled")
    environment_file = Path(args.results).parent.parent / "environment.json"
    source_environment = json.loads(environment_file.read_text()) if environment_file.exists() else {}
    report = {"purpose": "fixed-history diagnostic, not a benchmark or correctness pass",
              "diagnostic_version": "first_gdn_stages_v3",
              "arguments": vars(args), "source_commit": source_environment.get("git_commit"),
              "source_token_consistency_passed": source.get("token_consistency_passed"), "steps": []}
    try:
        for prompt_ids in ids:
            llm.add_request(prompt_ids,
                            SamplingParams(temperature=0, max_tokens=128, ignore_eos=True))
        seqs = list(llm.scheduler.waiting)
        while any(seq.num_cached_tokens < seq.num_prompt_tokens for seq in seqs):
            _, stats = llm.step()
            if stats.num_decode_tokens:
                raise RuntimeError("A request decoded before diagnostic prefill completed")
        # Pending anchor IDs and every subsequent input follow the same saved
        # baseline history, regardless of any path's alternative prediction.
        for seq, tokens in zip(seqs, oracle):
            if seq.num_completion_tokens != 1:
                raise RuntimeError("Diagnostic requires exactly one pending completion token")
            seq.token_ids[-1] = seq.last_token = tokens[0]
        for step in range(max(args.probe_at) + 1):
            for seq in seqs:
                seq.is_prefill = False
                seq.num_scheduled_tokens = 1
                # Compact TP payload must retain the pending logical token.
                seq.mtp_num_speculative_tokens = 1
            future = [tokens[step + 1:step + 1 + args.drafts] for tokens in oracle]
            probe = step in args.probe_at
            value = llm.model_runner.call("diagnose_hybrid_numerics", seqs, future, probe)
            if probe:
                value["pending_completion_index"] = step
                value["predicts_completion_index"] = step + 1
                report["steps"].append(value)
                write_json(args.output, report)
                rank0 = value["ranks"][0]["comparisons"]
                print(f"probe input={step}, prediction={step+1}: " + "; ".join(
                    f"{mode}: argmax_diff={result['logits'].get('argmax_mismatches')}, "
                    f"hidden_max={result['hidden'].get('max_abs')}, "
                    f"first_layer={result['first_nonidentical_layer']}, "
                    f"first_gdn={result['first_nonidentical_gdn_stage']}" +
                    (f", vs_graph_hidden_exact={result['against_graph']['hidden']['exact']}, "
                     f"vs_graph_logits_exact={result['against_graph']['logits']['exact']}"
                     if 'against_graph' in result else '')
                    for mode, result in rank0.items()), flush=True)
                for mode in ("verify_one", "verify_block"):
                    print(f"  {mode} first GDN: " + "; ".join(
                        f"{row['stage']}={'exact' if row.get('exact') else row.get('max_abs', 'shape/nonfinite')}"
                        for row in rank0[mode]["first_gdn_stages"]), flush=True)
            llm.scheduler.postprocess(seqs, [tokens[step + 1] for tokens in oracle], False)
        report["diagnostic_completed"] = True
        write_json(args.output, report)
        print(f"Diagnostic complete: {args.output}; this is not a throughput measurement.")
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()
