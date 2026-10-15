"""Manual local-checkpoint DFlash entry; no automatic downloads."""
import argparse
import atexit
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen3.5-9B target directory")
    parser.add_argument("--draft", help="Local z-lab/Qwen3.5-9B-DFlash directory")
    parser.add_argument("--prompt", default="Explain how GPU shared memory differs from global memory.")
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--prefill-chunk", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--graph", action="store_true", help="Enable target verification CUDA Graphs")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--backend", choices=("auto", "reference", "cuda"), default="auto")
    args = parser.parse_args()
    if not args.baseline and not args.draft:
        parser.error("--draft is required unless --baseline is selected")
    if (not 1 <= args.draft_tokens <= 15 or args.batch < 1 or args.max_tokens < 1
            or args.max_model_len < 2 or args.prefill_chunk < args.draft_tokens + 1
            or args.temperature < 0):
        parser.error("Use 1..15 drafts, positive sizes, nonnegative temperature, and a sufficient token budget")

    import torch
    from nanovllm import LLM, SamplingParams
    llm = LLM(args.model, tensor_parallel_size=args.tp, max_num_seqs=args.batch,
        max_num_batched_tokens=args.prefill_chunk, max_model_len=args.max_model_len,
        enforce_eager=not args.graph,
        hybrid_cuda_graph_batch_sizes=tuple(n for n in (1, 2, 4, 8, 16, 32, 64, 128) if n <= args.batch),
        speculative_method="mtp" if args.baseline else "dflash",
        dflash_model=None if args.baseline else args.draft,
        dflash_num_speculative_tokens=args.draft_tokens,
        mtp_gdn_replay_backend=args.backend, mtp_gdn_verification_backend=args.backend,
        hybrid_prefix_cache_mode="opportunistic" if args.prefix_cache else "disabled",
        prefix_checkpoint_interval_blocks=1)
    try:
        prompts = [args.prompt] * args.batch
        parameters = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
        torch.cuda.synchronize()
        start = perf_counter()
        outputs = llm.generate(prompts, parameters, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        print(json.dumps({"method": "baseline" if args.baseline else "dflash",
            "outputs": outputs, "elapsed_seconds_including_prefill": elapsed,
            "speculative_stats": dict(llm.model_runner.mtp_stats)}, ensure_ascii=False, indent=2))
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()
