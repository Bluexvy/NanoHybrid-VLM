"""Manual GPU entry point; does not download weights or run a benchmark matrix."""
import argparse
import atexit
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local dense Qwen3.5 checkpoint directory")
    parser.add_argument("--prompt", default="Explain why the sky is blue in three sentences.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument("--baseline", action="store_true", help="Disable MTP for a later manual comparison")
    parser.add_argument("--mtp-tokens", type=int, default=1, help="1..15 draft tokens using the trained native head")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--graph", action="store_true", help="Allow bounded verification CUDA Graphs")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--image", help="Optional local image for one request")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--gdn-rollback", choices=("replay_inputs", "recompute"), default="replay_inputs")
    parser.add_argument("--gdn-replay-backend", choices=("auto", "reference", "cuda"), default="auto")
    parser.add_argument("--gdn-verification-backend", choices=("auto", "reference", "cuda"), default="auto")
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.max_model_len <= 1 or args.prefill_chunk < 2:
        parser.error("Use positive max-tokens, max-model-len > 1 and prefill-chunk >= 2")
    if not 1 <= args.mtp_tokens <= 15 or args.batch < 1 or args.temperature < 0:
        parser.error("Use 1..15 draft tokens, positive batch and non-negative temperature")
    if not args.baseline and args.prefill_chunk < args.mtp_tokens + 1:
        parser.error("prefill-chunk must cover pending plus all configured drafts")

    from nanovllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        tensor_parallel_size=args.tp,
        max_num_seqs=args.batch,
        max_num_batched_tokens=args.prefill_chunk,
        max_model_len=args.max_model_len,
        enforce_eager=not args.graph,
        hybrid_cuda_graph_batch_sizes=tuple(size for size in (1, 2, 4, 8, 16, 32, 64, 128)
                                           if size <= args.batch),
        gdn_decode_backend="fla",
        hybrid_prefix_cache_mode="opportunistic" if args.prefix_cache else "disabled",
        prefix_checkpoint_interval_blocks=1,
        mtp_num_speculative_tokens=0 if args.baseline else args.mtp_tokens,
        mtp_gdn_rollback=args.gdn_rollback,
        mtp_gdn_replay_backend=args.gdn_replay_backend,
        mtp_gdn_verification_backend=args.gdn_verification_backend,
    )
    try:
        prompts = [args.prompt] * args.batch
        image = None
        if args.image:
            from PIL import Image
            image = Image.open(args.image).convert("RGB")
            prompts[-1] = {"prompt": args.prompt, "multi_modal_data": {"image": image}}
        output = llm.generate(prompts, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens))
        report = {"mtp_enabled": not args.baseline, "outputs": output}
        if not args.baseline:
            counters = dict(llm.model_runner.mtp_stats)
            proposed = counters["proposed_tokens"]
            counters["acceptance_rate"] = counters["accepted_tokens"] / proposed if proposed else None
            report["mtp_stats"] = counters
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if 'image' in locals() and image is not None:
            image.close()
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()
