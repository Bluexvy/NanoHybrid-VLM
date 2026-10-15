"""Opt-in full-checkpoint GPU alignment matrix; never downloads weights.

Each case owns a fresh process so the NCCL group and CUDA pools cannot leak
between baseline, MTP, graph and prefix-cache runs.
"""
import argparse
import atexit
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CASES = ("baseline", "mtp_eager", "mtp_graph", "mtp_prefix",
         "dflash_eager", "dflash_graph", "dflash_prefix")


def worker(args):
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    seed = tokenizer.encode("This document explains GPU memory and model inference. ", add_special_tokens=False)
    shared = (seed * ((args.prefix_tokens + len(seed) - 1) // len(seed)))[:args.prefix_tokens]
    prompts = [shared + tokenizer.encode(f" Explain point {index} briefly.", add_special_tokens=False)
               for index in range(args.batch)]
    image = None
    if args.image:
        from PIL import Image
        image = Image.open(args.image).convert("RGB")
        prompts[-1] = {"prompt": "Describe the image briefly.", "multi_modal_data": {"image": image}}
    buckets = tuple(size for size in (1, 2, 4, 8, 16, 32, 64, 128) if size <= args.batch)
    prefix = args.case.endswith("_prefix")
    speculative = {}
    if args.case != "baseline":
        if args.method == "dflash":
            speculative = dict(speculative_method="dflash", dflash_model=args.draft_model,
                               dflash_num_speculative_tokens=args.drafts)
        else:
            speculative = dict(mtp_num_speculative_tokens=args.drafts)
    llm = LLM(args.model, tensor_parallel_size=args.tp, max_num_seqs=args.batch,
        max_num_batched_tokens=256, max_model_len=args.max_model_len,
        enforce_eager=args.case == "baseline" or args.case.endswith("_eager"),
        hybrid_cuda_graph_batch_sizes=buckets, gdn_decode_backend="fla",
        mtp_gdn_replay_backend=args.backend, mtp_gdn_verification_backend=args.backend,
        hybrid_prefix_cache_mode="opportunistic" if prefix else "disabled",
        prefix_checkpoint_interval_blocks=1, prefix_recurrent_snapshot_dtype="float32", **speculative)
    try:
        sampling = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True)
        first = llm.generate(prompts, sampling, use_tqdm=False)
        second = llm.generate(prompts, sampling, use_tqdm=False) if prefix else None
        if second is not None and [x["token_ids"] for x in first] != [x["token_ids"] for x in second]:
            raise AssertionError("Prefix restoration changed greedy output")
        if prefix and args.prefix_tokens > 256 and llm.scheduler.num_prefix_hit_tokens == 0:
            raise AssertionError("Prefix test did not exercise any actual prefix hit")
        report = {"case": args.case, "tp": args.tp, "batch": args.batch,
            "outputs": first, "mtp_stats": dict(llm.model_runner.mtp_stats),
            "prefix_hit_tokens": llm.scheduler.num_prefix_hit_tokens}
        Path(args.worker_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        if image is not None: image.close()
        atexit.unregister(llm.exit)
        llm.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=("mtp", "dflash"), default="mtp")
    parser.add_argument("--draft-model", help="Local DFlash checkpoint directory")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--drafts", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--prefix-tokens", type=int, default=384)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--backend", choices=("auto", "reference", "cuda"), default="auto")
    parser.add_argument("--image")
    parser.add_argument("--report", default="artifacts/mtp/model_validation.json")
    parser.add_argument("--require-graph", action="store_true")
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-report", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.method == "dflash" and (not args.draft_model or args.image):
        parser.error("DFlash requires --draft-model and currently supports text only")
    if (not 1 <= args.drafts <= 15 or args.batch <= 0 or args.output_tokens <= 0
            or args.prefix_tokens <= 256 or args.prefix_tokens + args.output_tokens + 32 >= args.max_model_len):
        parser.error("Use 1..15 drafts, positive batch/output, and a >256-token prefix within context capacity")
    if args.case:
        worker(args)
        return
    reports = []
    with tempfile.TemporaryDirectory() as directory:
        for case in ("baseline", f"{args.method}_eager", f"{args.method}_graph", f"{args.method}_prefix"):
            destination = str(Path(directory) / f"{case}.json")
            command = [sys.executable, str(Path(__file__).resolve()), "--model", args.model,
                "--tp", str(args.tp), "--batch", str(args.batch), "--drafts", str(args.drafts),
                "--output-tokens", str(args.output_tokens), "--prefix-tokens", str(args.prefix_tokens),
                "--max-model-len", str(args.max_model_len), "--backend", args.backend,
                "--case", case, "--worker-report", destination]
            command.extend(("--method", args.method))
            if args.draft_model: command.extend(("--draft-model", args.draft_model))
            if args.image: command.extend(("--image", args.image))
            subprocess.run(command, cwd=ROOT, check=True)
            reports.append(json.loads(Path(destination).read_text(encoding="utf-8")))
    expected = [output["token_ids"] for output in reports[0]["outputs"]]
    errors = []
    for report in reports[1:]:
        actual = [output["token_ids"] for output in report["outputs"]]
        if actual != expected: errors.append(f"{report['case']}: greedy tokens differ from baseline")
        if args.require_graph and "graph" in report["case"] and report["mtp_stats"]["verification_graph_replays"] == 0:
            errors.append(f"{report['case']}: no actual verification graph replay")
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"passed": not errors, "errors": errors, "cases": reports},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    if errors: raise AssertionError("; ".join(errors))
    print(f"PASS complete-checkpoint {args.method} alignment matrix: {destination}")


if __name__ == "__main__":
    main()
