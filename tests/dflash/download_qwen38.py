"""Download the BF16 Qwen3.8-27B / DFlash2 pair without importing torch/CUDA."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

PAIR = (("Qwen/Qwen3.8-27B", "Qwen3.8-27B"),
        ("incoai/Qwen3.8-27B-DFlash2", "Qwen3.8-27B-DFlash2"))
PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken"]


def describe_pair(target, draft, tp=2):
    text = target["text_config"]
    extra = draft["dflash_config"]
    if target.get("model_type") != "qwen3_5" or draft.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("Expected the Qwen3.8 hybrid target and a DFlash2 draft")
    if (text["hidden_size"] != draft["hidden_size"] or text["vocab_size"] != draft["vocab_size"]
            or text["num_hidden_layers"] != draft["num_target_layers"]):
        raise ValueError("Target and draft dimensions do not match")
    types = text["layer_types"]
    if len(types) != text["num_hidden_layers"] or set(types) != {"linear_attention", "full_attention"}:
        raise ValueError("Invalid hybrid layer layout")
    dimensions = (text["num_attention_heads"], text["num_key_value_heads"],
        text["linear_num_key_heads"], text["linear_num_value_heads"], text["intermediate_size"],
        draft["num_attention_heads"], draft["num_key_value_heads"], draft["intermediate_size"])
    if tp <= 0 or any(value % tp for value in dimensions):
        raise ValueError("Head/intermediate dimensions do not divide TP size")
    if any(i < 0 or i >= len(types) for i in extra["target_layer_ids"]):
        raise ValueError("Target feature layer is out of range")
    return {"target_layers": len(types), "gdn_layers": types.count("linear_attention"),
        "attention_layers": types.count("full_attention"), "hidden_size": text["hidden_size"],
        "intermediate_size": text["intermediate_size"], "tp": tp,
        "local_q_heads": text["num_attention_heads"] // tp,
        "local_kv_heads": text["num_key_value_heads"] // tp,
        "local_gdn_key_heads": text["linear_num_key_heads"] // tp,
        "local_gdn_value_heads": text["linear_num_value_heads"] // tp,
        "draft_layers": draft["num_hidden_layers"], "draft_block_size": extra["block_size"],
        "draft_feature_layers": extra["target_layer_ids"], "draft_mask_token_id": extra["mask_token_id"],
        "selector_top_k": extra["selector_top_k"], "selector_rank": extra["selector_rank"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Persistent model disk directory")
    parser.add_argument("--only", choices=("both", "target", "draft"), default="both")
    parser.add_argument("--dry-run", action="store_true", help="Print plan; no dependencies/network/writes")
    args = parser.parse_args()
    selected = PAIR if args.only == "both" else (PAIR[0 if args.only == "target" else 1],)
    for repo, name in selected:
        print(f"{repo} -> {args.root / name}", flush=True)
    if args.dry_run:
        return
    # Downloading files needs only huggingface_hub, never a GPU environment.
    from huggingface_hub import HfApi, snapshot_download
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "models": {}}
    for repo, name in selected:
        destination = args.root / name
        config = destination / "config.json"
        if config.exists():
            previous = json.loads(config.read_text(encoding="utf-8"))
            if name == "Qwen3.8-27B" and previous.get("text_config", {}).get("num_hidden_layers") != 64:
                raise ValueError(f"Directory contains a different target: {destination}")
            if name.endswith("DFlash2") and previous.get("architectures") != ["DFlash2DraftModel"]:
                raise ValueError(f"Directory contains a different draft: {destination}")
        info = HfApi().model_info(repo, files_metadata=True)
        weight_bytes = sum(file.size or 0 for file in info.siblings if file.rfilename.endswith(".safetensors"))
        print(f"Pinned revision: {info.sha}; weight files: {weight_bytes / 1e9:.2f} GB", flush=True)
        snapshot_download(repo_id=repo, revision=info.sha, local_dir=str(destination), allow_patterns=PATTERNS)
        manifest["models"][repo] = {"revision": info.sha, "directory": str(destination.resolve()),
                                   "weight_bytes": weight_bytes}
        (args.root / (name + ".download.json")).write_text(
            json.dumps(manifest["models"][repo], indent=2) + "\n", encoding="utf-8")
    target_file, draft_file = (args.root / name / "config.json" for _, name in PAIR)
    if target_file.exists() and draft_file.exists():
        manifest["layout_tp2"] = describe_pair(json.loads(target_file.read_text(encoding="utf-8")),
                                               json.loads(draft_file.read_text(encoding="utf-8")))
        print(json.dumps(manifest["layout_tp2"], indent=2), flush=True)
    (args.root / "qwen38-dflash2-download.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Download finished. GPU allocation is only needed for subsequent tests.")


if __name__ == "__main__":
    main()
