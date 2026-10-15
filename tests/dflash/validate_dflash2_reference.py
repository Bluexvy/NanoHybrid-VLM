"""Compare tiny v2 forward/selector with a locally inspected official model.py.

No network access or full weights are needed. --reference-file explicitly
executes the provided Python file; use only the inspected z-lab/dflash source.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
from helpers import DFlashDraftModel, tiny_config
from test_dflash_v2 import checkpoint_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-file", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtypes", nargs="+", default=["float32"], choices=("float32", "float16", "bfloat16"))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("official_dflash_reference", args.reference_file)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    # Older Transformers infers causal=True for any multi-token query without
    # a mask. The official draft expects module.is_causal=False to be honored.
    # Make the intended semantics explicit in THIS reference process only.
    sdpa = reference.ALL_ATTENTION_FUNCTIONS["sdpa"]
    def explicit_sdpa(module, query, key, value, attention_mask, **kwargs):
        kwargs["is_causal"] = bool(module.is_causal) if attention_mask is None else False
        return sdpa(module, query, key, value, attention_mask, **kwargs)
    reference.ALL_ATTENTION_FUNCTIONS = {"sdpa": explicit_sdpa}
    from transformers import Qwen3Config
    torch.set_num_threads(1)
    results = []
    for dtype_name in args.dtypes:
        torch.manual_seed(802)
        cfg = tiny_config(2)
        cfg.sliding_window = 32
        local = DFlashDraftModel(cfg, tp_rank=0, tp_size=1).eval()
        with torch.no_grad():
            for value in local.parameters():
                value.normal_(std=.2)
        weights = checkpoint_weights(local)
        raw = vars(cfg).copy()
        raw.pop("target_layer_ids")  # Official reads nested dflash_config.
        official_cfg = Qwen3Config(**raw)
        official_cfg._attn_implementation = "sdpa"
        official = reference.DFlash2DraftModel(official_cfg).eval()
        converted = {name + ".weight" if name in (
            "candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook") else name: value
            for name, value in weights.items()}
        official.load_state_dict(converted, strict=True)
        dtype = getattr(torch, dtype_name)
        local, official = local.to(args.device, dtype), official.to(args.device, dtype)
        features = torch.randn(8, 32, device=args.device, dtype=dtype)
        cache, start = None, 0
        with torch.inference_mode():
            for added in (5, 2, 1):
                end = start + added
                cache = local.append_context(features[start:end], torch.arange(start, end, device=args.device), cache)
                noise = torch.randn(4, 16, device=args.device, dtype=dtype)
                actual = local(noise, torch.arange(end, end+4, device=args.device), cache)
                expected = official(position_ids=torch.arange(end+4, device=args.device)[None],
                    noise_embedding=noise[None], target_hidden=features[:end][None])[0]
                tolerance = 2e-5 if dtype_name == "float32" else .03
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                # Match selector q rows along the SAME selected path, with a
                # fixed RNG so both implementations choose identical tokens.
                logits = torch.randn(3, 32, device=args.device, dtype=dtype)
                hidden = actual[1:]
                for temperature in (0, .8):
                    torch.manual_seed(903)
                    ids, q = local.candidate_selector.select(hidden, logits, 7, temperature)
                    torch.manual_seed(903)
                    ref_ids, indices, ref_q = official.candidate_selector.select(
                        hidden[None], logits[None], torch.tensor([7], device=args.device), temperature)
                    assert ids == ref_ids[0].tolist()
                    if temperature:
                        dense = torch.zeros(3, 32, device=args.device).scatter(1, indices[0], ref_q[0])
                        torch.testing.assert_close(torch.stack(q), dense, rtol=tolerance, atol=tolerance)
                results.append({"device": args.device, "dtype": dtype_name, "context": end,
                    "max_abs_error": float((actual-expected).abs().max()), "passed": True})
                start = end
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
