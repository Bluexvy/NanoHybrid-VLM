"""Compare with a separately downloaded official z-lab/dflash model.py.

python tests/dflash/compare_official.py --reference /path/to/official/model.py
No remote code is fetched or executed automatically by the engine.
"""
import argparse
import importlib.util
from pathlib import Path
import tempfile

import torch
import transformers
from transformers import Qwen3Config
from safetensors.torch import save_file
from helpers import tiny_raw, tiny_config, DFlashDraftModel
from nanovllm.utils.loader import load_model


class Cache:
    def __init__(self):
        self.layers = {}

    def update(self, k, v, index, kwargs):
        if index in self.layers:
            old = self.layers[index]
            k, v = torch.cat((old[0], k), -2), torch.cat((old[1], v), -2)
        self.layers[index] = k, v
        return k, v

    def crop(self, length):
        self.layers = {i: (k[:, :, :length].clone(), v[:, :, :length].clone())
                       for i, (k, v) in self.layers.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("dflash_official_reference", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    if int(transformers.__version__.split(".")[0]) < 5:
        # The public checkpoint requires Transformers 5.7. Old SDPA wrappers
        # ignore module.is_causal=False when mask=None. Supply the modern
        # explicit flag for this reference-only comparison; never patch runtime.
        original = reference.ALL_ATTENTION_FUNCTIONS["sdpa"]
        def compatible_sdpa(module, q, k, v, attention_mask, **kwargs):
            return original(module, q, k, v, attention_mask,
                is_causal=bool(module.is_causal and attention_mask is None and q.shape[2] > 1), **kwargs)
        reference.ALL_ATTENTION_FUNCTIONS["sdpa"] = compatible_sdpa
        print("Reference compatibility: explicit is_causal for Transformers <5")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = Qwen3Config(**tiny_raw())
    cfg.is_causal = None
    official = reference.DFlashDraftModel(cfg).eval().to(args.device)
    native = DFlashDraftModel(tiny_config(), tp_rank=0, tp_size=1).eval().to(args.device)
    with tempfile.TemporaryDirectory() as directory:
        save_file({k: v.detach().cpu() for k, v in official.state_dict().items()},
                  str(Path(directory) / "model.safetensors"))
        load_model(native, directory)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        official.to(dtype=dtype)
        native.to(dtype=dtype)
        cache, committed, start = Cache(), None, 0
        with torch.inference_mode():
            # Prefix, chunk continuation, and post-verification continuation.
            for added in (5, 2, 1):
                features = torch.randn(1, added, 32, device=args.device, dtype=dtype)
                noise = torch.randn(1, 4, 16, device=args.device, dtype=dtype)
                positions = torch.arange(start, start + added + 4, device=args.device)
                committed = native.append_context(features[0], positions[:added], committed)
                actual = native(noise[0], positions[added:], committed)
                expected = official(position_ids=positions[None], target_hidden=features,
                    noise_embedding=noise, past_key_values=cache, use_cache=True)[0]
                torch.testing.assert_close(actual, expected, atol=2e-5 if dtype == torch.float32 else .02,
                                           rtol=2e-5 if dtype == torch.float32 else .02)
                start += added
                cache.crop(start)
                for index, (k, v) in enumerate(committed):
                    torch.testing.assert_close(k, cache.layers[index][0][0], atol=.002, rtol=.002)
                    torch.testing.assert_close(v, cache.layers[index][1][0], atol=.002, rtol=.002)
        print(f"Official DFlash equivalence PASS: {args.device}, {dtype}, 3 cache continuations")


if __name__ == "__main__":
    main()
