"""Small, offline DFlash fixtures; never downloads model weights."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mtp"))
import cpu_runtime
from nanovllm.models.dflash import DFlashDraftModel, read_dflash_config


def tiny_raw():
    return dict(architectures=["DFlashDraftModel"], model_type="qwen3", hidden_size=16,
        intermediate_size=24, num_hidden_layers=2, num_target_layers=6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        rms_norm_eps=1e-6, attention_bias=False, hidden_act="silu", vocab_size=32,
        max_position_embeddings=128, layer_types=["sliding_attention", "full_attention"],
        sliding_window=4, use_sliding_window=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.},
        dflash_config={"block_size": 4, "mask_token_id": 31, "target_layer_ids": [1, 3]})


def tiny_config():
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_text(json.dumps(tiny_raw()))
        return read_dflash_config(directory)
