"""Import CPU-capable runtime modules without the public CUDA-only LLM entry."""
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
for name, relative in (("nanovllm", "nanovllm"), ("nanovllm.engine", "nanovllm/engine"),
                       ("nanovllm.models", "nanovllm/models"), ("nanovllm.utils", "nanovllm/utils")):
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(ROOT / relative)]
        sys.modules[name] = package

from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.mtp_runner import MTPRunnerMixin
from nanovllm.engine.hybrid_state import HybridCacheSpec, HybridStateManager
from nanovllm.engine.prefix_cache import PrefixKey, PrefixStateCache
from nanovllm.engine.block_manager import BlockManager
from nanovllm.models.qwen3_5_mtp import Qwen3_5MTP
