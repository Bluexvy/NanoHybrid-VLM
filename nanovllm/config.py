import os
from dataclasses import dataclass

from transformers import AutoConfig
# PretrainedConfig
# → Qwen3Config、Qwen3_5Config 等配置类的共同父类
from transformers.configuration_utils import PretrainedConfig

@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    # Prefill 请求允许持续等待的最长时间。
    # 超过该时间后，Scheduler 会强制为最老的
    # Prefill 请求保留一个 chunk 的执行机会。
    max_prefill_wait_ms: float = 50.0
    # Decode-first：
    #   每轮先调度正在生成的请求。
    #
    # Prefill-first：
    #   只要 waiting 中存在 Prefill，
    #   本轮就暂停 Decode。
    scheduler_policy: str = "decode_first"
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    # 完整模型的根配置
    hf_config: PretrainedConfig | None = None
    # 语言模型部分的配置
    text_config: PretrainedConfig | None = None
    # 视觉模型配置；纯文本模型时为 None
    vision_config: PretrainedConfig | None = None 
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # ModelRunner 根据实际剩余显存计算，
    # 然后 Scheduler 使用它创建整数 slot 分配器。
    num_state_slots: int = -1

    # 自动分配时，最多使用多少 Cache 预算保存 GDN state。
    #
    # 用户显式指定 num_state_slots 时，该比例不生效。
    gdn_state_memory_fraction: float = 0.25

    # None 表示根据模型类型自动选择：
    #
    # 纯 Attention 模型 → 开启
    # 含 GDN 的混合模型 → 关闭
    enable_prefix_cache: bool | None = None
        
    # Qwen3.5 联合 KV + GDN Prefix Cache 模式。
    #
    # disabled：
    #     完全关闭联合 Prefix State Cache。
    #
    # opportunistic：
    #     只在 Prefill chunk 自然结束于合法
    #     checkpoint 时创建 Entry，不额外切分 Forward。
    #
    # aligned_debug 和 adaptive 会在后续 Part 实现。
    hybrid_prefix_cache_mode: str = "disabled"

    # 每隔多少个完整 token block，才允许保存一个
    # GDN Prefix checkpoint。
    #
    # block_size=256、interval=4 时：
    #     checkpoint 间隔 = 1024 tokens。
    prefix_checkpoint_interval_blocks: int = 4

    # Prefix Entry 中休眠 recurrent state 的 dtype。
    #
    # float32：
    #     正确性基线，约 49.5 MiB/Entry。
    #
    # bfloat16：
    #     压缩实验，约 25.5 MiB/Entry。
    #
    # active recurrent state 始终仍为 FP32。
    prefix_recurrent_snapshot_dtype: str = "float32"

    # 单条请求最多创建多少个新 Prefix snapshot。
    #
    # 先限制为 1，防止一个长 Prompt 保存大量
    # 49.5 MiB 的 GDN states。
    max_new_prefix_snapshots_per_request: int = 1
    
    # Config是dataclass 
    # __post_init__() 是 @dataclass 在自动执行完 __init__() 后调用的初始化hook
    # 但有些初始化工作不能只是赋值 还需要检查参数是否合法 所以不能只用dataclass
    def __post_init__(self):
        
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        
        if self.scheduler_policy not in {
            "decode_first",
            "prefill_first",
        }:
            raise ValueError(
                "scheduler_policy must be "
                "'decode_first' or 'prefill_first'"
            )
        
        if self.max_prefill_wait_ms < 0:
            raise ValueError(
                "max_prefill_wait_ms must be non-negative"
        )
            
        if (
            self.num_state_slots == 0
            or self.num_state_slots < -1
        ):
            raise ValueError(
                "num_state_slots must be -1 or positive"
            )

        if not (
            0.0
            < self.gdn_state_memory_fraction
            < 1.0
        ):
            raise ValueError(
                "gdn_state_memory_fraction must be "
                "between 0 and 1"
            )


        # 读取完整模型的 config.json
        root_config = AutoConfig.from_pretrained(self.model)

        # 对纯文本模型返回根配置本身；
        # 对多模态模型返回内部的 text_config
        text_config = root_config.get_text_config(decoder=True)

        layer_types = getattr(
            text_config,
            "layer_types",
            (),
        )

        has_gdn_layers = any(
            layer_type == "linear_attention"
            for layer_type in layer_types
        )
        
        supported_hybrid_prefix_modes = {
            "disabled",
            "opportunistic",
        }

        if (
            self.hybrid_prefix_cache_mode
            not in supported_hybrid_prefix_modes
        ):
            raise ValueError(
                "hybrid_prefix_cache_mode must be "
                "'disabled' or 'opportunistic' "
                "in the current implementation"
            )

        if self.prefix_checkpoint_interval_blocks <= 0:
            raise ValueError(
                "prefix_checkpoint_interval_blocks "
                "must be positive"
            )

        supported_snapshot_dtypes = {
            "float32",
            "bfloat16",
        }

        if (
            self.prefix_recurrent_snapshot_dtype
            not in supported_snapshot_dtypes
        ):
            raise ValueError(
                "prefix_recurrent_snapshot_dtype must "
                "be 'float32' or 'bfloat16'"
            )

        if (
            self.max_new_prefix_snapshots_per_request
            <= 0
        ):
            raise ValueError(
                "max_new_prefix_snapshots_per_request "
                "must be positive"
            )

        hybrid_prefix_cache_enabled = (
            self.hybrid_prefix_cache_mode
            != "disabled"
        )

        if (
            hybrid_prefix_cache_enabled
            and not has_gdn_layers
        ):
            raise ValueError(
                "hybrid_prefix_cache_mode is only "
                "valid for models containing GDN layers"
            )

        if (
            hybrid_prefix_cache_enabled
            and self.tensor_parallel_size != 1
        ):
            raise NotImplementedError(
                "Hybrid Prefix State Cache currently "
                "supports TP=1 only"
            )

        if self.enable_prefix_cache is None:
            # Qwen3 保持原来的 Prefix Cache。
            #
            # Qwen3.5 首版关闭 Prefix Cache。
            self.enable_prefix_cache = (
                not has_gdn_layers
            )

        elif (
            has_gdn_layers
            and self.enable_prefix_cache
        ):
            raise ValueError(
                "Prefix Cache is not supported for "
                "models containing GDN layers"
            )
        # 纯文本模型通常没有 vision_config
        vision_config = getattr(
            root_config,
            "vision_config",
            None,)

        self.hf_config = root_config
        self.text_config = text_config
        self.vision_config = vision_config

        # 最大上下文长度属于语言模型，因此读取 text_config
        self.max_model_len = min(
            self.max_model_len,
            text_config.max_position_embeddings,
        )