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
    
    
    
    # Config是dataclass 
    # __post_init__() 是 @dataclass 在自动执行完 __init__() 后调用的初始化hook
    # 但有些初始化工作不能只是赋值 还需要检查参数是否合法 所以不能只用dataclass
    def __post_init__(self):
        
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8

        # 读取完整模型的 config.json
        root_config = AutoConfig.from_pretrained(self.model)

        # 对纯文本模型返回根配置本身；
        # 对多模态模型返回内部的 text_config
        text_config = root_config.get_text_config(decoder=True)

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