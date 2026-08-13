import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor


MODEL_ID = "Qwen/Qwen3.5-0.8B"
PROMPT = "一句话解释什么是KV cache"
MAX_NEW_TOKENS = 2048
DEVICE = "cuda"


# ---------- 1. 基础环境 ----------
assert torch.cuda.is_available()

# greedy 不使用随机数，但固定 seed 是良好的实验习惯
torch.manual_seed(0)

print("transformers:", transformers.__version__)
print("torch:", torch.__version__)
print("gpu:", torch.cuda.get_device_name(0))


# ---------- 2. 加载 Processor ----------
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("processor class:", type(processor).__name__)
print("tokenizer class:", type(processor.tokenizer).__name__)


# ---------- 3. 加载官方模型 ----------
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,

    # Qwen3.5 官方权重使用 BF16
    dtype=torch.bfloat16,

    # 先让 Full Attention 使用最容易排查的 eager 实现
    attn_implementation="eager",

    # 降低模型加载时的 CPU 内存峰值
    low_cpu_mem_usage=True,
)

model = model.eval().to(DEVICE)

print("model class:", type(model).__name__)
print("checkpoint revision:", getattr(model.config, "_commit_hash", None))
print(
    "model memory:",
    round(torch.cuda.memory_allocated() / 1024**3, 3),
    "GiB",
)


# ---------- 4. 构造聊天输入 ----------
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": PROMPT,
            }
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,

    # 自动在结尾添加 assistant 起始标记
    add_generation_prompt=True,

    # 直接执行 tokenizer，返回 token ID
    tokenize=True,

    # 返回包含 input_ids、attention_mask 等字段的字典
    return_dict=True,

    # 把结果变成 PyTorch Tensor
    return_tensors="pt",

    # 暂时关闭思考模式，让输出更短、更容易对齐
    enable_thinking=False,
)

# Processor 默认产生 CPU Tensor，需要搬到模型所在的 GPU
inputs = {
    name: tensor.to(DEVICE)
    for name, tensor in inputs.items()
}

print("\n=== Model inputs ===")

for name, tensor in inputs.items():
    print(
        name,
        "shape =", tuple(tensor.shape),
        "dtype =", tensor.dtype,
        "device =", tensor.device,
    )

prompt_token_ids = inputs["input_ids"][0].tolist()
prompt_length = inputs["input_ids"].shape[1]

print("prompt length:", prompt_length)
print("prompt token IDs:", prompt_token_ids)


# ---------- 5. Greedy generation ----------
torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    generation = model.generate(
        **inputs,

        # 最多生成 8 个新 token
        max_new_tokens=MAX_NEW_TOKENS,

        # Hugging Face 中真正的 greedy 是 do_sample=False
        do_sample=False,

        # 启用官方缓存，走 Prefill + Decode
        use_cache=True,

        # 返回结构化结果，而不只是一条 Tensor
        return_dict_in_generate=True,

        # 保存每轮用于选择 token 的分数
        output_scores=True,

        # 单请求影响不大，显式指定可以避免 padding 警告
        pad_token_id=processor.tokenizer.eos_token_id,
    )

torch.cuda.synchronize()

print(
    "peak generation memory:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    "GiB",
)


# ---------- 6. 拆出新生成的 token ----------
all_token_ids = generation.sequences[0]

# generation.sequences 包含：
# [原始 prompt token] + [新生成 token]
generated_token_ids = all_token_ids[prompt_length:].tolist()

raw_text = processor.decode(
    generated_token_ids,
    skip_special_tokens=False,
)

clean_text = processor.decode(
    generated_token_ids,
    skip_special_tokens=True,
)

print("\n=== Generation result ===")
print("all token IDs:", all_token_ids.tolist())
print("generated token IDs:", generated_token_ids)
print("raw text:", repr(raw_text))
print("clean text:", repr(clean_text))


# ---------- 7. 逐个观察生成 token ----------
print("\n=== Generated tokens ===")

for step, token_id in enumerate(generated_token_ids):
    token_piece = processor.decode(
        [token_id],
        skip_special_tokens=False,
    )

    print(
        f"step={step:02d}",
        f"token_id={token_id}",
        f"text={token_piece!r}",
    )


# ---------- 8. 保存第一份 golden 结果 ----------
golden_result = {
    "model_id": MODEL_ID,
    "checkpoint_revision": getattr(model.config, "_commit_hash", None),
    "transformers_version": transformers.__version__,
    "torch_version": torch.__version__,
    "dtype": "bfloat16",
    "attention_implementation": "eager",
    "prompt": PROMPT,
    "prompt_token_ids": prompt_token_ids,
    "generated_token_ids": generated_token_ids,
    "raw_text": raw_text,
    "clean_text": clean_text,
}

output_path = Path("artifacts/golden/qwen35_08b_text_greedy.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(
        golden_result,
        file,
        ensure_ascii=False,
        indent=2,
    )

print("\ngolden result saved to:", output_path)