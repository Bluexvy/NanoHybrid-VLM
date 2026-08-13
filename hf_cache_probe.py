import json

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from pathlib import Path

from safetensors.torch import save_file

MODEL_ID = "Qwen/Qwen3.5-0.8B"
PROMPT = "一句话解释什么是KV cache"
DEVICE = "cuda"


def tensor_mib(tensor):
    """计算一个 Tensor 实际占用多少 MiB。"""

    if tensor is None:
        return 0.0

    num_bytes = tensor.numel() * tensor.element_size()
    return num_bytes / 1024**2


def show_cache(cache, layer_types, title):
    """
    打印 DynamicCache 中每一层实际保存的内容。

    Full Attention 层：
        layer.keys
        layer.values

    GDN 层：
        layer.conv_states[0]
        layer.recurrent_states[0]
    """

    kv_mib = 0.0
    conv_mib = 0.0
    recurrent_mib = 0.0

    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print("cache class:", type(cache).__name__)
    print("number of cache layers:", len(cache.layers))
    print("cached sequence length:", cache.get_seq_length())

    for layer_idx, layer_type in enumerate(layer_types):
        cache_layer = cache.layers[layer_idx]

        if layer_type == "full_attention":
            keys = cache_layer.keys
            values = cache_layer.values

            layer_memory = tensor_mib(keys) + tensor_mib(values)
            kv_mib += layer_memory

            print(
                f"layer {layer_idx:02d}",
                "type=full_attention",
                f"cache_class={type(cache_layer).__name__}",
                f"K={tuple(keys.shape)}",
                f"V={tuple(values.shape)}",
                f"dtype={keys.dtype}",
                f"memory={layer_memory:.6f} MiB",
            )

        elif layer_type == "linear_attention":
            conv_state = cache_layer.conv_states[0]
            recurrent_state = cache_layer.recurrent_states[0]

            conv_memory = tensor_mib(conv_state)
            recurrent_memory = tensor_mib(recurrent_state)

            conv_mib += conv_memory
            recurrent_mib += recurrent_memory

            print(
                f"layer {layer_idx:02d}",
                "type=linear_attention",
                f"cache_class={type(cache_layer).__name__}",
                f"conv={tuple(conv_state.shape)}",
                f"conv_dtype={conv_state.dtype}",
                f"recurrent={tuple(recurrent_state.shape)}",
                f"recurrent_dtype={recurrent_state.dtype}",
            )

        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    print("\n--- Cache memory summary ---")
    print(f"Paged-style KV content: {kv_mib:.6f} MiB")
    print(f"GDN conv states:        {conv_mib:.6f} MiB")
    print(f"GDN recurrent states:   {recurrent_mib:.6f} MiB")
    print(
        "Total cache content:   "
        f"{kv_mib + conv_mib + recurrent_mib:.6f} MiB"
    )


def cache_metadata(cache, layer_types):
    """
    只保存缓存的结构信息，不保存巨大的缓存内容。

    返回：
        当前缓存 token 数
        每一层的缓存类型
        每个状态的 shape、dtype 和字节数
    """

    result = {
        "sequence_length": cache.get_seq_length(),
        "layers": [],
    }

    for layer_idx, layer_type in enumerate(layer_types):
        cache_layer = cache.layers[layer_idx]

        if layer_type == "full_attention":
            keys = cache_layer.keys
            values = cache_layer.values

            layer_info = {
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "cache_class": type(cache_layer).__name__,
                "key_shape": list(keys.shape),
                "value_shape": list(values.shape),
                "key_dtype": str(keys.dtype),
                "value_dtype": str(values.dtype),
                "num_bytes": (
                    keys.numel() * keys.element_size()
                    + values.numel() * values.element_size()
                ),
            }

        elif layer_type == "linear_attention":
            conv_state = cache_layer.conv_states[0]
            recurrent_state = cache_layer.recurrent_states[0]

            layer_info = {
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "cache_class": type(cache_layer).__name__,
                "conv_shape": list(conv_state.shape),
                "recurrent_shape": list(recurrent_state.shape),
                "conv_dtype": str(conv_state.dtype),
                "recurrent_dtype": str(recurrent_state.dtype),
                "num_bytes": (
                    conv_state.numel() * conv_state.element_size()
                    + recurrent_state.numel()
                    * recurrent_state.element_size()
                ),
            }

        else:
            raise ValueError(
                f"Unknown layer type: {layer_type}"
            )

        result["layers"].append(layer_info)

    result["total_num_bytes"] = sum(
        layer["num_bytes"]
        for layer in result["layers"]
    )

    return result



def make_topk_summary(logits, processor, k=10):
    """
    从完整词表 logits 中，选出分数最高的 k 个 token。
    """

    # 转成 float32 后再做统计，更方便观察数值
    logits_fp32 = logits.float()

    top_values, top_token_ids = torch.topk(
        logits_fp32,
        k=k,
    )

    token_ids = top_token_ids.tolist()
    values = top_values.tolist()

    token_texts = [
        processor.decode(
            [token_id],
            skip_special_tokens=False,
        )
        for token_id in token_ids
    ]

    return {
        "argmax_token_id": token_ids[0],
        "argmax_token_text": token_texts[0],
        "top_token_ids": token_ids,
        "top_token_texts": token_texts,
        "top_values": values,
        "minimum": logits_fp32.min().item(),
        "maximum": logits_fp32.max().item(),
        "mean": logits_fp32.mean().item(),
        "std": logits_fp32.std().item(),
    }
# ============================================================
# 1. 加载 Processor 和官方模型
# ============================================================

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)

model = model.eval().to(DEVICE)

layer_types = model.config.text_config.layer_types

print("number of decoder layers:", len(layer_types))
print(
    "linear attention layers:",
    layer_types.count("linear_attention"),
)
print(
    "full attention layers:",
    layer_types.count("full_attention"),
)


# ============================================================
# 2. 构造与 golden baseline 完全相同的输入
# ============================================================

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
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
)

inputs = {
    name: tensor.to(DEVICE)
    for name, tensor in inputs.items()
}

prompt_length = inputs["input_ids"].shape[1]

print("\ninput_ids shape:", tuple(inputs["input_ids"].shape))
print("prompt length:", prompt_length)
print("prompt token IDs:", inputs["input_ids"][0].tolist())


# ============================================================
# 3. 手动执行 Prefill
# ============================================================

with torch.inference_mode():
    prefill_output = model(
        **inputs,

        # 要求模型创建并返回缓存
        use_cache=True,

        # 只计算最后一个位置的 logits
        # 因为只有最后位置用于预测下一个 token
        logits_to_keep=1,

        return_dict=True,
    )

prefill_logits = prefill_output.logits
cache = prefill_output.past_key_values

print("\n=== Prefill output ===")
print("logits shape:", tuple(prefill_logits.shape))
print("cache class:", type(cache).__name__)

first_token_id = prefill_logits[:, -1, :].argmax(
    dim=-1,
    keepdim=True,
)

print("first token shape:", tuple(first_token_id.shape))
print("first token ID:", first_token_id.item())
print(
    "first token text:",
    repr(processor.decode(first_token_id[0])),
)

show_cache(
    cache,
    layer_types,
    title="AFTER PREFILL",
)

# 保存 Prefill 最后位置的完整 logits。
# detach：脱离计算图。
# cpu：从 GPU 搬到 CPU。
# contiguous：确保保存时内存连续。
prefill_last_logits_golden = (
    prefill_output.logits[0, -1]
    .detach()
    .cpu()
    .contiguous()
)

# 必须在 Decode 前记录。
# 因为 Decode 会原地修改同一个 cache 对象。
prefill_cache_metadata = cache_metadata(
    cache,
    layer_types,
)

# 在 Decode 前保存 GDN Tensor 的显存地址。
# 后面用它判断状态是否被原地更新。
gdn_pointers_before_decode = {}

for layer_idx, layer_type in enumerate(layer_types):
    if layer_type != "linear_attention":
        continue

    cache_layer = cache.layers[layer_idx]

    gdn_pointers_before_decode[layer_idx] = {
        "conv": cache_layer.conv_states[0].data_ptr(),
        "recurrent": cache_layer.recurrent_states[0].data_ptr(),
    }


# ============================================================
# 4. 手动执行一次 Decode
# ============================================================

# Prefill 的 attention_mask 长度是 17。
# 现在又向模型输入第一个生成 token，因此总上下文长度变成 18。
decode_attention_mask = torch.cat(
    [
        inputs["attention_mask"],
        torch.ones(
            (inputs["attention_mask"].shape[0], 1),
            dtype=inputs["attention_mask"].dtype,
            device=DEVICE,
        ),
    ],
    dim=1,
)

print("\n=== Decode input ===")
print("decode input_ids:", first_token_id.tolist())
print(
    "decode input_ids shape:",
    tuple(first_token_id.shape),
)
print(
    "decode attention_mask shape:",
    tuple(decode_attention_mask.shape),
)

with torch.inference_mode():
    decode_output = model(
        # Decode 只输入第一个生成 token
        input_ids=first_token_id,

        # mask 描述完整的 18-token 上下文
        attention_mask=decode_attention_mask,

        # 复用 Prefill 得到的混合缓存
        past_key_values=cache,

        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )

second_token_id = decode_output.logits[:, -1, :].argmax(
    dim=-1,
    keepdim=True,
)

print("\n=== Decode output ===")
print("logits shape:", tuple(decode_output.logits.shape))
print("second token ID:", second_token_id.item())
print(
    "second token text:",
    repr(processor.decode(second_token_id[0])),
)

print(
    "cache object reused:",
    cache is decode_output.past_key_values,
)

show_cache(
    decode_output.past_key_values,
    layer_types,
    title="AFTER ONE DECODE",
)

decode_last_logits_golden = (
    decode_output.logits[0, -1]
    .detach()
    .cpu()
    .contiguous()
)

decode_cache_metadata = cache_metadata(
    decode_output.past_key_values,
    layer_types,
)

# ============================================================
# 5. 检查 GDN 状态是否原地更新
# ============================================================

all_gdn_addresses_unchanged = True

for layer_idx, old_pointers in gdn_pointers_before_decode.items():
    cache_layer = cache.layers[layer_idx]

    new_conv_pointer = cache_layer.conv_states[0].data_ptr()
    new_recurrent_pointer = cache_layer.recurrent_states[0].data_ptr()

    conv_same = new_conv_pointer == old_pointers["conv"]
    recurrent_same = (
        new_recurrent_pointer == old_pointers["recurrent"]
    )

    print(
        f"layer {layer_idx:02d}",
        f"conv address unchanged={conv_same}",
        f"recurrent address unchanged={recurrent_same}",
    )

    all_gdn_addresses_unchanged &= conv_same
    all_gdn_addresses_unchanged &= recurrent_same

print(
    "all GDN states updated in-place:",
    all_gdn_addresses_unchanged,
)


# ============================================================
# 6. 与之前保存的 golden token 对齐
# ============================================================

with open(
    "artifacts/golden/qwen35_08b_text_greedy.json",
    "r",
    encoding="utf-8",
) as file:
    golden = json.load(file)

golden_ids = golden["generated_token_ids"]

assert first_token_id.item() == golden_ids[0]
assert second_token_id.item() == golden_ids[1]

print("\nGolden comparison: PASS")
print("manual first token:", first_token_id.item())
print("golden first token:", golden_ids[0])
print("manual second token:", second_token_id.item())
print("golden second token:", golden_ids[1])

golden_directory = Path("artifacts/golden")
golden_directory.mkdir(
    parents=True,
    exist_ok=True,
)

logits_path = (
    golden_directory
    / "qwen35_08b_prefill_decode_logits.safetensors"
)

save_file(
    {
        "prefill_last_logits": (
            prefill_last_logits_golden
        ),
        "decode_last_logits": (
            decode_last_logits_golden
        ),
    },
    str(logits_path),
)


summary = {
    "model_id": MODEL_ID,
    "checkpoint_revision": getattr(
        model.config,
        "_commit_hash",
        None,
    ),
    "dtype": str(model.dtype),
    "prompt": PROMPT,
    "prompt_token_ids": (
        inputs["input_ids"][0].tolist()
    ),
    "prefill": {
        "input_shape": list(
            inputs["input_ids"].shape
        ),
        "logits_shape": list(
            prefill_output.logits.shape
        ),
        "logits": make_topk_summary(
            prefill_last_logits_golden,
            processor,
        ),
        "cache": prefill_cache_metadata,
    },
    "decode": {
        "input_token_id": first_token_id.item(),
        "input_shape": list(first_token_id.shape),
        "logits_shape": list(
            decode_output.logits.shape
        ),
        "logits": make_topk_summary(
            decode_last_logits_golden,
            processor,
        ),
        "cache": decode_cache_metadata,
    },
}

summary_path = (
    golden_directory
    / "qwen35_08b_prefill_decode_summary.json"
)

with summary_path.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        ensure_ascii=False,
        indent=2,
    )

print("\n=== Golden artifacts ===")
print("full logits:", logits_path)
print("readable summary:", summary_path)