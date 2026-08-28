import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    
    path = "/workspace/models/Qwen3.5-9B"
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=3,
        num_state_slots=3,
        gpu_memory_utilization=0.9,
    )
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
    
    prompts = [
        "你好。",
        "请列出三个质数，只要结果。",
        "请用一句话简单解释什么是线性注意力。",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
