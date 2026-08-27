import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = (
        "/workspace/.cache/huggingface/hub/"
        "models--Qwen--Qwen3.5-0.8B/"
        "snapshots/"
        "2fc06364715b967f1860aea9cf38778875588b17"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        num_state_slots=4,
    )
    
    sampling_params = SamplingParams(temperature=0.6, max_tokens=2048)
    prompts = [
        "你是什么模型？",
        "写出100以内所有的素数，只要结果。",
        "你是什么模型？",
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
