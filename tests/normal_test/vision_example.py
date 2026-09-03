from PIL import Image, ImageDraw

from nanovllm import LLM, SamplingParams


MODEL_PATH = "/workspace/models/Qwen3.5-9B"


def create_test_image() -> Image.Image:
    """
    创建一张内容确定的测试图片：

    - 白色背景
    - 左侧红色正方形
    - 右侧蓝色圆形
    """

    image = Image.new(
        mode="RGB",
        size=(320, 320),
        color="white",
    )

    draw = ImageDraw.Draw(image)

    # 红色正方形
    draw.rectangle(
        xy=(40, 100, 140, 200),
        fill="red",
    )

    # 蓝色圆形
    draw.ellipse(
        xy=(180, 100, 280, 200),
        fill="blue",
    )

    return image


def main() -> None:
    image = create_test_image()

    llm = LLM(
        MODEL_PATH,

        # Qwen3.5 Hybrid Runtime 暂时只支持 eager。
        enforce_eager=True,

        # 当前首版使用单卡。
        tensor_parallel_size=1,

        # 这张测试图的视觉 token 数不多，
        # 1024 足够容纳图文 prompt 和输出。
        max_model_len=1024,

        # 第一次先让整个图文 prompt 尽量在
        # 一个 Prefill microbatch 中处理。
        max_num_batched_tokens=64,

        # 当前只有一个请求。
        max_num_seqs=1,

        # 当前只有一个活跃请求，
        # 只需要一个 GDN state slot。
        num_state_slots=1,

        # 给 Vision Tower 的临时激活多留一些显存。
        gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(
        # Greedy decoding，便于重复运行和 HF 对齐。
        temperature=0.0,
        max_tokens=512,
    )

    prompts = [
        {
            "prompt": (
                "请描述图片中的颜色和形状具体是什么，"
                "只用一句话简洁回答。"
            ),
            "multi_modal_data": {
                "image": image,
            },
        },
    ]

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    print("\n生成结果：")
    print(outputs[0]["text"])

    print("\n生成 token IDs：")
    print(outputs[0]["token_ids"])

    runner = llm.model_runner

    print("\n视觉缓存统计：")
    print(
        "current bytes:",
        runner.visual_cache_bytes,
    )
    print(
        "peak bytes:",
        runner.peak_visual_cache_bytes,
    )
    print(
        "vision forwards:",
        runner.num_vision_forwards,
    )

    print(
        "cache misses:",
        runner.num_visual_cache_misses,
    )

    print(
        "cache hits:",
        runner.num_visual_cache_hits,
    )

if __name__ == "__main__":
    main()