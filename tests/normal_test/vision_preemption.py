import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from nanovllm import LLM, SamplingParams


MODEL_PATH = "/workspace/models/Qwen3.5-9B"

BASELINE_PATH = Path(
    "artifacts/equivalence/"
    "qwen35_9b_vision_full.json"
)

IMAGE_PROMPT = (
    "请描述图片中的颜色和形状，"
    "只用一句话简洁回答。"
)


def create_test_image() -> Image.Image:
    image = Image.new(
        mode="RGB",
        size=(320, 320),
        color="white",
    )

    draw = ImageDraw.Draw(image)

    draw.rectangle(
        xy=(40, 100, 140, 200),
        fill="red",
    )

    draw.ellipse(
        xy=(180, 100, 280, 200),
        fill="blue",
    )

    return image


def load_baseline_token_ids() -> list[int]:
    with BASELINE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        baseline = json.load(file)

    return baseline[
        "generated_token_ids"
    ]


def collect_finished_outputs(
    completed: dict[int, list[int]],
    outputs: list[
        tuple[int, list[int]]
    ],
) -> None:
    for seq_id, token_ids in outputs:
        completed[seq_id] = token_ids


def main() -> None:
    baseline_token_ids = (
        load_baseline_token_ids()
    )

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,

        max_model_len=1024,
        max_num_batched_tokens=512,

        # 同时只能有一个活跃请求。
        # 因此 B 想运行，就必须抢占 A。
        max_num_seqs=1,
        num_state_slots=1,

        # 等待超过10ms后触发 Prefill 保留机制。
        max_prefill_wait_ms=10.0,

        gpu_memory_utilization=0.8,
    )

    image_request = {
        "prompt": IMAGE_PROMPT,
        "multi_modal_data": {
            "image": create_test_image(),
        },
    }

    image_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=64,
    )

    # =====================================
    # 第一阶段：A 完成第一次 Prefill
    # =====================================

    image_seq_id = llm.add_request(
        image_request,
        image_sampling,
    )

    completed: dict[
        int,
        list[int],
    ] = {}

    outputs, _ = llm.step()

    collect_finished_outputs(
        completed,
        outputs,
    )

    if image_seq_id in completed:
        raise AssertionError(
            "Image request must remain active "
            "after its first Prefill"
        )

    runner = llm.model_runner
    scheduler = llm.scheduler

    if (
        image_seq_id
        not in runner.visual_embedding_cache
    ):
        raise AssertionError(
            "Image request did not create "
            "a visual cache"
        )

    if runner.visual_cache_bytes <= 0:
        raise AssertionError(
            "Visual cache must be active "
            "after image Prefill"
        )

    print("\nA 首次 Prefill 后：")
    print(
        "visual cache bytes:",
        runner.visual_cache_bytes,
    )
    print(
        "vision forwards:",
        runner.num_vision_forwards,
    )
    print(
        "preemptions:",
        scheduler.num_preemptions,
    )

    # =====================================
    # 第二阶段：加入只生成一个 token 的 B
    # =====================================

    text_prompt = (
        llm.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "1加1等于多少？"
                        "只回答结果。"
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    text_seq_id = llm.add_request(
        text_prompt,
        SamplingParams(
            temperature=0.0,

            # B 在 Prefill 采样出首个 token 后
            # 立即结束，不会持续占用资源。
            max_tokens=1,
        ),
    )

    # 让 B 的等待时间超过10ms。
    time.sleep(0.05)

    outputs, _ = llm.step()

    collect_finished_outputs(
        completed,
        outputs,
    )

    # 这一轮 B 为了获得唯一的 active slot，
    # 必须抢占 A。
    if scheduler.num_preemptions != 1:
        raise AssertionError(
            "Expected exactly one preemption, "
            f"got {scheduler.num_preemptions}"
        )

    if text_seq_id not in completed:
        raise AssertionError(
            "The one-token text request "
            "must finish immediately"
        )

    # Engine 收到 SchedulePlan.preempted_seq_ids
    # 后，应当通知 ModelRunner 删除 A 的视觉缓存。
    if (
        image_seq_id
        in runner.visual_embedding_cache
    ):
        raise AssertionError(
            "Preempted image request still "
            "owns a visual cache"
        )

    if runner.visual_cache_bytes != 0:
        raise AssertionError(
            "Visual cache bytes must return "
            "to zero after preemption"
        )

    print("\nA 被抢占后：")
    print(
        "preemptions:",
        scheduler.num_preemptions,
    )
    print(
        "recomputed tokens:",
        scheduler.num_recomputed_tokens,
    )
    print(
        "visual cache bytes:",
        runner.visual_cache_bytes,
    )

    # =====================================
    # 第三阶段：继续运行，A 从头重算
    # =====================================

    while not llm.is_finished():
        outputs, _ = llm.step()

        collect_finished_outputs(
            completed,
            outputs,
        )

    if image_seq_id not in completed:
        raise AssertionError(
            "Image request did not finish"
        )

    actual_token_ids = completed[
        image_seq_id
    ]

    if (
        actual_token_ids
        != baseline_token_ids
    ):
        common_length = min(
            len(actual_token_ids),
            len(baseline_token_ids),
        )

        first_mismatch = None

        for index in range(
            common_length
        ):
            if (
                actual_token_ids[index]
                != baseline_token_ids[index]
            ):
                first_mismatch = index
                break

        raise AssertionError(
            "Preemption changed greedy output. "
            f"First mismatch: {first_mismatch}"
        )

    # A 首次执行一次 Vision；
    # 抢占释放后，重新 Prefill 再执行一次。
    if runner.num_vision_forwards != 2:
        raise AssertionError(
            "Expected two Vision forwards, "
            f"got {runner.num_vision_forwards}"
        )

    if runner.num_visual_cache_misses != 2:
        raise AssertionError(
            "Expected two visual cache misses, "
            f"got "
            f"{runner.num_visual_cache_misses}"
        )

    if runner.visual_cache_bytes != 0:
        raise AssertionError(
            "Visual cache leaked after completion"
        )

    print(
        "\nPart 18 passed: preemption and "
        "deterministic recomputation are correct."
    )

    print(
        "compared image tokens:",
        len(actual_token_ids),
    )

    print(
        "preemptions:",
        scheduler.num_preemptions,
    )

    print(
        "recomputed tokens:",
        scheduler.num_recomputed_tokens,
    )

    print(
        "vision forwards:",
        runner.num_vision_forwards,
    )

    print(
        "visual cache misses:",
        runner.num_visual_cache_misses,
    )

    print(
        "final visual cache bytes:",
        runner.visual_cache_bytes,
    )


if __name__ == "__main__":
    main()