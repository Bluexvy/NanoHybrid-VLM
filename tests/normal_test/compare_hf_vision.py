import json
from pathlib import Path


NANO_PATH = Path(
    "artifacts/equivalence/"
    "qwen35_9b_vision_full.json"
)

HF_PATH = Path(
    "artifacts/golden/"
    "qwen35_9b_vision_hf.json"
)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def find_first_mismatch(
    left: list[int],
    right: list[int],
) -> int | None:
    common_length = min(
        len(left),
        len(right),
    )

    for index in range(
        common_length
    ):
        if left[index] != right[index]:
            return index

    if len(left) != len(right):
        return common_length

    return None


def main() -> None:
    nano = load_json(NANO_PATH)
    hf = load_json(HF_PATH)

    nano_prompt = nano[
        "prompt_token_ids"
    ]

    hf_prompt = hf[
        "prompt_token_ids"
    ]

    prompt_mismatch = (
        find_first_mismatch(
            nano_prompt,
            hf_prompt,
        )
    )

    if prompt_mismatch is not None:
        raise AssertionError(
            "Nano and HF preprocessing "
            "produced different prompt tokens. "
            f"First mismatch: "
            f"{prompt_mismatch}"
        )

    print(
        "Prompt token IDs match:",
        len(nano_prompt),
        "tokens",
    )

    nano_generated = nano[
        "generated_token_ids"
    ]

    hf_generated = hf[
        "generated_token_ids"
    ]

    generation_mismatch = (
        find_first_mismatch(
            nano_generated,
            hf_generated,
        )
    )

    if generation_mismatch is not None:
        nano_token = (
            nano_generated[
                generation_mismatch
            ]
            if generation_mismatch
            < len(nano_generated)
            else None
        )

        hf_token = (
            hf_generated[
                generation_mismatch
            ]
            if generation_mismatch
            < len(hf_generated)
            else None
        )

        raise AssertionError(
            "Nano and HF greedy generation "
            "diverged. "
            f"First mismatch step: "
            f"{generation_mismatch}; "
            f"nano token: {nano_token}; "
            f"HF token: {hf_token}"
        )

    print(
        "Part 17B passed: Nano and HF "
        "generated identical greedy tokens."
    )

    print(
        "Compared generated tokens:",
        len(nano_generated),
    )


if __name__ == "__main__":
    main()