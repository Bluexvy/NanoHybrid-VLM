import json
from pathlib import Path


FULL_PATH = Path(
    "artifacts/equivalence/"
    "qwen35_9b_vision_full.json"
)

CHUNKED_PATH = Path(
    "artifacts/equivalence/"
    "qwen35_9b_vision_chunked.json"
)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    full = load_json(FULL_PATH)
    chunked = load_json(CHUNKED_PATH)

    full_ids = full["generated_token_ids"]
    chunked_ids = chunked[
        "generated_token_ids"
    ]

    if full_ids != chunked_ids:
        common_length = min(
            len(full_ids),
            len(chunked_ids),
        )

        first_mismatch = None

        for index in range(common_length):
            if (
                full_ids[index]
                != chunked_ids[index]
            ):
                first_mismatch = index
                break

        raise AssertionError(
            "Full and Chunked Prefill generated "
            "different tokens. "
            f"First mismatch: {first_mismatch}; "
            f"full={full_ids[first_mismatch]}; "
            f"chunked={chunked_ids[first_mismatch]}"
        )

    if full["vision_forwards"] != 1:
        raise AssertionError(
            "Full Prefill must run Vision once"
        )

    if chunked["vision_forwards"] != 1:
        raise AssertionError(
            "Chunked Prefill must run Vision once"
        )

    if chunked["visual_cache_hits"] < 1:
        raise AssertionError(
            "Chunked Prefill must reuse the "
            "visual embedding cache"
        )

    if (
        full["current_visual_cache_bytes"] != 0
        or chunked[
            "current_visual_cache_bytes"
        ] != 0
    ):
        raise AssertionError(
            "Visual cache was not released"
        )

    print(
        "Part 17A passed: Full and Chunked "
        "Prefill generated identical tokens."
    )

    print(
        "compared tokens:",
        len(full_ids),
    )

    print(
        "full prefill microbatches:",
        full["prefill_microbatches"],
    )

    print(
        "chunked prefill microbatches:",
        chunked["prefill_microbatches"],
    )

    print(
        "chunked visual cache hits:",
        chunked["visual_cache_hits"],
    )


if __name__ == "__main__":
    main()