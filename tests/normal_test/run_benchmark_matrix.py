import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RESULT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "bench"
    / "quick"
)

SUMMARY_CSV = (
    RESULT_DIR
    / "summary.csv"
)

SUMMARY_MD = (
    RESULT_DIR
    / "summary.md"
)


CASES = [
    {
        "name": "text_p128_o64_c1_b2048",
        "workload": "text",
        "prompt_tokens": 128,
        "image_size": 512,
        "output_tokens": 64,
        "concurrency": 1,
        "token_budget": 2048,
    },
    {
        "name": "text_p2048_o64_c4_b2048",
        "workload": "text",
        "prompt_tokens": 2048,
        "image_size": 512,
        "output_tokens": 64,
        "concurrency": 4,
        "token_budget": 2048,
    },
    {
        "name": "image_s512_o64_c1_b2048",
        "workload": "image",
        "prompt_tokens": 128,
        "image_size": 512,
        "output_tokens": 64,
        "concurrency": 1,
        "token_budget": 2048,
    },
    {
        "name": "image_s1024_o64_c4_b512",
        "workload": "image",
        "prompt_tokens": 128,
        "image_size": 1024,
        "output_tokens": 64,
        "concurrency": 4,
        "token_budget": 512,
    },
    {
        "name": "mixed_p128_s512_o64_c4_b512",
        "workload": "mixed",
        "prompt_tokens": 128,
        "image_size": 512,
        "output_tokens": 64,
        "concurrency": 4,
        "token_budget": 512,
    },
    {
        "name": "mixed_p2048_s1024_o64_c8_b2048",
        "workload": "mixed",
        "prompt_tokens": 2048,
        "image_size": 1024,
        "output_tokens": 64,
        "concurrency": 8,
        "token_budget": 2048,
    },
]


SUMMARY_FIELDS = [
    "name",
    "workload",
    "prompt_tokens",
    "image_size",
    "output_tokens",
    "concurrency",
    "token_budget",
    "requests_per_second",
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "e2e_p95_ms",
    "queue_p95_ms",
    "num_preemptions",
    "num_recomputed_tokens",
    "vision_forwards",
    "visual_cache_peak_gib",
    "cuda_peak_allocated_gib",
]


def run_case(case: dict) -> Path:
    output_path = (
        RESULT_DIR
        / f"{case['name']}.json"
    )

    command = [
        sys.executable,
        "tests/benchmark_hybrid.py",

        "--workload",
        case["workload"],

        "--prompt-tokens",
        str(case["prompt_tokens"]),

        "--image-size",
        str(case["image_size"]),

        "--output-tokens",
        str(case["output_tokens"]),

        "--concurrency",
        str(case["concurrency"]),

        "--token-budget",
        str(case["token_budget"]),

        "--output",
        str(output_path),
    ]

    print(
        "\n================================"
    )
    print("Running:", case["name"])
    print("================================")

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    return output_path


def load_result(
    case: dict,
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        result = json.load(file)

    config = result["config"]
    throughput = result["throughput"]
    latency = result["latency_ms"]
    scheduler = result["scheduler"]
    vision = result["vision"]
    memory = result["memory_gib"]

    return {
        "name": case["name"],
        "workload": config["workload"],
        "prompt_tokens": (
            config["prompt_tokens"]
        ),
        "image_size": (
            config["image_size"]
        ),
        "output_tokens": (
            config["output_tokens"]
        ),
        "concurrency": (
            config["concurrency"]
        ),
        "token_budget": (
            config["token_budget"]
        ),
        "requests_per_second": (
            throughput[
                "requests_per_second"
            ]
        ),
        "prefill_tokens_per_second": (
            throughput[
                "prefill_tokens_per_second"
            ]
        ),
        "decode_tokens_per_second": (
            throughput[
                "decode_tokens_per_second"
            ]
        ),
        "ttft_p50_ms": (
            latency["ttft"]["p50"]
        ),
        "ttft_p95_ms": (
            latency["ttft"]["p95"]
        ),
        "tpot_p50_ms": (
            latency["tpot"]["p50"]
        ),
        "tpot_p95_ms": (
            latency["tpot"]["p95"]
        ),
        "e2e_p95_ms": (
            latency["e2e"]["p95"]
        ),
        "queue_p95_ms": (
            latency["queue"]["p95"]
        ),
        "num_preemptions": (
            scheduler["num_preemptions"]
        ),
        "num_recomputed_tokens": (
            scheduler[
                "num_recomputed_tokens"
            ]
        ),
        "vision_forwards": (
            vision["num_vision_forwards"]
        ),
        "visual_cache_peak_gib": (
            memory[
                "visual_cache_peak_gib"
            ]
        ),
        "cuda_peak_allocated_gib": (
            memory[
                "cuda_peak_allocated_gib"
            ]
        ),
    }


def format_number(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"

    return str(value)


def write_csv(
    rows: list[dict],
) -> None:
    with SUMMARY_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    rows: list[dict],
) -> None:
    markdown_columns = [
        ("Case", "name"),
        ("Req/s", "requests_per_second"),
        (
            "Prefill tok/s",
            "prefill_tokens_per_second",
        ),
        (
            "Decode tok/s",
            "decode_tokens_per_second",
        ),
        ("TTFT p50", "ttft_p50_ms"),
        ("TTFT p95", "ttft_p95_ms"),
        ("TPOT p50", "tpot_p50_ms"),
        ("TPOT p95", "tpot_p95_ms"),
        ("E2E p95", "e2e_p95_ms"),
        (
            "Peak GPU GiB",
            "cuda_peak_allocated_gib",
        ),
    ]

    lines = []

    lines.append(
        "| "
        + " | ".join(
            title
            for title, _
            in markdown_columns
        )
        + " |"
    )

    lines.append(
        "| "
        + " | ".join(
            "---"
            for _ in markdown_columns
        )
        + " |"
    )

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                format_number(
                    row[field]
                )
                for _, field
                in markdown_columns
            )
            + " |"
        )

    SUMMARY_MD.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for case in CASES:
        result_path = run_case(case)

        row = load_result(
            case,
            result_path,
        )

        rows.append(row)

    write_csv(rows)
    write_markdown(rows)

    print("\nBenchmark matrix complete.")
    print("CSV:", SUMMARY_CSV)
    print("Markdown:", SUMMARY_MD)


if __name__ == "__main__":
    main()