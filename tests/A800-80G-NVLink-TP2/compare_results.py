from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if lhs != rhs:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def compare_output_groups(
    name: str,
    reference: list[list[int]],
    candidate: list[list[int]],
    mismatches: list[dict[str, Any]],
) -> None:
    if len(reference) != len(candidate):
        mismatches.append(
            {
                "name": name,
                "reason": "request_count",
                "reference": len(reference),
                "candidate": len(candidate),
            }
        )
        return

    for request_index, (left, right) in enumerate(zip(reference, candidate)):
        difference = first_difference(left, right)
        if difference is not None:
            mismatches.append(
                {
                    "name": name,
                    "request_index": request_index,
                    "first_different_token": difference,
                    "reference_length": len(left),
                    "candidate_length": len(right),
                    "reference_token": left[difference] if difference < len(left) else None,
                    "candidate_token": right[difference] if difference < len(right) else None,
                }
            )


def main() -> None:
    args = parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8-sig"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8-sig"))
    mismatches: list[dict[str, Any]] = []
    compared: list[str] = []

    common_cases = sorted(set(reference["cases"]) & set(candidate["cases"]))
    if not common_cases:
        raise RuntimeError("No common benchmark cases to compare")

    for case_name in common_cases:
        left = reference["cases"][case_name]["iterations"][0]["outputs"]
        right = candidate["cases"][case_name]["iterations"][0]["outputs"]
        compare_output_groups(case_name, left, right, mismatches)
        compared.append(case_name)

    if "dynamic" in reference and "dynamic" in candidate:
        compare_output_groups(
            "dynamic",
            reference["dynamic"]["outputs"],
            candidate["dynamic"]["outputs"],
            mismatches,
        )
        compared.append("dynamic")

    left_vision = reference.get("vision")
    right_vision = candidate.get("vision")
    if (
        isinstance(left_vision, dict)
        and isinstance(right_vision, dict)
        and not left_vision.get("skipped", False)
        and not right_vision.get("skipped", False)
    ):
        compare_output_groups(
            "vision",
            left_vision["outputs"],
            right_vision["outputs"],
            mismatches,
        )
        compared.append("vision")

    payload = {
        "label": args.label,
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "compared": compared,
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
