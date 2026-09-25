from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def load_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def mean_case_metric(case: dict[str, Any], *keys: str) -> float:
    values: list[float] = []
    for iteration in case["iterations"]:
        value: Any = iteration
        for key in keys:
            value = value[key]
        values.append(float(value))
    return statistics.mean(values)


def format_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}x"


def comparison_status(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "未运行"
    return "通过" if payload.get("passed") else "失败"


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    preflight = load_optional(result_dir / "preflight.json")
    nccl = load_optional(result_dir / "nccl_smoke.json")
    tp1 = load_optional(result_dir / "tp1_eager_cuda.json")
    tp2 = load_optional(result_dir / "tp2_eager_cuda.json")
    graph = load_optional(result_dir / "tp2_graph_cuda.json")
    prefix = load_optional(result_dir / "prefix_tp2.json")
    compare_tp = load_optional(result_dir / "compare_tp1_tp2.json")
    compare_graph = load_optional(result_dir / "compare_eager_graph.json")

    lines = ["# NanoHybrid-VLM TP=2 测试报告", ""]

    if preflight is not None:
        lines.extend(
            [
                "## 环境",
                "",
                f"- PyTorch：`{preflight['torch']}`",
                f"- CUDA：`{preflight['torch_cuda']}`",
                f"- NCCL：`{preflight['nccl_version']}`",
                f"- 可见 GPU：`{preflight['visible_device_count']}`",
            ]
        )
        for device in preflight["devices"]:
            lines.append(
                f"- GPU {device['index']}：{device['name']}，"
                f"空闲 {device['free_memory_gib']:.2f} GiB / "
                f"总计 {device['total_memory_gib']:.2f} GiB"
            )
        lines.extend(
            [
                f"- P2P：`{preflight['peer_access']}`",
                f"- 模型维度：`{preflight['model_geometry']['tp_divisibility']}`",
                "",
            ]
        )

    if nccl is not None:
        lines.extend(
            [
                "## NCCL AllReduce",
                "",
                "| Payload | 单次平均延迟 | 64 次估算 |",
                "|---:|---:|---:|",
            ]
        )
        for case in nccl["cases"].values():
            lines.append(
                f"| {case['payload_bytes'] / 1024:.0f} KiB "
                f"| {case['mean_allreduce_us']:.2f} us "
                f"| {case['estimated_64_allreduces_ms']:.3f} ms |"
            )
        lines.append("")

    prefix_status = "通过" if prefix and prefix.get("passed") else "未通过或未运行"
    lines.extend(
        [
            "## 正确性",
            "",
            f"- TP1 Eager vs TP2 Eager：**{comparison_status(compare_tp)}**",
            f"- TP2 Eager vs TP2 CUDA Graph：**{comparison_status(compare_graph)}**",
            f"- TP2 联合 Prefix Cache：**{prefix_status}**",
            "",
        ]
    )

    summary_cases: dict[str, Any] = {}
    if tp1 is not None or tp2 is not None or graph is not None:
        all_case_names: set[str] = set()
        for payload in (tp1, tp2, graph):
            if payload is not None:
                all_case_names.update(payload["cases"])

        lines.extend(
            [
                "## 性能",
                "",
                "| Case | TP1 Eager tok/s | TP2 Eager tok/s | TP 加速比 | TP2 Graph tok/s | Graph 加速比 | TP1 TTFT ms | TP2 TTFT ms |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for case_name in sorted(all_case_names):
            tp1_case = tp1["cases"].get(case_name) if tp1 else None
            tp2_case = tp2["cases"].get(case_name) if tp2 else None
            graph_case = graph["cases"].get(case_name) if graph else None

            tp1_rate = (
                mean_case_metric(tp1_case, "decode_tokens_per_second")
                if tp1_case
                else None
            )
            tp2_rate = (
                mean_case_metric(tp2_case, "decode_tokens_per_second")
                if tp2_case
                else None
            )
            graph_rate = (
                mean_case_metric(graph_case, "decode_tokens_per_second")
                if graph_case
                else None
            )
            tp_speedup = tp2_rate / tp1_rate if tp1_rate and tp2_rate else None
            graph_speedup = graph_rate / tp2_rate if tp2_rate and graph_rate else None
            tp1_ttft = (
                mean_case_metric(tp1_case, "ttft_ms", "mean")
                if tp1_case
                else None
            )
            tp2_ttft = (
                mean_case_metric(tp2_case, "ttft_ms", "mean")
                if tp2_case
                else None
            )

            lines.append(
                f"| {case_name} "
                f"| {format_number(tp1_rate)} "
                f"| {format_number(tp2_rate)} "
                f"| {format_ratio(tp_speedup)} "
                f"| {format_number(graph_rate)} "
                f"| {format_ratio(graph_speedup)} "
                f"| {format_number(tp1_ttft)} "
                f"| {format_number(tp2_ttft)} |"
            )
            summary_cases[case_name] = {
                "tp1_decode_tokens_per_second": tp1_rate,
                "tp2_decode_tokens_per_second": tp2_rate,
                "tp_speedup": tp_speedup,
                "tp2_graph_decode_tokens_per_second": graph_rate,
                "graph_speedup": graph_speedup,
                "tp1_ttft_ms": tp1_ttft,
                "tp2_ttft_ms": tp2_ttft,
            }
        lines.append("")

    if prefix is not None:
        cold_prefill = int(prefix["cold"]["total_prefill_tokens"])
        hot_prefill = int(prefix["hot"]["total_prefill_tokens"])
        lines.extend(
            [
                "## TP2 Prefix Cache",
                "",
                f"- PrefixKey 边界：{prefix['prefix_key']['num_cached_tokens']} tokens",
                f"- Cold Prefill：{cold_prefill} tokens",
                f"- Hot Prefill：{hot_prefill} tokens",
                f"- 跳过计算：{cold_prefill - hot_prefill} tokens",
                f"- Rank 0 GDN Snapshot：{prefix['entry']['gdn_snapshot_mib_rank0']:.2f} MiB",
                f"- 命中数：{prefix['after_hot']['num_hits']}",
                f"- GDN 恢复数：{prefix['after_hot']['num_gdn_restores']}",
                "",
            ]
        )

    status_path = result_dir / "status.tsv"
    if status_path.exists():
        lines.extend(["## 阶段状态", "", "```text"])
        lines.extend(status_path.read_text(encoding="utf-8-sig").splitlines())
        lines.extend(["```", ""])

    report_path = result_dir / "summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (result_dir / "summary.json").write_text(
        json.dumps(
            {
                "correctness": {
                    "tp1_vs_tp2": compare_tp,
                    "eager_vs_graph": compare_graph,
                    "prefix_passed": bool(prefix and prefix.get("passed")),
                },
                "cases": summary_cases,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
