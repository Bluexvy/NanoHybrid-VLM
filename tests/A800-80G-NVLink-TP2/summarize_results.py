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


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def mean_metric(case: dict[str, Any], *keys: str) -> float:
    values: list[float] = []
    for iteration in case["iterations"]:
        value: Any = iteration
        for key in keys:
            value = value[key]
        values.append(float(value))
    return statistics.mean(values)


def pct(ratio: float | None) -> str:
    return "-" if ratio is None else f"{(ratio - 1.0) * 100.0:+.2f}%"


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def case_value(payload: dict[str, Any] | None, name: str, *keys: str) -> float | None:
    if payload is None or name not in payload.get("cases", {}):
        return None
    return mean_metric(payload["cases"][name], *keys)


def extrema_text(values: list[float], suffix: str = "%") -> str:
    if not values:
        return "未生成"
    return f"{min(values):.2f}{suffix}～{max(values):.2f}{suffix}"


def main() -> None:
    args = parse_args()
    root = args.result_dir
    preflight = load(root / "preflight.json")
    nccl = load(root / "nccl_stress.json")
    tp1 = load(root / "headline_tp1_eager_cuda.json")
    tp2 = load(root / "headline_tp2_eager_cuda.json")
    fla = load(root / "headline_tp2_eager_fla.json")
    graph = load(root / "headline_tp2_graph_cuda.json")
    long_context = load(root / "long_context_tp2_eager.json")
    prefix = load(root / "prefix_64k_48k_tp2.json")
    soak = load(root / "soak_tp2_graph.json")
    capacity = load(root / "capacity" / "capacity_sweep.json")

    lines = ["# 2×A800 80GB NVLink TP=2 超负载测试报告", ""]
    summary: dict[str, Any] = {"headline": {}, "resume_candidates": {}}

    if preflight:
        lines += [
            "## 环境与链路证据",
            "",
            f"- PyTorch / CUDA / NCCL：`{preflight['torch']}` / `{preflight['torch_cuda']}` / `{preflight['nccl_version']}`",
            f"- NVLink 检测：**{'PASS' if preflight['nvlink_pair_detected'] else 'FAIL'}**",
            f"- P2P：`{preflight['peer_access']}`",
            f"- 模型最大上下文：`{preflight['model_geometry'].get('max_position_embeddings')}`",
        ]
        for device in preflight["devices"][:2]:
            lines.append(
                f"- GPU {device['index']}：{device['name']}，"
                f"{device['free_memory_gib']:.2f}/{device['total_memory_gib']:.2f} GiB free/total"
            )
        lines.append("")

    if nccl:
        lines += [
            "## NVLink / NCCL AllReduce",
            "",
            "| Payload | Latency | Alg BW | Bus BW | 64 layers estimate |",
            "|---:|---:|---:|---:|---:|",
        ]
        for item in nccl["cases"].values():
            size = item["payload_bytes"]
            size_text = f"{size / 1024:.0f} KiB" if size < 1024**2 else f"{size / 1024**2:.0f} MiB"
            lines.append(
                f"| {size_text} | {item['mean_allreduce_us']:.2f} us | "
                f"{item['algorithm_bandwidth_gbps']:.2f} GB/s | "
                f"{item['bus_bandwidth_gbps']:.2f} GB/s | "
                f"{item['estimated_64_allreduces_ms']:.3f} ms |"
            )
        lines.append("")

    headline_names: set[str] = set()
    for payload in (tp1, tp2, fla, graph):
        if payload:
            headline_names.update(payload.get("cases", {}))
    tp_ratios: list[float] = []
    custom_ratios: list[float] = []
    graph_ratios: list[float] = []
    if headline_names:
        lines += [
            "## 简历主矩阵：大 Batch Decode",
            "",
            "| Case | TP1 Eager | TP2 Eager | TP scaling | TP2 FLA | Custom/FLA | TP2 Graph | Graph/Eager | Graph TPOT |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name in sorted(headline_names):
            tp1_rate = case_value(tp1, name, "decode_tokens_per_second")
            tp2_rate = case_value(tp2, name, "decode_tokens_per_second")
            fla_rate = case_value(fla, name, "decode_tokens_per_second")
            graph_rate = case_value(graph, name, "decode_tokens_per_second")
            graph_tpot = case_value(graph, name, "tpot_ms", "mean")
            tp_ratio = ratio(tp2_rate, tp1_rate)
            custom_ratio = ratio(tp2_rate, fla_rate)
            graph_ratio = ratio(graph_rate, tp2_rate)
            if tp_ratio is not None:
                tp_ratios.append(tp_ratio)
            if custom_ratio is not None:
                custom_ratios.append(custom_ratio)
            if graph_ratio is not None:
                graph_ratios.append(graph_ratio)
            fmt = lambda value: "-" if value is None else f"{value:.2f}"
            lines.append(
                f"| {name} | {fmt(tp1_rate)} | {fmt(tp2_rate)} | {pct(tp_ratio)} | "
                f"{fmt(fla_rate)} | {pct(custom_ratio)} | {fmt(graph_rate)} | "
                f"{pct(graph_ratio)} | {fmt(graph_tpot)} ms |"
            )
            summary["headline"][name] = {
                "tp1_eager_toks": tp1_rate,
                "tp2_eager_toks": tp2_rate,
                "tp_scaling": tp_ratio,
                "tp2_fla_toks": fla_rate,
                "custom_vs_fla": custom_ratio,
                "tp2_graph_toks": graph_rate,
                "graph_vs_eager": graph_ratio,
                "graph_tpot_ms": graph_tpot,
            }
        lines.append("")

    if long_context:
        lines += [
            "## 长上下文吞吐",
            "",
            "| Case | Total prompt tokens | Prefill tok/s | Decode tok/s | TTFT | Peak rank0 allocated |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        memory = long_context.get("rank0_memory_mib", {})
        peak = memory.get("peak_allocated_mib") or memory.get("max_memory_allocated_mib")
        for name, case in long_context["cases"].items():
            prompt = int(case["prompt_tokens"])
            batch = int(case["batch_size"])
            lines.append(
                f"| {name} | {prompt * batch} | "
                f"{mean_metric(case, 'prefill_tokens_per_second'):.2f} | "
                f"{mean_metric(case, 'decode_tokens_per_second'):.2f} | "
                f"{mean_metric(case, 'ttft_ms', 'mean'):.2f} ms | "
                f"{peak if peak is not None else '-'} MiB |"
            )
        lines.append("")

    prefix_reduction: float | None = None
    if prefix:
        cold_ttft = float(prefix["cold"]["ttft_ms"]["mean"])
        hot_runs = prefix.get("hot_runs", [prefix["hot"]])
        hot_ttft = statistics.mean(float(item["ttft_ms"]["mean"]) for item in hot_runs)
        prefix_reduction = (1.0 - hot_ttft / cold_ttft) * 100.0 if cold_ttft else None
        lines += [
            "## 64K / 48K 联合 Prefix Cache",
            "",
            f"- 请求级命中率：{prefix.get('request_hit_rate', 0.5) * 100:.0f}%",
            f"- Cold/Hot 实际 Prefill：{prefix['cold']['total_prefill_tokens']} / {prefix['hot']['total_prefill_tokens']} tokens",
            f"- Cold/Hot TTFT：{cold_ttft:.2f} / {hot_ttft:.2f} ms",
            f"- 平均 Hot TTFT 降低：**{prefix_reduction:.2f}%**" if prefix_reduction is not None else "- TTFT 降低：-",
            f"- GDN state restores：{prefix['after_hot']['num_gdn_restores']}",
            f"- Rank0 GDN snapshot：{prefix['entry']['gdn_snapshot_mib_rank0']:.2f} MiB",
            "",
        ]

    if soak:
        aggregate = soak["aggregate"]
        lines += [
            "## 长稳态混合负载",
            "",
            f"- 请求数 / 生成 token：{soak['total_requests']} / {soak['total_generated_tokens']}",
            f"- Decode throughput mean/min/max：{aggregate['decode_tokens_per_second_mean']:.2f} / {aggregate['decode_tokens_per_second_min']:.2f} / {aggregate['decode_tokens_per_second_max']:.2f} tok/s",
            f"- Throughput CV：{aggregate['throughput_cv'] * 100:.2f}%",
            f"- Mean TPOT / TTFT：{aggregate['tpot_ms_mean']:.2f} / {aggregate['ttft_ms_mean']:.2f} ms",
            f"- Graph replay / eager fallback：{soak['graph']['replays']} / {soak['graph']['eager_fallbacks']}",
            f"- Preemption / recomputed tokens：{soak['scheduler']['num_preemptions']} / {soak['scheduler']['num_recomputed_tokens']}",
            "",
        ]

    if capacity:
        lines += [
            "## 容量与极限边界",
            "",
            "容量项失败或 OOM 是有效边界数据，不计为主矩阵失败。",
            "",
            "| Case | Group | Prompt | Batch | Output | Total prompt | Status | Decode tok/s |",
            "|---|---|---:|---:|---:|---:|---|---:|",
        ]
        for record in capacity["cases"]:
            case_result = load(root / "capacity" / f"{record['name']}.json")
            rate = None
            if case_result and case_result.get("cases"):
                only_case = next(iter(case_result["cases"].values()))
                rate = mean_metric(only_case, "decode_tokens_per_second")
            lines.append(
                f"| {record['name']} | {record.get('group', '-')} | "
                f"{record.get('prompt_tokens', '-')} | {record.get('batch_size', '-')} | "
                f"{record.get('output_tokens', '-')} | {record.get('total_prompt_tokens', '-')} | "
                f"{record['status']} | {'-' if rate is None else f'{rate:.2f}'} |"
            )
        lines.append("")

    graph_improvements = [(value - 1.0) * 100.0 for value in graph_ratios]
    custom_improvements = [(value - 1.0) * 100.0 for value in custom_ratios]
    tp_improvements = [(value - 1.0) * 100.0 for value in tp_ratios]
    resume_lines = [
        "# A800 实验完成后的简历候选数字（仅取自本次产物）",
        "",
        f"- TP1→TP2 Decode 吞吐变化：{extrema_text(tp_improvements)}",
        f"- State-Aware CUDA 相对 FLA Decode 吞吐变化：{extrema_text(custom_improvements)}",
        f"- CUDA Graph 相对 Eager Decode 吞吐变化：{extrema_text(graph_improvements)}",
        f"- 64K/48K Prefix Cache Hot TTFT 降低：{'未生成' if prefix_reduction is None else f'{prefix_reduction:.2f}%'}",
    ]
    if graph and graph.get("cases"):
        graph_rates = [mean_metric(case, "decode_tokens_per_second") for case in graph["cases"].values()]
        graph_tpots = [mean_metric(case, "tpot_ms", "mean") for case in graph["cases"].values()]
        resume_lines += [
            f"- Graph 峰值 Decode：{max(graph_rates):.2f} tok/s",
            f"- Graph 最低平均 TPOT：{min(graph_tpots):.2f} ms",
        ]
    summary["resume_candidates"] = {
        "tp_scaling_percent": tp_improvements,
        "custom_vs_fla_percent": custom_improvements,
        "graph_vs_eager_percent": graph_improvements,
        "prefix_ttft_reduction_percent": prefix_reduction,
    }

    status = root / "status.tsv"
    if status.exists():
        lines += ["## 阶段状态", "", "```text", status.read_text(encoding="utf-8-sig").rstrip(), "```", ""]

    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "resume_candidates.md").write_text("\n".join(resume_lines) + "\n", encoding="utf-8")
    print((root / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
