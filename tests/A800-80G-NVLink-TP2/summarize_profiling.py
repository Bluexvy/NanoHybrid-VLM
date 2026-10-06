from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiling-dir", type=Path, required=True)
    return parser.parse_args()


def read_optional(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip() if path.exists() else "not available"


def file_rows(root: Path, suffix: str) -> list[dict[str, object]]:
    return [
        {
            "name": path.name,
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "mib": path.stat().st_size / 1024**2,
        }
        for path in sorted(root.rglob(f"*{suffix}"))
    ]


def main() -> None:
    args = parse_args()
    root = args.profiling_dir
    nsys_reports = file_rows(root, ".nsys-rep")
    ncu_reports = file_rows(root, ".ncu-rep")
    permission_blocked = (root / "ncu" / "PERMISSION_BLOCKED.txt").exists()
    payload = {
        "nsys_version": read_optional(root / "nsys" / "version.txt"),
        "ncu_version": read_optional(root / "ncu" / "version.txt"),
        "nsys_status": read_optional(root / "nsys" / "status.tsv"),
        "ncu_status": read_optional(root / "ncu" / "status.tsv"),
        "ncu_permission_blocked": permission_blocked,
        "nsys_reports": nsys_reports,
        "ncu_reports": ncu_reports,
    }
    lines = [
        "# A800 Nsight Profiling 产物清单",
        "",
        "## 工具版本",
        "",
        "```text",
        payload["nsys_version"],
        payload["ncu_version"],
        "```",
        "",
        "## Nsight Systems",
        "",
        "`.nsys-rep` 下载到本机后使用 Nsight Systems GUI 打开。GUI 版本必须不旧于生成报告的 CLI 版本。",
        "",
        "| Report | Size |",
        "|---|---:|",
    ]
    for item in nsys_reports:
        lines.append(f"| `{item['relative_path']}` | {item['mib']:.2f} MiB |")
    if not nsys_reports:
        lines.append("| 未生成 | - |")

    lines += [
        "",
        "## Nsight Compute",
        "",
        "`.ncu-rep` 使用 Nsight Compute GUI 打开；`*_details.csv` 可直接用于表格分析。",
        "",
        f"- GPU Performance Counter 权限：**{'被宿主机阻止' if permission_blocked else '未检测到权限阻止'}**",
        "",
        "| Report | Size |",
        "|---|---:|",
    ]
    for item in ncu_reports:
        lines.append(f"| `{item['relative_path']}` | {item['mib']:.2f} MiB |")
    if not ncu_reports:
        lines.append("| 未生成 | - |")

    lines += [
        "",
        "## 阶段状态",
        "",
        "### nsys",
        "",
        "```text",
        payload["nsys_status"],
        "```",
        "",
        "### ncu",
        "",
        "```text",
        payload["ncu_status"],
        "```",
        "",
    ]
    root.mkdir(parents=True, exist_ok=True)
    (root / "profiling_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "profiling_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((root / "profiling_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

