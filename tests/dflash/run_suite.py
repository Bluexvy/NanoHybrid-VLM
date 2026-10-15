"""Run isolated real-model cases; retain logs/reports even on failure."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from benchmark_utils import compare_outputs, write_json

ROOT = Path(__file__).resolve().parents[2]


def build_plan(batches, drafts, state_checks=True):
    cases = []
    for batch in batches:
        for graph in (False, True):
            cases.append(dict(name=f"b{batch}_baseline_{'graph' if graph else 'eager'}",
                kind="benchmark", method="baseline", graph=graph, batch=batch, drafts=0))
            for count in drafts:
                cases.append(dict(name=f"b{batch}_dflash{count}_{'graph' if graph else 'eager'}",
                    kind="benchmark", method="dflash", graph=graph, batch=batch, drafts=count))
    if state_checks:
        for method in ("baseline", "dflash"):
            cases.append(dict(name=f"{method}_state_checks", kind="state_checks", method=method,
                              graph=method == "dflash", batch=2, drafts=drafts[0] if method == "dflash" else 0))
    return cases


def compare_reports(reports):
    errors, comparisons = [], []
    for report in reports:
        case = report["case"]
        if not report.get("passed"):
            errors.append({"case": case["name"], "error": report.get("error", "worker failed")})
            continue
        if case["kind"] == "state_checks":
            reference = next((r for r in reports if r["case"]["kind"] == "state_checks"
                              and r["case"]["method"] == "baseline" and r.get("passed")), None)
            if case["method"] != "baseline":
                if reference is None:
                    errors.append({"case": case["name"], "error": "missing state-check baseline"})
                else:
                    if len(reference["check_outputs"]) != len(report["check_outputs"]):
                        errors.append({"case": case["name"], "error": "state-check phase counts differ"})
                    for phase, (expected, actual) in enumerate(zip(reference["check_outputs"], report["check_outputs"])):
                        mismatch = compare_outputs(expected, actual)
                        if mismatch: errors.append(dict(mismatch, case=case["name"], phase=phase))
            continue
        baseline = next((r for r in reports if r["case"]["kind"] == "benchmark" and r.get("passed")
            and r["case"]["batch"] == case["batch"] and r["case"]["method"] == "baseline"
            and not r["case"]["graph"]), None)
        if baseline is None:
            errors.append({"case": case["name"], "error": "missing Eager baseline"})
            continue
        if len(report["trials"]) != len(baseline["trials"]):
            errors.append({"case": case["name"], "error": "trial counts differ"})
        for index, (expected, actual) in enumerate(zip(baseline["trials"], report["trials"])):
            if expected["workloads"] != actual["workloads"]:
                errors.append({"case": case["name"], "trial": index, "error": "input workloads differ"})
            mismatch = compare_outputs(expected["outputs"], actual["outputs"])
            if mismatch: errors.append(dict(mismatch, case=case["name"], trial=index))
        # Compare to both matching Graph mode and the Eager baseline.
        for matching_graph in (False, True):
            reference = next((r for r in reports if r.get("passed") and r["case"]["kind"] == "benchmark"
                and r["case"]["batch"] == case["batch"] and r["case"]["method"] == "baseline"
                and r["case"]["graph"] == matching_graph), None)
            if reference is None: continue
            current, base = report["summary"], reference["summary"]
            row = {"case": case["name"], "reference": reference["case"]["name"]}
            for key in ("end_to_end_output_tokens_per_second", "pure_decode_output_tokens_per_second"):
                row[key + "_speedup"] = current[key] / base[key] if base[key] and current[key] else None
            comparisons.append(row)
    return errors, comparisons


def render_markdown(reports, errors, comparisons):
    lines = ["# DFlash A800 test report", "", f"Overall: {'FAIL' if errors else 'PASS'}", "",
        "Fixed-budget Greedy generation. Warmups excluded. Built-in prompts are a smoke workload, not a standard benchmark.", "",
        "| Case | E2E tok/s | Decode-only tok/s | Accepted drafts | Mean output/verify/request | Mean TTFT ms | Mean TPOT ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def number(value, percent=False):
        if value is None: return "N/A"
        return f"{value:.2%}" if percent else f"{value:.3f}"
    for report in reports:
        if "summary" not in report: continue
        s, name = report["summary"], report["case"]["name"]
        spec = s["speculation"]
        lines.append(f"| {name} | {number(s['end_to_end_output_tokens_per_second'])} | "
            f"{number(s['pure_decode_output_tokens_per_second'])} | {number(spec['acceptance_rate'], True)} | "
            f"{number(spec['mean_emitted_tokens_per_request_round'])} | {number(s['mean_request_ttft_ms'])} | "
            f"{number(s['mean_request_tpot_ms'])} |")
    lines += ["", "## Speedups", "", "| Case | Reference | E2E | Decode-only |", "| --- | --- | ---: | ---: |"]
    for row in comparisons:
        if row["case"] == row["reference"]: continue
        lines.append(f"| {row['case']} | {row['reference']} | "
            f"{number(row['end_to_end_output_tokens_per_second_speedup'])} | "
            f"{number(row['pure_decode_output_tokens_per_second_speedup'])} |")
    if errors:
        lines += ["", "## Errors", "", "```json", json.dumps(errors, indent=2, ensure_ascii=False), "```"]
    lines += ["", "Per-rank peak memory, Graph captures/fallbacks, outputs, and workload hashes are in summary.json.",
        "Request TPOT is an average over output tokens; no per-token ITL percentiles are inferred from speculative bursts."]
    return "\n".join(lines) + "\n"


def launch(command, log, timeout):
    with open(log, "w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")
        try:
            return process.wait(timeout=timeout), None
        except subprocess.TimeoutExpired:
            return 124, f"case exceeded {timeout} seconds"
        finally:
            if os.name != "nt":
                # Only this worker's newly created process group is targeted.
                # This also removes NCCL children after worker errors/timeouts.
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                if process.poll() is None:
                    try: process.wait(timeout=3)
                    except subprocess.TimeoutExpired: pass
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            elif process.poll() is None:
                process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--drafts", nargs="+", type=int, default=[3])
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--backend", choices=("auto", "cuda", "reference"), default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--memory-fraction", type=float, default=.82)
    parser.add_argument("--case-timeout", type=int, default=1200)
    parser.add_argument("--prompts-jsonl")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--skip-state-checks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (any(b not in (1, 2, 4, 8, 16, 32, 64, 128) for b in args.batches)
            or any(not 1 <= d <= 15 for d in args.drafts)
            or min(args.repeats, args.warmups, args.output_tokens, args.case_timeout, args.tp) < 1
            or args.token_budget < max(args.drafts) + 1 or args.max_model_len < 512 + args.output_tokens
            or not 0 < args.memory_fraction < 1):
        parser.error("Invalid batches/drafts/budgets: use positive sizes, warmups >=1 and sufficient context")
    plan = build_plan(args.batches, args.drafts, not args.skip_state_checks)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    destination = Path(args.output).resolve()
    if (destination / "manifest.json").exists():
        parser.error("Output already has a manifest; use a fresh output directory to avoid mixing runs")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                "arguments": vars(args), "plan": plan}
    write_json(destination / "manifest.json", manifest)
    reports = []
    for index, case in enumerate(plan):
        print(f"[{index + 1}/{len(plan)}] {case['name']}", flush=True)
        spec, report_file = destination / (case["name"] + ".spec.json"), destination / (case["name"] + ".json")
        write_json(spec, case)
        command = [sys.executable, str(ROOT / "tests/dflash/a800_worker.py"), "--model", args.model,
            "--draft", args.draft, "--spec", str(spec), "--report", str(report_file)]
        for name in ("tp", "backend", "repeats", "warmups", "output_tokens", "max_model_len", "token_budget", "memory_fraction"):
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        if args.prompts_jsonl: command += ["--prompts-jsonl", str(Path(args.prompts_jsonl).resolve())]
        if args.thinking: command += ["--thinking"]
        code, failure = launch(command, destination / (case["name"] + ".log"), args.case_timeout)
        report = (json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists()
                  else {"case": case, "passed": False, "error": failure or f"worker exited {code}; see log"})
        if code:
            report["passed"] = False
            report["error"] = failure or report.get("error", f"worker exited {code}")
        reports.append(report)
        errors, comparisons = compare_reports(reports)
        overall = not errors and len(reports) == len(plan)
        write_json(destination / "summary.json", dict(passed=overall, completed=len(reports), planned=len(plan),
            errors=errors, comparisons=comparisons, cases=reports, manifest=manifest))
        (destination / "summary.md").write_text(render_markdown(reports, errors, comparisons), encoding="utf-8")
        if code or not report.get("passed") or errors:
            print(f"FAIL: {case['name']}; retained reports/logs in {destination}", flush=True)
            raise SystemExit(1)
    print(f"PASS: all cases and Greedy token comparisons; report: {destination / 'summary.md'}")


if __name__ == "__main__":
    main()
