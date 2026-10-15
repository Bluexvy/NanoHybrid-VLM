from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from benchmark_utils import counters_delta, speculation_metrics, run_requests, compare_outputs
from run_suite import build_plan, compare_reports, launch


class Engine:
    def __init__(self):
        self.request_metrics, self.active = {}, {}
        self.next_id = 0

    def add_request(self, prompt, budget):
        self.next_id += 1
        self.active[self.next_id] = {"budget": budget, "produced": 0}
        self.request_metrics[self.next_id] = SimpleNamespace(ttft_ms=1., tpot_ms=2.)

    def is_finished(self):
        return not self.active

    def step(self):
        output, prefill, decode = [], 0, 0
        for seq_id, value in list(self.active.items()):
            if value["produced"] == 0:
                prefill += 1
                value["produced"] = 1
            else:
                count = min(2, value["budget"] - value["produced"])
                value["produced"] += count
                decode += count
            if value["produced"] == value["budget"]:
                output.append((seq_id, list(range(value["budget"]))))
                del self.active[seq_id]
        return output, SimpleNamespace(num_prefill_tokens=prefill, num_decode_tokens=decode)


def make_report(case, outputs=None):
    trial = {"workloads": [{"input_sha256": "same"}], "outputs": outputs or [[1, 2, 3]]}
    return {"case": case, "passed": True, "trials": [trial], "summary": {
        "end_to_end_output_tokens_per_second": 10., "pure_decode_output_tokens_per_second": 20.}}


class HarnessTests(unittest.TestCase):
    def test_acceptance_and_advancement_use_different_denominators(self):
        value = speculation_metrics(dict(proposed_tokens=10, accepted_tokens=3,
            verification_request_rounds=4, emitted_tokens=7))
        self.assertEqual(value["acceptance_rate"], .3)
        self.assertEqual(value["mean_emitted_tokens_per_request_round"], 1.75)
        self.assertIsNone(speculation_metrics({})["acceptance_rate"])
        with self.assertRaises(ValueError):
            speculation_metrics(dict(proposed_tokens=1, accepted_tokens=2))

    def test_warmup_counters_are_subtracted_and_peaks_are_not_differenced(self):
        delta = counters_delta(dict(proposed_tokens=100, accepted_tokens=60, gdn_replay_buffer_bytes_peak=99),
                               dict(proposed_tokens=80, accepted_tokens=50, gdn_replay_buffer_bytes_peak=90))
        self.assertEqual(delta, dict(proposed_tokens=20, accepted_tokens=10))

    def test_dynamic_arrivals_bursts_and_completed_order(self):
        engine = Engine()
        result = run_requests(engine, ["A", "B"], [3, 5], lambda: None,
                              injected=(["C", "D"], [2, 4]))
        self.assertEqual(result["outputs"], [list(range(n)) for n in (3, 5, 2, 4)])
        self.assertEqual(result["output_tokens"], 14)
        self.assertEqual(result["injected_requests"], 2)
        self.assertGreater(result["mixed_steps"], 0)
        self.assertEqual(result["pure_decode_output_tokens"], 8)
        self.assertTrue(engine.is_finished())

    def test_report_detects_mismatched_inputs_and_greedy_outputs(self):
        plan = build_plan([1], [3], state_checks=False)
        reports = [make_report(plan[0]), make_report(plan[1], [[1, 8, 3]])]
        errors, _ = compare_reports(reports)
        self.assertEqual(errors[0]["token_index"], 1)
        reports[1] = make_report(plan[1])
        reports[1]["trials"][0]["workloads"][0]["input_sha256"] = "different"
        errors, _ = compare_reports(reports)
        self.assertEqual(errors[0]["error"], "input workloads differ")
        self.assertIsNone(compare_outputs([[1, 2]], [[1, 2]]))

    def test_plan_includes_matched_graph_baseline_and_state_checks(self):
        plan = build_plan([1, 4], [3])
        self.assertEqual(len(plan), 10)
        self.assertEqual(len({case["name"] for case in plan}), len(plan))
        self.assertEqual(plan[-1]["kind"], "state_checks")
        self.assertTrue(any(case["method"] == "baseline" and case["graph"] for case in plan))

    def test_worker_failures_and_timeouts_preserve_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            code, error = launch([sys.executable, "-c", "print('diagnostic',flush=True); raise SystemExit(3)"], log, 10)
            self.assertEqual(code, 3)
            self.assertIn("diagnostic", log.read_text())
            code, error = launch([sys.executable, "-c", "import time; print('waiting',flush=True); time.sleep(10)"], log, .2)
            self.assertEqual(code, 124)
            self.assertIn("exceeded", error)


if __name__ == "__main__":
    unittest.main()
