"""Pure-Python workload, timing, and report helpers shared by A800 tests."""
import hashlib
import json
from pathlib import Path
import statistics
from time import perf_counter


WORKLOADS = [
    {"id": "qa-memory", "task": "qa", "prompt": "Explain the difference between GPU shared memory and global memory, with one simple example."},
    {"id": "code-search", "task": "code", "prompt": "Write a Python binary search function returning the first index of a target in a sorted list. Explain edge cases."},
    {"id": "math-workers", "task": "math", "prompt": "Three machines make 180 parts in two hours. At the same constant rate, how many parts can five machines make in four hours? Explain your calculation."},
    {"id": "qa-prefix", "task": "qa", "prompt": "What is prefix caching in a language model inference server? Explain why sharing a prefix can reduce first-token latency."},
    {"id": "code-lru", "task": "code", "prompt": "Implement a small LRU cache in Python with get and put using collections.OrderedDict. Include a usage example."},
    {"id": "math-probability", "task": "math", "prompt": "A fair six-sided die is rolled twice. What is the probability that the sum equals seven? List the possible outcomes."},
    {"id": "qa-tp", "task": "qa", "prompt": "Explain tensor parallelism to a student. Use a matrix multiplication split over two GPUs as your example."},
    {"id": "code-merge", "task": "code", "prompt": "Write a Python function that merges two sorted integer lists without sorting them again. Explain its time complexity."},
    {"id": "math-system", "task": "math", "prompt": "Solve the simultaneous equations 2x + y = 11 and x - y = 1. Show your steps and check the result."},
]


def read_workloads(path=None):
    rows = WORKLOADS if path is None else [json.loads(line) for line in
        Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row.get("prompt"), str) or not row["prompt"].strip() for row in rows):
        raise ValueError("Workloads must contain nonempty prompt strings")
    return [dict(row, id=str(row.get("id", i)), task=str(row.get("task", "custom"))) for i, row in enumerate(rows)]


def chat_ids(tokenizer, text, thinking):
    # New Transformers versions default to BatchEncoding; the engine takes list[int].
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
        add_generation_prompt=True, enable_thinking=thinking, return_dict=False, return_tensors=None)
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in ids)):
        raise TypeError("Chat template must produce a nonempty list of integer token IDs")
    return ids


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def counters_delta(after, before):
    return {key: value - before.get(key, 0) for key, value in after.items()
            if isinstance(value, (int, float)) and not key.endswith("_peak")}


def speculation_metrics(counters):
    proposed, accepted = counters.get("proposed_tokens", 0), counters.get("accepted_tokens", 0)
    rounds, emitted = counters.get("verification_request_rounds", 0), counters.get("emitted_tokens", 0)
    if accepted > proposed or min(proposed, accepted, rounds, emitted) < 0:
        raise ValueError("Invalid speculative counters")
    return {"proposed_tokens": proposed, "accepted_draft_tokens": accepted,
        "acceptance_rate": accepted / proposed if proposed else None,
        "verification_request_rounds": rounds,
        "mean_accepted_drafts_per_request_round": accepted / rounds if rounds else None,
        "mean_emitted_tokens_per_request_round": emitted / rounds if rounds else None,
        "counters": counters}


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(trials):
    def total(key):
        return sum(trial[key] for trial in trials)
    wall, decode = total("wall_seconds"), total("pure_decode_seconds")
    output, decode_tokens = total("output_tokens"), total("pure_decode_output_tokens")
    ttft = [row["ttft_ms"] for trial in trials for row in trial["request_metrics"] if row["ttft_ms"] is not None]
    tpot = [row["tpot_ms"] for trial in trials for row in trial["request_metrics"] if row["tpot_ms"] is not None]
    counters = {}
    for trial in trials:
        for key, value in trial["counter_delta"].items():
            counters[key] = counters.get(key, 0) + value
    return {"trials": len(trials), "requests": sum(len(t["outputs"]) for t in trials),
        "output_tokens": output, "end_to_end_seconds": wall,
        "end_to_end_output_tokens_per_second": output / wall if wall else None,
        "pure_decode_seconds": decode, "pure_decode_output_tokens": decode_tokens,
        "pure_decode_output_tokens_per_second": decode_tokens / decode if decode else None,
        "mean_request_ttft_ms": statistics.mean(ttft) if ttft else None,
        "p50_request_ttft_ms": percentile(ttft, .5), "p95_request_ttft_ms": percentile(ttft, .95),
        "mean_request_tpot_ms": statistics.mean(tpot) if tpot else None,
        "p50_request_tpot_ms": percentile(tpot, .5), "p95_request_tpot_ms": percentile(tpot, .95),
        "mixed_steps": total("mixed_steps"), "speculation": speculation_metrics(counters)}


def run_requests(engine, prompts, parameters, synchronize, injected=None):
    """Measure synchronized engine steps; inject a second wave after decode begins.

    No per-token ITL percentiles: speculative output arrives in bursts.
    Request TPOT is (last output time - first output time)/(output count - 1).
    """
    requests = []
    def add(items, settings):
        for prompt, parameter in zip(items, settings):
            previous = set(engine.request_metrics)
            engine.add_request(prompt, parameter)
            created = set(engine.request_metrics) - previous
            if len(created) != 1:
                raise RuntimeError("Unable to identify newly enqueued request")
            requests.append(created.pop())
    synchronize()
    start = perf_counter()
    add(prompts, parameters)
    results, steps = {}, []
    pending = injected is not None
    while not engine.is_finished() or pending:
        step_start = perf_counter()
        output, stats = engine.step()
        if stats.num_decode_tokens == 0 and stats.num_prefill_tokens == 0:
            raise RuntimeError("Scheduler made no progress with unfinished requests")
        synchronize()
        seconds = perf_counter() - step_start
        steps.append({"seconds": seconds, "decode_tokens": stats.num_decode_tokens,
                      "prefill_tokens": stats.num_prefill_tokens})
        results.update({seq_id: list(tokens) for seq_id, tokens in output})
        if pending and (stats.num_decode_tokens > 0 or engine.is_finished()):
            add(*injected)
            pending = False
    synchronize()
    wall = perf_counter() - start
    if set(results) != set(requests):
        raise AssertionError("Completed request IDs do not match submitted requests")
    metrics = []
    for seq_id in requests:
        value = engine.request_metrics[seq_id]
        metrics.append({"seq_id": seq_id, "ttft_ms": value.ttft_ms, "tpot_ms": value.tpot_ms})
    pure = [step for step in steps if step["decode_tokens"] > 0 and step["prefill_tokens"] == 0]
    return {"wall_seconds": wall, "output_tokens": sum(len(results[i]) for i in requests),
        "pure_decode_seconds": sum(step["seconds"] for step in pure),
        "pure_decode_output_tokens": sum(step["decode_tokens"] for step in pure),
        "mixed_steps": sum(step["decode_tokens"] > 0 and step["prefill_tokens"] > 0 for step in steps),
        "step_count": len(steps), "outputs": [results[i] for i in requests], "request_metrics": metrics,
        "injected_requests": 0 if injected is None else len(injected[0])}


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def compare_outputs(expected, actual):
    if len(expected) != len(actual):
        return {"error": "request count differs", "expected": len(expected), "actual": len(actual)}
    for request, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            position = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), min(len(left), len(right)))
            return {"error": "greedy token mismatch", "request": request, "token_index": position,
                "expected": left[max(0, position - 3):position + 4],
                "actual": right[max(0, position - 3):position + 4]}
    return None
