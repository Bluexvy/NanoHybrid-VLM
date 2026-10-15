"""Opt-in fixed-history diagnostics. Never used by normal generation or timing."""
import math
from contextlib import contextmanager
import torch
import torch.distributed as dist


def estimate_decode_splits(batch, q_heads, kv_heads, head_dim, capacity, num_sms):
    """FA 2.8.3 heuristic estimate for this plain GQA Decode call, not a device trace.

    Reference: Dao-AILab/flash-attention v2.8.3, flash_api.cpp,
    set_params_splitkv / num_splits_heuristic / mha_fwd_kvcache.
    """
    grouped = q_heads > kv_heads and head_dim % 8 == 0
    heads, queries = (kv_heads, q_heads // kv_heads) if grouped else (q_heads, 1)
    tile = 256 if head_dim <= 64 else 128 if head_dim <= 128 else 64
    blocks = math.ceil(capacity / tile)
    parallel = batch * heads * math.ceil(queries / 64)
    sm_target = 2 * num_sms
    if parallel >= .8 * sm_target:
        return {"num_splits_estimate": 1, "tokens_per_split_estimate": blocks * tile}
    scores = {}
    for splits in range(1, min(128, sm_target, blocks) + 1):
        if splits > 1 and math.ceil(blocks / splits) == math.ceil(blocks / (splits - 1)):
            continue
        waves = parallel * splits / sm_target
        scores[splits] = waves / math.ceil(waves)
    threshold = .85 * max(scores.values())
    selected = next(splits for splits, score in scores.items() if score >= threshold)
    return {"num_splits_estimate": selected,
            "tokens_per_split_estimate": math.ceil(blocks / selected) * tile}


def tensor_difference(expected, actual):
    a, b = expected.detach().float(), actual.detach().float()
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"finite": False, "exact": False}
    delta = a - b
    return {"finite": True, "exact": torch.equal(a, b),
            "max_abs": float(delta.abs().max()),
            "relative_l2": float(torch.linalg.vector_norm(delta) /
                                 torch.linalg.vector_norm(a).clamp_min(1e-12))}


def logits_difference(expected, actual):
    result = tensor_difference(expected, actual)
    if result.get("shape_mismatch") or not result.get("finite"):
        return result
    a, b = expected.float().cpu(), actual.float().cpu()
    left, right = a.argmax(-1), b.argmax(-1)
    rows = []
    for row in range(a.shape[0]):
        av, ai = a[row].topk(2)
        bv, bi = b[row].topk(2)
        rows.append({"row": row, "expected_argmax": int(left[row]), "actual_argmax": int(right[row]),
                     "expected_top2_ids": ai.tolist(), "actual_top2_ids": bi.tolist(),
                     "expected_top2_logits": av.tolist(), "actual_top2_logits": bv.tolist(),
                     "expected_margin": float(av[0] - av[1]), "actual_margin": float(bv[0] - bv[1])})
    result.update(argmax_mismatches=int((left != right).sum()), rows=rows)
    return result


def state_difference(expected, actual):
    rows = []
    for layer, (a, b) in enumerate(zip(expected, actual)):
        if a is None and b is None:
            continue
        if a is None or b is None:
            rows.append({"layer": layer, "missing_state": True})
        else:
            rows.append({"layer": layer,
                         "conv": tensor_difference(a.conv_state, b.conv_state),
                         "recurrent": tensor_difference(a.recurrent_state, b.recurrent_state)})
    return rows


def anchor_rows(tensor, batch):
    if tensor.shape[0] % batch:
        raise ValueError("Diagnostic rows do not align with request boundaries")
    return tensor[::tensor.shape[0] // batch]


@contextmanager
def padded_decode_table(runner, columns):
    """Diagnostic ablation: change table capacity only, retaining live KV IDs.

    This pads page metadata, NOT requests/tokens. Live context lengths, slots,
    and ordinary Eager execution stay unchanged. Extra pages are never live.
    """
    from nanovllm.utils.context import get_context
    original = runner.prepare_decode

    def prepare(seqs):
        result = original(seqs)
        context = get_context()
        table = context.block_tables
        if table is None or table.ndim != 2 or columns < table.shape[1]:
            raise ValueError("Diagnostic table width must cover all live pages")
        padded = table.new_zeros((table.shape[0], columns))
        padded[:, :table.shape[1]].copy_(table)
        context.block_tables = padded
        return result

    runner.prepare_decode = prepare
    try:
        yield
    finally:
        runner.prepare_decode = original


@torch.inference_mode()
def diagnose_hybrid_step(runner, seqs, lookahead, probe=True):
    """Compare paths from one checkpoint; advance active GDN by baseline ONE step.

    The caller advances token metadata using an oracle trajectory. KV rows at the
    current/lookahead positions are scratch and overwritten by each path; the
    historical KV prefix is retained. This helper is restricted to a single page
    per request so Decode cannot address a future reserved page by mistake.
    """
    from nanovllm.utils.context import get_context, reset_context, set_context
    manager, batch = runner.hybrid_state_manager, len(seqs)
    if not seqs or len(lookahead) != batch or len({s.state_slot for s in seqs}) != batch:
        raise ValueError("Diagnostics require unique initialized request slots")
    if any(s.num_cached_tokens != s.num_tokens - 1 or len(s.block_table) != 1
           or s.num_tokens + len(tokens) > runner.block_size for s, tokens in zip(seqs, lookahead)):
        raise ValueError("Fixed-history diagnostic requires one pending token and one KV page per request")
    slots = [s.state_slot for s in seqs]
    original_eager = runner.enforce_eager
    if not probe:
        try:
            runner.enforce_eager = True
            ids = runner.run_hybrid_decode(seqs)
            return {"baseline_ids": ids} if runner.rank == 0 else None
        finally:
            runner.enforce_eager = original_eager
    checkpoint = manager.read_batched_states(slots)
    baseline_states = None
    original_compute = runner.model.compute_logits
    handles, trace, observed, comparisons = [], {}, {}, {}
    language = runner.model.model.language_model

    def record_logits(hidden):
        output = original_compute(hidden)
        observed["hidden"] = hidden.detach().float().cpu().clone()
        if output is not None:
            observed["logits"] = output.detach().float().cpu().clone()
        return output

    def hook_for(name):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            trace[name] = anchor_rows(hidden, batch).detach().float().cpu().clone()
        return hook

    def compare(name, hidden, logits, states):
        layer_errors = [{"layer": key, **tensor_difference(base_trace[key], value)}
                        for key, value in trace.items() if key in base_trace]
        comparisons[name] = {"hidden": tensor_difference(base_hidden, hidden),
                             "states": state_difference(baseline_states, states),
                             "layers": layer_errors,
                             "first_nonidentical_layer": next((r["layer"] for r in layer_errors
                                                              if not r.get("exact", False)), None)}
        if runner.rank == 0:
            comparisons[name]["logits"] = logits_difference(base_logits, logits)

    try:
        runner.model.compute_logits = record_logits
        for index, layer in enumerate(language.layers):
            handles.append(layer.register_forward_hook(hook_for(f"decoder.{index}")))
        handles.append(language.norm.register_forward_hook(hook_for("final_norm")))
        runner.enforce_eager = True
        baseline_ids = runner.run_hybrid_decode(seqs)
        baseline_states = manager.read_batched_states(slots)
        base_hidden, base_logits, base_trace = observed["hidden"], observed.get("logits"), dict(trace)

        if batch in runner.hybrid_graphs:
            manager.write_batched_states(slots, checkpoint)
            trace.clear()
            runner.enforce_eager = False
            graph_ids = runner.run_hybrid_decode(seqs)
            compare("decode_graph", observed["hidden"], observed.get("logits"), manager.read_batched_states(slots))
            comparisons["decode_graph"]["sampled_ids"] = graph_ids
            graph_hidden, graph_logits = observed["hidden"], observed.get("logits")

            # Keep ordinary Eager allocations/execution; change ONLY table width.
            # Compare directly with Graph, rather than inferring equality from
            # coincidentally equal max-error summaries against the baseline.
            manager.write_batched_states(slots, checkpoint)
            trace.clear()
            runner.enforce_eager = True
            columns = runner.hybrid_graph_workspace.block_tables.shape[1]
            with padded_decode_table(runner, columns):
                runner.run_hybrid_decode(seqs)
            compare("decode_padded_table", observed["hidden"], observed.get("logits"),
                    manager.read_batched_states(slots))
            against_graph = {"hidden": tensor_difference(graph_hidden, observed["hidden"])}
            if runner.rank == 0:
                against_graph["logits"] = logits_difference(graph_logits, observed["logits"])
            comparisons["decode_padded_table"]["against_graph"] = against_graph

            # Same static tensors as Graph, but executed by Python, with hooks.
            manager.write_batched_states(slots, checkpoint)
            trace.clear()
            ids, positions = runner.prepare_decode(seqs)
            dynamic = get_context()
            workspace = runner.hybrid_graph_workspace
            workspace.copy_decode_inputs(input_ids=ids, positions=positions,
                slot_mapping=dynamic.slot_mapping, context_lens=dynamic.context_lens,
                block_tables=dynamic.block_tables, state_slot_ids=slots)
            set_context(False, slot_mapping=workspace.slot_mapping[:batch],
                context_lens=workspace.context_lens[:batch], block_tables=workspace.block_tables[:batch])
            try:
                runner.run_hybrid_graph_body(batch)
                runner.model.compute_logits(workspace.hidden_states[:batch])
            finally:
                reset_context()
            compare("decode_static_eager", observed["hidden"], observed.get("logits"), manager.read_batched_states(slots))

        for name, drafts in (("verify_one", [[] for _ in seqs]), ("verify_block", lookahead)):
            manager.write_batched_states(slots, checkpoint)
            trace.clear()
            logits, hidden, updated, records = runner._verify_mtp_batch(seqs, drafts)
            positions, offset = [], 0
            for tokens in drafts:
                positions.append(offset)
                offset += len(tokens) + 1
            hidden = hidden[positions].detach().float().cpu()
            logits = logits[positions].detach().float().cpu() if logits is not None else None
            if name == "verify_block":
                # Compare state after the SAME anchor, not the whole future block.
                parts = [runner._replay_gdn_inputs(seq,
                    {layer: value[row] if isinstance(value, tuple) else value for layer, value in records.items()}, 1)
                    for row, seq in enumerate(seqs)]
                updated = [None if checkpoint[layer] is None else type(checkpoint[layer])(
                    conv_state=torch.cat([part[layer].conv_state for part in parts], 0),
                    recurrent_state=torch.cat([part[layer].recurrent_state for part in parts], 0))
                    for layer in range(len(checkpoint))]
            compare(name, hidden, logits, updated)
        metadata = {"dynamic_table_columns": max(len(seq.block_table) for seq in seqs),
                    "static_table_columns": runner.hybrid_graph_workspace.block_tables.shape[1]
                                            if batch in runner.hybrid_graphs else None,
                    "min_context_len": min(seq.num_tokens for seq in seqs),
                    "max_context_len": max(seq.num_tokens for seq in seqs),
                    "block_size": runner.block_size}
        attention = next((module for module in runner.model.modules()
                          if module.__class__.__name__ == "Attention"), None)
        if attention is not None and torch.cuda.is_available():
            sms = torch.cuda.get_device_properties(runner.rank).multi_processor_count
            metadata["num_sms"] = sms
            for label in ("dynamic", "static"):
                columns = metadata[f"{label}_table_columns"]
                if columns:
                    metadata[f"{label}_splitkv_estimate"] = estimate_decode_splits(batch,
                        attention.num_heads, attention.num_kv_heads, attention.head_dim,
                        columns * runner.block_size, sms)
        local = {"rank": runner.rank, "attention_metadata": metadata, "comparisons": comparisons}
        ranks = [None] * runner.world_size
        if runner.world_size > 1:
            dist.all_gather_object(ranks, local)
        else:
            ranks = [local]
        return {"baseline_ids": baseline_ids, "ranks": ranks} if runner.rank == 0 else None
    finally:
        runner.enforce_eager = original_eager
        runner.model.compute_logits = original_compute
        for handle in handles:
            handle.remove()
        # Alternate paths must not contaminate the continuation history.
        manager.write_batched_states(slots, baseline_states if baseline_states is not None else checkpoint)
        reset_context()
