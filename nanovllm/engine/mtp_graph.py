"""Bounded, lazy CUDA Graphs for uniform packed MTP verification.

Only verification is captured. Sampling, ragged requests and state commit remain
outside the graph; checkpoint State Pool tensors are read-only during capture.
"""
from dataclasses import dataclass
import math
import torch
import torch.distributed as dist


@dataclass
class MTPVerificationWorkspace:
    batch: int
    length: int
    context_capacity: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    state_slots: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    gdn_cu: torch.Tensor

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.input_ids, self.positions,
            self.slot_mapping, self.block_tables, self.state_slots, self.cu_q, self.cu_k, self.gdn_cu))

    @classmethod
    def allocate(cls, batch, length, capacity, block_size, position_axes, device):
        if batch <= 0 or length <= 0 or capacity < length:
            raise ValueError("Invalid MTP verification workspace shape")
        count = batch * length
        shape = (position_axes, count) if position_axes else (count,)
        cu_q = torch.arange(0, count + 1, length, dtype=torch.int32, device=device)
        return cls(batch, length, capacity,
            torch.zeros(count, dtype=torch.long, device=device),
            torch.zeros(shape, dtype=torch.long, device=device),
            torch.full((count,), -1, dtype=torch.int32, device=device),
            torch.full((batch, math.ceil(capacity / block_size)), -1, dtype=torch.int32, device=device),
            torch.zeros(batch, dtype=torch.long, device=device), cu_q,
            torch.zeros(batch + 1, dtype=torch.int32, device=device), cu_q.to(torch.long))

    def copy_inputs(self, ids, positions, context, slots):
        if (ids.shape != self.input_ids.shape or positions.shape != self.positions.shape
                or context.slot_mapping.shape != self.slot_mapping.shape
                or len(slots) != self.batch or len(set(slots)) != self.batch):
            raise ValueError("MTP graph inputs must match its exact verification bucket")
        if (context.block_tables is None or context.block_tables.shape[0] != self.batch
                or context.block_tables.shape[1] > self.block_tables.shape[1]):
            raise ValueError("MTP graph block table exceeds its context bucket")
        self.input_ids.copy_(ids)
        self.positions.copy_(positions)
        self.slot_mapping.copy_(context.slot_mapping)
        self.cu_k.copy_(context.cu_seqlens_k)
        self.block_tables.fill_(-1)
        self.block_tables[:, :context.block_tables.shape[1]].copy_(context.block_tables)
        self.state_slots.copy_(torch.tensor(slots, dtype=torch.long, device=self.state_slots.device))

    def bind_context(self):
        from nanovllm.utils.context import set_context, get_context
        set_context(True, cu_seqlens_q=self.cu_q, cu_seqlens_k=self.cu_k,
            max_seqlen_q=self.length, max_seqlen_k=self.context_capacity,
            slot_mapping=self.slot_mapping, block_tables=self.block_tables,
            prefill_seqlens=(self.length,) * self.batch, gdn_cu_seqlens=self.gdn_cu)
        context = get_context()
        context.gdn_mtp_verification = True
        context.gdn_replay_capture_tokens = self.length
        context.gdn_replay_records = {}
        return context


class MTPVerificationGraph:
    def __init__(self, runner, workspace):
        self.runner, self.workspace = runner, workspace
        self.graph = torch.cuda.CUDAGraph()
        self.outputs = None

    def body(self):
        context = self.workspace.bind_context()
        context.gdn_mtp_verification_backend = getattr(self.runner.config, "mtp_gdn_verification_backend", "reference")
        states = self.runner.hybrid_state_manager.gather_batched_states_for_graph(self.workspace.state_slots)
        hidden, updated = self.runner.model(input_ids=self.workspace.input_ids,
            positions=self.workspace.positions, gdn_states=states)
        self.outputs = hidden, updated, context.gdn_replay_records
        self.draft_features = self.runner._mtp_verification_features(hidden)

    def capture(self):
        from nanovllm.utils.context import reset_context
        stream, current = torch.cuda.Stream(), torch.cuda.current_stream()
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.body()
            current.wait_stream(stream)
            torch.cuda.synchronize()
            # Do not retain warmup's full updated states alongside capture.
            self.outputs = None
            self.draft_features = None
            torch.cuda.empty_cache()
            reserved_before = torch.cuda.memory_reserved()
            if self.runner.world_size > 1:
                dist.barrier()
            with torch.cuda.graph(self.graph, stream=stream):
                self.body()
            current.wait_stream(stream)
            self.reserved_bytes = max(0, torch.cuda.memory_reserved() - reserved_before)
        finally:
            reset_context()


@dataclass
class MTPDraftWorkspace:
    capacity: int
    positions: torch.Tensor
    embeddings: torch.Tensor
    features: torch.Tensor
    past_length: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor

    @classmethod
    def allocate(cls, model, positions, capacity):
        attention = model.layers[0].self_attn
        options = dict(device=positions.device, dtype=model.fc.weight.dtype)
        return cls(capacity, torch.zeros_like(positions),
            torch.zeros(1, model.fc.out_features, **options),
            torch.zeros(1, model.fc.out_features, **options),
            torch.zeros((), dtype=torch.long, device=positions.device),
            torch.zeros(attention.kv_heads, capacity, attention.dim, **options),
            torch.zeros(attention.kv_heads, capacity, attention.dim, **options))

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.positions, self.embeddings, self.features, self.past_length, self.key, self.value))

    def copy_inputs(self, positions, embeddings, features, past):
        length = 0 if past is None else past[0].shape[1]
        if length >= self.capacity or embeddings.shape != self.embeddings.shape:
            raise ValueError("Draft graph inputs exceed the fixed KV capacity")
        self.positions.copy_(positions)
        self.embeddings.copy_(embeddings)
        self.features.copy_(features)
        self.past_length.fill_(length)
        self.key.zero_()
        self.value.zero_()
        if past is not None:
            self.key[:, :length].copy_(past[0])
            self.value[:, :length].copy_(past[1])


class MTPDraftGraph(MTPVerificationGraph):
    def body(self):
        workspace = self.workspace
        self.outputs = self.runner.model.mtp.forward_step(workspace.positions, workspace.embeddings,
            workspace.features, (workspace.key, workspace.value), workspace.past_length, self.runner._mtp_rotary())


class MTPVerificationGraphMixin:
    def mtp_head_with_graph(self, positions, embeddings, features, past):
        config = self.config
        length = 0 if past is None else past[0].shape[1]
        eligible = (not config.enforce_eager and embeddings.is_cuda and embeddings.shape[0] == 1
                    and length < config.max_model_len)
        if not eligible:
            self.mtp_stats["draft_graph_fallbacks"] += 1
            return self.model.mtp(positions, embeddings, features, past, self._mtp_rotary())
        capacity = min(config.max_model_len, 1 << length.bit_length())
        key = capacity, tuple(positions.shape)
        entry = self.mtp_draft_graphs.get(key)
        if entry is None:
            attention = self.model.mtp.layers[0].self_attn
            scalar_bytes = self.model.mtp.fc.weight.element_size()
            estimate = ((2 * attention.kv_heads + 2 * attention.heads) * capacity * attention.dim * scalar_bytes
                        + 8 * self.model.mtp.fc.out_features * scalar_bytes)
            free, _ = torch.cuda.mem_get_info()
            available = int(estimate <= self.mtp_graph_memory_budget - self.mtp_graph_memory_bytes
                and estimate * 1.2 <= free
                and len(self.mtp_draft_graphs) + len(self.mtp_verification_graphs) < config.mtp_cuda_graph_max_entries)
            if self.world_size > 1:
                flag = torch.tensor(available, device=embeddings.device, dtype=torch.int32)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                available = int(flag)
            if not available:
                self.mtp_stats["draft_graph_fallbacks"] += 1
                return self.model.mtp(positions, embeddings, features, past, self._mtp_rotary())
            workspace = MTPDraftWorkspace.allocate(self.model.mtp, positions, capacity)
            workspace.copy_inputs(positions, embeddings, features, past)
            entry = MTPDraftGraph(self, workspace)
            entry.capture()
            self.mtp_graph_memory_bytes += workspace.nbytes + entry.reserved_bytes
            self.mtp_draft_graphs[key] = entry
            self.mtp_stats["draft_graph_captures"] += 1
        else:
            entry.workspace.copy_inputs(positions, embeddings, features, past)
        entry.graph.replay()
        self.mtp_stats["draft_graph_replays"] += 1
        # Each request gets an independent cache; a later replay may reuse buffer.
        updated = (entry.workspace.key[:, :length + 1].clone(), entry.workspace.value[:, :length + 1].clone())
        return entry.outputs, updated

    def verify_mtp_with_graph(self, seqs, drafts):
        lengths = [1 + len(tokens) for tokens in drafts]
        config = self.config
        # Exact buckets only: no padded row may write a live request's state.
        eligible = (not config.enforce_eager and len(set(lengths)) == 1
                    and len(seqs) in config.hybrid_cuda_graph_batch_sizes
                    and self.hybrid_state_manager.device.type == "cuda"
                    and config.mtp_gdn_rollback == "replay_inputs")
        if not eligible:
            self.mtp_stats["verification_graph_fallbacks"] += 1
            return self._verify_mtp_batch(seqs, drafts)
        capacity = max(seq.num_tokens + len(tokens) for seq, tokens in zip(seqs, drafts))
        capacity = min(config.max_model_len, 1 << (capacity - 1).bit_length())
        key = len(seqs), lengths[0], capacity
        entry = self.mtp_verification_graphs.get(key)
        if entry is None:
            spec = self.hybrid_state_manager.spec
            scalar_bytes = spec.dtype_nbytes(spec.conv_dtype)
            record_per_token = (spec.conv_dim * scalar_bytes
                + spec.num_gdn_value_heads * (spec.gdn_key_head_dim + spec.gdn_value_head_dim) * scalar_bytes
                + spec.num_gdn_value_heads * (4 + scalar_bytes))
            estimate = ((2 * len(seqs) + 2) * spec.state_bytes_per_slot
                        + record_per_token * spec.num_gdn_layers * len(seqs) * lengths[0])
            if getattr(config, "speculative_method", "mtp") == "dflash":
                estimate += len(seqs) * lengths[0] * self.dflash.fc.in_features * scalar_bytes
            budget = self.mtp_graph_memory_budget - self.mtp_graph_memory_bytes
            free, _ = torch.cuda.mem_get_info()
            available = int(len(self.mtp_verification_graphs) + len(self.mtp_draft_graphs) < config.mtp_cuda_graph_max_entries
                            and estimate <= budget and estimate * 1.2 <= free)
            if self.world_size > 1:
                flag = torch.tensor(available, device=self.hybrid_state_manager.device, dtype=torch.int32)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                available = int(flag)
            if not available:
                self.mtp_stats["verification_graph_fallbacks"] += 1
                return self._verify_mtp_batch(seqs, drafts)
        from nanovllm.utils.context import get_context, reset_context
        views = [self._mtp_verify_view(seq, tokens) for seq, tokens in zip(seqs, drafts)]
        ids, positions = self.prepare_prefill(views)
        try:
            if entry is None:
                workspace = MTPVerificationWorkspace.allocate(len(seqs), lengths[0], capacity,
                    self.block_size, positions.shape[0] if positions.ndim == 2 else 0, ids.device)
                workspace.copy_inputs(ids, positions, get_context(), [seq.state_slot for seq in seqs])
                entry = MTPVerificationGraph(self, workspace)
                entry.capture()
                self.mtp_graph_memory_bytes += workspace.nbytes + entry.reserved_bytes
                self.mtp_verification_graphs[key] = entry
                self.mtp_stats["verification_graph_captures"] += 1
            else:
                entry.workspace.copy_inputs(ids, positions, get_context(), [seq.state_slot for seq in seqs])
            entry.graph.replay()
            hidden, states, records = entry.outputs
            self.mtp_stats["verification_graph_replays"] += 1
            return self._mtp_logits(hidden), entry.draft_features, states, records
        finally:
            reset_context()
