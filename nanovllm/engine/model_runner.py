import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.registry import get_model_class
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.engine.hybrid_state import (
    HybridCacheSpec,
    HybridStateManager,
)
from nanovllm.engine.hybrid_cuda_graph import (
    HybridDecodeGraphPolicy,
    HybridDecodeStaticWorkspace,
)

class ModelRunner:
    
    IMAGE_TOKEN_TYPE = 1

    def __init__(    
        self,
        config: Config,
        rank: int,
        event: Event | list[Event],
    ):
        
        self.config = config
        
        root_config = config.hf_config
        text_config = config.text_config
        # 如果 Config 还没有完成初始化，ModelRunner 不允许继续运行。
        if root_config is None or text_config is None:
            raise RuntimeError(
                "Config must initialize hf_config and "
                "text_config before ModelRunner construction"
            )

        # 这里返回的是一个类，还没有创建对象
        model_class = get_model_class(root_config)

        model_dtype = getattr(
            text_config,
            "dtype",
            None,
        )

        if model_dtype is None:
            raise ValueError(
                "The text model config does not define dtype"
            )

        self.root_config = root_config
        self.text_config = text_config
        self.model_dtype = model_dtype

        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        
        # 当前 Qwen3.5 多模态模型使用
        # temporal/height/width 三轴 RoPE。
        #
        # 普通 Qwen3 没有 vision_config，
        # 继续使用一维 positions。
        self.uses_multimodal_rope = (
            config.vision_config is not None
        )
        self.image_token_id = getattr(
            root_config,
            "image_token_id",
            None,
        )
        # seq_id -> 完整视觉 token embeddings
        self.visual_embedding_cache: dict[
            int,
            torch.Tensor,
        ] = {}

        # 当前视觉缓存实际持有的字节数。
        self.visual_cache_bytes = 0

        # ModelRunner 生命周期内的峰值视觉缓存字节数。
        self.peak_visual_cache_bytes = 0
        
        # Vision Tower 实际成功执行了多少次。
        self.num_vision_forwards = 0

        # 查询视觉缓存时命中的次数。
        self.num_visual_cache_hits = 0

        # 查询视觉缓存时未命中的次数。
        self.num_visual_cache_misses = 0
        
        # 实际执行了多少个 Prefill microbatch。
        self.num_prefill_microbatches = 0

        # 单个 Prefill microbatch 中最多包含多少条请求。
        self.max_observed_prefill_batch_size = 0

        # 同时包含纯文本和图文请求的 Prefill 数量。
        self.num_mixed_prefill_microbatches = 0

        if (
            self.uses_multimodal_rope
            and not isinstance(
                self.image_token_id,
                int,
            )
        ):
            raise ValueError(
                "A multimodal model must provide "
                "image_token_id"
            )

        layer_types = tuple(
            getattr(
                text_config,
                "layer_types",
                (),
            )
        )

        # 判断是否是混合模型 去遍历层，只要遍历到一个是"linear_attention" 那就是混合模型
        self.is_hybrid_model = any(
            layer_type == "linear_attention"
            for layer_type in layer_types
        )
        
        if self.is_hybrid_model:
            self.hybrid_cache_spec = (
                HybridCacheSpec.from_text_config(
                    text_config,
                    tensor_parallel_size=self.world_size,
                )
            )
        else:
            self.hybrid_cache_spec = None

        self.hybrid_state_manager = None
        
        # Hybrid CUDA Graph 路由策略。
        self.hybrid_graph_policy = None

        # 固定地址输入、输出和 GDN 状态。
        self.hybrid_graph_workspace = None

        # batch_size -> torch.cuda.CUDAGraph
        self.hybrid_graphs: dict[
            int,
            torch.cuda.CUDAGraph,
        ] = {}

        # 多张 Graph 共用的 CUDA memory pool handle。
        self.hybrid_graph_pool = None

        # 运行时统计。
        self.num_hybrid_graph_replays = 0
        self.num_hybrid_graph_eager_fallbacks = 0

        self.hybrid_graph_fallback_reasons: dict[
            str,
            int,
        ] = {}

        # Capture 前后 torch.cuda.memory_allocated()
        # 的差值。包括 Workspace 和 Graph 中长期存活
        # 的一部分 Tensor。
        self.hybrid_graph_capture_allocated_bytes = 0

        if self.is_hybrid_model:
            self.hybrid_graph_policy = (
                HybridDecodeGraphPolicy(
                    batch_sizes=(
                        config
                        .hybrid_cuda_graph_batch_sizes
                    ),
                    max_num_seqs=(
                        config.max_num_seqs
                    ),
                )
            )

        dist.init_process_group(
            "nccl",
            "tcp://localhost:2333",
            world_size=self.world_size,
            rank=rank,
        )

        torch.cuda.set_device(rank)

        default_dtype = torch.get_default_dtype()

        torch.set_default_dtype(model_dtype)
        torch.set_default_device("cuda")

        # 根据配置创建模型结构和参数空间
        self.model = model_class(root_config)
        # 加载权重
        load_model(self.model, config.model)
        # 创建 Sampler
        self.sampler = Sampler()
        # 模型预热
        # 注意需要先预热再分配cache 测量一些需要的额外或者临时显存（先测模型运行峰值） 然后再计算真正可以分配的显存
        self.warmup_model()
        # 计算并分配KV cache
        self.allocate_model_caches()

        if not self.enforce_eager:
            if self.is_hybrid_model:
                self.capture_hybrid_cudagraph()
            else:
                self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def get_memory_stats(
        self,
    ) -> dict[str, int]:
        """
        返回当前 ModelRunner 的主要显存组成。
        """

        torch.cuda.synchronize()

        # 共享权重可能被多个 Parameter 引用。
        # 例如 Embedding 和 LM Head tied weights。
        #
        # 使用 storage.data_ptr() 去重，
        # 避免重复统计同一块显存。
        seen_storages: set[int] = set()

        model_parameter_bytes = 0

        for parameter in self.model.parameters():
            storage = (
                parameter.untyped_storage()
            )

            storage_ptr = storage.data_ptr()

            if storage_ptr in seen_storages:
                continue

            seen_storages.add(
                storage_ptr
            )

            model_parameter_bytes += (
                storage.nbytes()
            )

        kv_cache_bytes = (
            self.kv_cache.numel()
            * self.kv_cache.element_size()
        )

        if (
            self.hybrid_cache_spec is not None
        ):
            gdn_state_bytes = (
                self.hybrid_cache_spec
                .state_bytes_per_slot
                * self.config.num_state_slots
            )
        else:
            gdn_state_bytes = 0

        cuda_current_allocated = (
            torch.cuda.memory_allocated(
                self.rank
            )
        )

        cuda_peak_allocated = (
            torch.cuda.max_memory_allocated(
                self.rank
            )
        )

        cuda_reserved = (
            torch.cuda.memory_reserved(
                self.rank
            )
        )

        # 从 Benchmark 开始到峰值期间，
        # 相对最终常驻内存额外出现的显存。
        runtime_peak_extra_bytes = max(
            0,
            (
                cuda_peak_allocated
                - cuda_current_allocated
            ),
        )

        return {
            "model_parameter_bytes": (
                model_parameter_bytes
            ),
            "kv_cache_bytes": (
                kv_cache_bytes
            ),
            "gdn_state_bytes": (
                gdn_state_bytes
            ),
            "visual_cache_current_bytes": (
                self.visual_cache_bytes
            ),
            "visual_cache_peak_bytes": (
                self.peak_visual_cache_bytes
            ),
            "cuda_current_allocated_bytes": (
                cuda_current_allocated
            ),
            "cuda_peak_allocated_bytes": (
                cuda_peak_allocated
            ),
            "cuda_reserved_bytes": (
                cuda_reserved
            ),
            "runtime_peak_extra_bytes": (
                runtime_peak_extra_bytes
            ),
            "kv_cache_block_bytes": (
                self.kv_cache_block_bytes
            ),
        }

    @property
    def kv_cache_block_bytes(self) -> int:
        """
        一个物理 Paged KV Block 在完整 KV Cache
        Tensor 中对应的字节数。

        kv_cache shape：

            [
                2,
                num_full_attention_layers,
                num_physical_blocks,
                block_size,
                num_kv_heads,
                head_dim,
            ]
        """

        if self.kv_cache.ndim != 6:
            raise RuntimeError(
                "KV Cache must have six dimensions"
            )

        if self.kv_cache.shape[2] <= 0:
            raise RuntimeError(
                "KV Cache does not contain any "
                "physical blocks"
            )

        # 固定物理 block 维度的下标为 0，
        # 剩下的 Tensor 正好代表一个完整物理 Block。
        one_block = self.kv_cache[
            :,
            :,
            0,
            :,
            :,
            :,
        ]

        return (
            one_block.numel()
            * one_block.element_size()
        )

    def exit(self):
        self.release_visual_embedding_cache(
            list(
                self.visual_embedding_cache.keys()
            )
        )
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            if self.is_hybrid_model:
                self.hybrid_graphs.clear()

                self.hybrid_graph_workspace = None
                self.hybrid_graph_pool = None

            else:
                del self.graphs
                del self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if self.is_hybrid_model:
            self.warmup_hybrid_model()
        else:
            self.warmup_attention_model()

        # 等待 warmup 中所有异步 CUDA Kernel 完成，
        # 确保 peak memory 已经记录完整。
        torch.cuda.synchronize()

        # 释放 warmup 产生但已经不再使用的缓存块。
        #
        # peak memory 统计不会因此消失。
        torch.cuda.empty_cache()

    def warmup_attention_model(self):
        max_num_batched_tokens = (
            self.config.max_num_batched_tokens
        )

        max_model_len = self.config.max_model_len

        seq_len = min(
            max_num_batched_tokens,
            max_model_len,
        )

        num_seqs = min(
            max_num_batched_tokens // seq_len,
            self.config.max_num_seqs,
        )

        seqs = [
            Sequence([0] * seq_len)
            for _ in range(num_seqs)
        ]

        for seq in seqs:
            seq.num_scheduled_tokens = seq_len

        self.run(
            seqs,
            is_prefill=True,
        )


    @torch.inference_mode()
    def warmup_hybrid_model(self):
        seq_len = min(
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )

        # 当前 Qwen3.5 正确性路径一次只处理一条
        # Sequence，因此 warmup 也使用单条最大长度请求。
        seq = Sequence(
            [0] * seq_len
        )

        seq.num_scheduled_tokens = seq_len

        input_ids, positions = (
            self.prepare_prefill([seq])
        )

        try:
            (
                hidden_states,
                updated_gdn_states,
            ) = self.model(
                input_ids=input_ids,
                positions=positions,
                gdn_states=None,
            )

            logits = self.model.compute_logits(
                hidden_states
            )

            # 明确解除 Python 引用，让这些临时 Tensor
            # 在 empty_cache() 前可以被回收。
            del logits
            del hidden_states
            del updated_gdn_states

        finally:
            reset_context()

    def allocate_model_caches(self):
        config = self.config
        text_config = self.text_config

        torch.cuda.synchronize()
        
        # 当前 GPU 的驱动级显存情况。
        free_bytes, total_bytes = (
            torch.cuda.mem_get_info()
        )

        used_bytes = total_bytes - free_bytes

        # warmup 期间 PyTorch allocator 观察到的峰值和当前值。
        memory_stats = torch.cuda.memory_stats()

        peak_bytes = memory_stats[
            "allocated_bytes.all.peak"
        ]

        current_bytes = memory_stats[
            "allocated_bytes.all.current"
        ]

        # warmup 中临时增加的显存：
        #
        # peak = 常驻显存 + 临时激活峰值
        # current = warmup 后仍然保留的常驻显存
        activation_headroom_bytes = (
            peak_bytes - current_bytes
        )

        # 最多允许本进程使用的显存。
        memory_limit_bytes = int(
            total_bytes
            * config.gpu_memory_utilization
        )

        # 分配 Cache 后仍然必须给运行时激活留出
        # warmup 测得的峰值空间。
        cache_budget_bytes = (
            memory_limit_bytes
            - used_bytes
            - activation_headroom_bytes
        )

        if cache_budget_bytes <= 0:
            raise RuntimeError(
                "No GPU memory remains for model caches"
            )

        if self.is_hybrid_model:
            spec = self.hybrid_cache_spec

            if spec is None:
                raise RuntimeError(
                    "Hybrid model is missing "
                    "HybridCacheSpec"
                )

            state_bytes_per_slot = (
                spec.state_bytes_per_slot
            )

            block_bytes = spec.kv_block_bytes(
                self.block_size
            )

            # Qwen3.5 只给 Full Attention 层分配 KV。
            num_cache_layers = (
                spec.num_full_attention_layers
            )

            if config.num_state_slots == -1:
                # 自动模式：
                # 最多使用规定比例的 Cache 预算保存状态。
                state_budget_bytes = int(
                    cache_budget_bytes
                    * config.gdn_state_memory_fraction
                )

                num_state_slots = min(
                    config.max_num_seqs,
                    state_budget_bytes
                    // state_bytes_per_slot,
                )
            else:
                # 手动模式：
                # 用户明确指定所需 slot 数量。
                num_state_slots = min(
                    config.num_state_slots,
                    config.max_num_seqs,
                )

            if num_state_slots <= 0:
                raise RuntimeError(
                    "GPU memory cannot hold even one "
                    "GDN state slot"
                )

            state_reserved_bytes = (
                num_state_slots
                * state_bytes_per_slot
            )

            kv_budget_bytes = (
                cache_budget_bytes
                - state_reserved_bytes
            )

            if kv_budget_bytes <= 0:
                raise RuntimeError(
                    "GDN state slots consume the entire "
                    "cache memory budget"
                )

            num_kvcache_blocks = (
                kv_budget_bytes // block_bytes
            )

            if num_kvcache_blocks <= 0:
                raise RuntimeError(
                    "GPU memory cannot hold even one "
                    "KV cache block after reserving "
                    "GDN states"
                )

            # 把实际计算结果写回 Config。
            #
            # LLMEngine 会在 ModelRunner 构造完成后，
            # 再使用同一个 Config 创建 Scheduler。
            config.num_state_slots = num_state_slots
            config.num_kvcache_blocks = (
                num_kvcache_blocks
            )

            # 真正在 GPU 上分配固定大小的状态池。
            self.hybrid_state_manager = (
                HybridStateManager(
                    spec=spec,
                    num_slots=num_state_slots,
                    device=f"cuda:{self.rank}",
                )
            )

            num_kv_heads = spec.num_kv_heads
            head_dim = spec.attention_head_dim

        else:
            # 原 Qwen3 路径：所有 DecoderLayer 都是
            # Full Attention，所以剩余预算全部用于 KV。
            num_cache_layers = (
                text_config.num_hidden_layers
            )

            num_kv_heads = (
                text_config.num_key_value_heads
                // self.world_size
            )

            head_dim = getattr(
                text_config,
                "head_dim",
                (
                    text_config.hidden_size
                    // text_config.num_attention_heads
                ),
            )

            dtype_bytes = torch.empty(
                (),
                dtype=text_config.dtype,
            ).element_size()

            block_bytes = (
                2
                * num_cache_layers
                * self.block_size
                * num_kv_heads
                * head_dim
                * dtype_bytes
            )

            config.num_kvcache_blocks = (
                cache_budget_bytes // block_bytes
            )

            if config.num_kvcache_blocks <= 0:
                raise RuntimeError(
                    "GPU memory cannot hold even one "
                    "KV cache block"
                )

        # 两种模型都会分配 KV Cache。
        #
        # Qwen3：
        # [2, num_hidden_layers, ...]
        #
        # Qwen3.5：
        # [2, num_full_attention_layers, ...]
        self.kv_cache = torch.empty(
            (
                2,
                num_cache_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
            ),
            dtype=text_config.dtype,
            device=f"cuda:{self.rank}",
        )

        # 按模型中的出现顺序，把每个 Attention 模块
        # 绑定到紧凑 KV Cache 的对应层。
        cache_layer_idx = 0

        for module in self.model.modules():
            if (
                hasattr(module, "k_cache")
                and hasattr(module, "v_cache")
            ):
                if cache_layer_idx >= num_cache_layers:
                    raise RuntimeError(
                        "Model contains more Attention "
                        "modules than allocated cache layers"
                    )

                module.k_cache = self.kv_cache[
                    0,
                    cache_layer_idx,
                ]

                module.v_cache = self.kv_cache[
                    1,
                    cache_layer_idx,
                ]

                cache_layer_idx += 1

        if cache_layer_idx != num_cache_layers:
            raise RuntimeError(
                "Attention module count does not match "
                "allocated KV cache layers: "
                f"expected {num_cache_layers}, "
                f"found {cache_layer_idx}"
            )

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        if self.uses_multimodal_rope:
            # 三行分别保存 T/H/W。
            positions = [
                [],
                [],
                [],
            ]
        else:
            # 原 Qwen3 保持一维位置。
            positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        prefill_seqlens = []
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            if seqlen_q <= 0:
                raise ValueError(
                    f"Sequence {seq.seq_id} has no "
                    "scheduled Prefill tokens"
                )

            prefill_seqlens.append(seqlen_q)
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            if self.uses_multimodal_rope:
                if (
                    seq.mrope_position_ids
                    is None
                ):
                    # Qwen3.5 纯文本请求。
                    #
                    # 三个轴使用完全相同的一维位置。
                    text_positions = list(
                        range(start, end)
                    )

                    for axis in range(3):
                        positions[axis].extend(
                            text_positions
                        )

                else:
                    # Qwen3.5 图文请求。
                    #
                    # Sequence 保存的是整条请求：
                    # [3, total_sequence_length]
                    #
                    # 本轮只取当前 Prefill chunk。
                    if (
                        end
                        > seq.mrope_position_ids
                        .shape[1]
                    ):
                        raise RuntimeError(
                            f"Sequence {seq.seq_id} "
                            "does not have enough "
                            "mRoPE positions for its "
                            "scheduled Prefill chunk"
                        )

                    chunk_positions = (
                        seq.mrope_position_ids[
                            :,
                            start:end,
                        ]
                    )

                    if chunk_positions.shape != (
                        3,
                        seqlen_q,
                    ):
                        raise RuntimeError(
                            "Chunked mRoPE position "
                            "shape mismatch: "
                            f"{tuple(chunk_positions.shape)}"
                        )

                    for axis in range(3):
                        positions[axis].extend(
                            chunk_positions[
                                axis
                            ].tolist()
                        )

            else:
                # 原 Qwen3 一维 RoPE。
                positions.extend(
                    range(start, end)
                )
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(
            input_ids,
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True)

        positions = torch.tensor(
            positions,
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True)

        # FLA GDN使用torch.long边界。
        # 必须在cu_seqlens_q这个Python列表被覆盖前构造。
        gdn_cu_seqlens = torch.tensor(
            cu_seqlens_q,
            dtype=torch.long,
            pin_memory=True,
        ).cuda(non_blocking=True)

        # Flash Attention使用torch.int32边界。
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=block_tables,
            prefill_seqlens=tuple(prefill_seqlens),
            gdn_cu_seqlens=gdn_cu_seqlens,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        if self.uses_multimodal_rope:
            positions = [
                [],
                [],
                [],
            ]
        else:
            positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(
                seq.last_token
            )

            token_index = len(seq) - 1

            if self.uses_multimodal_rope:
                if (
                    seq.mrope_position_delta
                    is None
                ):
                    # Qwen3.5 纯文本请求。
                    token_position = (
                        token_index
                    )

                else:
                    # Qwen3.5 图文请求。
                    token_position = (
                        token_index
                        + seq.mrope_position_delta
                    )

                # Decode 生成的是文本 token，
                # 所以 T/H/W 三轴位置相同。
                for axis in range(3):
                    positions[axis].append(
                        token_position
                    )

            else:
                # 原 Qwen3。
                positions.append(
                    token_index
                )

            context_lens.append(
                len(seq)
            )
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    # 把多个 temperature 拼成 Tensor
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def get_or_create_visual_embeddings(
        self,
        seq: Sequence,
    ) -> torch.Tensor:
        """
        返回一条请求完整的视觉 embeddings。

        缓存命中：
            直接返回 GPU Tensor。

        缓存未命中：
            运行 Vision Tower，保存后返回。
        """
        cached = self.visual_embedding_cache.get(
            seq.seq_id
        )

        if cached is not None:
            self.num_visual_cache_hits += 1
            return cached

        self.num_visual_cache_misses += 1
        
        if (
            seq.pixel_values is None
            or seq.image_grid_thw is None
            or seq.mm_token_type_ids is None
        ):
            raise RuntimeError(
                f"Sequence {seq.seq_id} does not "
                "contain complete image inputs"
            )

        pixel_values = seq.pixel_values.to(
            device=torch.device(
                "cuda",
                self.rank,
            ),
            dtype=self.model_dtype,
            non_blocking=True,
        )

        image_grid_thw = (
            seq.image_grid_thw.to(
                device=torch.device(
                    "cuda",
                    self.rank,
                ),
                dtype=torch.long,
                non_blocking=True,
            )
        )

        visual_embeddings = (
            self.model.get_visual_embeddings(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        )
        self.num_vision_forwards += 1

        expected_visual_tokens = sum(
            token_type == self.IMAGE_TOKEN_TYPE
            for token_type
            in seq.mm_token_type_ids
        )

        if (
            visual_embeddings.ndim != 2
            or visual_embeddings.shape[0]
            != expected_visual_tokens
        ):
            raise RuntimeError(
                "Vision Tower output token count "
                "mismatch: "
                f"expected {expected_visual_tokens}, "
                f"got "
                f"{tuple(visual_embeddings.shape)}"
            )

        # 推理阶段不需要 autograd graph。
        visual_embeddings = (
            visual_embeddings.detach()
        )

        self.visual_embedding_cache[
            seq.seq_id
        ] = visual_embeddings

        tensor_bytes = (
            visual_embeddings.numel()
            * visual_embeddings.element_size()
        )

        self.visual_cache_bytes += tensor_bytes

        self.peak_visual_cache_bytes = max(
            self.peak_visual_cache_bytes,
            self.visual_cache_bytes,
        )

        return visual_embeddings

    def release_visual_embedding_cache(
        self,
        seq_ids: list[int],
    ) -> None:
        """
        释放指定请求持有的视觉 embeddings。
        """

        for seq_id in seq_ids:
            visual_embeddings = (
                self.visual_embedding_cache.pop(
                    seq_id,
                    None,
                )
            )

            # 纯文本请求或尚未建立缓存的请求，
            # 释放时什么也不做。
            if visual_embeddings is None:
                continue

            tensor_bytes = (
                visual_embeddings.numel()
                * visual_embeddings.element_size()
            )

            self.visual_cache_bytes -= tensor_bytes

        if self.visual_cache_bytes < 0:
            raise RuntimeError(
                "Visual cache byte counter "
                "became negative"
            )

    @torch.inference_mode()
    def prepare_multimodal_embeddings(
        self,
        seqs: list[Sequence],
        input_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        为 Packed Prefill 构造图文 embeddings。

        如果当前 microbatch 没有任何图像 token，
        返回 None，调用方继续走 input_ids 路径。

        如果有图像 token：
            1. 为全部 packed tokens 查询文本 embedding；
            2. 运行对应请求的 Vision Tower；
            3. 用 visual embedding 替换图像位置。
        """

        replacement_plans = []

        packed_start = 0

        # =====================================
        # 第一阶段：定位所有需要替换的位置
        # =====================================

        for seq in seqs:
            start = seq.num_cached_tokens

            end = (
                start
                + seq.num_scheduled_tokens
            )

            chunk_length = (
                seq.num_scheduled_tokens
            )

            if chunk_length <= 0:
                raise ValueError(
                    f"Sequence {seq.seq_id} has no "
                    "scheduled Prefill tokens"
                )

            if (
                seq.mm_token_type_ids
                is None
            ):
                # 纯文本请求。
                packed_start += chunk_length
                continue

            if (
                len(seq.mm_token_type_ids)
                < end
            ):
                raise RuntimeError(
                    f"Sequence {seq.seq_id} does "
                    "not have enough multimodal "
                    "token types"
                )

            chunk_token_types = (
                seq.mm_token_type_ids[
                    start:end
                ]
            )

            # 当前 chunk 内部，哪些位置是图像 token。
            local_image_positions = [
                local_index
                for local_index, token_type
                in enumerate(
                    chunk_token_types
                )
                if token_type
                == self.IMAGE_TOKEN_TYPE
            ]

            # 当前 chunk 不包含图像 token。
            #
            # 例如图片已经在前一个 chunk 处理完，
            # 这一轮只处理图片后的文本。
            if not local_image_positions:
                packed_start += chunk_length
                continue

            if self.image_token_id is None:
                raise RuntimeError(
                    "image_token_id has not been "
                    "initialized"
                )

            chunk_token_ids = seq[
                start:end
            ]

            # 再次确认 mm type=1 的位置确实是
            # <|image_pad|> token。
            for local_index in (
                local_image_positions
            ):
                if (
                    chunk_token_ids[local_index]
                    != self.image_token_id
                ):
                    raise RuntimeError(
                        "Image token type does not "
                        "point to image_token_id"
                    )

            if (
                seq.pixel_values is None
                or seq.image_grid_thw is None
            ):
                raise RuntimeError(
                    f"Sequence {seq.seq_id} contains "
                    "image tokens but does not carry "
                    "image tensors"
                )

            # 当前 chunk 之前已经跳过多少个
            # image token。
            #
            # 它决定应该从 visual_embeddings
            # 的哪一行开始读取。
            visual_start = sum(
                token_type
                == self.IMAGE_TOKEN_TYPE
                for token_type
                in seq.mm_token_type_ids[
                    :start
                ]
            )

            visual_end = (
                visual_start
                + len(local_image_positions)
            )

            # 将 chunk 内部下标转换成 packed batch
            # 中的全局行号。
            packed_image_positions = [
                packed_start + local_index
                for local_index
                in local_image_positions
            ]

            replacement_plans.append(
                (
                    seq,
                    packed_image_positions,
                    visual_start,
                    visual_end,
                )
            )

            packed_start += chunk_length

        if packed_start != input_ids.numel():
            raise RuntimeError(
                "Packed token count does not match "
                "input_ids: "
                f"{packed_start} != "
                f"{input_ids.numel()}"
            )

        # 当前 batch 没有图像 token。
        #
        # 返回 None 后，模型继续使用 input_ids，
        # 保持原纯文本路径完全不变。
        if not replacement_plans:
            return None

        # =====================================
        # 第二阶段：查询全部文本 embeddings
        # =====================================

        inputs_embeds = (
            self.model.embed_input_ids(
                input_ids
            )
        )

        if inputs_embeds.ndim != 2:
            raise RuntimeError(
                "Text embeddings must have shape "
                "[num_tokens, hidden_size]"
            )

        if (
            inputs_embeds.shape[0]
            != input_ids.numel()
        ):
            raise RuntimeError(
                "Text embedding token count "
                "does not match input_ids"
            )

        # =====================================
        # 第三阶段：运行 Vision Tower 并替换
        # =====================================

        for (
            seq,
            packed_image_positions,
            visual_start,
            visual_end,
        ) in replacement_plans:
            visual_embeddings = (
                self.get_or_create_visual_embeddings(
                    seq
                )
            )

            if (
                visual_embeddings.shape[1]
                != inputs_embeds.shape[1]
            ):
                raise RuntimeError(
                    "Vision output hidden size "
                    "does not match text hidden size"
                )

            selected_visual_embeddings = (
                visual_embeddings[
                    visual_start:visual_end
                ]
            )

            if (
                selected_visual_embeddings
                .shape[0]
                != len(
                    packed_image_positions
                )
            ):
                raise RuntimeError(
                    "Selected visual embedding "
                    "count does not match packed "
                    "image positions"
                )

            destination_indices = (
                torch.tensor(
                    packed_image_positions,
                    dtype=torch.long,
                    device=input_ids.device,
                )
            )

            # 把图像位置的普通 token embedding
            # 替换成真实 visual embedding。
            inputs_embeds.index_copy_(
                0,
                destination_indices,
                selected_visual_embeddings.to(
                    dtype=inputs_embeds.dtype,
                ),
            )

        return inputs_embeds


    @torch.inference_mode()
    def run_hybrid_prefill(
        self,
        seqs: list[Sequence],
    ) -> list[int]:
        """
        一次模型Forward处理整个Variable-length
        Prefill microbatch。
        """

        if not seqs:
            raise ValueError(
                "Prefill batch must not be empty"
            )
            
        self.num_prefill_microbatches += 1

        self.max_observed_prefill_batch_size = max(
            self.max_observed_prefill_batch_size,
            len(seqs),
        )

        has_multimodal_request = any(
            seq.is_multimodal
            for seq in seqs
        )

        has_text_only_request = any(
            not seq.is_multimodal
            for seq in seqs
        )

        if (
            has_multimodal_request
            and has_text_only_request
        ):
            self.num_mixed_prefill_microbatches += 1

        state_manager = (
            self.hybrid_state_manager
        )

        if state_manager is None:
            raise RuntimeError(
                "HybridStateManager has not been "
                "allocated"
            )

        # 整个 Prefill microbatch 是否都是从 token 0
        # 开始计算的新请求或抢占重算请求。
        #
        # 如果所有 Sequence 的 num_cached_tokens 都为 0，
        # 就没有任何需要从 State Pool 中读取的旧 GDN 状态。
        all_fresh_prefill = all(
            seq.num_cached_tokens == 0
            for seq in seqs
        )

        state_slots = []


        # =====================================
        # 第一阶段：收集每条请求的state slot
        # =====================================

        for seq in seqs:
            if seq.state_slot is None:
                raise RuntimeError(
                    f"Sequence {seq.seq_id} does not "
                    "own a GDN state slot"
                )

            state_slot = seq.state_slot

            if seq.num_cached_tokens == 0:
                # 两种情况：
                #
                # 1. 新请求首次Prefill；
                # 2. 请求被抢占后从token 0重算。
                #
                # slot可能以前属于其他请求，
                # 所以先标记为未初始化。
                state_manager.reset_slot(
                    state_slot
                )

            elif not (
                state_manager.is_slot_initialized(
                    state_slot
                )
            ):
                # 如果已经缓存了一部分token，说明这是
                # Chunked Prefill。此时必须存在上一个
                # chunk保存的GDN状态。
                raise RuntimeError(
                    f"Sequence {seq.seq_id} has "
                    f"{seq.num_cached_tokens} cached "
                    "tokens, but GDN state slot "
                    f"{state_slot} is not initialized"
                )

            state_slots.append(
                state_slot
            )

        # =====================================
        # 第二阶段：构造Packed Prefill输入
        # =====================================

        input_ids, positions = (
            self.prepare_prefill(seqs)
        )
        inputs_embeds = (
            self.prepare_multimodal_embeddings(
                seqs,
                input_ids,
            )
        )

        try:
            # =================================
            # 第三阶段：批量Gather旧GDN状态
            # =================================

            if all_fresh_prefill:
                # 所有请求都是首次 Prefill，或者抢占后从
                # token 0 开始重算。
                #
                # 它们没有需要继承的历史 GDN 状态。
                # Qwen3_5Model.forward() 接收到 None 后，
                # 会让所有 GDN 层从零状态开始计算。
                #
                # 这样可以避免把全部 GDN 层的空旧状态
                # 从 State Pool Gather 成大体积临时 Tensor。
                old_gdn_states = None

            else:
                # 至少有一条请求需要继承历史状态。
                #
                # 可能的情况：
                # 1. Chunked Prefill 的后续 chunk；
                # 2. Prefix Cache 命中后的 suffix Prefill；
                # 3. fresh 和 continuation 混合的 Prefill batch。
                old_gdn_states = (
                    state_manager
                    .read_batched_states(
                        state_slots
                    )
                )

            # =================================
            # 第四阶段：一次模型Forward
            # =================================

            (
                hidden_states,
                updated_gdn_states,
            ) = self.model(
                input_ids=(
                    input_ids
                    if inputs_embeds is None
                    else None
                ),
                positions=positions,
                gdn_states=old_gdn_states,
                inputs_embeds=(
                    inputs_embeds
                ),
            )

            # ParallelLMHead会根据：
            #
            # context.cu_seqlens_q[1:] - 1
            #
            # 自动选择每条Sequence本轮最后一个token。
            logits = self.model.compute_logits(
                hidden_states
            )

            expected_batch_size = len(seqs)

            if (
                logits.shape[0]
                != expected_batch_size
            ):
                raise RuntimeError(
                    "Prefill logits batch size does "
                    "not match the number of "
                    "sequences: "
                    f"expected {expected_batch_size}, "
                    f"got {logits.shape[0]}"
                )

            # =================================
            # 第五阶段：批量Scatter新GDN状态
            # =================================

            state_manager.write_batched_states(
                state_slots,
                updated_gdn_states,
            )

            # =================================
            # 第六阶段：一次Batch Sampling
            # =================================

            temperatures = self.prepare_sample(
                seqs
            )

            token_ids = self.sampler(
                logits,
                temperatures,
            )

            return token_ids.tolist()

        finally:
            # Context是当前模型Forward的临时元数据。
            # 即使中途报错也必须清空，不能影响下一轮。
            reset_context()

    @torch.inference_mode()
    def run_hybrid_decode(
        self,
        seqs: list[Sequence],
    ) -> list[int]:

        state_manager = (
            self.hybrid_state_manager
        )

        if state_manager is None:
            raise RuntimeError(
                "HybridStateManager has not "
                "been allocated"
            )

        state_slots: list[int] = []

        for seq in seqs:
            if seq.state_slot is None:
                raise RuntimeError(
                    f"Sequence {seq.seq_id} does "
                    "not own a GDN state slot"
                )

            if seq.num_cached_tokens <= 0:
                raise RuntimeError(
                    f"Sequence {seq.seq_id} entered "
                    "Decode before Prefill completed"
                )

            if not (
                state_manager
                .is_slot_initialized(
                    seq.state_slot
                )
            ):
                raise RuntimeError(
                    f"GDN state slot "
                    f"{seq.state_slot} for "
                    f"Sequence {seq.seq_id} is "
                    "not initialized"
                )

            state_slots.append(
                seq.state_slot
            )

        input_ids, positions = (
            self.prepare_decode(seqs)
        )

        try:
            policy = self.hybrid_graph_policy

            if policy is None:
                raise RuntimeError(
                    "Hybrid CUDA Graph policy "
                    "has not been initialized"
                )

            route = policy.route(
                batch_size=len(seqs),
                is_prefill=False,
                enforce_eager=(
                    self.enforce_eager
                ),
            )

            if route.use_graph:
                # Graph 路径：
                #
                # Gather、模型执行和 Scatter
                # 都在 captured CUDA Graph 内完成。
                hidden_states = (
                    self.replay_hybrid_cudagraph(
                        batch_size=len(seqs),
                        input_ids=input_ids,
                        positions=positions,
                        state_slots=state_slots,
                    )
                )

            else:
                self.num_hybrid_graph_eager_fallbacks += 1

                old_count = (
                    self
                    .hybrid_graph_fallback_reasons
                    .get(
                        route.reason,
                        0,
                    )
                )

                self.hybrid_graph_fallback_reasons[
                    route.reason
                ] = old_count + 1

                # Eager 路径仍然使用原来的
                # Python list[int] Gather 接口。
                old_gdn_states = (
                    state_manager
                    .read_batched_states(
                        state_slots
                    )
                )

                (
                    hidden_states,
                    updated_gdn_states,
                ) = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    gdn_states=old_gdn_states,
                )

                # Eager 模型执行不包含 Scatter，
                # 因此仍然需要显式写回状态池。
                state_manager.write_batched_states(
                    state_slots,
                    updated_gdn_states,
                )

            # LM Head 暂时不进入 Graph。
            logits = self.model.compute_logits(
                hidden_states
            )

            # 首版 Sampling 不进入 Graph。
            temperatures = self.prepare_sample(
                seqs
            )

            token_ids = self.sampler(
                logits,
                temperatures,
            )

            return token_ids.tolist()

        finally:
            reset_context()
            
    def run_hybrid(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> list[int]:

        if is_prefill:
            return self.run_hybrid_prefill(
                seqs
            )

        return self.run_hybrid_decode(
            seqs
        )

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> list[int]:

        if self.is_hybrid_model:
            return self.run_hybrid(
                seqs,
                is_prefill,
            )

        input_ids, positions = (
            self.prepare_prefill(seqs)
            if is_prefill
            else self.prepare_decode(seqs)
        )

        temperatures = (
            self.prepare_sample(seqs)
            if self.rank == 0
            else None
        )

        logits = self.run_model(
            input_ids,
            positions,
            is_prefill,
        )

        token_ids = (
            self.sampler(
                logits,
                temperatures,
            ).tolist()
            if self.rank == 0
            else None
        )

        reset_context()

        return token_ids

    @torch.inference_mode()
    def run_hybrid_graph_body(
        self,
        batch_size: int,
    ) -> None:
        """
        Hybrid CUDA Graph 真正捕获的 GPU 工作。

        捕获内容：
        1. 根据固定地址 state_slot_ids Gather GDN 状态；
        2. 执行 Qwen3.5 Hybrid Decode；
        3. 将更新后的 GDN 状态 Scatter 回状态池；
        4. 保存 Decoder hidden states。

        LM Head 和 Sampling 暂时仍在 Graph 外。
        """

        workspace = (
            self.hybrid_graph_workspace
        )

        if workspace is None:
            raise RuntimeError(
                "Hybrid CUDA Graph Workspace "
                "has not been allocated"
            )

        state_manager = (
            self.hybrid_state_manager
        )

        if state_manager is None:
            raise RuntimeError(
                "HybridStateManager has not "
                "been allocated"
            )

        # 这个 Tensor 的地址在 Capture 和 Replay
        # 期间不变，但其中保存的 slot ID 会在每次
        # Replay 前根据真实请求更新。
        slot_indices = (
            workspace.state_slot_ids[
                :batch_size
            ]
        )

        # Graph 内 Gather：
        #
        # active state pool
        #     -> 按 slot_indices 选择请求
        #     -> 连续的 batch GDN states
        input_states = (
            state_manager
            .gather_batched_states_for_graph(
                slot_indices
            )
        )

        (
            hidden_states,
            updated_gdn_states,
        ) = self.model(
            input_ids=(
                workspace.input_ids[
                    :batch_size
                ]
            ),
            positions=(
                workspace.positions[
                    :,
                    :batch_size,
                ]
            ),
            gdn_states=input_states,
        )

        # Graph 内 Scatter：
        #
        # updated batch GDN states
        #     -> 根据相同 slot_indices
        #     -> 写回 active state pool
        state_manager.scatter_batched_states_for_graph(
            slot_indices,
            updated_gdn_states,
        )

        # LM Head 仍在 Graph 外执行，因此保留一份
        # 固定地址的 hidden_states 输出。
        workspace.hidden_states[
            :batch_size
        ].copy_(
            hidden_states
        )

    @torch.inference_mode()
    def capture_hybrid_cudagraph(
        self,
    ) -> None:
        """
        为配置中的 Hybrid Decode batch sizes
        捕获 CUDA Graph。

        首版建议只配置：
            hybrid_cuda_graph_batch_sizes=(1,)
        """

        spec = self.hybrid_cache_spec

        if spec is None:
            raise RuntimeError(
                "Hybrid CUDA Graph requires "
                "HybridCacheSpec"
            )

        if self.hybrid_state_manager is None:
            raise RuntimeError(
                "Hybrid CUDA Graph requires "
                "HybridStateManager"
            )

        batch_sizes = (
            self.config
            .hybrid_cuda_graph_batch_sizes
        )

        max_batch_size = batch_sizes[-1]

        max_num_blocks = (
            self.config.max_model_len
            + self.block_size
            - 1
        ) // self.block_size

        if (
            self.config.num_kvcache_blocks
            < max_batch_size
        ):
            raise RuntimeError(
                "CUDA Graph capture requires at "
                "least one scratch KV block per "
                "captured batch row"
            )

        memory_before = (
            torch.cuda.memory_allocated(
                self.rank
            )
        )

        workspace = (
            HybridDecodeStaticWorkspace
            .allocate(
                spec=spec,
                max_batch_size=max_batch_size,
                max_num_blocks=max_num_blocks,
                hidden_size=(
                    self.text_config.hidden_size
                ),
                device=f"cuda:{self.rank}",
            )
        )

        # Capture 的虚拟 Decode 从确定的零状态开始。
        workspace.hidden_states.zero_()

        self.hybrid_graph_workspace = (
            workspace
        )

        self.hybrid_graphs = {}
        self.hybrid_graph_pool = None

        # 和原 nano-vLLM 一样，从大 batch 向小 batch
        # 捕获，便于共享 Graph memory pool。
        for batch_size in reversed(
            batch_sizes
        ):
            # ---------------------------------
            # 构造合法的虚拟 Decode metadata
            # ---------------------------------

            workspace.input_ids[
                :batch_size
            ].zero_()

            workspace.positions[
                :,
                :batch_size,
            ].zero_()

            # 第 i 条虚拟请求使用第 i 个物理 KV block。
            scratch_block_ids = torch.arange(
                batch_size,
                dtype=torch.int32,
                device=workspace.device,
            )

            workspace.slot_mapping.fill_(-1)

            workspace.slot_mapping[
                :batch_size
            ].copy_(
                scratch_block_ids
                * self.block_size
            )

            workspace.context_lens.zero_()

            # 每条 Capture 请求当前上下文只有一个 token。
            workspace.context_lens[
                :batch_size
            ].fill_(1)

            workspace.block_tables.fill_(-1)

            workspace.block_tables[
                :batch_size,
                0,
            ].copy_(
                scratch_block_ids
            )

            workspace.state_slot_ids.fill_(-1)
            
            # Graph Capture 使用 slot 0...B-1
            # 作为虚拟请求状态槽。
            #
            # 状态池由 torch.empty() 分配，因此捕获
            # 前必须初始化，避免读取未定义值。
            #
            # 这里只发生在 Engine 初始化期间，
            # 此时还没有真实请求占用 state slot。
            workspace.state_slot_ids[
                :batch_size
            ].copy_(
                torch.arange(
                    batch_size,
                    dtype=torch.long,
                    device=workspace.device,
                )
            )
            self.hybrid_state_manager.conv_state_pool[
                :batch_size
            ].zero_()

            self.hybrid_state_manager.recurrent_state_pool[
                :batch_size
            ].zero_()


            # Attention.forward() 在 Capture 时通过
            # Context 取得这些固定地址。
            set_context(
                is_prefill=False,
                slot_mapping=(
                    workspace.slot_mapping[
                        :batch_size
                    ]
                ),
                context_lens=(
                    workspace.context_lens[
                        :batch_size
                    ]
                ),
                block_tables=(
                    workspace.block_tables[
                        :batch_size
                    ]
                ),
            )

            try:
                # -----------------------------
                # 在非默认 Stream 上预热
                # -----------------------------

                warmup_stream = (
                    torch.cuda.Stream(
                        device=self.rank
                    )
                )

                warmup_stream.wait_stream(
                    torch.cuda.current_stream(
                        self.rank
                    )
                )

                with torch.cuda.stream(
                    warmup_stream
                ):
                    for _ in range(3):
                        self.run_hybrid_graph_body(
                            batch_size
                        )

                torch.cuda.current_stream(
                    self.rank
                ).wait_stream(
                    warmup_stream
                )

                torch.cuda.synchronize(
                    self.rank
                )

                # -----------------------------
                # 捕获真正的 CUDA Graph
                # -----------------------------

                graph = torch.cuda.CUDAGraph()

                with torch.cuda.graph(
                    graph,
                    self.hybrid_graph_pool,
                ):
                    self.run_hybrid_graph_body(
                        batch_size
                    )

                if self.hybrid_graph_pool is None:
                    self.hybrid_graph_pool = (
                        graph.pool()
                    )

                self.hybrid_graphs[
                    batch_size
                ] = graph

                torch.cuda.synchronize(
                    self.rank
                )

            finally:
                reset_context()

        memory_after = (
            torch.cuda.memory_allocated(
                self.rank
            )
        )

        self.hybrid_graph_capture_allocated_bytes = (
            max(
                0,
                memory_after - memory_before,
            )
        )

    @torch.inference_mode()
    def replay_hybrid_cudagraph(
        self,
        *,
        batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        state_slots: list[int],
    ):
        """
        把本轮动态 Decode 数据复制进固定 Workspace，
        然后 Replay 对应 batch size 的 Graph。
        """

        workspace = (
            self.hybrid_graph_workspace
        )

        if workspace is None:
            raise RuntimeError(
                "Hybrid CUDA Graph Workspace "
                "has not been allocated"
            )

        graph = self.hybrid_graphs.get(
            batch_size
        )

        if graph is None:
            raise RuntimeError(
                "No captured Hybrid CUDA Graph "
                f"for batch size {batch_size}"
            )

        dynamic_context = get_context()

        if (
            dynamic_context.slot_mapping is None
            or dynamic_context.context_lens is None
            or dynamic_context.block_tables is None
        ):
            raise RuntimeError(
                "prepare_decode() did not create "
                "complete Paged KV metadata"
            )

        workspace.copy_decode_inputs(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=(
                dynamic_context.slot_mapping
            ),
            context_lens=(
                dynamic_context.context_lens
            ),
            block_tables=(
                dynamic_context.block_tables
            ),
            state_slot_ids=state_slots,
        )

        # Context 切换到 Capture 时使用的固定地址。
        #
        # Replay 本身不会重新执行 Python，
        # 但这样可以保证当前逻辑 Context 与 Graph
        # 中记录的 Tensor 完全一致。
        set_context(
            is_prefill=False,
            slot_mapping=(
                workspace.slot_mapping[
                    :batch_size
                ]
            ),
            context_lens=(
                workspace.context_lens[
                    :batch_size
                ]
            ),
            block_tables=(
                workspace.block_tables[
                    :batch_size
                ]
            ),
        )

        graph.replay()

        self.num_hybrid_graph_replays += 1

        hidden_states = (
            workspace.hidden_states[
                :batch_size
            ]
        )

        # GDN 状态已经在 Graph 内 Scatter 回
        # active state pool，因此这里只返回
        # LM Head 所需的 hidden states。
        return hidden_states

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        text_config = self.text_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, text_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
