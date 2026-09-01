import atexit
from dataclasses import (
    dataclass,
    field,
    fields,
)
from time import perf_counter
from tqdm.auto import tqdm
import torch
import torch.multiprocessing as mp


from nanovllm.engine.prefix_cache import (
    PrefixKey,
    PrefixStateCache,
)
from nanovllm.config import Config
from nanovllm.inputs import (
    InputProcessor,
    PromptInput,
)
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

@dataclass(slots=True)
class StepStats:
    """
    一轮逻辑 step 中两个 microbatch 的统计。
    """

    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0

    decode_elapsed: float = 0.0
    prefill_elapsed: float = 0.0

    @property
    def decode_throughput(self) -> float:
        if (
            self.num_decode_tokens == 0
            or self.decode_elapsed == 0
        ):
            return 0.0

        return (
            self.num_decode_tokens
            / self.decode_elapsed
        )

    @property
    def prefill_throughput(self) -> float:
        if (
            self.num_prefill_tokens == 0
            or self.prefill_elapsed == 0
        ):
            return 0.0

        return (
            self.num_prefill_tokens
            / self.prefill_elapsed
        )
        

@dataclass(slots=True)
class RequestMetrics:
    """
    一条请求从进入 Engine 到完成的生命周期指标。
    """

    seq_id: int
    num_prompt_tokens: int

    arrival_time: float
    enqueue_time: float

    first_scheduled_time: (
        float | None
    ) = None

    first_token_time: (
        float | None
    ) = None

    finish_time: (
        float | None
    ) = None

    num_completion_tokens: int = 0
    
    # 每一个生成 token 被 Engine 观察到的时间。
    #
    # token_timestamps[0]：
    #     第一个输出 token 的时间。
    #
    # token_timestamps[1]：
    #     第二个输出 token 的时间。
    token_timestamps: list[float] = (
        field(default_factory=list)
    )

    @property
    def preprocessing_ms(
        self,
    ) -> float:
        return (
            self.enqueue_time
            - self.arrival_time
        ) * 1000.0

    @property
    def queue_ms(
        self,
    ) -> float | None:
        if (
            self.first_scheduled_time
            is None
        ):
            return None

        return (
            self.first_scheduled_time
            - self.enqueue_time
        ) * 1000.0

    @property
    def ttft_ms(
        self,
    ) -> float | None:
        if self.first_token_time is None:
            return None

        return (
            self.first_token_time
            - self.arrival_time
        ) * 1000.0

    @property
    def tpot_ms(
        self,
    ) -> float | None:
        if (
            self.first_token_time is None
            or self.finish_time is None
        ):
            return None

        if self.num_completion_tokens <= 1:
            return 0.0

        return (
            (
                self.finish_time
                - self.first_token_time
            )
            * 1000.0
            / (
                self.num_completion_tokens
                - 1
            )
        )

    @property
    def e2e_ms(
        self,
    ) -> float | None:
        if self.finish_time is None:
            return None

        return (
            self.finish_time
            - self.arrival_time
        ) * 1000.0
        
class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(
            config,
            0,
            self.events,
        )

        self.input_processor = (
            InputProcessor(config)
        )

        # 保持 generate() 最后的 decode 逻辑不变。
        self.tokenizer = (
            self.input_processor.tokenizer
        )
        
        model_eos_token_id = getattr(
            config.text_config,
            "eos_token_id",
            None,
        )

        if model_eos_token_id is None:
            model_eos_token_id = (
                self.tokenizer.eos_token_id
            )

        if not isinstance(model_eos_token_id, int):
            raise NotImplementedError(
                "Multiple EOS token IDs are not supported"
            )

        config.eos = model_eos_token_id
        self.scheduler = Scheduler(config)
        
        self.hybrid_prefix_cache_mode = (
            config.hybrid_prefix_cache_mode
        )

        self.prefix_checkpoint_interval_blocks = (
            config.prefix_checkpoint_interval_blocks
        )

        self.max_new_prefix_snapshots_per_request = (
            config.max_new_prefix_snapshots_per_request
        )

        self.prefix_state_cache: (
            PrefixStateCache | None
        ) = None

        if (
            self.hybrid_prefix_cache_mode
            != "disabled"
        ):
            state_manager = (
                self.model_runner
                .hybrid_state_manager
            )

            if state_manager is None:
                raise RuntimeError(
                    "Hybrid Prefix Cache requires an "
                    "initialized HybridStateManager"
                )

            snapshot_dtype_by_name = {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }

            recurrent_snapshot_dtype = (
                snapshot_dtype_by_name[
                    config.prefix_recurrent_snapshot_dtype
                ]
            )

            spec = state_manager.spec

            # Prefix Cache 目前位于进程内存中，不会跨进程
            # 持久化，但仍然为所有影响状态布局的参数
            # 建立明确 namespace。
            self.prefix_model_namespace = (
                "hybrid-prefix-v1"
                f"|model={config.model}"
                f"|model_dtype={self.model_runner.model_dtype}"
                f"|active_recurrent_dtype="
                f"{spec.recurrent_dtype}"
                f"|snapshot_recurrent_dtype="
                f"{recurrent_snapshot_dtype}"
                f"|conv_dtype={spec.conv_dtype}"
                f"|block_size={config.kvcache_block_size}"
                f"|gdn_layers={spec.num_gdn_layers}"
            )

            self.prefix_state_cache = (
                PrefixStateCache(
                    block_manager=(
                        self.scheduler.block_manager
                    ),
                    state_manager=state_manager,
                    model_namespace=(
                        self.prefix_model_namespace
                    ),
                    recurrent_snapshot_dtype=(
                        recurrent_snapshot_dtype
                    ),
                )
            )
        else:
            self.prefix_model_namespace = None
            
        # 无论 Cache 是否开启，都显式通知 Scheduler。
        #
        # disabled 模式传入 None；
        # opportunistic 模式传入 PrefixStateCache。
        self.scheduler.set_prefix_state_cache(
            self.prefix_state_cache
        )        

        # seq_id -> 请求生命周期指标
        self.request_metrics: dict[
            int,
            RequestMetrics,
        ] = {}
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self,
        prompt: PromptInput,
        sampling_params: SamplingParams,
    ):
        arrival_time = perf_counter()
        
        processed_prompt = (
            self.input_processor.process(
                prompt
            )
        )

        seq = Sequence(
            processed_prompt.token_ids,
            sampling_params,
            mm_token_type_ids=(
                processed_prompt
                .mm_token_type_ids
            ),
            pixel_values=(
                processed_prompt
                .pixel_values
            ),
            image_grid_thw=(
                processed_prompt
                .image_grid_thw
            ),
            mrope_position_ids=(
                processed_prompt
                .mrope_position_ids
            ),
            mrope_position_delta=(
                processed_prompt
                .mrope_position_delta
            ),
        )

        enqueue_time = perf_counter()

        self.scheduler.add(seq)

        if seq.seq_id in self.request_metrics:
            raise RuntimeError(
                f"Duplicate request metrics for "
                f"Sequence {seq.seq_id}"
            )

        self.request_metrics[
            seq.seq_id
        ] = RequestMetrics(
            seq_id=seq.seq_id,
            num_prompt_tokens=(
                seq.num_prompt_tokens
            ),
            arrival_time=arrival_time,
            enqueue_time=enqueue_time,
        )

        return seq.seq_id

    def _record_first_scheduled(
        self,
        seqs: list[Sequence],
    ) -> None:
        if not seqs:
            return

        scheduled_time = perf_counter()

        for seq in seqs:
            metrics = self.request_metrics.get(
                seq.seq_id
            )

            if metrics is None:
                raise RuntimeError(
                    f"Sequence {seq.seq_id} has "
                    "no request metrics"
                )

            # Chunked Prefill 会多次调度，
            # 这里只记录第一次。
            if (
                metrics.first_scheduled_time
                is None
            ):
                metrics.first_scheduled_time = (
                    scheduled_time
                )

    def _record_request_progress(
        self,
        seqs: list[Sequence],
    ) -> None:
        if not seqs:
            return

        progress_time = perf_counter()

        for seq in seqs:
            metrics = self.request_metrics.get(
                seq.seq_id
            )

            if metrics is None:
                raise RuntimeError(
                    f"Sequence {seq.seq_id} has "
                    "no request metrics"
                )

            completion_tokens = (
                seq.num_completion_tokens
            )

            # 当前 Sequence 已经有多少输出 token，
            # 与目前保存了多少个时间戳之间的差值，
            # 就是本轮新产生的 token 数量。
            num_new_timestamps = (
                completion_tokens
                - len(metrics.token_timestamps)
            )

            if num_new_timestamps < 0:
                raise RuntimeError(
                    f"Sequence {seq.seq_id} completion "
                    "token count moved backwards"
                )

            # 当前引擎每条请求每轮最多产生一个 token。
            #
            # 这里仍然用 range，是为了让统计逻辑能够处理
            # 将来一次返回多个 token 的情况。
            for _ in range(
                num_new_timestamps
            ):
                metrics.token_timestamps.append(
                    progress_time
                )

            # Prefill 完成后会产生第一个 token。
            if (
                completion_tokens >= 1
                and metrics.first_token_time
                is None
            ):
                metrics.first_token_time = (
                    progress_time
                )

            metrics.num_completion_tokens = (
                completion_tokens
            )

            if (
                seq.is_finished
                and metrics.finish_time is None
            ):
                metrics.finish_time = (
                    progress_time
                )

    def _commit_opportunistic_prefixes(
        self,
        seqs: list[Sequence],
    ) -> None:
        """
        为本轮 Prefill 自然结束在合法 checkpoint
        的纯文本请求创建联合 KV + GDN Prefix Entry。

        调用时必须满足：

            1. ModelRunner 已经写回最新 GDN state；
            2. BlockManager 已经记录完整 block hash；
            3. Scheduler.postprocess() 尚未释放资源。
        """

        cache = self.prefix_state_cache

        if cache is None:
            return

        if (
            self.hybrid_prefix_cache_mode
            != "opportunistic"
        ):
            return

        checkpoint_interval_tokens = (
            self.prefix_checkpoint_interval_blocks
            * self.scheduler.block_size
        )

        for seq in seqs:
            # 首版只支持纯文本 Prefix Cache。
            #
            # 不同图片可能拥有相同的 image placeholder
            # token IDs，因此不能只按 token hash 共享。
            if seq.is_multimodal:
                continue

            # 每条请求限制创建的新 snapshot 数量。
            if (
                seq.num_prefix_snapshots_created
                >= self.max_new_prefix_snapshots_per_request
            ):
                continue

            if seq.state_slot is None:
                raise RuntimeError(
                    f"Sequence {seq.seq_id} lost its "
                    "GDN state slot before Prefix commit"
                )

            checkpoint_tokens = (
                seq.num_cached_tokens
                + seq.num_scheduled_tokens
            )

            # Prefix Entry 不保存 logits。
            #
            # 因此命中后至少要保留一个 Prompt token
            # 交给模型执行，以重新产生最后位置的 logits。
            if (
                checkpoint_tokens
                >= seq.num_prompt_tokens
            ):
                continue

            # 只接受完整 token-block 边界。
            if (
                checkpoint_tokens
                % self.scheduler.block_size
                != 0
            ):
                continue

            # 进一步应用稀疏 checkpoint 间隔。
            #
            # block_size=256，interval_blocks=4：
            # checkpoint 必须位于 1024、2048、3072……
            if (
                checkpoint_tokens
                % checkpoint_interval_tokens
                != 0
            ):
                continue

            (
                kv_block_ids,
                block_hash,
            ) = (
                self.scheduler.block_manager
                .prefix_metadata_at_boundary(
                    seq,
                    num_cached_tokens=(
                        checkpoint_tokens
                    ),
                )
            )

            key = PrefixKey(
                model_namespace=(
                    self.prefix_model_namespace
                ),
                block_hash=block_hash,
                num_cached_tokens=(
                    checkpoint_tokens
                ),
            )

            _, created = cache.commit(
                key=key,
                kv_block_ids=kv_block_ids,
                state_slot=seq.state_slot,
            )

            if created:
                seq.num_prefix_snapshots_created += 1

    def step(
        self,
    ) -> tuple[
        list[tuple[int, list[int]]],
        StepStats,
    ]:
        plan = self.scheduler.schedule()
        if plan.preempted_seq_ids:
            self.model_runner.call(
                "release_visual_embedding_cache",
                plan.preempted_seq_ids,
            ) 
            
        self._record_first_scheduled(
            plan.prefill_seqs
        )
        stats = StepStats(
            num_decode_tokens=(
                plan.num_decode_tokens
            ),
            num_prefill_tokens=(
                plan.num_prefill_tokens
            ),
        )

        outputs: list[
            tuple[int, list[int]]
        ] = []

        # =====================================
        # 第一阶段：Decode microbatch
        # =====================================

        if plan.decode_seqs:
            start = perf_counter()

            decode_token_ids = (
                self.model_runner.call(
                    "run",
                    plan.decode_seqs,
                    False,
                )
            )

            self.scheduler.record_computed_block_metadata(
                plan.decode_seqs
            )

            self.scheduler.postprocess(
                plan.decode_seqs,
                decode_token_ids,
                False,
            )
            self._record_request_progress(
                plan.decode_seqs
            )
            
            finished_decode_seq_ids = [
                seq.seq_id
                for seq in plan.decode_seqs
                if seq.is_finished
            ]

            if finished_decode_seq_ids:
                self.model_runner.call(
                    "release_visual_embedding_cache",
                    finished_decode_seq_ids,
                )

            stats.decode_elapsed = (
                perf_counter() - start
            )

            outputs.extend(
                (
                    seq.seq_id,
                    seq.completion_token_ids,
                )
                for seq in plan.decode_seqs
                if seq.is_finished
            )

        # =====================================
        # 第二阶段：Prefill microbatch
        # =====================================

        if plan.prefill_seqs:
            start = perf_counter()

            prefill_token_ids = (
                self.model_runner.call(
                    "run",
                    plan.prefill_seqs,
                    True,
                )
            )

            self.scheduler.record_computed_block_metadata(
                plan.prefill_seqs
            )

            self._commit_opportunistic_prefixes(
                plan.prefill_seqs
            )

            self.scheduler.postprocess(
                plan.prefill_seqs,
                prefill_token_ids,
                True,
            )
            
            self._record_request_progress(
                plan.prefill_seqs
            )

            finished_prefill_seq_ids = [
                seq.seq_id
                for seq in plan.prefill_seqs
                if seq.is_finished
            ]

            if finished_prefill_seq_ids:
                self.model_runner.call(
                    "release_visual_embedding_cache",
                    finished_prefill_seq_ids,
                )

            stats.prefill_elapsed = (
                perf_counter() - start
            )

            outputs.extend(
                (
                    seq.seq_id,
                    seq.completion_token_ids,
                )
                for seq in plan.prefill_seqs
                if seq.is_finished
            )

        return outputs, stats

    def get_completed_request_metrics(
        self,
    ) -> list[RequestMetrics]:
        """
        按 seq_id 返回所有已完成请求的指标。
        """

        completed = [
            metrics
            for metrics
            in self.request_metrics.values()
            if metrics.finish_time is not None
        ]

        return sorted(
            completed,
            key=lambda metrics: (
                metrics.seq_id
            ),
        )

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[PromptInput],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            output, stats = self.step()

            if stats.num_prefill_tokens > 0:
                prefill_throughput = (
                    stats.prefill_throughput
                )

            if stats.num_decode_tokens > 0:
                decode_throughput = (
                    stats.decode_throughput
                )

            pbar.set_postfix({
                "Prefill": (
                    f"{int(prefill_throughput)}tok/s"
                ),
                "Decode": (
                    f"{int(decode_throughput)}tok/s"
                ),
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
