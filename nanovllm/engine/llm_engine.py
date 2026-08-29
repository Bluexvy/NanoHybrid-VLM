import atexit
from dataclasses import dataclass, fields
from time import perf_counter
from tqdm.auto import tqdm
import torch.multiprocessing as mp

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

        self.scheduler.add(seq)

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

            self.scheduler.postprocess(
                plan.decode_seqs,
                decode_token_ids,
                False,
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

            self.scheduler.postprocess(
                plan.prefill_seqs,
                prefill_token_ids,
                True,
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
