"""vLLM-backed generator used by the public inference pipeline."""

from dataclasses import dataclass, field
from typing import List

from vllm import LLM, SamplingParams

from source.module.generate.base import BaseGenerator, BaseGeneratorConfig


@dataclass
class LlamaGeneratorConfig(BaseGeneratorConfig):
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_total_tokens: int = 4096
    max_new_tokens: int = 128
    min_new_tokens: int = 1
    temperature: float = 0.0
    stop: List[str] = field(default_factory=lambda: ["<|eot_id|>"])
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1


class LlamaGenerator(BaseGenerator):
    def __init__(self, cfg: LlamaGeneratorConfig) -> None:
        super().__init__(cfg)
        self.model = LLM(
            model=cfg.model_name,
            max_model_len=cfg.max_total_tokens,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            tensor_parallel_size=cfg.tensor_parallel_size,
        )

    def _generate(self, input_texts: List[str]) -> List[str]:
        sampling = SamplingParams(
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_new_tokens,
            min_tokens=self.cfg.min_new_tokens,
            stop=self.cfg.stop,
        )
        results = self.model.generate(input_texts, sampling, use_tqdm=False)
        return [result.outputs[0].text.strip() for result in results]
