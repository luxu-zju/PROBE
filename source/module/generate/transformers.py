"""Transformers generators used by inference and offline data synthesis."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TransformersGeneratorConfig:
    model_name: str
    device_ids: tuple[int, ...] = (1, 2)
    batch_size_per_device: int = 8
    max_total_tokens: int = 4096
    max_new_tokens: int = 128
    min_new_tokens: int = 1
    temperature: float = 0.0


class _SingleDeviceGenerator:
    def __init__(self, cfg: TransformersGeneratorConfig, device_id: int) -> None:
        self.cfg = cfg
        self.device = torch.device(f"cuda:{device_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        added_pad_token = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            added_pad_token = True
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        if added_pad_token:
            self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def generate(self, prompts: list[str]) -> list[str]:
        outputs = []
        for start in range(0, len(prompts), self.cfg.batch_size_per_device):
            batch = prompts[start : start + self.cfg.batch_size_per_device]
            model_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=self.cfg.max_total_tokens,
                truncation=True,
                padding=True,
            )
            model_inputs = {
                name: tensor.to(self.device) for name, tensor in model_inputs.items()
            }
            generation_args = {
                "max_new_tokens": self.cfg.max_new_tokens,
                # Preserve the reference Transformers invocation. It used
                # min_length (total sequence length), not min_new_tokens.
                "min_length": self.cfg.min_new_tokens,
                "do_sample": self.cfg.temperature > 0,
                "temperature": self.cfg.temperature,
                "num_return_sequences": 1,
                "repetition_penalty": 1.0,
                "length_penalty": 1.0,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "use_cache": True,
            }
            sequences = self.model.generate(**model_inputs, **generation_args)
            generated_tokens = sequences[:, model_inputs["input_ids"].shape[1] :]
            outputs.extend(
                text.strip()
                for text in self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
            )
        return outputs


class TransformersGenerator:
    """Generate on one device with a Transformers causal language model."""

    def __init__(self, cfg: TransformersGeneratorConfig, device_id: int = 0) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Transformers generation requires CUDA")
        if device_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"Generation requested CUDA device {device_id}, but only "
                f"{torch.cuda.device_count()} devices are visible"
            )
        self.generator = _SingleDeviceGenerator(cfg, device_id)

    def __call__(self, prompts: list[str]) -> list[str]:
        return self.generator.generate(prompts)


class ParallelTransformersGenerator:
    """Load one full model replica on each GPU and split every batch in two."""

    def __init__(self, cfg: TransformersGeneratorConfig) -> None:
        if len(cfg.device_ids) != 2:
            raise ValueError("Parallel Transformers generation requires exactly two GPUs")
        if not torch.cuda.is_available():
            raise RuntimeError("Offline synthesis requires CUDA for Transformers generation")
        required_devices = max(cfg.device_ids) + 1
        if torch.cuda.device_count() < required_devices:
            raise RuntimeError(
                f"Offline synthesis needs visible GPUs {cfg.device_ids}, but only "
                f"{torch.cuda.device_count()} CUDA devices are available"
            )
        self.generators = [
            _SingleDeviceGenerator(cfg, device_id) for device_id in cfg.device_ids
        ]

    def __call__(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        midpoint = len(prompts) // 2
        batches = [prompts[:midpoint], prompts[midpoint:]]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(generator.generate, batch)
                for generator, batch in zip(self.generators, batches)
            ]
            return futures[0].result() + futures[1].result()
