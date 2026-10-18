"""Base interface for batched text generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class BaseGeneratorConfig:
    batch_size: int = 4


class BaseGenerator(ABC):
    def __init__(self, cfg: BaseGeneratorConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def _generate(self, input_texts: List[str]) -> List[str]:
        raise NotImplementedError

    def generate(self, input_texts: List[str]) -> List[str]:
        if not isinstance(input_texts, list):
            raise TypeError(f"input_texts must be a list, got {type(input_texts).__name__}")

        outputs: List[str] = []
        for start in range(0, len(input_texts), self.cfg.batch_size):
            outputs.extend(self._generate(input_texts[start : start + self.cfg.batch_size]))
        return outputs

    def __call__(self, input_texts: List[str]) -> List[str]:
        return self.generate(input_texts)
