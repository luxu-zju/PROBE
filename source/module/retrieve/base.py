"""Shared batched embedding interface for retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from tqdm import tqdm


@dataclass
class BaseRetrieverConfig:
    batch_size: int = 4
    training_strategy: Literal["query_only"] | None = None
    use_fp16: bool = True
    return_tensors: Literal["pt", "np"] = "pt"


class BaseRetriever(ABC):
    def __init__(self, cfg: BaseRetrieverConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def _embed_queries(self, input_texts: list[str]):
        raise NotImplementedError

    @abstractmethod
    def _embed_passages(self, input_texts: list[str]):
        raise NotImplementedError

    def embed(
        self,
        input_texts: list[str],
        input_type: Literal["query", "passage"] = "query",
        verbose: bool = False,
    ):
        if not input_texts:
            raise ValueError("input_texts must not be empty")
        embed_batch = self._embed_queries if input_type == "query" else self._embed_passages
        requires_grad = input_type == "query" and self.cfg.training_strategy == "query_only"
        starts = range(0, len(input_texts), self.cfg.batch_size)
        if verbose:
            starts = tqdm(starts, desc=f"Embedding {input_type}s")
        outputs = []
        for start in starts:
            batch = input_texts[start : start + self.cfg.batch_size]
            if requires_grad:
                outputs.append(embed_batch(batch))
            else:
                with torch.no_grad():
                    outputs.append(embed_batch(batch))
        output = torch.cat(outputs)
        if self.cfg.return_tensors == "np":
            return output.detach().cpu().numpy().astype("float32")
        return output

    def __call__(self, input_texts: list[str], input_type: Literal["query", "passage"] = "query"):
        return self.embed(input_texts, input_type)
