"""Contriever-style dense text encoder."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import AutoModel, AutoTokenizer

from .base import BaseRetriever, BaseRetrieverConfig


@dataclass
class DenseRetrieverConfig(BaseRetrieverConfig):
    query_model_name_or_path: str = "facebook/contriever-msmarco"
    pooling: Literal["average", "cls"] = "average"
    max_length: int = 512
    normalize: bool = False
    device_id: int | None = None


class DenseRetriever(BaseRetriever):
    def __init__(self, cfg: DenseRetrieverConfig | None = None) -> None:
        super().__init__(cfg or DenseRetrieverConfig())
        self.device = torch.device(
            f"cuda:{self.cfg.device_id}"
            if torch.cuda.is_available() and self.cfg.device_id is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.query_model_name_or_path
        )
        self.query_model = AutoModel.from_pretrained(
            self.cfg.query_model_name_or_path
        ).to(self.device)

        if self.cfg.training_strategy == "query_only":
            self.query_model.train()
            self.passage_model = copy.deepcopy(self.query_model).eval()
            self.passage_model.requires_grad_(False)
            self.passage_tokenizer = self.query_tokenizer
            if self.cfg.use_fp16 and self.device.type == "cuda":
                self.passage_model.half()
        else:
            self.query_model.eval()
            self.passage_model = self.query_model
            self.passage_tokenizer = self.query_tokenizer
            if self.cfg.use_fp16 and self.device.type == "cuda":
                self.query_model.half()

    def _encode(self, input_texts: list[str], model, tokenizer) -> torch.Tensor:
        model_inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=self.cfg.max_length,
            padding=True,
            truncation=True,
        )
        model_inputs = {name: tensor.to(self.device) for name, tensor in model_inputs.items()}
        token_embeddings = model(**model_inputs).last_hidden_state
        return pool_embeddings(
            token_embeddings,
            model_inputs["attention_mask"],
            strategy=self.cfg.pooling,
            normalize=self.cfg.normalize,
        )

    def _embed_queries(self, input_texts: list[str]) -> torch.Tensor:
        return self._encode(input_texts, self.query_model, self.query_tokenizer)

    def _embed_passages(self, input_texts: list[str]) -> torch.Tensor:
        return self._encode(input_texts, self.passage_model, self.passage_tokenizer)


def pool_embeddings(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: Literal["average", "cls"] = "average",
    normalize: bool = False,
) -> torch.Tensor:
    if strategy == "average":
        masked = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    elif strategy == "cls":
        embeddings = token_embeddings[:, 0]
    else:
        raise ValueError(f"Unsupported pooling strategy: {strategy}")
    if normalize:
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings
