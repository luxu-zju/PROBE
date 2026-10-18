"""BGE cross-encoder scoring for offline supervision synthesis."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class CrossEncoderConfig:
    model_name_or_path: str
    batch_size: int = 32
    max_length: int = 512
    use_fp16: bool = True
    device_id: int | None = None


class CrossEncoder:
    def __init__(self, cfg: CrossEncoderConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(
            f"cuda:{cfg.device_id}"
            if torch.cuda.is_available() and cfg.device_id is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name_or_path
        ).to(self.device).eval()
        if cfg.use_fp16 and self.device.type == "cuda":
            self.model.half()

    def score(self, queries: list[str], documents: list[list[str]]) -> list[list[float]]:
        all_scores = []
        for query, query_documents in zip(queries, documents):
            scores = []
            for start in range(0, len(query_documents), self.cfg.batch_size):
                batch = query_documents[start : start + self.cfg.batch_size]
                inputs = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.max_length,
                    return_tensors="pt",
                )
                inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    if logits.shape[-1] == 1:
                        logits = logits.flatten()
                    else:
                        logits = logits[:, -1]
                    scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
            all_scores.append(scores)
        return all_scores

    def rerank(self, queries: list[str], documents: list[list], top_k: int) -> list[list]:
        document_texts = [[str(document) for document in group] for group in documents]
        score_groups = self.score(queries, document_texts)
        outputs = []
        for group, scores in zip(documents, score_groups):
            ranked = sorted(zip(group, scores), key=lambda item: item[1], reverse=True)[:top_k]
            for document, score in ranked:
                document.relevance_score = float(score)
            outputs.append([document for document, _ in ranked])
        return outputs
