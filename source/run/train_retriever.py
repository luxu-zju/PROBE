"""Train a dense query encoder from relevance-scored retrieval candidates.

Each JSONL record must contain a ``query`` string and a ``details`` list.  Each
detail needs a ``doc_text`` and a ``bge_score`` in the range [0, 1].  The script
keeps the passage encoder frozen and trains only the query encoder with a
pairwise Margin-MSE objective.

Example:
    python -m source.run.train_retriever \
        --pseudo-label-path data/pseudo_labels.jsonl \
        --model-path facebook/contriever-msmarco \
        --output-dir checkpoints/probe
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from source.module.retrieve.dense import DenseRetriever, DenseRetrieverConfig


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass(frozen=True)
class TrainingConfig:
    pseudo_label_path: Path
    model_path: str
    output_dir: Path
    batch_size: int
    learning_rate: float
    epochs: int
    gradient_accumulation_steps: int
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    min_confidence_gap: float
    num_workers: int
    log_interval: int
    save_interval: int
    seed: int


@dataclass(frozen=True)
class TrainingExample:
    query: str
    documents: List[str]
    teacher_scores: List[float]


class PseudoLabelDataset(Dataset[TrainingExample]):
    """Validated training examples loaded from offline supervision JSONL."""

    def __init__(
        self,
        path: Path | str,
        minimum_gap: float,
    ) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Pseudo-label file does not exist: {path}")

        self.examples: List[TrainingExample] = []
        skipped_records = 0
        skipped_by_gap = 0
        skipped_unknown = 0

        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if str(record.get("model_answer", "")).strip().lower() == "unknown":
                        skipped_unknown += 1
                        continue
                    example = self._parse_record(record)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    skipped_records += 1
                    print(f"Skipping malformed record {line_number}: {error}")
                    continue

                if self._confidence_gap(example.teacher_scores) < minimum_gap:
                    skipped_by_gap += 1
                    continue
                self.examples.append(example)

        if not self.examples:
            raise ValueError(
                "No usable training examples remain. Check the JSONL schema and "
                "--min-confidence-gap."
            )

        print(
            f"Loaded {len(self.examples)} examples "
            f"({skipped_by_gap} filtered by confidence gap, "
            f"{skipped_unknown} unknown answers, "
            f"{skipped_records} malformed)."
        )

    @staticmethod
    def _parse_record(record: Dict[str, Any]) -> TrainingExample:
        query = record["query"].strip()
        details = record["details"]
        if not query or not isinstance(details, list):
            raise ValueError("record requires a non-empty query and a details list")

        documents: List[str] = []
        scores: List[float] = []
        for detail in details:
            document = detail["doc_text"]
            score = float(detail["bge_score"])
            if (
                not isinstance(document, str)
                or not document.strip()
                or not math.isfinite(score)
                or not 0.0 <= score <= 1.0
            ):
                continue
            documents.append(document.strip())
            scores.append(score)

        if len(documents) < 2:
            raise ValueError("record needs at least two valid candidate documents")
        return TrainingExample(query=query, documents=documents, teacher_scores=scores)

    @staticmethod
    def _confidence_gap(scores: Sequence[float]) -> float:
        best, second_best = sorted(scores, reverse=True)[:2]
        return best - second_best

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]


def collate_examples(batch: Sequence[TrainingExample]) -> List[TrainingExample]:
    """Keep variable-sized document candidate lists intact."""

    return list(batch)


def margin_mse_loss(
    query_embedding: torch.Tensor,
    document_embeddings: torch.Tensor,
    teacher_scores: torch.Tensor,
) -> torch.Tensor:
    """Match pairwise student score margins to pairwise teacher score margins."""

    student_scores = document_embeddings.float() @ query_embedding.float()
    student_margins = student_scores[:, None] - student_scores[None, :]
    teacher_margins = teacher_scores[:, None] - teacher_scores[None, :]

    # Per-example normalization makes supervision independent of encoder scale.
    def normalize(margins: torch.Tensor) -> torch.Tensor:
        minimum, maximum = margins.min(), margins.max()
        if torch.isclose(minimum, maximum):
            return margins
        return 2 * (margins - minimum) / (maximum - minimum) - 1

    return functional.mse_loss(normalize(student_margins), normalize(teacher_margins))


class RetrieverTrainer:
    def __init__(self, config: TrainingConfig, retriever: DenseRetriever) -> None:
        self.config = config
        self.retriever = retriever
        self.device = next(retriever.query_model.parameters()).device
        self.optimizer = AdamW(
            retriever.query_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.global_step = 0
        self.best_loss = float("inf")
        self.epoch_losses: List[float] = []
        self.scheduler: SequentialLR | None = None
        self.accumulated_batches = 0

    def _build_scheduler(self, batches_per_epoch: int) -> None:
        """Reproduce the per-epoch warmup/cosine schedule of the reference run."""

        warmup_steps = max(1, int(batches_per_epoch * self.config.warmup_ratio))
        cosine_steps = max(1, batches_per_epoch - warmup_steps)
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    def _train_batch(self, batch: Sequence[TrainingExample]) -> float:
        queries = [example.query for example in batch]
        flat_documents = [document for example in batch for document in example.documents]
        document_counts = [len(example.documents) for example in batch]

        with torch.no_grad():
            document_embeddings = self.retriever.embed(flat_documents, input_type="passage")
        query_embeddings = self.retriever.embed(queries, input_type="query")

        loss = torch.zeros((), device=self.device)
        offset = 0
        for example, query_embedding, count in zip(batch, query_embeddings, document_counts):
            scores = torch.tensor(example.teacher_scores, device=self.device, dtype=torch.float32)
            loss = loss + margin_mse_loss(
                query_embedding,
                document_embeddings[offset : offset + count],
                scores,
            )
            offset += count
        loss = loss / len(batch)
        # The reference experiment accumulates the unscaled batch losses.
        loss.backward()
        self.accumulated_batches += 1
        if self.accumulated_batches == self.config.gradient_accumulation_steps:
            self._optimizer_step()

        self.global_step += 1
        return loss.detach().item()

    def _optimizer_step(self) -> None:
        torch.nn.utils.clip_grad_norm_(
            self.retriever.query_model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulated_batches = 0

    def train(self, dataset: PseudoLabelDataset) -> None:
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_examples,
        )
        if not len(loader):
            raise ValueError("The filtered dataset is too small for the selected batch size.")

        self.optimizer.zero_grad(set_to_none=True)
        all_batch_losses: List[float] = []

        for epoch in range(1, self.config.epochs + 1):
            self._build_scheduler(len(loader))
            batch_losses: List[float] = []
            progress = tqdm(loader, desc=f"Epoch {epoch}/{self.config.epochs}")
            for batch_index, batch in enumerate(progress, start=1):
                loss = self._train_batch(batch)
                batch_losses.append(loss)
                all_batch_losses.append(loss)

                progress.set_postfix(loss=f"{loss:.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
                if self.global_step % self.config.log_interval == 0:
                    recent = np.mean(batch_losses[-self.config.log_interval :])
                    print(f"Step {self.global_step}: mean loss {recent:.4f}")
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}")

            # Flush the final partial accumulation before saving checkpoints.
            if epoch == self.config.epochs and self.accumulated_batches > 0:
                self._optimizer_step()

            # The reference checkpoint criterion uses the cumulative mean.
            epoch_loss = float(np.mean(all_batch_losses))
            self.epoch_losses.append(epoch_loss)
            print(f"Epoch {epoch} complete: mean loss {epoch_loss:.4f}")
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint("best_model")

        self.save_checkpoint("final_model")
        print(f"Training complete. Best loss: {self.best_loss:.4f}")

    def save_checkpoint(self, name: str) -> None:
        checkpoint_dir = self.config.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.query_model.save_pretrained(checkpoint_dir)
        self.retriever.query_tokenizer.save_pretrained(checkpoint_dir)
        (checkpoint_dir / "training_config.json").write_text(
            json.dumps(asdict(self.config), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        torch.save(
            {
                "global_step": self.global_step,
                "best_loss": self.best_loss,
                "epoch_losses": self.epoch_losses,
            },
            checkpoint_dir / "training_state.pt",
        )
        print(f"Saved checkpoint: {checkpoint_dir}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-label-path", required=True, type=Path)
    parser.add_argument("--model-path", default="facebook/contriever-msmarco")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-confidence-gap", "--min-gap", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    if args.batch_size < 1 or args.epochs < 1 or args.gradient_accumulation_steps < 1:
        parser.error("batch size, epochs, and gradient accumulation steps must be positive")
    if args.min_confidence_gap < 0:
        parser.error("minimum confidence gap must be non-negative")
    return TrainingConfig(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    print(json.dumps(asdict(config), indent=2, default=str))

    retriever = DenseRetriever(
        DenseRetrieverConfig(
            query_model_name_or_path=config.model_path,
            batch_size=config.batch_size,
            training_strategy="query_only",
            use_fp16=torch.cuda.is_available(),
        )
    )
    dataset = PseudoLabelDataset(
        config.pseudo_label_path,
        config.min_confidence_gap,
    )
    RetrieverTrainer(config, retriever).train(dataset)


if __name__ == "__main__":
    main()
