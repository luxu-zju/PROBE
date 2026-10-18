"""Configuration for the public multi-hop inference pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from source.utility.system_utils import seed_everything


@dataclass
class PipelineConfig:
    running_name: str
    dataset: Literal["hotpotqa", "2wikimultihopqa", "musique"]
    data_root: Path = Path("data")
    index_root: Path = Path("data/database/contriever_msmarco")
    output_root: Path = Path("predictions")
    dataset_split: Literal["train", "dev", "test"] = "dev"
    method: Literal["base", "probe"] = "probe"
    batch_size: int = 8
    seed: int = 100

    prompt_set: int = 1
    prompt_document_from: Literal["last_only", "full"] = "last_only"
    prompt_max_para_count: int = 15

    retrieval_count: int = 8
    retrieval_query_type: Literal["last_only", "full"] = "full"
    retrieval_buffer_size: int = 32
    retrieval_no_duplicates: bool = False

    # T_max: additional thought/next-hop-query rounds after initial retrieval.
    # The maximum number of retrieval hops is therefore T_max + 1.
    max_thought_rounds: int = 6

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.index_root = Path(self.index_root)
        self.output_root = Path(self.output_root)
        if self.max_thought_rounds < 0:
            raise ValueError("max_thought_rounds must be non-negative")
        self.min_num_thought = 1 if self.method == "probe" else 0
        seed_everything(self.seed)

    @property
    def database_path(self) -> Path:
        return self.index_root / self.dataset

    @property
    def prediction_file_dir(self) -> Path:
        return self.output_root / self.dataset / self.running_name

    @property
    def configuration_file_path(self) -> Path:
        return self.prediction_file_dir / "configuration.json"

    @property
    def data_file_path(self) -> Path:
        dataset_dir = self.data_root / "processed_data" / self.dataset
        if self.dataset_split == "test":
            # The paper's test set is a fixed subset sampled from development.
            candidates = [dataset_dir / "test_subsampled.jsonl"]
        else:
            candidates = [dataset_dir / f"{self.dataset_split}.jsonl"]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"No {self.dataset_split} data found in {dataset_dir}. "
            "Run script/download/multihop_raw_data.sh and preprocessing first."
        )

    @property
    def logging_file_path(self) -> Path:
        return self.prediction_file_dir / f"{self.dataset_split}_logging.json"

    @property
    def ground_truth_file_path(self) -> Path:
        return self.prediction_file_dir / f"{self.dataset_split}_ground_truth.json"

    @property
    def prediction_file_path(self) -> Path:
        return self.prediction_file_dir / f"{self.dataset_split}_prediction.json"

    @property
    def official_evaluation_file_path(self) -> Path:
        return self.prediction_file_dir / f"{self.dataset_split}_official_evaluation.json"

    @property
    def prompt_file_dir(self) -> Path:
        return Path("prompts") / f"prompt_set__{self.prompt_set}"

    @property
    def answer_gen_prompt_file_path(self) -> Path:
        return self.prompt_file_dir / "answer_gen.txt"

    @property
    def thought_gen_prompt_file_path(self) -> Path:
        return self.prompt_file_dir / "thought_gen_new.txt"

    def save(self) -> None:
        self.prediction_file_dir.mkdir(parents=True, exist_ok=True)
        self.configuration_file_path.write_text(
            json.dumps(asdict(self), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
