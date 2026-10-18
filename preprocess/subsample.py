"""Create the fixed development and test subsets used by the paper runs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SUBSAMPLE_SEED = 13370 # Don't change this.


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_records(records: list[dict], count: int) -> list[dict]:
    # The reference script runs as a fresh process for every subset, resetting
    # its module-level seed before each random.sample call.
    return random.Random(SUBSAMPLE_SEED).sample(records, min(count, len(records)))


def create_subsets(dataset_dir: Path) -> None:
    records = read_jsonl(dataset_dir / "dev.jsonl")
    dev_records = sample_records(records, 100)
    dev_ids = {record["question_id"] for record in dev_records}
    test_candidates = [
        record for record in records if record["question_id"] not in dev_ids
    ]
    test_records = sample_records(test_candidates, 500)
    write_jsonl(dataset_dir / "dev_subsampled.jsonl", dev_records)
    write_jsonl(dataset_dir / "test_subsampled.jsonl", test_records)
    print(
        f"Wrote {len(dev_records)} development and {len(test_records)} test "
        f"examples under {dataset_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["hotpotqa", "2wikimultihopqa", "musique"],
    )
    parser.add_argument("--processed-root", required=True, type=Path)
    args = parser.parse_args()
    create_subsets(args.processed_root / args.dataset)


if __name__ == "__main__":
    main()
