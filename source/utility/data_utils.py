"""Dataset and passage collection loading helpers."""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def format_drop_answer(answer: Dict) -> list[str]:
    if answer.get("number"):
        return [str(answer["number"])]
    if answer.get("spans"):
        return [str(span) for span in answer["spans"]]
    date = answer.get("date", {})
    return ["-".join(str(date.get(part, "")) for part in ("day", "month", "year"))]


def load_data_from_jsonl(file_path: str, return_contexts: bool = False):
    path = Path(file_path)
    with path.open(encoding="utf-8") as handle:
        first_character = handle.read(1)
        handle.seek(0)
        records = json.load(handle) if first_character == "[" else [
            json.loads(line) for line in handle if line.strip()
        ]

    questions = {}
    ground_truths = {}
    contexts = {}
    for record in records:
        question_id = record.get("id") or record.get("_id") or record.get("question_id")
        if question_id is None:
            raise ValueError(f"Record in {path} has no question identifier")
        questions[question_id] = record.get("question") or record.get("question_text", "")
        answer = record.get("answer", "")
        if not answer and record.get("answers_objects"):
            formatted = [tuple(format_drop_answer(item)) for item in record["answers_objects"]]
            answer = list(Counter(formatted).most_common(1)[0][0])
        ground_truths[question_id] = answer
        if return_contexts:
            contexts[question_id] = (
                record.get("context") or record.get("contexts") or record.get("paragraphs") or []
            )
    if return_contexts:
        return questions, ground_truths, contexts
    return questions, ground_truths


def load_passages(path: str, shard_id: int = 0, num_shards: int = 1) -> List[Dict]:
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("shard_id must be between 0 and num_shards - 1")
    passage_path = Path(path)
    if not passage_path.is_file():
        raise FileNotFoundError(passage_path)

    passages = []
    with passage_path.open(encoding="utf-8") as handle:
        if passage_path.suffix == ".jsonl":
            passages = [json.loads(line) for line in handle if line.strip()]
        else:
            for row in csv.DictReader(handle, delimiter="\t"):
                passages.append(
                    {"id": row["id"], "title": row.get("title", ""), "text": row["text"]}
                )

    start = len(passages) * shard_id // num_shards
    end = len(passages) * (shard_id + 1) // num_shards
    return passages[start:end]
