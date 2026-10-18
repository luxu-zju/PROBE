"""Convert raw MHQA question files into the common JSONL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def answer_object(answer: str):
    return [{"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": [answer]}]


def write_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_hotpotqa(raw_root: Path, output_root: Path) -> None:
    files = {"train": "hotpot_train_v1.1.json", "dev": "hotpot_dev_distractor_v1.json"}
    for split, filename in files.items():
        raw_records = json.loads((raw_root / "hotpotqa" / filename).read_text(encoding="utf-8"))
        processed = []
        for record in raw_records:
            supporting_titles = {title for title, _ in record["supporting_facts"]}
            contexts = [
                {
                    "idx": index,
                    "title": title,
                    "paragraph_text": " ".join(sentences),
                    "is_supporting": title in supporting_titles,
                }
                for index, (title, sentences) in enumerate(record["context"])
            ]
            processed.append(
                {
                    "dataset": "hotpotqa",
                    "question_id": record["_id"],
                    "question_text": record["question"],
                    "level": record["level"],
                    "type": record["type"],
                    "answers_objects": answer_object(record["answer"]),
                    "contexts": contexts,
                }
            )
        write_jsonl(processed, output_root / "hotpotqa" / f"{split}.jsonl")


def process_2wiki(raw_root: Path, output_root: Path) -> None:
    for split in ("train", "dev"):
        raw_records = json.loads(
            (raw_root / "2wikimultihopqa" / f"{split}.json").read_text(encoding="utf-8")
        )
        processed = []
        for record in raw_records:
            supporting_titles = {title for title, _ in record["supporting_facts"]}
            processed.append(
                {
                    "question_id": record["_id"],
                    "question_text": record["question"],
                    "answers_objects": answer_object(record["answer"]),
                    "reasoning_steps": [" ".join(evidence) for evidence in record["evidences"]],
                    "contexts": [
                        {
                            "idx": index,
                            "title": title,
                            "paragraph_text": " ".join(sentences),
                            "is_supporting": title in supporting_titles,
                        }
                        for index, (title, sentences) in enumerate(record["context"])
                    ],
                }
            )
        write_jsonl(processed, output_root / "2wikimultihopqa" / f"{split}.jsonl")


def process_musique(raw_root: Path, output_root: Path) -> None:
    for split in ("train", "dev"):
        path = raw_root / "musique" / f"musique_ans_v1.0_{split}.jsonl"
        processed = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                previous_answers = {}
                reasoning_steps = []
                for step_number, step in enumerate(record["question_decomposition"], start=1):
                    step_question = step["question"]
                    for referenced_number, referenced_answer in previous_answers.items():
                        step_question = step_question.replace(
                            f"#{referenced_number}", referenced_answer
                        )
                    previous_answers[step_number] = step["answer"]
                    reasoning_steps.append(
                        f"{step_question.strip()} >>>> {step['answer'].strip()}"
                    )
                processed.append(
                    {
                        "question_id": record["id"],
                        "question_text": record["question"],
                        "answers_objects": answer_object(record["answer"]),
                        "reasoning_steps": reasoning_steps,
                        "contexts": [
                            {
                                "idx": index,
                                "title": paragraph["title"],
                                "paragraph_text": paragraph["paragraph_text"],
                                "is_supporting": paragraph["is_supporting"],
                            }
                            for index, paragraph in enumerate(record["paragraphs"])
                        ],
                    }
                )
        write_jsonl(processed, output_root / "musique" / f"{split}.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["hotpotqa", "2wikimultihopqa", "musique", "all"], default="all")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    processors = {
        "hotpotqa": process_hotpotqa,
        "2wikimultihopqa": process_2wiki,
        "musique": process_musique,
    }
    selected = processors if args.dataset == "all" else {args.dataset: processors[args.dataset]}
    for name, processor in selected.items():
        processor(args.raw_root, args.output_root)
        print(f"Processed {name} questions")


if __name__ == "__main__":
    main()
