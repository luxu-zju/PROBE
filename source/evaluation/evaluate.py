"""Adapters for the official HotpotQA, 2WikiMultiHopQA, and MuSiQue evaluators."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


OFFICIAL_DIR = Path(__file__).resolve().parent / "official_evaluation"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str]) -> str:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    result = subprocess.run(
        command,
        cwd=OFFICIAL_DIR,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Official evaluator failed with exit code {result.returncode}:\n{result.stderr}"
        )
    return result.stdout.strip()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _select(records: list[dict], id_key: str, predictions: dict[str, str]) -> list[dict]:
    keyed = {str(record[id_key]): record for record in records}
    missing = set(predictions) - set(keyed)
    if missing:
        preview = ", ".join(sorted(missing)[:3])
        raise ValueError(f"{len(missing)} prediction IDs are missing from official gold data: {preview}")
    return [keyed[question_id] for question_id in predictions]


def official_evaluate(
    dataset: str,
    predictions: dict[str, str],
    raw_root: str | Path,
    split: str = "dev",
) -> dict[str, float | int]:
    """Evaluate answer predictions with the benchmark's official implementation."""
    if not predictions:
        raise ValueError("Cannot evaluate an empty prediction dictionary")
    raw_root = Path(raw_root).resolve()

    with tempfile.TemporaryDirectory(prefix="probe-official-eval-") as temporary:
        temporary_dir = Path(temporary)
        prediction_path = temporary_dir / "predictions.json"
        gold_path = temporary_dir / "gold.json"

        if dataset == "hotpotqa":
            if split != "dev":
                raise ValueError("HotpotQA official answer evaluation is available for the dev split")
            raw_path = raw_root / "hotpotqa" / "hotpot_dev_distractor_v1.json"
            gold = _select(json.loads(raw_path.read_text(encoding="utf-8")), "_id", predictions)
            gold_path.write_text(json.dumps(gold), encoding="utf-8")
            payload = {
                "answer": {key: str(value) for key, value in predictions.items()},
                "sp": {key: [["", 0]] for key in predictions},
            }
            prediction_path.write_text(json.dumps(payload), encoding="utf-8")
            output = _run(
                [sys.executable, str(OFFICIAL_DIR / "hotpot_evaluate_v1.py"), str(prediction_path), str(gold_path)]
            )
            raw_metrics = ast.literal_eval(output.splitlines()[-1])
            return {
                "em": round(raw_metrics["em"], 3),
                "f1": round(raw_metrics["f1"], 3),
                "precision": round(raw_metrics["prec"], 3),
                "recall": round(raw_metrics["recall"], 3),
                "count": len(predictions),
            }

        if dataset == "2wikimultihopqa":
            raw_path = raw_root / dataset / f"{split}.json"
            gold = _select(json.loads(raw_path.read_text(encoding="utf-8")), "_id", predictions)
            gold_path.write_text(json.dumps(gold), encoding="utf-8")
            payload = {
                "answer": {key: str(value) for key, value in predictions.items()},
                "sp": {key: [["", 0]] for key in predictions},
                "evidence": {key: ["", "", ""] for key in predictions},
            }
            prediction_path.write_text(json.dumps(payload), encoding="utf-8")
            alias_path = raw_root / dataset / "id_aliases.json"
            output = _run(
                [
                    sys.executable,
                    str(OFFICIAL_DIR / "2wikimultihop_evaluate_v1.1.py"),
                    str(prediction_path),
                    str(gold_path),
                    str(alias_path),
                ]
            )
            raw_metrics = json.loads(output)
            return {
                "em": round(raw_metrics["em"] / 100, 3),
                "f1": round(raw_metrics["f1"] / 100, 3),
                "precision": round(raw_metrics["prec"] / 100, 3),
                "recall": round(raw_metrics["recall"] / 100, 3),
                "count": len(predictions),
            }

        if dataset == "musique":
            raw_path = raw_root / "musique" / f"musique_ans_v1.0_{split}.jsonl"
            gold = _select(_read_jsonl(raw_path), "id", predictions)
            gold_path = temporary_dir / "gold.jsonl"
            prediction_path = temporary_dir / "predictions.jsonl"
            metrics_path = temporary_dir / "metrics.json"
            _write_jsonl(gold_path, gold)
            _write_jsonl(
                prediction_path,
                [
                    {
                        "id": question_id,
                        "predicted_answer": str(predictions[question_id]),
                        "predicted_support_idxs": [0, 1],
                        "predicted_answerable": True,
                    }
                    for question_id in predictions
                ],
            )
            _run(
                [
                    sys.executable,
                    str(OFFICIAL_DIR / "evaluate_v1.0.py"),
                    str(prediction_path),
                    str(gold_path),
                    "--output_filepath",
                    str(metrics_path),
                ]
            )
            raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            return {
                "em": round(raw_metrics["answer_em"], 3),
                "f1": round(raw_metrics["answer_f1"], 3),
                "count": len(predictions),
            }

    raise ValueError(f"Unsupported dataset: {dataset}")
