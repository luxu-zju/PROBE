"""Convert raw dataset corpora into passage TSV files for dense indexing."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
from pathlib import Path
from typing import Iterable, Tuple

from tqdm import tqdm


def document_id(title: str, text: str) -> str:
    return hashlib.blake2b(f"{title}\n{text}".encode("utf-8"), digest_size=16).hexdigest()


def clean(text: str) -> str:
    return " ".join(text.replace("\t", " ").replace("\n", " ").split())


def hotpot_documents(raw_root: Path) -> Iterable[Tuple[str, str]]:
    corpus_root = raw_root / "hotpotqa" / "wikipedia-paragraphs"
    files = sorted(corpus_root.glob("*/wiki_*.bz2"))
    if not files:
        raise FileNotFoundError(f"HotpotQA Wikipedia corpus not found under {corpus_root}")
    for path in tqdm(files, desc="HotpotQA corpus files"):
        with bz2.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                yield record["title"], " ".join(record["text"])


def wiki2_documents(raw_root: Path) -> Iterable[Tuple[str, str]]:
    dataset_root = raw_root / "2wikimultihopqa"
    for split in ("train", "dev", "test"):
        path = dataset_root / f"{split}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        for record in json.loads(path.read_text(encoding="utf-8")):
            for title, sentences in record["context"]:
                yield title, " ".join(sentences)


def musique_documents(raw_root: Path) -> Iterable[Tuple[str, str]]:
    dataset_root = raw_root / "musique"
    for variant in ("ans", "full"):
        for split in ("train", "dev", "test"):
            path = dataset_root / f"musique_{variant}_v1.0_{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    for paragraph in record["paragraphs"]:
                        yield paragraph["title"], paragraph["paragraph_text"]


def write_collection(dataset: str, raw_root: Path, output_path: Path) -> int:
    loaders = {
        "hotpotqa": hotpot_documents,
        "2wikimultihopqa": wiki2_documents,
        "musique": musique_documents,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        output.write("id\ttext\ttitle\n")
        for title, text in tqdm(loaders[dataset](raw_root), desc=f"Writing {dataset}"):
            title, text = clean(title), clean(text)
            identifier = document_id(title, text)
            if identifier in seen:
                continue
            seen.add(identifier)
            output.write(f"{identifier}\t{text}\t{title}\n")
            count += 1
    print(f"Wrote {count} passages to {output_path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"])
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    write_collection(args.dataset, args.raw_root, args.output_path)


if __name__ == "__main__":
    main()
