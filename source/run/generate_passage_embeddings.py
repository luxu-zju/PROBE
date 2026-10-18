"""Encode a shard of a TSV passage collection."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from source.module.index.docstore import Document
from source.module.retrieve.dense import DenseRetriever, DenseRetrieverConfig
from source.utility.data_utils import load_passages


def encode_passages(args) -> Path:
    retriever = DenseRetriever(
        DenseRetrieverConfig(
            query_model_name_or_path=args.model_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            training_strategy=None,
            use_fp16=not args.no_fp16 and torch.cuda.is_available(),
        )
    )
    passages = load_passages(
        str(args.passages), shard_id=args.shard_id, num_shards=args.num_shards
    )
    documents = []
    for passage in passages:
        text = f"{passage.get('title', '')} {passage['text']}".strip()
        documents.append(
            Document(id=str(passage["id"]), content=text, metadata=passage)
        )

    embeddings = retriever.embed(
        [document.content for document in documents], input_type="passage", verbose=True
    ).detach().float().cpu().numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"embeddings_{args.shard_id:02d}"
    with output_path.open("wb") as handle:
        pickle.dump((embeddings, documents), handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(documents)} passage embeddings to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--passages", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-fp16", action="store_true")
    encode_passages(parser.parse_args())


if __name__ == "__main__":
    main()
