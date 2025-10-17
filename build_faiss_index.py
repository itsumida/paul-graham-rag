#!/usr/bin/env python3
import argparse
import json
import os
from typing import List

import faiss
import numpy as np

from embedding_utils import BGEEmbedder


def read_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_index(chunks_path: str, out_dir: str, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 128, verbose: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    data = read_jsonl(chunks_path)
    if verbose:
        print(f"Loaded {len(data)} chunks from {chunks_path}")
    if not data:
        raise ValueError("No data found in chunks.jsonl. Ensure chunking step produced content.")

    texts = [d.get("text", "") for d in data]
    meta = [{k: v for k, v in d.items() if k != "text"} for d in data]

    embedder = BGEEmbedder(model_name=model_name)

    # Embed in batches
    macro = max(batch_size * 8, 1024)
    all_embs: List[np.ndarray] = []
    for i in range(0, len(texts), macro):
        slice_texts = texts[i:i + macro]
        if verbose:
            print(f"Embedding {i+1}-{min(i+len(slice_texts), len(texts))} / {len(texts)}...")
        embs_part = embedder.embed_texts(slice_texts, batch_size=batch_size, is_query=False).astype("float32")
        all_embs.append(embs_part)
    embs = np.vstack(all_embs)

    if verbose:
        print(f"Embeddings shape: {embs.shape}")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(embs)

    if verbose:
        print("Writing artifacts...")
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    np.save(os.path.join(out_dir, "embeddings.npy"), embs)
    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "count": len(data)}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks with BGE and build FAISS index")
    parser.add_argument("--chunks", default="chunks.jsonl", help="Path to chunks.jsonl")
    parser.add_argument("--out", default="index_bge", help="Output directory for FAISS and artifacts")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="Model name for BGE")
    parser.add_argument("--batch_size", type=int, default=128, help="Embedding batch size")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    try:
        build_index(args.chunks, args.out, args.model, args.batch_size, verbose=not args.quiet)
        print(f"Index built at {args.out}")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()


