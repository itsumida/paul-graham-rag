#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Essay:
    url: str
    title: str
    text: str


def iter_essay_jsons(input_dir: str) -> Iterable[Essay]:
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(input_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            url = data.get("url", "")
            title = data.get("title", "")
            text = data.get("text", "")
            if text:
                yield Essay(url=url, title=title, text=text)
        except Exception as e:
            # Skip bad files but continue
            print(f"Warning: failed to load {path}: {e}")


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step
    return chunks


def write_jsonl(chunks: List[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in chunks:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_chunks(input_dir: str, out_path: str, chunk_size: int, overlap: int) -> int:
    all_entries: List[dict] = []
    for essay in iter_essay_jsons(input_dir):
        parts = split_into_chunks(essay.text, chunk_size=chunk_size, overlap=overlap)
        for i, content in enumerate(parts):
            all_entries.append(
                {
                    "url": essay.url,
                    "title": essay.title,
                    "chunk_id": i,
                    "text": content,
                }
            )
    write_jsonl(all_entries, out_path)
    return len(all_entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk PG essays JSONs into a chunks.jsonl for RAG")
    parser.add_argument("--input", default="pg_essays_json", help="Directory with essay JSON files")
    parser.add_argument("--out", default="chunks.jsonl", help="Output JSONL file path")
    parser.add_argument("--chunk_size", type=int, default=300, help="Chunk size in words")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap in words between consecutive chunks")
    args = parser.parse_args()

    total = build_chunks(args.input, args.out, args.chunk_size, args.overlap)
    print(f"Wrote {total} chunks to {args.out}")


if __name__ == "__main__":
    main()


