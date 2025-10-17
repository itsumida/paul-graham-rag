#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Tuple

import faiss
import numpy as np
import requests

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


def load_index(index_dir: str) -> tuple[faiss.Index, List[dict]]:
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    meta: List[dict] = []
    with open(os.path.join(index_dir, "meta.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    return index, meta


def search(index: faiss.Index, qvecs: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    return index.search(qvecs.astype("float32"), top_k)


def simple_local_rerank(query: str, docs: List[dict], top_n: int | None = None) -> List[tuple[int, float]]:
    """Simple local reranker based on keyword overlap and text length."""
    query_words = set(query.lower().split())
    
    scores = []
    for i, doc in enumerate(docs):
        text = doc.get("text", "").lower()
        text_words = set(text.split())
        
        # Simple scoring: keyword overlap + length penalty
        overlap = len(query_words.intersection(text_words))
        total_query_words = len(query_words)
        keyword_score = overlap / max(total_query_words, 1)
        
        # Prefer shorter, more focused texts
        length_penalty = min(1.0, 1000 / max(len(text), 100))
        
        # Combined score
        score = keyword_score * 0.7 + length_penalty * 0.3
        scores.append((i, score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    if top_n is not None:
        scores = scores[:top_n]
    
    return scores


def cohere_rerank(query: str, docs: List[dict], api_key: str, model: str = "rerank-english-v3.0", top_n: int | None = None) -> List[tuple[int, float]]:
    url = "https://api.cohere.com/v1/rerank"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Cohere expects documents as list of {text} or strings. We pass texts.
    payload = {
        "query": query,
        "documents": [d["text"] for d in docs],
        "model": model,
    }
    if top_n is not None:
        payload["top_n"] = int(top_n)
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    # results: list with {index, relevance_score}
    ranked = [(r["index"], float(r.get("relevance_score", 0.0))) for r in results]
    # If top_n is set, results already truncated. Otherwise, sort all.
    if not ranked:
        return []
    if top_n is None:
        ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def zerank_rerank(query: str, docs: List[dict], url: str, api_key: str | None = None, top_n: int | None = None, model: str | None = None, mode: str = "objects") -> List[tuple[int, float]]:
    headers = {"Content-Type": "application/json"}
    # Send both header styles for compatibility
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
    # Truncate texts to keep payload size reasonable
    def _truncate(t: str, max_len: int = 2000) -> str:
        return t[:max_len]

    if mode == "strings":
        documents = [_truncate(d["text"]) for d in docs]
    else:
        documents = [{"id": i, "text": _truncate(d["text"]) } for i, d in enumerate(docs)]

    payload = {"query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = int(top_n)
        payload["top_k"] = int(top_n)
    if model:
        payload["model"] = model
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        try:
            print(f"[ZeRank] HTTP {resp.status_code}: {resp.text}")
        finally:
            resp.raise_for_status()
    data = resp.json()
    # Support different response shapes: {results:[{id,score}]} or {data:[{index/idx,score}]}
    results = data.get("results")
    if results is None:
        results = data.get("data", [])
    ranked: List[tuple[int, float]] = []
    for r in results:
        rid = r.get("id")
        if rid is None:
            rid = r.get("index", r.get("idx", 0))
        score = r.get("score")
        if score is None:
            score = r.get("relevance", r.get("relevance_score", 0.0))
        ranked.append((int(rid), float(score)))
    if not ranked:
        return []
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def print_results(query: str, meta: List[dict], ids: np.ndarray, scores: np.ndarray, k: int) -> None:
    print("\n=== Query ===")
    print(query)
    print(f"\nTop-{k} results (BGE)")
    for i in range(k):
        idx = ids[0, i]
        score = scores[0, i]
        m = meta[int(idx)] if idx >= 0 else {}
        title = (m.get("title", "") or "").strip()[:80]
        url = m.get("url", "")
        print(f"{i+1:>2}. {title} ({score:>6.3f})\n    {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a BGE-built FAISS index with optional reranking (Cohere/ZeRank)")
    parser.add_argument("--index_dir", default="index_bge", help="Directory with index.faiss and meta.jsonl built with BGE")
    parser.add_argument("--chunks", default="chunks.jsonl", help="Path to chunks.jsonl (needed for reranking text)")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k results to display")
    parser.add_argument("--faiss_k", type=int, default=30, help="Candidate size from FAISS before rerank")
    parser.add_argument("--bge_model", default="BAAI/bge-small-en-v1.5", help="BGE model for queries")
    parser.add_argument("--rerank", choices=["cohere", "zerank", "none"], default="none", help="Reranker to use")
    parser.add_argument("--cohere_model", default="rerank-english-v3.0", help="Cohere rerank model")
    parser.add_argument("--cohere_api_key", default=os.environ.get("COHERE_API_KEY", ""), help="Cohere API key")
    parser.add_argument("--zerank_url", default=os.environ.get("ZERANK_URL", ""), help="ZeRank HTTP endpoint")
    parser.add_argument("--zerank_api_key", default=os.environ.get("ZERANK_API_KEY", ""), help="ZeRank API key (optional)")
    parser.add_argument("--zerank_model", default=os.environ.get("ZERANK_MODEL", ""), help="ZeRank model name (optional)")
    parser.add_argument("--zerank_mode", choices=["objects", "strings"], default=os.environ.get("ZERANK_MODE", "objects"), help="Payload shape for ZeRank")
    parser.add_argument("queries", nargs="*", help="Queries to run; if empty, prompts interactively")
    args = parser.parse_args()

    index, meta = load_index(args.index_dir)
    # Load texts for reranking, aligned to index rows
    texts: List[str] = []
    if args.rerank != "none":
        if not os.path.exists(args.chunks):
            raise FileNotFoundError("Reranking requires --chunks pointing to chunks.jsonl with text")
        chunks = read_jsonl(args.chunks)
        texts = [c.get("text", "") for c in chunks]

    bge = BGEEmbedder(model_name=args.bge_model)

    def run_query(q: str):
        q_bge = bge.embed_texts([q], is_query=True)
        # Retrieve FAISS candidates (top faiss_k)
        D_cand, I_cand = search(index, q_bge, args.faiss_k)

        # If no rerank, just print top_k from FAISS
        if args.rerank == "none":
            print_results(q, meta, I_cand, D_cand, args.top_k)
            return

        # Build candidate docs for reranker
        candidates = []
        for rank in range(I_cand.shape[1]):
            idx = int(I_cand[0, rank])
            m = meta[idx]
            txt = texts[idx] if idx < len(texts) else ""
            candidates.append({"index": idx, "title": m.get("title", ""), "url": m.get("url", ""), "text": txt})

        # Call reranker
        reranked: List[tuple[int, float]] = []
        if args.rerank == "cohere":
            if not args.cohere_api_key:
                print("[Warn] --rerank cohere requested but COHERE_API_KEY missing; skipping rerank.")
            else:
                # Cohere returns indices relative to provided docs list
                pairs = cohere_rerank(q, candidates, api_key=args.cohere_api_key, model=args.cohere_model, top_n=args.top_k)
                # Map back to global indices
                reranked = [(candidates[i]["index"], score) for i, score in pairs]
        elif args.rerank == "zerank":
            if not args.zerank_url:
                print("[Warn] --rerank zerank requested but --zerank_url missing; skipping rerank.")
            else:
                try:
                    pairs = zerank_rerank(q, candidates, url=args.zerank_url, api_key=args.zerank_api_key, top_n=args.top_k, model=(args.zerank_model or None), mode=args.zerank_mode)
                except requests.HTTPError:
                    # Retry with alternate payload mode if unprocessable
                    alt_mode = "strings" if args.zerank_mode == "objects" else "objects"
                    print(f"[ZeRank] Retry with mode={alt_mode}")
                    pairs = zerank_rerank(q, candidates, url=args.zerank_url, api_key=args.zerank_api_key, top_n=args.top_k, model=(args.zerank_model or None), mode=alt_mode)
                reranked = [(candidates[i]["index"], score) for i, score in pairs]

        # If reranker failed or not configured, fall back to FAISS order
        if not reranked:
            print("[Info] Reranker returned no results; falling back to FAISS order.")
            I_show, D_show = I_cand[:, :args.top_k], D_cand[:, :args.top_k]
            print_results(q, meta, I_show, D_show, args.top_k)
            return

        # Build arrays for printing in the same format
        top_indices = [idx for idx, _ in reranked][:args.top_k]
        top_scores = [score for _, score in reranked][:args.top_k]
        I_show = np.array([top_indices], dtype=np.int64)
        D_show = np.array([top_scores], dtype=np.float32)
        print_results(q, meta, I_show, D_show, args.top_k)

    if args.queries:
        for q in args.queries:
            run_query(q)
    else:
        print("Enter queries (empty line to exit):")
        while True:
            try:
                q = input("> ").strip()
            except EOFError:
                break
            if not q:
                break
            run_query(q)


if __name__ == "__main__":
    main()


