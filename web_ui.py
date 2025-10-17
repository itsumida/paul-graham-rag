#!/usr/bin/env python3
import json
import os
from typing import List

import faiss
import numpy as np
from flask import Flask, render_template, request, jsonify

from embedding_utils import BGEEmbedder
from query_compare import cohere_rerank, simple_local_rerank, load_index, read_jsonl

app = Flask(__name__)

# Global variables for loaded data
index = None
meta = None
texts = None
bge_embedder = None


def init_app(index_dir: str, chunks_path: str, bge_model: str = "BAAI/bge-small-en-v1.5"):
    """Initialize the app with FAISS index and text data."""
    global index, meta, texts, bge_embedder
    
    index, meta = load_index(index_dir)
    chunks = read_jsonl(chunks_path)
    texts = [c.get("text", "") for c in chunks]
    bge_embedder = BGEEmbedder(model_name=bge_model)


def search_and_rerank(query: str, faiss_k: int = 30, top_k: int = 5, cohere_api_key: str = None, use_local_rerank: bool = True) -> List[dict]:
    """Search FAISS and optionally rerank with Cohere."""
    global index, meta, texts, bge_embedder
    
    # Embed query
    q_emb = bge_embedder.embed_texts([query], is_query=True)
    
    # Search FAISS
    scores, indices = index.search(q_emb.astype("float32"), faiss_k)
    
    # Build candidates
    candidates = []
    for i in range(faiss_k):
        idx = int(indices[0, i])
        score = float(scores[0, i])
        m = meta[idx] if idx < len(meta) else {}
        text = texts[idx] if idx < len(texts) else ""
        
        candidates.append({
            "index": idx,
            "title": m.get("title", ""),
            "url": m.get("url", ""),
            "text": text,
            "faiss_score": score
        })
    
    # Apply local reranker first (if enabled) - score ALL candidates
    local_reranked = None
    if use_local_rerank:
        try:
            # Score all candidates, not just top-k
            local_reranked = simple_local_rerank(query, candidates, top_n=None)
            # Update ALL candidates with local reranker scores
            for candidate_idx, local_score in local_reranked:
                if candidate_idx < len(candidates):
                    candidates[candidate_idx]["local_score"] = local_score
        except Exception as e:
            pass  # Silently fall back to FAISS results
    
    # If no Cohere API key, return results with local reranker (if used)
    if not cohere_api_key:
        if use_local_rerank and local_reranked:
            # Return local reranked results
            results = []
            for candidate_idx, local_score in local_reranked:
                if candidate_idx < len(candidates):
                    results.append(candidates[candidate_idx])
            return results
        else:
            # Return FAISS results
            return candidates[:top_k]
    
    # Apply Cohere reranker on top of local reranker
    try:
        cohere_reranked = cohere_rerank(query, candidates, api_key=cohere_api_key, top_n=top_k)
        
        # Map back to candidates with both scores - preserve local scores
        results = []
        for candidate_idx, cohere_score in cohere_reranked:
            if candidate_idx < len(candidates):
                candidate = candidates[candidate_idx].copy()
                candidate["cohere_score"] = cohere_score
                # Keep the local_score if it exists
                if "local_score" in candidates[candidate_idx]:
                    candidate["local_score"] = candidates[candidate_idx]["local_score"]
                results.append(candidate)
        
        return results
    except Exception as e:
        # Fall back to local reranked results or FAISS
        if use_local_rerank and local_reranked:
            results = []
            for candidate_idx, local_score in local_reranked:
                if candidate_idx < len(candidates):
                    results.append(candidates[candidate_idx])
            return results
        else:
            return candidates[:top_k]


@app.route('/')
def index_page():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Get parameters
    faiss_k = int(data.get('faiss_k', 30))
    top_k = int(data.get('top_k', 5))
    cohere_api_key = data.get('cohere_api_key', '').strip()
    
    try:
        results = search_and_rerank(query, faiss_k=faiss_k, top_k=top_k, cohere_api_key=cohere_api_key, use_local_rerank=True)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Web UI for PG essays RAG with Cohere reranking")
    parser.add_argument("--index_dir", default="index_bge", help="Directory with FAISS index")
    parser.add_argument("--chunks", default="chunks.jsonl", help="Path to chunks.jsonl")
    parser.add_argument("--bge_model", default="BAAI/bge-small-en-v1.5", help="BGE model name")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Initialize app
    init_app(args.index_dir, args.chunks, args.bge_model)
    
    # Run app
    app.run(host=args.host, port=args.port, debug=args.debug)
