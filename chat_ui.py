#!/usr/bin/env python3
import json
import os
from typing import List, Dict, Any
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from embedding_utils import BGEEmbedder
from query_compare import cohere_rerank, simple_local_rerank, load_index, read_jsonl

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Global variables for loaded data
index = None
meta = None
texts = None
bge_embedder = None

# Default API keys (set these or use environment variables)
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Set your key here or use env var
DEFAULT_COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")  # Set your key here or use env var


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
        print("DEBUG: No Cohere API key, using local reranker only")
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
    print("DEBUG: Using Cohere reranker")
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


def generate_answer(query: str, context_chunks: List[dict], openai_api_key: str = None) -> tuple[str, List[dict]]:
    """Generate an answer using retrieved context chunks."""
    
    # Prepare context text and sources (use only top 3 chunks for consistency)
    context_chunks = context_chunks[:3]  # Limit to top 3 for consistency
    context_text = ""
    sources = []
    
    for i, chunk in enumerate(context_chunks):
        # Truncate long chunks for better context
        text = chunk['text'][:400] + "..." if len(chunk['text']) > 400 else chunk['text']
        context_text += f"[{i+1}] {text}\n\n"
        sources.append({
            "number": i + 1,
            "title": chunk.get("title", "Unknown"),
            "url": chunk.get("url", ""),
            "score": chunk.get("cohere_score", chunk.get("local_score", chunk.get("faiss_score", 0)))
        })
    
    # Check for OpenAI API key in environment or parameter
    if not openai_api_key:
        openai_api_key = DEFAULT_OPENAI_API_KEY
    
    # Debug: Print API key status
    print(f"DEBUG: OpenAI API key available: {bool(openai_api_key)}")
    if openai_api_key:
        print(f"DEBUG: API key starts with: {openai_api_key[:10]}...")
    
    # Use OpenAI API if key is available, otherwise use template
    if openai_api_key:
        # Use OpenAI API for better answers
        try:
            prompt = f"""You are an AI assistant helping users understand Paul Graham's ideas. Based on the following excerpts from his essays, provide a natural, conversational answer to the user's question.

User Question: {query}

Relevant Excerpts:
{context_text}

Instructions:
- Write a natural, conversational answer (2-3 paragraphs)
- Synthesize the information from the excerpts into coherent insights
- Reference sources as [1], [2], [3] when making specific points
- Write in Paul Graham's clear, direct style
- Make it sound like you're explaining Paul Graham's ideas, not just listing excerpts
- If the excerpts don't fully answer the question, acknowledge this

Answer:"""

            headers = {
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
            else:
                # Fallback to template answer
                key_insights = []
                for i, chunk in enumerate(context_chunks):
                    text = chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
                    sentences = text.split('.')
                    if len(sentences) > 1:
                        main_point = sentences[0].strip()
                        if len(main_point) > 20:
                            key_insights.append(f"• {main_point} [{i+1}]")
                
                if key_insights:
                    answer = f"""Based on Paul Graham's essays about "{query}":

{chr(10).join(key_insights)}

These insights come from Paul Graham's writings. Click the numbered references to read the full essays."""
                else:
                    answer = f"""I found some relevant content about "{query}" in Paul Graham's essays, but the excerpts are quite brief. 

**Sources:** {', '.join([f'[{s["number"]}]({s["url"]})' for s in sources[:3]])}"""
                
        except Exception as e:
            # Fallback to template answer
            key_insights = []
            for i, chunk in enumerate(context_chunks):
                text = chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
                sentences = text.split('.')
                if len(sentences) > 1:
                    main_point = sentences[0].strip()
                    if len(main_point) > 20:
                        key_insights.append(f"• {main_point} [{i+1}]")
            
            if key_insights:
                answer = f"""Based on Paul Graham's essays about "{query}":

{chr(10).join(key_insights)}

These insights come from Paul Graham's writings. Click the numbered references to read the full essays."""
            else:
                answer = f"""I found some relevant content about "{query}" in Paul Graham's essays, but the excerpts are quite brief. 

**Sources:** {', '.join([f'[{s["number"]}]({s["url"]})' for s in sources[:3]])}"""
    else:
        # Template-based answer (no API key)
        key_insights = []
        for i, chunk in enumerate(context_chunks):  # Use the already limited chunks
            text = chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
            # Extract the main point from the chunk
            sentences = text.split('.')
            if len(sentences) > 1:
                main_point = sentences[0].strip()
                if len(main_point) > 20:  # Only use substantial sentences
                    key_insights.append(f"• {main_point} [{i+1}]")
        
        if key_insights:
            answer = f"""Based on Paul Graham's essays about "{query}":

{chr(10).join(key_insights)}

These insights come from Paul Graham's writings. Click the numbered references to read the full essays."""
        else:
            answer = f"""I found some relevant content about "{query}" in Paul Graham's essays, but the excerpts are quite brief. 

**Sources:** {', '.join([f'[{s["number"]}]({s["url"]})' for s in sources[:3]])}"""
    
    return answer, sources


@app.route('/')
def index_page():
    return render_template('chat.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    # Get parameters
    faiss_k = int(data.get('faiss_k', 30))
    top_k = int(data.get('top_k', 5))
    cohere_api_key = data.get('cohere_api_key', '').strip() or DEFAULT_COHERE_API_KEY
    openai_api_key = data.get('openai_api_key', '').strip() or DEFAULT_OPENAI_API_KEY
    
    try:
        # Retrieve relevant chunks with reranking
        chunks = search_and_rerank(message, faiss_k=faiss_k, top_k=top_k, cohere_api_key=cohere_api_key, use_local_rerank=True)
        
        # Debug: Print reranking status
        print(f"DEBUG: Cohere API key available: {bool(cohere_api_key)}")
        if cohere_api_key:
            print(f"DEBUG: Cohere key starts with: {cohere_api_key[:10]}...")
        print(f"DEBUG: Retrieved {len(chunks)} chunks for reranking")
        
        # Generate answer
        answer, sources = generate_answer(message, chunks, openai_api_key=openai_api_key)
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'chunks_used': len(chunks)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Chat UI for PG essays RAG with source citations")
    parser.add_argument("--index_dir", default="index_bge", help="Directory with FAISS index")
    parser.add_argument("--chunks", default="chunks.jsonl", help="Path to chunks.jsonl")
    parser.add_argument("--bge_model", default="BAAI/bge-small-en-v1.5", help="BGE model name")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Initialize app
    init_app(args.index_dir, args.chunks, args.bge_model)
    
    # Run app
    app.run(host=args.host, port=args.port, debug=args.debug)
