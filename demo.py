#!/usr/bin/env python3
"""
Demo script for Paul Graham Essays RAG System

This script demonstrates the complete pipeline:
1. Scraping essays from paulgraham.com
2. Chunking into smaller pieces
3. Building FAISS index with BGE embeddings
4. Querying with optional Cohere reranking
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_file_exists(filepath, description):
    """Check if a file exists and return True if it does."""
    if Path(filepath).exists():
        print(f"✅ {description} already exists, skipping...")
        return True
    return False


def main():
    """Run the complete demo pipeline."""
    print("🚀 Paul Graham Essays RAG System Demo")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("scrape_pg_essays.py").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Scrape essays (skip if already done)
    if not check_file_exists("pg_essays_json", "Essay JSON files"):
        if not run_command(
            "python scrape_pg_essays.py --out pg_essays_json --limit 5",
            "Scraping first 5 essays (for demo)"
        ):
            return
    
    # Step 2: Chunk essays (skip if already done)
    if not check_file_exists("chunks.jsonl", "Chunked essays file"):
        if not run_command(
            "python chunk_pg_essays.py --input pg_essays_json --out chunks.jsonl --chunk_size 200 --overlap 50",
            "Chunking essays into smaller pieces"
        ):
            return
    
    # Step 3: Build FAISS index (skip if already done)
    if not check_file_exists("index_bge/index.faiss", "FAISS index"):
        if not run_command(
            "python build_faiss_index.py --chunks chunks.jsonl --out index_bge --batch_size 32",
            "Building FAISS index with BGE embeddings"
        ):
            return
    
    # Step 4: Test queries
    print("\n🔍 Testing queries...")
    test_queries = [
        "startup ideas",
        "good writing",
        "how to work hard"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        run_command(
            f'python query_compare.py --index_dir index_bge --top_k 3 "{query}"',
            f"Searching for: {query}"
        )
    
    # Step 5: Launch web UI
    print("\n🌐 Starting web interface...")
    print("The web UI will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the demo")
    
    try:
        subprocess.run([
            "python", "web_ui.py", 
            "--index_dir", "index_bge",
            "--chunks", "chunks.jsonl",
            "--port", "5000"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Demo completed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Web UI failed to start: {e}")


if __name__ == "__main__":
    main()
