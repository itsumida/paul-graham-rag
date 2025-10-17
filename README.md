# Paul Graham Essays RAG System

A complete Retrieval-Augmented Generation (RAG) system for Paul Graham's essays with multi-stage reranking capabilities.

## 🚀 Quick Start

Run the complete demo pipeline:

```bash
python demo.py
```

This will:
1. Scrape 5 essays from paulgraham.com
2. Chunk them into smaller pieces
3. Build a FAISS index with BGE embeddings
4. Test some queries
5. Launch the web interface at http://localhost:5000

## 📋 Features

- **Web Scraping**: Automated essay collection from paulgraham.com
- **Smart Chunking**: Overlapping text chunks for better retrieval
- **Vector Search**: FAISS index with BGE embeddings
- **Multi-Stage Reranking**: Local keyword-based + Cohere semantic reranking
- **Web Interface**: Interactive querying with relevancy scores
- **CLI Tools**: Command-line interface for all operations

## 🛠️ Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd paul-graham-rag

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch for better performance
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 📖 Usage

### 1. Scrape Essays

```bash
python scrape_pg_essays.py --out pg_essays_json --delay 0.5
```

### 2. Chunk for RAG

```bash
python chunk_pg_essays.py --input pg_essays_json --out chunks.jsonl --chunk_size 300 --overlap 50
```

### 3. Build FAISS Index

```bash
python build_faiss_index.py --chunks chunks.jsonl --out index_bge
```

### 4. Query the System

**Command Line:**
```bash
python query_compare.py --index_dir index_bge --top_k 5 "startup ideas"
```

**Web Interface:**
```bash
python web_ui.py --index_dir index_bge --chunks chunks.jsonl --port 5000
```

## 🎯 Multi-Stage Reranking

The system supports three scoring methods:

1. **FAISS Score** (Blue): Vector similarity from BGE embeddings
2. **Local Score** (Yellow): Keyword overlap + length preference
3. **Cohere Score** (Green): Advanced semantic reranking (requires API key)

### With Cohere Reranking

```bash
export COHERE_API_KEY=your_key
python web_ui.py --index_dir index_bge --chunks chunks.jsonl --port 5000
```

Then enter your Cohere API key in the web interface for enhanced results.

## 📁 Project Structure

```
paul-graham-rag/
├── scrape_pg_essays.py      # Web scraper
├── chunk_pg_essays.py       # Text chunking
├── build_faiss_index.py     # FAISS index builder
├── query_compare.py         # CLI query tool
├── web_ui.py               # Web interface
├── embedding_utils.py      # BGE embedding utilities
├── demo.py                 # Complete demo pipeline
├── requirements.txt        # Dependencies
├── templates/
│   └── index.html         # Web UI template
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

- `COHERE_API_KEY`: For Cohere reranking
- `ZERANK_URL`: For ZeRank reranking
- `ZERANK_API_KEY`: ZeRank API key

### Key Parameters

- `--chunk_size`: Words per chunk (default: 300)
- `--overlap`: Overlap between chunks (default: 50)
- `--faiss_k`: Candidates for reranking (default: 30)
- `--top_k`: Final results to return (default: 5)

## 🎨 Web Interface Features

- Clean, responsive design
- Real-time search with loading states
- Side-by-side relevancy scores
- Click-through links to original essays
- Optional Cohere API key integration
- Adjustable result count (1-20)

## 🚀 Production Deployment

For production use:

1. Use a production WSGI server (Gunicorn, uWSGI)
2. Set up proper logging
3. Configure environment variables securely
4. Use a production-grade vector database if needed
5. Implement rate limiting and caching

## 📊 Performance

- **Indexing**: ~2-3 minutes for full essay collection
- **Query Time**: ~100-500ms depending on reranking
- **Memory Usage**: ~500MB for full index
- **Storage**: ~50MB for complete system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and demonstration purposes. Please respect Paul Graham's content and website terms of service.


