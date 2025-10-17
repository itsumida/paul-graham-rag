import math
from typing import List, Tuple

import numpy as np

# We support two backends:
# 1) PyTorch + Transformers (if torch is available)
# 2) FastEmbed (no torch required)

def _has_torch() -> bool:
    try:
        import torch  # type: ignore[import-untyped]
        return True
    except Exception:
        return False


def _torch_backend_embed(texts: List[str], is_query: bool, model_name: str, batch_size: int) -> np.ndarray:
    import torch  # type: ignore
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    query_prefix = "Represent this sentence for searching relevant passages: "
    doc_prefix = "Represent this document for retrieval: "
    prefix = query_prefix if is_query else doc_prefix

    vecs: List[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i:i + batch_size]]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            pooled = mean_pooling(out.last_hidden_state, enc["attention_mask"])  # (B, D)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs.append(pooled.cpu().numpy())
    return np.vstack(vecs) if vecs else np.zeros((0, model.config.hidden_size), dtype=np.float32)


def _fastembed_backend_embed(texts: List[str], is_query: bool, model_name: str, batch_size: int) -> np.ndarray:
    # FastEmbed handles batching internally via iterator API; we still slice for memory friendliness.
    from fastembed import TextEmbedding

    query_prefix = "Represent this sentence for searching relevant passages: "
    doc_prefix = "Represent this document for retrieval: "
    prefix = query_prefix if is_query else doc_prefix

    embedder = TextEmbedding(model_name=model_name)
    all_vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i:i + batch_size]]
        # embed returns an iterator of vectors
        batch_vecs = list(embedder.embed(batch))
        arr = np.asarray(batch_vecs, dtype=np.float32)
        # Normalize to unit for cosine via inner product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        all_vecs.append(arr)
    return np.vstack(all_vecs) if all_vecs else np.zeros((0, 384), dtype=np.float32)


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        self.model_name = model_name
        self.use_torch = _has_torch()

    def embed_texts(self, texts: List[str], batch_size: int = 64, is_query: bool = False) -> np.ndarray:
        if self.use_torch:
            return _torch_backend_embed(texts, is_query=is_query, model_name=self.model_name, batch_size=batch_size)
        # Fallback to FastEmbed (no torch required)
        return _fastembed_backend_embed(texts, is_query=is_query, model_name=self.model_name, batch_size=batch_size)


