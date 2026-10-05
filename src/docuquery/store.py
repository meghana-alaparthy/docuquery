"""FAISS-backed vector store with a JSON sidecar for chunk text and sources."""

import json
from pathlib import Path

import numpy as np


class DocStore:
    def __init__(self, dim, index_path, meta_path):
        import faiss

        self._faiss = faiss
        self.dim = dim
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        # inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []  # list of {"id", "text", "source"}

    def add(self, chunks, embeddings):
        base = len(self.chunks)
        for i, ch in enumerate(chunks):
            self.chunks.append({"id": base + i, "text": ch["text"], "source": ch["source"]})
        vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
        if len(vecs):
            self.index.add(vecs)

    def search(self, query_vector, k=5):
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, ids = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            ch = self.chunks[int(idx)]
            results.append({"text": ch["text"], "source": ch["source"], "score": float(score)})
        return results

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.chunks))

    @classmethod
    def load(cls, dim, index_path, meta_path):
        import faiss

        store = cls.__new__(cls)
        store._faiss = faiss
        store.dim = dim
        store.index_path = Path(index_path)
        store.meta_path = Path(meta_path)
        store.index = faiss.read_index(str(index_path))
        store.chunks = json.loads(Path(meta_path).read_text())
        return store
