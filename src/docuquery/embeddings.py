"""Embedding wrapper. The model loads lazily on first use."""

import numpy as np


class Embedder:
    def __init__(self, model_name):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts, batch_size=32):
        """Embed a list of texts; returns L2-normalized float32 vectors."""
        model = self._ensure_model()
        vecs = model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text):
        return self.encode([text])[0]

    @property
    def dim(self):
        return self._ensure_model().get_sentence_embedding_dimension()
