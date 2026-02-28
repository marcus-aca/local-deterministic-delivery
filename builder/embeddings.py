"""Shared embedding backend with offline deterministic fallback."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
_BACKEND_CACHE: dict[str, "EmbeddingBackend"] = {}

# Keep local runs quiet and deterministic.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def deterministic_embedding(text: str, dim: int = EMBEDDING_DIMENSION) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []

    while len(values) < dim:
        digest = hashlib.sha256(digest).digest()
        for byte in digest:
            values.append((byte / 127.5) - 1.0)
            if len(values) == dim:
                break

    return values


@dataclass
class EmbeddingBackend:
    model_name: str
    using_fallback: bool = False

    def __post_init__(self) -> None:
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None or self.using_fallback:
            return

        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self.using_fallback = True
            self._model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._load_model()

        if self._model is None:
            return [deterministic_embedding(text) for text in texts]

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in embeddings]


def build_embedding_backend() -> EmbeddingBackend:
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    if model_name not in _BACKEND_CACHE:
        _BACKEND_CACHE[model_name] = EmbeddingBackend(model_name=model_name)
    return _BACKEND_CACHE[model_name]
