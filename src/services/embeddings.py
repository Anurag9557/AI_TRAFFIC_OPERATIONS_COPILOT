"""Sentence Transformer embedding and FAISS index utilities.

Only vector data is written to FAISS. Event metadata remains in SQLite, and
the Hugging Face model uses its normal user cache outside the project.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when embeddings or a FAISS index cannot be produced."""


class TextEncoder(Protocol):
    """Small encoder contract that keeps retrieval tests lightweight."""

    model_name: str

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Return normalized float32 embeddings for the supplied texts."""


class SentenceTransformerEncoder:
    """Lazy CPU-friendly wrapper around `all-MiniLM-L6-v2`."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: object | None = None

    def _load_model(self) -> object:
        """Download/load the model on first use without saving it in the repo."""

        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. Run "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        LOGGER.info("Loading embedding model %s on %s.", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Encode and L2-normalize texts for cosine search with FAISS."""

        if not texts:
            raise EmbeddingError("At least one text is required for embedding.")
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be positive.")

        model = self._load_model()
        try:
            vectors = model.encode(  # type: ignore[attr-defined]
                list(texts),
                batch_size=batch_size,
                show_progress_bar=len(texts) > batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

        embeddings = np.asarray(vectors, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise EmbeddingError(
                f"Encoder returned unexpected shape {embeddings.shape} for "
                f"{len(texts)} texts."
            )
        return np.ascontiguousarray(embeddings)


@dataclass(frozen=True)
class IndexMetadata:
    """Compact sidecar metadata required to map FAISS rows to SQLite IDs."""

    version: int
    model_name: str
    dimension: int
    event_count: int
    event_ids: list[str]


@dataclass(frozen=True)
class IndexBuildSummary:
    """Result returned after a successful persisted index build."""

    event_count: int
    dimension: int
    model_name: str
    index_path: str
    mapping_path: str
    index_bytes: int
    mapping_bytes: int


def _import_faiss() -> object:
    """Import FAISS lazily and provide an actionable dependency error."""

    try:
        import faiss
    except ImportError as exc:
        raise EmbeddingError(
            "faiss-cpu is not installed. Run "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return faiss


def load_embedding_source(database_path: Path) -> tuple[list[str], list[str]]:
    """Load ordered event IDs and canonical text without duplicating records."""

    if not database_path.exists():
        raise EmbeddingError(f"SQLite database does not exist: {database_path}")

    try:
        with sqlite3.connect(database_path) as connection:
            rows = connection.execute(
                """
                SELECT event_id, canonical_text
                FROM events
                WHERE canonical_text IS NOT NULL
                  AND LENGTH(TRIM(canonical_text)) > 0
                ORDER BY event_id
                """
            ).fetchall()
    except sqlite3.Error as exc:
        raise EmbeddingError(f"Unable to read events from SQLite: {exc}") from exc

    if not rows:
        raise EmbeddingError("No canonical event text is available for indexing.")

    event_ids = [str(row[0]) for row in rows]
    texts = [str(row[1]) for row in rows]
    if len(event_ids) != len(set(event_ids)):
        raise EmbeddingError("Duplicate event IDs found in embedding source.")
    return event_ids, texts


def create_faiss_index(embeddings: np.ndarray) -> object:
    """Create an exact cosine-similarity index for normalized embeddings."""

    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise EmbeddingError(f"Invalid embedding matrix shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise EmbeddingError("Embedding matrix contains non-finite values.")

    vectors = np.ascontiguousarray(vectors)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise EmbeddingError("Embedding matrix contains zero-length vectors.")
    vectors /= norms[:, np.newaxis]

    faiss = _import_faiss()
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)
    return index


def persist_faiss_index(
    index: object,
    metadata: IndexMetadata,
    index_path: Path,
    mapping_path: Path,
) -> None:
    """Atomically persist the FAISS index and compact ID mapping sidecar."""

    index_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    index_temp = index_path.with_name(f"{index_path.name}.tmp")
    mapping_temp = mapping_path.with_name(f"{mapping_path.name}.tmp")
    faiss = _import_faiss()

    try:
        faiss.write_index(index, str(index_temp))
        mapping_temp.write_text(
            json.dumps(asdict(metadata), separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(index_temp, index_path)
        os.replace(mapping_temp, mapping_path)
    except (OSError, ValueError, RuntimeError) as exc:
        index_temp.unlink(missing_ok=True)
        mapping_temp.unlink(missing_ok=True)
        raise EmbeddingError(f"Unable to persist FAISS artifacts: {exc}") from exc


def load_faiss_index(
    index_path: Path,
    mapping_path: Path,
) -> tuple[object, IndexMetadata]:
    """Load and cross-check persisted FAISS artifacts."""

    if not index_path.exists():
        raise EmbeddingError(f"FAISS index does not exist: {index_path}")
    if not mapping_path.exists():
        raise EmbeddingError(f"FAISS mapping does not exist: {mapping_path}")

    faiss = _import_faiss()
    try:
        index = faiss.read_index(str(index_path))
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        metadata = IndexMetadata(**payload)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise EmbeddingError(f"Unable to load FAISS artifacts: {exc}") from exc

    if metadata.version != 1:
        raise EmbeddingError(f"Unsupported FAISS mapping version: {metadata.version}")
    if index.ntotal != metadata.event_count:
        raise EmbeddingError(
            f"Index contains {index.ntotal} vectors but mapping declares "
            f"{metadata.event_count}."
        )
    if len(metadata.event_ids) != metadata.event_count:
        raise EmbeddingError("FAISS event-ID mapping length is inconsistent.")
    if index.d != metadata.dimension:
        raise EmbeddingError(
            f"Index dimension {index.d} does not match mapping "
            f"dimension {metadata.dimension}."
        )
    return index, metadata


def build_faiss_index(
    database_path: Path,
    index_path: Path,
    mapping_path: Path,
    encoder: TextEncoder,
    batch_size: int = 64,
) -> IndexBuildSummary:
    """Build and persist an exact FAISS index from SQLite canonical text."""

    event_ids, texts = load_embedding_source(database_path)
    LOGGER.info("Generating embeddings for %s historical events.", len(texts))
    embeddings = encoder.encode(texts, batch_size=batch_size)
    if embeddings.shape[0] != len(event_ids):
        raise EmbeddingError("Embedding count does not match source event count.")

    index = create_faiss_index(embeddings)
    metadata = IndexMetadata(
        version=1,
        model_name=encoder.model_name,
        dimension=int(embeddings.shape[1]),
        event_count=len(event_ids),
        event_ids=event_ids,
    )
    persist_faiss_index(index, metadata, index_path, mapping_path)

    summary = IndexBuildSummary(
        event_count=len(event_ids),
        dimension=int(embeddings.shape[1]),
        model_name=encoder.model_name,
        index_path=str(index_path.resolve()),
        mapping_path=str(mapping_path.resolve()),
        index_bytes=index_path.stat().st_size,
        mapping_bytes=mapping_path.stat().st_size,
    )
    LOGGER.info(
        "Built FAISS index with %s vectors and %s dimensions.",
        summary.event_count,
        summary.dimension,
    )
    return summary
