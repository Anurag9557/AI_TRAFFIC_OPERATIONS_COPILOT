"""Build the Phase 3 FAISS index from SQLite canonical event text."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.services.embeddings import (  # noqa: E402
    EmbeddingError,
    SentenceTransformerEncoder,
    build_faiss_index,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the lightweight FAISS build command."""

    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Build an exact FAISS index from SQLite historical events.",
    )
    parser.add_argument("--database", type=Path, default=settings.database_path)
    parser.add_argument("--index", type=Path, default=settings.faiss_index_path)
    parser.add_argument("--mapping", type=Path, default=settings.faiss_mapping_path)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=settings.log_level,
    )
    return parser


def main() -> int:
    """Build the index and print a compact JSON storage summary."""

    arguments = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    encoder = SentenceTransformerEncoder(
        model_name=arguments.model,
        device=arguments.device,
    )
    try:
        summary = build_faiss_index(
            database_path=arguments.database.expanduser().resolve(),
            index_path=arguments.index.expanduser().resolve(),
            mapping_path=arguments.mapping.expanduser().resolve(),
            encoder=encoder,
            batch_size=arguments.batch_size,
        )
    except EmbeddingError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
