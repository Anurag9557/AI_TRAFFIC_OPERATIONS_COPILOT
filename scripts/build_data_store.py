"""Build the local SQLite event store from the supplied historical CSV."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.data.ingestion import IngestionError, ingest_csv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the ingestion pipeline."""

    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Clean the historical event CSV and rebuild SQLite.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=settings.raw_csv_path,
        help=f"Source CSV path (default: {settings.raw_csv_path})",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=settings.database_path,
        help=f"SQLite path (default: {settings.database_path})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000,
        help="Number of events inserted per SQLite batch.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=settings.log_level,
        help="Console logging level.",
    )
    return parser


def main() -> int:
    """Run ingestion and print its validated JSON summary."""

    parser = build_parser()
    arguments = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    settings = get_settings()
    csv_path = arguments.csv.expanduser().resolve()
    database_path = arguments.database.expanduser().resolve()

    try:
        summary = ingest_csv(
            csv_path=csv_path,
            database_path=database_path,
            settings=settings,
            batch_size=arguments.batch_size,
        )
    except IngestionError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    print(summary.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

