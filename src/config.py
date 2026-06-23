"""Application configuration for the hackathon MVP.

The module centralizes filesystem paths and environment-driven settings so
ingestion scripts and later application phases use the same configuration.
No OpenAI client or other external service is initialized here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependencies are installed in setup.
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str, root: Path) -> Path:
    """Resolve an environment path relative to the project root."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


class Settings(BaseModel):
    """Validated settings shared by scripts and future application modules."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    project_root: Path = PROJECT_ROOT
    raw_csv_path: Path
    database_path: Path
    faiss_index_path: Path
    faiss_mapping_path: Path
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4-mini"
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # A deliberately broad operating-area check for Bengaluru event records.
    bengaluru_min_latitude: float = 12.6
    bengaluru_max_latitude: float = 13.3
    bengaluru_min_longitude: float = 77.2
    bengaluru_max_longitude: float = 78.0
    local_timezone: str = "Asia/Kolkata"

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load settings from `.env` and process environment variables."""

        if load_dotenv is not None:
            load_dotenv(PROJECT_ROOT / ".env", override=False)

        return cls(
            raw_csv_path=_resolve_path(
                os.getenv("TRAFFIC_DATA_CSV", "data/raw/events.csv"),
                PROJECT_ROOT,
            ),
            database_path=_resolve_path(
                os.getenv("TRAFFIC_DATABASE_PATH", "data/traffic_ops.db"),
                PROJECT_ROOT,
            ),
            faiss_index_path=_resolve_path(
                os.getenv("FAISS_INDEX_PATH", "data/events.faiss"),
                PROJECT_ROOT,
            ),
            faiss_mapping_path=_resolve_path(
                os.getenv("FAISS_MAPPING_PATH", "data/faiss_event_ids.json"),
                PROJECT_ROOT,
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            openai_api_key=(
                SecretStr(os.environ["OPENAI_API_KEY"])
                if os.getenv("OPENAI_API_KEY")
                else None
            ),
            openai_model=os.getenv("OPENAI_MODEL") or "gpt-5.4-mini",
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )

    def ensure_directories(self) -> None:
        """Create directories used by generated local artifacts."""

        self.raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_mapping_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one cached settings instance for the current process."""

    settings = Settings.from_environment()
    settings.ensure_directories()
    return settings
