from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final


ENV_PREFIX: Final[str] = "CC_"


def _getenv(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"{ENV_PREFIX}{name}", default)


def _getenv_int(name: str, default: int) -> int:
    raw = _getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid int for {ENV_PREFIX}{name}: {raw!r}") from exc


def _getenv_path(name: str, default: Path) -> Path:
    raw = _getenv(name)
    if raw is None or raw == "":
        return default
    return Path(raw)


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> bool:
    """
    Minimal `.env` loader (no external dependencies).

    - Supports `KEY=VALUE` lines and ignores comments/blank lines.
    - Does not expand variables.
    - By default does NOT override existing environment variables.
    """
    env_path = Path(path)
    if not env_path.exists():
        return False

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not override and key in os.environ:
            continue
        os.environ[key] = value
    return True


@dataclass(frozen=True)
class Settings:
    """
    Central config for API/UI/MCP servers.

    All env vars are prefixed with `CC_` (e.g., `CC_OPENAI_API_KEY`).
    """

    env: str

    model: str

    openai_api_key: str | None
    tavily_api_key: str | None

    aws_region: str
    s3_bucket: str | None
    s3_prefix: str

    web_cache_ttl_days: int

    data_dir: Path
    artifacts_dir: Path
    cache_dir: Path
    logs_dir: Path

    timeout_stats_s: int
    timeout_retrieval_s: int
    timeout_web_s: int
    timeout_sim_s: int
    timeout_fantasy_s: int

    router_mode: str
    router_heuristic_min_score: int
    router_heuristic_margin: int

    memory_db_path: Path
    memory_keep_last_n: int
    memory_summarize_chunk_n: int
    memory_summary_max_chars: int
    summary_model: str

    composer_model: str
    composer_max_chars: int

    @property
    def is_prod(self) -> bool:
        return self.env.lower() in {"prod", "production"}

    def validate_for_prod(self) -> None:
        if not self.openai_api_key:
            raise ValueError("Missing CC_OPENAI_API_KEY for production.")
        if not self.s3_bucket:
            raise ValueError("Missing CC_S3_BUCKET for production.")


@lru_cache(maxsize=1)
def get_settings(*, load_env_file: bool = True) -> Settings:
    """
    Returns a cached Settings instance.

    Set `load_env_file=False` if you want to manage environment variables yourself.
    """
    if load_env_file:
        load_dotenv(".env", override=False)

    env = (_getenv("ENV", "local") or "local").strip()

    data_dir = _getenv_path("DATA_DIR", Path("data"))
    artifacts_dir = _getenv_path("ARTIFACTS_DIR", Path("artifacts"))
    cache_dir = _getenv_path("CACHE_DIR", Path("cache"))
    logs_dir = _getenv_path("LOGS_DIR", Path("logs"))
    memory_db_path = _getenv_path("MEMORY_DB_PATH", Path("data/duckdb/cricket_companion_memory.duckdb"))

    router_mode = (_getenv("ROUTER_MODE", "real") or "real").strip().lower()
    default_min_score = 3 if router_mode == "easy" else 5
    default_margin = 0 if router_mode == "easy" else 3

    return Settings(
        env=env,
        model=_getenv("MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        openai_api_key=_getenv("OPENAI_API_KEY"),
        tavily_api_key=_getenv("TAVILY_API_KEY"),
        aws_region=_getenv("AWS_REGION", "us-east-1") or "us-east-1",
        s3_bucket=_getenv("S3_BUCKET"),
        s3_prefix=_getenv("S3_PREFIX", "cricket-companion") or "cricket-companion",
        web_cache_ttl_days=_getenv_int("WEB_CACHE_TTL_DAYS", 7),
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        cache_dir=cache_dir,
        logs_dir=logs_dir,
        timeout_stats_s=_getenv_int("TIMEOUT_STATS_S", 10),
        timeout_retrieval_s=_getenv_int("TIMEOUT_RETRIEVAL_S", 120),
        timeout_web_s=_getenv_int("TIMEOUT_WEB_S", 30),
        timeout_sim_s=_getenv_int("TIMEOUT_SIM_S", 20),
        timeout_fantasy_s=_getenv_int("TIMEOUT_FANTASY_S", 20),

        router_mode=router_mode,
        router_heuristic_min_score=_getenv_int("ROUTER_MIN_SCORE", default_min_score),
        router_heuristic_margin=_getenv_int("ROUTER_MARGIN", default_margin),

        memory_db_path=memory_db_path,
        memory_keep_last_n=_getenv_int("MEMORY_KEEP_LAST_N", 20),
        memory_summarize_chunk_n=_getenv_int("MEMORY_SUMMARIZE_CHUNK_N", 10),
        memory_summary_max_chars=_getenv_int("MEMORY_SUMMARY_MAX_CHARS", 2000),
        summary_model=_getenv("SUMMARY_MODEL", _getenv("MODEL", "gpt-4o-mini") or "gpt-4o-mini") or "gpt-4o-mini",

        composer_model=_getenv("COMPOSER_MODEL", _getenv("MODEL", "gpt-4o-mini") or "gpt-4o-mini") or "gpt-4o-mini",
        composer_max_chars=_getenv_int("COMPOSER_MAX_CHARS", 2000),
    )
