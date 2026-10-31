"""
Central configuration module for Agentic Data Analyst.
Uses Pydantic BaseSettings to load environment variables and default parameters.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    # Base paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent

    # Local LLM (Ollama)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"

    # Cloud Fallback (Optional)
    GEMINI_API_KEY: str = ""
    USE_CLOUD_FALLBACK: bool = False

    # Persistence Directories (Relative to PROJECT_ROOT)
    SQLITE_DB_PATH: str = "data/analysis_history.db"
    CHROMA_PERSIST_DIR: str = "data/vector_store"
    UPLOADS_DIR: str = "data/uploads"
    SAMPLES_DIR: str = "data/samples"

    # Execution & File limits
    MAX_RETRY_COUNT: int = 3
    MAX_QUERY_EXEC_SECONDS: int = 10
    MAX_RESULT_ROWS: int = 500
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list[str] = [".csv", ".xlsx", ".xls"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_sqlite_path(self) -> Path:
        p = self.PROJECT_ROOT / self.SQLITE_DB_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def get_uploads_path(self) -> Path:
        p = self.PROJECT_ROOT / self.UPLOADS_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_samples_path(self) -> Path:
        p = self.PROJECT_ROOT / self.SAMPLES_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_chroma_path(self) -> Path:
        p = self.PROJECT_ROOT / self.CHROMA_PERSIST_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p


# Singleton instance
settings = AppSettings()
