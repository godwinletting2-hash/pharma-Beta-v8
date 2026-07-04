"""
core/config.py
--------------
Central configuration for PharmaScanKE.
All environment-driven settings handled via pydantic-settings.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ───────────────────────────────────────────────────────────
    APP_TITLE: str = "PharmaScanKE"
    APP_VERSION: str = "2.1.0"
    DEBUG: bool = False

    # ── Server ────────────────────────────────────────────────────────────────
    PORT: int = int(os.environ.get("PORT", 8000))
    HOST: str = "0.0.0.0"

    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploaded_notes"
    TEMPLATES_DIR: Path = BASE_DIR / "templates"
    STATIC_DIR: Path = BASE_DIR / "static"

    # ── Database ──────────────────────────────────────────────────────────────
    # Use PHARMASCAN_DATABASE_URL so we don't collide with the workspace-level
    # DATABASE_URL env var (which points to the Node.js api-server's Postgres DB).
    PHARMASCAN_DATABASE_URL: str = f"sqlite+aiosqlite:///{Path(__file__).resolve().parent.parent}/pharmascan.db"

    @property
    def DATABASE_URL(self) -> str:
        return self.PHARMASCAN_DATABASE_URL

    # ── Upload Constraints ────────────────────────────────────────────────────
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt"}
    MAX_UPLOAD_SIZE_MB: int = 50

    # ── Semesters ─────────────────────────────────────────────────────────────
    VALID_SEMESTERS: list = ["Y1S1", "Y1S2", "Y2S1", "Y2S2", "Y3S1", "Y3S2"]

    # ── AI / Groq ─────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MAX_TOKENS: int = 8192
    GROQ_TEMPERATURE: float = 0.3

    # ── Community channels (optional) ─────────────────────────────────────────
    WHATSAPP_CHANNEL_URL: str = os.environ.get("WHATSAPP_CHANNEL_URL", "")
    TELEGRAM_CHANNEL_URL: str = os.environ.get("TELEGRAM_CHANNEL_URL", "")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# Guarantee the upload directory exists at import time
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
