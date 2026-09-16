from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.exists() else ".env",
        env_prefix="NPL_",
        extra="ignore",
    )

    app_mode: str = "demo"  # demo | production
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "https://npl-nowcasting.vercel.app",
    ]
    database_url: str = "sqlite:///./npl.db"
    data_dir: str = "data/demo"
    openai_api_key: Optional[str] = None
    news_api_key: Optional[str] = None
    # Google Gemini (Ask AI) — free-tier flash models
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.6-flash"
    # Legacy Qwen fields (ignored by Ask AI, kept for old .env compatibility)
    qwen_api_key: Optional[str] = None
    qwen_model: str = "qwen-plus"
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    default_target: str = "gross_npl_ratio"
    random_seed: int = 42


settings = Settings()
