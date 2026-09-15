from __future__ import annotations

from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_mode: str = "demo"  # demo | production
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    database_url: str = "sqlite:///./npl.db"
    data_dir: str = "data/demo"
    openai_api_key: Optional[str] = None
    news_api_key: Optional[str] = None
    # Qwen / DashScope (OpenAI-compatible chat completions)
    qwen_api_key: Optional[str] = None
    qwen_model: str = "qwen-plus"
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    default_target: str = "gross_npl_ratio"
    random_seed: int = 42

    class Config:
        env_file = ".env"
        env_prefix = "NPL_"


settings = Settings()
