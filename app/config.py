from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    host: str = '0.0.0.0'
    port: int = 8030

    data_dir: Path = Path('data')
    templates_dir: Path = Path('app/templates')

    enable_ocr: bool = Field(default=False, alias='ENABLE_OCR')
    resume_app_token: Optional[str] = Field(default=None, alias='RESUME_APP_TOKEN')

    openai_api_key: Optional[str] = Field(default=None, alias='OPENAI_API_KEY')
    openai_model: str = Field(default='gpt-4.1-mini', alias='OPENAI_MODEL')

    request_token_header: str = 'X-Resume-Token'

    def ensure_directories(self) -> None:
        required = [
            self.data_dir / 'resume',
            self.data_dir / 'vault' / 'items',
            self.data_dir / 'jobs',
            self.data_dir / 'outputs',
            self.data_dir / 'uploads',
        ]
        for path in required:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
