from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        populate_by_name=True,
    )

    host: str = '0.0.0.0'
    port: int = 8030

    data_dir: Path = Path('data')
    templates_dir: Path = Path('app/templates')
    sqlite_path: Path = Field(default=Path('data/app.db'), alias='SQLITE_PATH')

    enable_ocr: bool = Field(default=False, alias='ENABLE_OCR')
    resume_app_token: Optional[str] = Field(default=None, alias='RESUME_APP_TOKEN')
    allow_self_signup: bool = Field(default=False, alias='ALLOW_SELF_SIGNUP')
    max_upload_mb: int = Field(default=10, alias='MAX_UPLOAD_MB')

    openai_api_key: Optional[str] = Field(default=None, alias='OPENAI_API_KEY')
    openai_model: str = Field(default='gpt-4.1-mini', alias='OPENAI_MODEL')

    request_token_header: str = 'X-Resume-Token'
    app_env: str = Field(default='dev', alias='APP_ENV')
    app_secret_key: str = Field(default='dev-change-me', alias='APP_SECRET_KEY')
    session_cookie_name: str = Field(default='resume_session', alias='SESSION_COOKIE_NAME')
    session_ttl_seconds: int = Field(default=60 * 60 * 24 * 7, alias='SESSION_TTL_SECONDS')
    session_cookie_secure: bool = Field(default=False, alias='SESSION_COOKIE_SECURE')
    bootstrap_user_email: Optional[str] = Field(default=None, alias='BOOTSTRAP_USER_EMAIL')
    bootstrap_user_password: Optional[str] = Field(default=None, alias='BOOTSTRAP_USER_PASSWORD')

    @field_validator('session_cookie_secure', 'allow_self_signup', mode='before')
    @classmethod
    def _parse_bool_like(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {'1', 'true', 'yes', 'on'}:
                return True
            if normalized in {'0', 'false', 'no', 'off', ''}:
                return False
            raise ValueError('Expected a boolean-like value.')
        if isinstance(value, int):
            return bool(value)
        return value

    @model_validator(mode='after')
    def _validate_runtime_safety(self) -> 'Settings':
        if self.session_ttl_seconds <= 0:
            raise ValueError('SESSION_TTL_SECONDS must be greater than zero.')
        if self.max_upload_mb <= 0:
            raise ValueError('MAX_UPLOAD_MB must be greater than zero.')

        normalized_env = (self.app_env or '').strip().lower()
        if not normalized_env:
            raise ValueError('APP_ENV must not be empty.')
        if normalized_env not in {'dev', 'development', 'test'} and self.app_secret_key == 'dev-change-me':
            raise ValueError('APP_SECRET_KEY must be changed when APP_ENV is not dev/test.')
        return self

    @property
    def resolved_sqlite_path(self) -> Path:
        if self.sqlite_path.is_absolute():
            return self.sqlite_path
        parts = self.sqlite_path.parts
        if parts and parts[0] == 'data':
            if len(parts) == 1:
                return self.data_dir / 'app.db'
            return self.data_dir / Path(*parts[1:])
        return self.sqlite_path

    def ensure_directories(self) -> None:
        required = [
            self.data_dir / 'resume',
            self.data_dir / 'vault' / 'items',
            self.data_dir / 'jobs',
            self.data_dir / 'outputs',
            self.data_dir / 'uploads',
            self.data_dir / 'users',
        ]
        for path in required:
            path.mkdir(parents=True, exist_ok=True)
        self.resolved_sqlite_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
