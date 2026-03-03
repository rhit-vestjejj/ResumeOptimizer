from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


EMAIL_PATTERN = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_email(email: str) -> str:
    return (email or '').strip().lower()


def _is_valid_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.fullmatch(_normalize_email(email)))


def _hash_password(password: str, *, iterations: int = 260_000) -> str:
    if len(password) < 8:
        raise ValueError('Password must be at least 8 characters.')
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
    return f'pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}'


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iter_text, salt_hex, digest_hex = password_hash.split('$', 3)
    except ValueError:
        return False
    if algorithm != 'pbkdf2_sha256':
        return False
    try:
        iterations = int(iter_text)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
    return hmac.compare_digest(candidate, expected)


@dataclass
class AuthUser:
    id: str
    email: str
    password_hash: str
    created_at: str
    last_login_at: Optional[str]
    is_active: bool


class AuthStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1
                )
                '''
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS base_resumes (
                    user_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                '''
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS vault_items (
                    user_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    item_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, item_id),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                '''
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS jobs (
                    user_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    title TEXT,
                    company TEXT,
                    url TEXT,
                    path TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, job_id),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                '''
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS outputs (
                    user_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    path TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, job_id, timestamp),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                '''
            )
            connection.commit()

    def count_users(self) -> int:
        with self._connect() as connection:
            row = connection.execute('SELECT COUNT(*) AS count FROM users').fetchone()
            return int(row['count']) if row else 0

    def get_user_by_email(self, email: str) -> Optional[AuthUser]:
        normalized = _normalize_email(email)
        with self._connect() as connection:
            row = connection.execute(
                '''
                SELECT id, email, password_hash, created_at, last_login_at, is_active
                FROM users
                WHERE email = ?
                ''',
                (normalized,),
            ).fetchone()
        return self._row_to_user(row)

    def get_user_by_id(self, user_id: str) -> Optional[AuthUser]:
        with self._connect() as connection:
            row = connection.execute(
                '''
                SELECT id, email, password_hash, created_at, last_login_at, is_active
                FROM users
                WHERE id = ?
                ''',
                (user_id,),
            ).fetchone()
        return self._row_to_user(row)

    def create_user(self, *, email: str, password: str) -> AuthUser:
        normalized = _normalize_email(email)
        if not _is_valid_email(normalized):
            raise ValueError('Enter a valid email address.')
        if self.get_user_by_email(normalized):
            raise ValueError('An account with this email already exists.')
        user = AuthUser(
            id=secrets.token_hex(12),
            email=normalized,
            password_hash=_hash_password(password),
            created_at=_utc_now_iso(),
            last_login_at=None,
            is_active=True,
        )
        with self._connect() as connection:
            connection.execute(
                '''
                INSERT INTO users (id, email, password_hash, created_at, last_login_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    user.id,
                    user.email,
                    user.password_hash,
                    user.created_at,
                    user.last_login_at,
                    1 if user.is_active else 0,
                ),
            )
            connection.commit()
        return user

    def verify_credentials(self, *, email: str, password: str) -> Optional[AuthUser]:
        user = self.get_user_by_email(email)
        if user is None or not user.is_active:
            return None
        if not _verify_password(password, user.password_hash):
            return None
        return user

    def update_last_login(self, user_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                'UPDATE users SET last_login_at = ? WHERE id = ?',
                (_utc_now_iso(), user_id),
            )
            connection.commit()

    def ensure_bootstrap_user(self, *, email: Optional[str], password: Optional[str]) -> Optional[AuthUser]:
        normalized = _normalize_email(email or '')
        if not normalized or not password:
            return None
        existing = self.get_user_by_email(normalized)
        if existing:
            return existing
        return self.create_user(email=normalized, password=password)

    def upsert_base_resume(self, *, user_id: str, path: str) -> None:
        with self._connect() as connection:
            connection.execute(
                '''
                INSERT INTO base_resumes (user_id, path, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    path = excluded.path,
                    updated_at = excluded.updated_at
                ''',
                (user_id, path, _utc_now_iso()),
            )
            connection.commit()

    def upsert_vault_item(
        self,
        *,
        user_id: str,
        item_id: str,
        title: str,
        item_type: str,
        path: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                '''
                INSERT INTO vault_items (user_id, item_id, title, item_type, path, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, item_id) DO UPDATE SET
                    title = excluded.title,
                    item_type = excluded.item_type,
                    path = excluded.path,
                    updated_at = excluded.updated_at
                ''',
                (user_id, item_id, title, item_type, path, _utc_now_iso()),
            )
            connection.commit()

    def delete_vault_item(self, *, user_id: str, item_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                'DELETE FROM vault_items WHERE user_id = ? AND item_id = ?',
                (user_id, item_id),
            )
            connection.commit()

    def upsert_job(
        self,
        *,
        user_id: str,
        job_id: str,
        title: Optional[str],
        company: Optional[str],
        url: Optional[str],
        path: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                '''
                INSERT INTO jobs (user_id, job_id, title, company, url, path, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, job_id) DO UPDATE SET
                    title = excluded.title,
                    company = excluded.company,
                    url = excluded.url,
                    path = excluded.path,
                    updated_at = excluded.updated_at
                ''',
                (user_id, job_id, title, company, url, path, _utc_now_iso()),
            )
            connection.commit()

    def upsert_output(self, *, user_id: str, job_id: str, timestamp: str, path: str) -> None:
        with self._connect() as connection:
            connection.execute(
                '''
                INSERT INTO outputs (user_id, job_id, timestamp, path, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, job_id, timestamp) DO UPDATE SET
                    path = excluded.path,
                    updated_at = excluded.updated_at
                ''',
                (user_id, job_id, timestamp, path, _utc_now_iso()),
            )
            connection.commit()

    @staticmethod
    def _row_to_user(row: Optional[sqlite3.Row]) -> Optional[AuthUser]:
        if row is None:
            return None
        return AuthUser(
            id=str(row['id']),
            email=str(row['email']),
            password_hash=str(row['password_hash']),
            created_at=str(row['created_at']),
            last_login_at=str(row['last_login_at']) if row['last_login_at'] else None,
            is_active=bool(row['is_active']),
        )


class SessionManager:
    def __init__(self, secret_key: str, ttl_seconds: int = 60 * 60 * 24 * 7) -> None:
        if not secret_key:
            raise ValueError('secret_key is required for session signing.')
        self.secret_key = secret_key.encode('utf-8')
        self.ttl_seconds = max(1, int(ttl_seconds))

    @staticmethod
    def _b64_encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode('ascii').rstrip('=')

    @staticmethod
    def _b64_decode(value: str) -> bytes:
        padding = '=' * (-len(value) % 4)
        return base64.urlsafe_b64decode((value + padding).encode('ascii'))

    def _sign(self, payload_b64: str) -> str:
        digest = hmac.new(self.secret_key, payload_b64.encode('utf-8'), hashlib.sha256).digest()
        return self._b64_encode(digest)

    def issue(self, user_id: str) -> str:
        issued_at = int(time.time())
        payload = {'uid': user_id, 'iat': issued_at, 'exp': issued_at + self.ttl_seconds}
        payload_b64 = self._b64_encode(json.dumps(payload, separators=(',', ':')).encode('utf-8'))
        signature = self._sign(payload_b64)
        return f'{payload_b64}.{signature}'

    def parse(self, token: Optional[str]) -> Optional[str]:
        if not token or '.' not in token:
            return None
        payload_b64, signature = token.split('.', 1)
        expected = self._sign(payload_b64)
        if not hmac.compare_digest(expected, signature):
            return None
        try:
            payload = json.loads(self._b64_decode(payload_b64).decode('utf-8'))
        except Exception:
            return None
        exp = int(payload.get('exp', 0) or 0)
        user_id = str(payload.get('uid', '')).strip()
        if not user_id or exp <= int(time.time()):
            return None
        return user_id
