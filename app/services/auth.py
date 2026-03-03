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

from app.models import CanonicalResume, UserProfile


EMAIL_PATTERN = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
PROFILE_STEPS = ('basics', 'resume_import', 'evidence', 'first_job')
EXTENSION_RUN_STATES = {'queued', 'running', 'succeeded', 'failed'}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_email(email: str) -> str:
    return (email or '').strip().lower()


def _is_valid_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.fullmatch(_normalize_email(email)))


def _json_dump_list(values: list[str]) -> str:
    return json.dumps([str(item).strip() for item in values if str(item).strip()])


def _json_load_list(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    output: list[str] = []
    for item in parsed:
        cleaned = str(item or '').strip()
        if cleaned:
            output.append(cleaned)
    return output


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


@dataclass
class ExtensionApiKeyStatus:
    user_id: str
    key_id: str
    created_at: str
    rotated_at: str
    is_active: bool


@dataclass
class ExtensionRun:
    run_id: str
    user_id: str
    job_id: str
    status: str
    error: Optional[str]
    output_timestamp: Optional[str]
    created_at: str
    updated_at: str


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
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL DEFAULT '',
                    email TEXT NOT NULL DEFAULT '',
                    phone TEXT NOT NULL DEFAULT '',
                    location TEXT NOT NULL DEFAULT '',
                    links_json TEXT NOT NULL DEFAULT '[]',
                    headline TEXT NOT NULL DEFAULT '',
                    target_roles_json TEXT NOT NULL DEFAULT '[]',
                    years_experience TEXT NOT NULL DEFAULT '',
                    onboarding_state TEXT NOT NULL DEFAULT 'not_started',
                    completed_steps_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                '''
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS extension_api_keys (
                    user_id TEXT PRIMARY KEY,
                    key_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    rotated_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                '''
            )
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS extension_runs (
                    run_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    output_timestamp TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
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

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        with self._connect() as connection:
            row = connection.execute(
                '''
                SELECT
                    user_id,
                    display_name,
                    email,
                    phone,
                    location,
                    links_json,
                    headline,
                    target_roles_json,
                    years_experience,
                    onboarding_state,
                    completed_steps_json,
                    created_at,
                    updated_at
                FROM user_profiles
                WHERE user_id = ?
                ''',
                (user_id,),
            ).fetchone()
        return self._row_to_profile(row)

    def upsert_profile(self, profile: UserProfile) -> None:
        normalized_steps: list[str] = []
        seen_steps: set[str] = set()
        for step in profile.completed_steps:
            cleaned = str(step or '').strip().lower()
            if cleaned not in PROFILE_STEPS or cleaned in seen_steps:
                continue
            seen_steps.add(cleaned)
            normalized_steps.append(cleaned)

        normalized_links = [str(link).strip() for link in profile.links if str(link).strip()]
        normalized_roles = [str(role).strip() for role in profile.target_roles if str(role).strip()]
        now = _utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                '''
                INSERT INTO user_profiles (
                    user_id,
                    display_name,
                    email,
                    phone,
                    location,
                    links_json,
                    headline,
                    target_roles_json,
                    years_experience,
                    onboarding_state,
                    completed_steps_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    email = excluded.email,
                    phone = excluded.phone,
                    location = excluded.location,
                    links_json = excluded.links_json,
                    headline = excluded.headline,
                    target_roles_json = excluded.target_roles_json,
                    years_experience = excluded.years_experience,
                    onboarding_state = excluded.onboarding_state,
                    completed_steps_json = excluded.completed_steps_json,
                    updated_at = excluded.updated_at
                ''',
                (
                    profile.user_id,
                    profile.display_name.strip(),
                    _normalize_email(profile.email) or profile.email.strip(),
                    profile.phone.strip(),
                    profile.location.strip(),
                    _json_dump_list(normalized_links),
                    profile.headline.strip(),
                    _json_dump_list(normalized_roles),
                    profile.years_experience.strip(),
                    profile.onboarding_state,
                    _json_dump_list(normalized_steps),
                    profile.created_at,
                    now,
                ),
            )
            connection.commit()

    def ensure_profile_for_user(self, *, user: AuthUser, seed_resume: Optional[CanonicalResume] = None) -> UserProfile:
        existing = self.get_profile(user.id)
        if existing:
            return existing

        now = _utc_now_iso()
        completed_steps: list[str] = []
        display_name = ''
        email = user.email
        phone = ''
        location = ''
        links: list[str] = []
        headline = ''
        target_roles: list[str] = []

        if seed_resume:
            identity = seed_resume.identity
            display_name = identity.name
            email = identity.email or user.email
            phone = identity.phone
            location = identity.location
            links = list(identity.links or [])
            completed_steps = ['basics', 'resume_import']

        onboarding_state = 'in_progress' if completed_steps else 'not_started'
        profile = UserProfile(
            user_id=user.id,
            display_name=display_name,
            email=email,
            phone=phone,
            location=location,
            links=links,
            headline=headline,
            target_roles=target_roles,
            years_experience='',
            onboarding_state=onboarding_state,
            completed_steps=completed_steps,
            created_at=now,
            updated_at=now,
        )
        self.upsert_profile(profile)
        stored = self.get_profile(user.id)
        return stored or profile

    def mark_onboarding_step(self, *, user_id: str, step: str) -> Optional[UserProfile]:
        cleaned_step = str(step or '').strip().lower()
        if cleaned_step not in PROFILE_STEPS:
            return self.get_profile(user_id)
        profile = self.get_profile(user_id)
        if profile is None:
            return None
        if cleaned_step not in profile.completed_steps:
            profile.completed_steps.append(cleaned_step)
        if all(step_name in profile.completed_steps for step_name in PROFILE_STEPS):
            profile.onboarding_state = 'completed'
        elif profile.completed_steps:
            profile.onboarding_state = 'in_progress'
        else:
            profile.onboarding_state = 'not_started'
        self.upsert_profile(profile)
        return self.get_profile(user_id)

    @staticmethod
    def _hash_extension_api_key(api_key: str) -> str:
        return hashlib.sha256((api_key or '').encode('utf-8')).hexdigest()

    def regenerate_extension_api_key(self, *, user_id: str) -> str:
        now = _utc_now_iso()
        key_id = secrets.token_hex(6)
        secret = secrets.token_hex(24)
        plain_key = f'rox_{key_id}_{secret}'
        key_hash = self._hash_extension_api_key(plain_key)
        with self._connect() as connection:
            connection.execute(
                '''
                INSERT INTO extension_api_keys (user_id, key_id, key_hash, created_at, rotated_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    key_id = excluded.key_id,
                    key_hash = excluded.key_hash,
                    rotated_at = excluded.rotated_at,
                    is_active = excluded.is_active
                ''',
                (user_id, key_id, key_hash, now, now, 1),
            )
            connection.commit()
        return plain_key

    def get_extension_api_key_status(self, *, user_id: str) -> Optional[ExtensionApiKeyStatus]:
        with self._connect() as connection:
            row = connection.execute(
                '''
                SELECT user_id, key_id, created_at, rotated_at, is_active
                FROM extension_api_keys
                WHERE user_id = ?
                ''',
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return ExtensionApiKeyStatus(
            user_id=str(row['user_id']),
            key_id=str(row['key_id']),
            created_at=str(row['created_at']),
            rotated_at=str(row['rotated_at']),
            is_active=bool(row['is_active']),
        )

    def resolve_user_id_from_extension_api_key(self, api_key: str) -> Optional[str]:
        normalized = str(api_key or '').strip()
        if not normalized:
            return None
        key_hash = self._hash_extension_api_key(normalized)
        with self._connect() as connection:
            row = connection.execute(
                '''
                SELECT user_id
                FROM extension_api_keys
                WHERE key_hash = ? AND is_active = 1
                ''',
                (key_hash,),
            ).fetchone()
        if row is None:
            return None
        return str(row['user_id'])

    def create_extension_run(self, *, user_id: str, job_id: str) -> ExtensionRun:
        run_id = secrets.token_hex(12)
        now = _utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                '''
                INSERT INTO extension_runs (
                    run_id, user_id, job_id, status, error, output_timestamp, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (run_id, user_id, job_id, 'queued', None, None, now, now),
            )
            connection.commit()
        run = self.get_extension_run(run_id=run_id, user_id=user_id)
        if run is None:
            raise RuntimeError('Failed to create extension run.')
        return run

    def update_extension_run(
        self,
        *,
        run_id: str,
        status: str,
        error: Optional[str] = None,
        output_timestamp: Optional[str] = None,
    ) -> None:
        normalized_status = str(status or '').strip().lower()
        if normalized_status not in EXTENSION_RUN_STATES:
            raise ValueError(f'Invalid extension run status: {status}')
        with self._connect() as connection:
            connection.execute(
                '''
                UPDATE extension_runs
                SET status = ?, error = ?, output_timestamp = ?, updated_at = ?
                WHERE run_id = ?
                ''',
                (normalized_status, error, output_timestamp, _utc_now_iso(), run_id),
            )
            connection.commit()

    def get_extension_run(self, *, run_id: str, user_id: str) -> Optional[ExtensionRun]:
        with self._connect() as connection:
            row = connection.execute(
                '''
                SELECT run_id, user_id, job_id, status, error, output_timestamp, created_at, updated_at
                FROM extension_runs
                WHERE run_id = ? AND user_id = ?
                ''',
                (run_id, user_id),
            ).fetchone()
        return self._row_to_extension_run(row)

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

    @staticmethod
    def _row_to_profile(row: Optional[sqlite3.Row]) -> Optional[UserProfile]:
        if row is None:
            return None
        return UserProfile(
            user_id=str(row['user_id']),
            display_name=str(row['display_name'] or ''),
            email=str(row['email'] or ''),
            phone=str(row['phone'] or ''),
            location=str(row['location'] or ''),
            links=_json_load_list(row['links_json']),
            headline=str(row['headline'] or ''),
            target_roles=_json_load_list(row['target_roles_json']),
            years_experience=str(row['years_experience'] or ''),
            onboarding_state=str(row['onboarding_state'] or 'not_started'),
            completed_steps=_json_load_list(row['completed_steps_json']),
            created_at=str(row['created_at'] or _utc_now_iso()),
            updated_at=str(row['updated_at'] or _utc_now_iso()),
        )

    @staticmethod
    def _row_to_extension_run(row: Optional[sqlite3.Row]) -> Optional[ExtensionRun]:
        if row is None:
            return None
        return ExtensionRun(
            run_id=str(row['run_id']),
            user_id=str(row['user_id']),
            job_id=str(row['job_id']),
            status=str(row['status']),
            error=str(row['error']) if row['error'] else None,
            output_timestamp=str(row['output_timestamp']) if row['output_timestamp'] else None,
            created_at=str(row['created_at']),
            updated_at=str(row['updated_at']),
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
