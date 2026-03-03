from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from starlette.requests import Request

from app import main as app_main
from app.config import Settings
from app.models import CanonicalResume, Identity
from app.services.auth import AuthStore, SessionManager
from app.services.repository import DataRepository


def _configure_main_for_auth_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    secure_cookie: bool,
    ttl_seconds: int,
    allow_self_signup: bool = True,
) -> Settings:
    settings = Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / 'app.db',
        app_env='test',
        app_secret_key='test-secret',
        session_cookie_secure=secure_cookie,
        session_ttl_seconds=ttl_seconds,
        allow_self_signup=allow_self_signup,
    )
    settings.ensure_directories()

    monkeypatch.setattr(app_main, 'settings', settings)
    monkeypatch.setattr(app_main, 'repository', DataRepository(settings))
    monkeypatch.setattr(app_main, 'auth_store', AuthStore(settings.resolved_sqlite_path))
    monkeypatch.setattr(
        app_main,
        'session_manager',
        SessionManager(settings.app_secret_key, ttl_seconds=settings.session_ttl_seconds),
    )
    return settings


def _cookie_header_parts(header_value: str) -> tuple[str, set[str]]:
    segments = [segment.strip() for segment in (header_value or '').split(';') if segment.strip()]
    if not segments:
        return '', set()
    return segments[0], {segment.lower() for segment in segments[1:]}


def _request(method: str, path: str) -> Request:
    return Request(
        {
            'type': 'http',
            'http_version': '1.1',
            'method': method.upper(),
            'scheme': 'http',
            'path': path,
            'raw_path': path.encode('utf-8'),
            'query_string': b'',
            'headers': [],
            'client': ('testclient', 50000),
            'server': ('testserver', 80),
            'root_path': '',
        }
    )


def _post_request(path: str) -> Request:
    return _request('POST', path)


def _get_request(path: str) -> Request:
    return _request('GET', path)


def test_auth_store_create_and_verify(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / 'auth.db')
    user = store.create_user(email='user@example.com', password='supersecure123')
    assert user.email == 'user@example.com'

    verified = store.verify_credentials(email='user@example.com', password='supersecure123')
    assert verified is not None
    assert verified.id == user.id

    assert store.verify_credentials(email='user@example.com', password='wrong-password') is None


def test_auth_store_rejects_invalid_inputs(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / 'auth.db')
    with pytest.raises(ValueError):
        store.create_user(email='not-an-email', password='supersecure123')
    with pytest.raises(ValueError):
        store.create_user(email='a@example.com', password='short')


def test_session_manager_issue_and_parse() -> None:
    manager = SessionManager('test-secret', ttl_seconds=60)
    token = manager.issue('user123')
    assert manager.parse(token) == 'user123'


def test_session_manager_expires() -> None:
    manager = SessionManager('test-secret', ttl_seconds=1)
    token = manager.issue('user123')
    time.sleep(1.2)
    assert manager.parse(token) is None


def test_settings_reject_non_positive_session_ttl() -> None:
    with pytest.raises(ValueError, match='SESSION_TTL_SECONDS'):
        Settings(session_ttl_seconds=0)


def test_settings_reject_default_secret_in_prod() -> None:
    with pytest.raises(ValueError, match='APP_SECRET_KEY'):
        Settings(app_env='prod')


def test_settings_parses_session_cookie_secure_from_string() -> None:
    settings = Settings(session_cookie_secure='1')
    assert settings.session_cookie_secure is True


def test_settings_parses_allow_self_signup_from_string() -> None:
    settings = Settings(allow_self_signup='1')
    assert settings.allow_self_signup is True


def test_settings_parses_enable_profile_rewrite_from_string() -> None:
    settings = Settings(enable_profile_rewrite='1')
    assert settings.enable_profile_rewrite is True


def test_settings_parses_enable_extension_api_from_string() -> None:
    settings = Settings(enable_extension_api='1')
    assert settings.enable_extension_api is True


def test_settings_reject_non_positive_upload_limit() -> None:
    with pytest.raises(ValueError, match='MAX_UPLOAD_MB'):
        Settings(max_upload_mb=0)


def test_settings_uses_tmp_data_dir_on_vercel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('VERCEL', '1')
    settings = Settings()
    assert settings.data_dir == Path('/tmp/data')
    assert settings.resolved_sqlite_path == Path('/tmp/data/app.db')


def test_settings_moves_relative_sqlite_path_under_tmp_on_vercel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('VERCEL', '1')
    settings = Settings(sqlite_path=Path('db/runtime.db'))
    assert settings.resolved_sqlite_path == Path('/tmp/data/db/runtime.db')


def test_register_sets_expected_cookie_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _configure_main_for_auth_routes(
        tmp_path,
        monkeypatch,
        secure_cookie=True,
        ttl_seconds=123,
        allow_self_signup=True,
    )
    response = asyncio.run(
        app_main.auth_register(
            _post_request('/auth/register'),
            email='new-user@example.com',
            password='supersecure123',
            confirm_password='supersecure123',
            next='/',
        )
    )

    assert response.status_code == 303
    cookie_kv, attributes = _cookie_header_parts(response.headers.get('set-cookie', ''))
    assert cookie_kv.startswith(f'{settings.session_cookie_name}=')
    assert 'httponly' in attributes
    assert 'samesite=lax' in attributes
    assert 'path=/' in attributes
    assert f'max-age={settings.session_ttl_seconds}' in attributes
    assert 'secure' in attributes


def test_login_sets_cookie_without_secure_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _configure_main_for_auth_routes(
        tmp_path,
        monkeypatch,
        secure_cookie=False,
        ttl_seconds=321,
    )
    app_main.auth_store.create_user(email='login-user@example.com', password='supersecure123')

    response = asyncio.run(
        app_main.auth_login(
            _post_request('/auth/login'),
            email='login-user@example.com',
            password='supersecure123',
            next='/',
        )
    )

    assert response.status_code == 303
    cookie_kv, attributes = _cookie_header_parts(response.headers.get('set-cookie', ''))
    assert cookie_kv.startswith(f'{settings.session_cookie_name}=')
    assert 'httponly' in attributes
    assert 'samesite=lax' in attributes
    assert 'path=/' in attributes
    assert f'max-age={settings.session_ttl_seconds}' in attributes
    assert 'secure' not in attributes


def test_logout_clears_session_cookie(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _configure_main_for_auth_routes(
        tmp_path,
        monkeypatch,
        secure_cookie=False,
        ttl_seconds=120,
    )
    logout_response = asyncio.run(app_main.auth_logout())

    assert logout_response.status_code == 303
    cookie_kv, attributes = _cookie_header_parts(logout_response.headers.get('set-cookie', ''))
    assert cookie_kv.startswith(f'{settings.session_cookie_name}=')
    assert 'max-age=0' in attributes
    assert 'path=/' in attributes


def test_register_page_available_even_when_signup_flag_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_main_for_auth_routes(
        tmp_path,
        monkeypatch,
        secure_cookie=False,
        ttl_seconds=300,
        allow_self_signup=False,
    )

    response = asyncio.run(app_main.auth_register_page(_get_request('/auth/register')))
    assert response.status_code == 200


def test_max_upload_bytes_uses_setting_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _configure_main_for_auth_routes(
        tmp_path,
        monkeypatch,
        secure_cookie=False,
        ttl_seconds=300,
        allow_self_signup=True,
    )
    settings.max_upload_mb = 3
    assert app_main._max_upload_bytes() == 3 * 1024 * 1024


def test_auth_store_profile_round_trip(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / 'auth.db')
    user = store.create_user(email='profile@example.com', password='supersecure123')

    profile = store.ensure_profile_for_user(user=user)
    assert profile.user_id == user.id
    assert profile.email == user.email
    assert profile.onboarding_state == 'not_started'

    profile.display_name = 'Profile User'
    profile.completed_steps = ['basics']
    profile.onboarding_state = 'in_progress'
    store.upsert_profile(profile)

    loaded = store.get_profile(user.id)
    assert loaded is not None
    assert loaded.display_name == 'Profile User'
    assert 'basics' in loaded.completed_steps

    marked = store.mark_onboarding_step(user_id=user.id, step='resume_import')
    assert marked is not None
    assert 'resume_import' in marked.completed_steps


def test_auth_store_profile_seeded_from_resume(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / 'auth.db')
    user = store.create_user(email='seeded@example.com', password='supersecure123')
    resume = CanonicalResume(
        identity=Identity(
            name='Seeded User',
            email='seeded@example.com',
            phone='555-123-9876',
            location='Chicago, IL',
            links=['https://example.com'],
        ),
        summary='Backend engineer focused on reliability.',
    )

    profile = store.ensure_profile_for_user(user=user, seed_resume=resume)
    assert profile.display_name == 'Seeded User'
    assert profile.phone == '555-123-9876'
    assert 'resume_import' in profile.completed_steps


def test_auth_store_extension_api_key_round_trip(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / 'auth.db')
    user = store.create_user(email='ext@example.com', password='supersecure123')
    assert store.get_extension_api_key_status(user_id=user.id) is None

    first_key = store.regenerate_extension_api_key(user_id=user.id)
    first_status = store.get_extension_api_key_status(user_id=user.id)
    assert first_status is not None
    assert first_status.is_active is True
    assert store.resolve_user_id_from_extension_api_key(first_key) == user.id

    second_key = store.regenerate_extension_api_key(user_id=user.id)
    assert second_key != first_key
    assert store.resolve_user_id_from_extension_api_key(first_key) is None
    assert store.resolve_user_id_from_extension_api_key(second_key) == user.id


def test_auth_store_extension_run_lifecycle(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / 'auth.db')
    user = store.create_user(email='run@example.com', password='supersecure123')
    run = store.create_extension_run(user_id=user.id, job_id='job123')
    assert run.status == 'queued'
    assert run.output_timestamp is None

    store.update_extension_run(run_id=run.run_id, status='running')
    running = store.get_extension_run(run_id=run.run_id, user_id=user.id)
    assert running is not None
    assert running.status == 'running'

    store.update_extension_run(
        run_id=run.run_id,
        status='succeeded',
        output_timestamp='20260303-010203',
    )
    complete = store.get_extension_run(run_id=run.run_id, user_id=user.id)
    assert complete is not None
    assert complete.status == 'succeeded'
    assert complete.output_timestamp == '20260303-010203'
