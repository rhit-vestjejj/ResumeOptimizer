from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from starlette.requests import Request
from starlette.responses import Response

from app.main import app
from app import main as app_main
from app.config import Settings
from app.services.auth import AuthStore, SessionManager
from app.services.repository import DataRepository


def test_required_public_api_routes_exist() -> None:
    expected = {
        '/api/upload_resume',
        '/api/upload_job_description',
        '/api/lint_resume',
        '/api/parse_mirror',
        '/api/build_canonical',
        '/api/score_parse_quality',
        '/api/score_match',
        '/api/generate_patches',
        '/api/apply_patches',
        '/api/render_outputs',
        '/api/export_bundle/{job_id}/{timestamp}',
    }
    actual = {route.path for route in app.routes}
    assert expected.issubset(actual)


def test_audit_routes_exist() -> None:
    expected = {
        '/audit',
        '/audit/run',
    }
    actual = {route.path for route in app.routes}
    assert expected.issubset(actual)


def test_ops_routes_exist() -> None:
    expected = {
        '/healthz',
        '/readyz',
    }
    actual = {route.path for route in app.routes}
    assert expected.issubset(actual)


def test_extension_api_routes_exist() -> None:
    expected = {
        '/api/ext/v1/key/status',
        '/api/ext/v1/key/regenerate',
        '/api/ext/v1/tailor-runs',
        '/api/ext/v1/tailor-runs/{run_id}',
        '/api/ext/v1/tailor-runs/{run_id}/resume.pdf',
    }
    actual = {route.path for route in app.routes}
    assert expected.issubset(actual)


def test_mvp_surface_routes_exist() -> None:
    expected = {
        '/',
        '/generate',
        '/advance',
        '/profile',
        '/profile/step/{step_id}',
        '/auth/login',
        '/auth/register',
        '/auth/logout',
    }
    actual = {route.path for route in app.routes}
    assert expected.issubset(actual)


def test_app_uses_lifespan_startup() -> None:
    assert app.router.on_startup == []


def test_timestamp_helper_format() -> None:
    stamp = app_main.datetime_now_stamp()
    assert re.fullmatch(r'\d{8}-\d{6}-\d{6}', stamp)


def test_clean_payload_text_handles_none_and_strips() -> None:
    assert app_main._clean_payload_text(None) == ''
    assert app_main._clean_payload_text('  value  ') == 'value'
    assert app_main._clean_payload_text(42) == '42'


def _get_request(path: str) -> Request:
    return Request(
        {
            'type': 'http',
            'http_version': '1.1',
            'method': 'GET',
            'scheme': 'http',
            'path': path,
            'raw_path': path.encode('utf-8'),
            'query_string': b'',
            'headers': [],
            'client': ('testclient', 50000),
            'server': ('testserver', 80),
            'root_path': '',
            'state': {},
        }
    )


def test_generate_page_renders_with_setup_checklist() -> None:
    response = asyncio.run(app_main.generate_page(_get_request('/')))
    assert response.status_code == 200


def test_generate_page_renders_guided_home_when_flag_enabled(tmp_path, monkeypatch) -> None:
    settings = Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / 'app.db',
        app_env='test',
        app_secret_key='test-secret',
        enable_profile_rewrite=True,
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
    response = asyncio.run(app_main.generate_page(_get_request('/')))
    assert response.status_code == 200


def test_advance_page_renders_with_setup_checklist() -> None:
    response = asyncio.run(app_main.advance_page(_get_request('/advance')))
    assert response.status_code == 200


def test_legacy_dashboard_routes_redirect_to_workspace() -> None:
    response = asyncio.run(app_main.legacy_dashboard_redirect())
    assert response.status_code == 303
    assert response.headers.get('location') == '/advance'

    alias_response = asyncio.run(app_main.advanced_dashboard_redirect())
    assert alias_response.status_code == 303
    assert alias_response.headers.get('location') == '/advance'


def test_request_id_header_added_to_public_route() -> None:
    async def _call_next(_request):
        return Response('ok')

    request = _get_request('/auth/login')
    response = asyncio.run(app_main.attach_request_id(request, _call_next))
    request_id = response.headers.get('X-Request-ID', '')
    assert request_id
    assert len(request_id) == 12


def test_readyz_payload_shape() -> None:
    response = asyncio.run(app_main.readiness())
    body = json.loads(response.body.decode('utf-8'))
    assert 'status' in body
    assert 'checks' in body
    assert 'sqlite' in body['checks']


def test_setup_checklist_completion_count(monkeypatch) -> None:
    class _RepoStub:
        def load_base_resume(self):
            return object()

        def list_vault_items(self):
            return [('item', object())]

        def list_jobs(self):
            return []

    class _LLMStub:
        available = True

    monkeypatch.setattr(app_main, 'repository', _RepoStub())
    monkeypatch.setattr(app_main, 'llm_service', _LLMStub())
    checklist = app_main._build_setup_checklist()
    assert checklist['completed_count'] == 3
    assert checklist['total'] == 4


def test_derive_jd_signal_extracts_role_and_skills() -> None:
    jd = '''
Machine Learning Engineer
Required:
- 3+ years of Python and SQL
- Experience with AWS and Docker
Preferred:
- FastAPI
'''
    signal = app_main._derive_jd_signal(jd)
    assert signal['role_title']
    assert 'Python' in signal['skills']
    assert 'SQL' in signal['skills']


def test_update_profile_focus_from_jd_updates_profile(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / 'app.db',
        app_env='test',
        app_secret_key='test-secret',
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

    user = app_main.auth_store.create_user(email='focus@example.com', password='supersecure123')
    app_main.auth_store.ensure_profile_for_user(user=user, seed_resume=None)
    app_main._update_profile_focus_from_jd(
        user.id,
        jd_text='Backend Engineer role requiring Python, SQL, and Docker.',
        fallback_title='Backend Engineer',
    )

    profile = app_main.auth_store.get_profile(user.id)
    assert profile is not None
    assert profile.target_roles
    assert profile.headline
