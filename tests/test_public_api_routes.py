from __future__ import annotations

import asyncio
import json
import re

from starlette.requests import Request
from starlette.responses import Response

from app.main import app
from app import main as app_main


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


def test_mvp_surface_routes_exist() -> None:
    expected = {
        '/',
        '/generate',
        '/advance',
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
