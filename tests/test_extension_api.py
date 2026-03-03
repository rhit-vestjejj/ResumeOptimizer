from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException
import pytest
from starlette.requests import Request

from app import main as app_main
from app.config import Settings
from app.services.auth import AuthStore, SessionManager
from app.services.repository import DataRepository


def _configure_extension_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    enable_extension_api: bool = True,
) -> None:
    settings = Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / 'app.db',
        app_env='test',
        app_secret_key='test-secret',
        enable_extension_api=enable_extension_api,
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
    app_main.EXTENSION_RUN_TASKS.clear()


def _request(
    *,
    method: str,
    path: str,
    json_body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Request:
    body_bytes = b''
    if json_body is not None:
        body_bytes = json.dumps(json_body).encode('utf-8')

    sent = False

    async def receive() -> Dict[str, Any]:
        nonlocal sent
        if sent:
            return {'type': 'http.request', 'body': b'', 'more_body': False}
        sent = True
        return {'type': 'http.request', 'body': body_bytes, 'more_body': False}

    normalized_headers: list[tuple[bytes, bytes]] = []
    for key, value in (headers or {}).items():
        normalized_headers.append((key.lower().encode('utf-8'), str(value).encode('utf-8')))
    if json_body is not None:
        normalized_headers.append((b'content-type', b'application/json'))

    scope = {
        'type': 'http',
        'http_version': '1.1',
        'method': method.upper(),
        'scheme': 'http',
        'path': path,
        'raw_path': path.encode('utf-8'),
        'query_string': b'',
        'headers': normalized_headers,
        'client': ('testclient', 50000),
        'server': ('testserver', 80),
        'root_path': '',
        'state': state or {},
    }
    return Request(scope, receive)


def _json_response_body(response) -> Dict[str, Any]:
    return json.loads(response.body.decode('utf-8'))


def test_extension_api_user_resolves_from_bearer_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_extension_app(tmp_path, monkeypatch)
    user = app_main.auth_store.create_user(email='ext-api@example.com', password='supersecure123')
    key_v1 = app_main.auth_store.regenerate_extension_api_key(user_id=user.id)
    req_v1 = _request(
        method='GET',
        path='/api/ext/v1/key/status',
        headers={'Authorization': f'Bearer {key_v1}'},
    )
    resolved_v1 = app_main._extension_api_user(req_v1)
    assert resolved_v1 is not None
    assert resolved_v1.id == user.id

    key_v2 = app_main.auth_store.regenerate_extension_api_key(user_id=user.id)
    req_stale = _request(
        method='GET',
        path='/api/ext/v1/key/status',
        headers={'Authorization': f'Bearer {key_v1}'},
    )
    assert app_main._extension_api_user(req_stale) is None

    req_v2 = _request(
        method='GET',
        path='/api/ext/v1/key/status',
        headers={'Authorization': f'Bearer {key_v2}'},
    )
    resolved_v2 = app_main._extension_api_user(req_v2)
    assert resolved_v2 is not None
    assert resolved_v2.id == user.id


def test_extension_api_user_disabled_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_extension_app(tmp_path, monkeypatch, enable_extension_api=False)
    user = app_main.auth_store.create_user(email='ext-disabled@example.com', password='supersecure123')
    api_key = app_main.auth_store.regenerate_extension_api_key(user_id=user.id)
    request = _request(
        method='GET',
        path='/api/ext/v1/key/status',
        headers={'Authorization': f'Bearer {api_key}'},
    )
    assert app_main._extension_api_user(request) is None


def test_extension_key_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_extension_app(tmp_path, monkeypatch)
    user = app_main.auth_store.create_user(email='ext-endpoints@example.com', password='supersecure123')

    status_request = _request(
        method='GET',
        path='/api/ext/v1/key/status',
        state={'current_user_id': user.id},
    )
    status_response = asyncio.run(app_main.extension_key_status(status_request))
    status_payload = _json_response_body(status_response)
    assert status_payload['has_key'] is False

    rotate_request = _request(
        method='POST',
        path='/api/ext/v1/key/regenerate',
        state={'current_user_id': user.id},
    )
    rotate_response = asyncio.run(app_main.extension_key_regenerate(rotate_request))
    rotate_payload = _json_response_body(rotate_response)
    assert rotate_payload['api_key'].startswith('rox_')
    assert rotate_payload['key_id']

    status_after_request = _request(
        method='GET',
        path='/api/ext/v1/key/status',
        state={'current_user_id': user.id},
    )
    status_after_response = asyncio.run(app_main.extension_key_status(status_after_request))
    status_after_payload = _json_response_body(status_after_response)
    assert status_after_payload['has_key'] is True
    assert status_after_payload['key_id'] == rotate_payload['key_id']


def test_extension_tailor_run_lifecycle_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_extension_app(tmp_path, monkeypatch)
    user = app_main.auth_store.create_user(email='run-api@example.com', password='supersecure123')

    def _create_task_stub(coro):
        coro.close()
        loop = asyncio.get_running_loop()
        done = loop.create_future()
        done.set_result(None)
        return done

    monkeypatch.setattr(app_main.asyncio, 'create_task', _create_task_stub)

    create_request = _request(
        method='POST',
        path='/api/ext/v1/tailor-runs',
        json_body={
            'jd_text': 'Backend Engineer role requiring Python, SQL, and Docker.',
            'job_title': 'Backend Engineer',
            'company': 'ExampleCo',
            'source_url': 'https://example.com/jobs/backend',
        },
        state={'current_user_id': user.id},
    )
    create_response = asyncio.run(app_main.extension_create_tailor_run(create_request))
    assert create_response.status_code == 202
    create_payload = _json_response_body(create_response)
    assert create_payload['status'] == 'queued'

    run_id = create_payload['run_id']
    run_status_request = _request(
        method='GET',
        path=f'/api/ext/v1/tailor-runs/{run_id}',
        state={'current_user_id': user.id},
    )
    run_status_response = asyncio.run(app_main.extension_tailor_run_status(run_status_request, run_id))
    run_status_payload = _json_response_body(run_status_response)
    assert run_status_payload['status'] == 'queued'

    run = app_main.auth_store.get_extension_run(run_id=run_id, user_id=user.id)
    assert run is not None
    assert run.job_id == create_payload['job_id']

    timestamp = '20260303-120000'
    app_main.auth_store.update_extension_run(
        run_id=run_id,
        status='succeeded',
        error=None,
        output_timestamp=timestamp,
    )

    outputs_root = app_main.repository.outputs_dir_for(user.id)
    pdf_path = outputs_root / run.job_id / timestamp / 'resume.pdf'
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b'%PDF-1.4\n%%EOF\n')

    ready_request = _request(
        method='GET',
        path=f'/api/ext/v1/tailor-runs/{run_id}',
        state={'current_user_id': user.id},
    )
    ready_response = asyncio.run(app_main.extension_tailor_run_status(ready_request, run_id))
    ready_payload = _json_response_body(ready_response)
    assert ready_payload['status'] == 'succeeded'
    assert ready_payload['pdf_download_url'].endswith(f'/api/ext/v1/tailor-runs/{run_id}/resume.pdf')

    pdf_request = _request(
        method='GET',
        path=f'/api/ext/v1/tailor-runs/{run_id}/resume.pdf',
        state={'current_user_id': user.id},
    )
    pdf_response = asyncio.run(app_main.extension_tailor_run_pdf(pdf_request, run_id))
    assert pdf_response.status_code == 200
    assert str(pdf_response.media_type).lower() == 'application/pdf'


def test_extension_tailor_run_requires_jd_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_extension_app(tmp_path, monkeypatch)
    user = app_main.auth_store.create_user(email='run-required@example.com', password='supersecure123')

    request = _request(
        method='POST',
        path='/api/ext/v1/tailor-runs',
        json_body={},
        state={'current_user_id': user.id},
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_main.extension_create_tailor_run(request))
    assert exc_info.value.status_code == 400


def test_extension_status_downgrades_succeeded_when_pdf_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_extension_app(tmp_path, monkeypatch)
    user = app_main.auth_store.create_user(email='missing-pdf@example.com', password='supersecure123')
    run = app_main.auth_store.create_extension_run(user_id=user.id, job_id='jobmissingpdf')
    app_main.auth_store.update_extension_run(
        run_id=run.run_id,
        status='succeeded',
        error=None,
        output_timestamp='20260303-230000',
    )

    status_request = _request(
        method='GET',
        path=f'/api/ext/v1/tailor-runs/{run.run_id}',
        state={'current_user_id': user.id},
    )
    status_response = asyncio.run(app_main.extension_tailor_run_status(status_request, run.run_id))
    payload = _json_response_body(status_response)
    assert payload['status'] == 'failed'
    assert 'resume.pdf is missing' in payload['error']

    stored = app_main.auth_store.get_extension_run(run_id=run.run_id, user_id=user.id)
    assert stored is not None
    assert stored.status == 'failed'
    assert stored.output_timestamp is None
