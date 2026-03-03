from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app import main as app_main
from app.config import Settings
from app.services.auth import AuthStore, SessionManager
from app.services.repository import DataRepository


def _configure_main_for_output_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
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
    return settings


def _request(path: str, method: str = 'GET') -> Request:
    return Request(
        {
            'type': 'http',
            'http_version': '1.1',
            'method': method,
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


def test_output_latex_editor_loads_and_saves_tex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_main_for_output_routes(tmp_path, monkeypatch)
    output_dir = app_main.repository.create_output_dir('job123')
    tex_path = output_dir / 'resume.tex'
    tex_path.write_text('\\documentclass{article}\n', encoding='utf-8')

    load_response = asyncio.run(
        app_main.output_latex_editor(_request(f'/outputs/job123/{output_dir.name}/latex'), 'job123', output_dir.name)
    )
    assert load_response.status_code == 200

    updated_tex = '\\documentclass{article}\n\\begin{document}\nUpdated\n\\end{document}\n'
    save_response = asyncio.run(
        app_main.output_latex_editor_save(
            _request(f'/outputs/job123/{output_dir.name}/latex', method='POST'),
            'job123',
            output_dir.name,
            tex_content=updated_tex,
            action='save',
        )
    )
    assert save_response.status_code == 200
    assert tex_path.read_text(encoding='utf-8') == updated_tex


def test_output_latex_editor_rejects_empty_tex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_main_for_output_routes(tmp_path, monkeypatch)
    output_dir = app_main.repository.create_output_dir('job123')
    tex_path = output_dir / 'resume.tex'
    original_tex = '\\documentclass{article}\n\\begin{document}\nOriginal\n\\end{document}\n'
    tex_path.write_text(original_tex, encoding='utf-8')

    response = asyncio.run(
        app_main.output_latex_editor_save(
            _request(f'/outputs/job123/{output_dir.name}/latex', method='POST'),
            'job123',
            output_dir.name,
            tex_content='   ',
            action='save',
        )
    )
    assert response.status_code == 200
    assert tex_path.read_text(encoding='utf-8') == original_tex


def test_resolve_output_paths_raises_for_missing_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_main_for_output_routes(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as exc:
        app_main._resolve_output_paths('missing-job', 'missing-run')
    assert exc.value.status_code == 404
