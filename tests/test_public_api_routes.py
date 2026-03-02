from __future__ import annotations

from app.main import app


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


def test_mvp_surface_routes_exist() -> None:
    expected = {
        '/',
        '/generate',
        '/advance',
    }
    actual = {route.path for route in app.routes}
    assert expected.issubset(actual)
