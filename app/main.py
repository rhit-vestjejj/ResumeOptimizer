from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import quote

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.models import CanonicalResume, JobRecord, TailorMode, UserProfile, VaultItem
from app.public_api import (
    apply_patches as public_apply_patches,
    build_canonical as public_build_canonical,
    export_bundle as public_export_bundle,
    generate_patches as public_generate_patches,
    lint_resume as public_lint_resume,
    parse_mirror as public_parse_mirror,
    render_outputs as public_render_outputs,
    score_match as public_score_match,
    score_parse_quality as public_score_parse_quality,
    upload_job_description as public_upload_job_description,
    upload_resume as public_upload_resume,
)
from app.services.extractors import canonicalize_resume_text, extract_resume_text
from app.services.ats_engine import (
    build_requirement_graph,
    compare_versions,
    compute_match_score,
    parse_job_description,
    version_resume,
)
from app.services.auth import AuthStore, AuthUser, SessionManager
from app.services.latex import LatexRenderError, LatexService
from app.services.llm import LLMService, LLMUnavailableError
from app.services.repository import DataRepository, get_current_user_id, reset_current_user_id, set_current_user_id
from app.services.scraper import ScrapeError, scrape_job_posting
from app.services.tailoring import (
    MAX_PROJECT_ITEMS,
    MIN_PROJECT_ITEMS,
    expand_resume_with_projects,
    tailor_resume,
    tighten_resume_for_one_page,
)
from app.services.vault_sync import sync_base_resume_to_vault
from app.services.vault_ingest import VaultIngestError, parse_uploaded_text, parse_vault_source_text
from app.storage import load_json, save_json
from app.utils import ensure_within, normalize_token

settings: Settings = get_settings()
repository = DataRepository(settings)
auth_store = AuthStore(settings.resolved_sqlite_path)
session_manager = SessionManager(settings.app_secret_key, ttl_seconds=settings.session_ttl_seconds)
latex_service = LatexService(settings.templates_dir)

llm_init_error: Optional[str] = None
try:
    llm_service = LLMService(settings.openai_api_key, settings.openai_model)
except LLMUnavailableError as exc:
    llm_service = None
    llm_init_error = str(exc)

@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _bootstrap_startup_user()
    yield


app = FastAPI(title='Local Resume Tailor', version='1.0.0', lifespan=_lifespan)
app.mount('/static', StaticFiles(directory='app/static'), name='static')
templates = Jinja2Templates(directory='app/templates')
extension_origins = [item.strip() for item in settings.extension_allowed_origins.split(',') if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=extension_origins,
    allow_origin_regex=r'chrome-extension://.*',
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['Authorization', 'Content-Type', settings.request_token_header],
    allow_credentials=False,
)

DEFAULT_TAILOR_MODE = TailorMode.HARD_TRUTH
TERM_DISPLAY_OVERRIDES: Dict[str, str] = {
    'ml': 'Machine Learning',
    'ai': 'AI',
    'llm': 'LLM',
    'nlp': 'NLP',
    'sql': 'SQL',
    'api': 'API',
    'etl': 'ETL',
    'aws': 'AWS',
    'gcp': 'GCP',
    'dbt': 'dbt',
    'xgboost': 'XGBoost',
    'pytorch': 'PyTorch',
    'tensorflow': 'TensorFlow',
    'postgresql': 'PostgreSQL',
    'kafka': 'Kafka',
    'sklearn': 'scikit-learn',
}
UPPERCASE_DISPLAY_TERMS = {'ai', 'llm', 'nlp', 'sql', 'api', 'etl', 'aws', 'gcp', 'dbt'}
logger = logging.getLogger(__name__)
GUIDED_PROFILE_STEPS: tuple[Dict[str, str], ...] = (
    {
        'id': 'basics',
        'label': 'Profile Basics',
        'description': 'Confirm your identity and contact details.',
        'href': '/profile/step/basics',
    },
    {
        'id': 'resume_import',
        'label': 'Resume Import',
        'description': 'Upload and parse your latest resume into canonical data.',
        'href': '/profile/step/resume_import',
    },
    {
        'id': 'evidence',
        'label': 'Evidence Vault',
        'description': 'Add at least one evidence item so tailoring has proof.',
        'href': '/profile/step/evidence',
    },
    {
        'id': 'first_job',
        'label': 'First Job',
        'description': 'Ingest a role and run your first tailored output.',
        'href': '/profile/step/first_job',
    },
)
REQUIRED_SETUP_STEP_IDS = {'basics', 'resume_import', 'evidence'}
EXTENSION_ROUTE_PREFIX = '/api/ext/v1/'
EXTENSION_RUN_TASKS: Dict[str, asyncio.Task] = {}


def _request_id(request: Request) -> str:
    value = getattr(request.state, 'request_id', '')
    return str(value).strip()


def _with_request_id(request: Request, message: str) -> str:
    request_id = _request_id(request)
    if not request_id:
        return message
    return f'{message} (request id: {request_id})'


def _flash_message(request: Request) -> Optional[Dict[str, str]]:
    key = (request.query_params.get('flash') or '').strip()
    mapping = {
        'resume_saved': 'Base resume saved and synced to vault.',
        'vault_synced': 'Base resume synced into vault items.',
        'vault_saved': 'Vault item saved.',
        'vault_deleted': 'Vault item deleted.',
        'job_ingested': 'Job description ingested and ready to tailor.',
        'jd_saved': 'Job description text saved.',
        'profile_saved': 'Profile basics saved.',
        'profile_resume_imported': 'Resume imported and profile data refreshed.',
    }
    message = mapping.get(key)
    if not message:
        return None
    return {'kind': 'success', 'message': message}


def render(request: Request, template_name: str, context: Dict[str, Any]) -> HTMLResponse:
    current_user = getattr(request.state, 'current_user', None)
    base_context = {
        'request': request,
        'llm_available': bool(llm_service and llm_service.available),
        'token_header': settings.request_token_header,
        'llm_init_error': llm_init_error,
        'current_user': current_user,
        'is_authenticated': bool(current_user),
        'request_id': _request_id(request),
        'flash': _flash_message(request),
        'enable_profile_rewrite': settings.enable_profile_rewrite,
    }
    base_context.update(context)
    return templates.TemplateResponse(template_name, base_context)


@app.middleware('http')
async def attach_request_id(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex[:12]
    response = await call_next(request)
    response.headers['X-Request-ID'] = _request_id(request)
    return response


@app.middleware('http')
async def require_token_header(request: Request, call_next):
    required_token = settings.resume_app_token
    if request.url.path.startswith('/auth/') or request.url.path.startswith(EXTENSION_ROUTE_PREFIX):
        return await call_next(request)
    if required_token and request.method.upper() == 'POST':
        provided_token = request.headers.get(settings.request_token_header)
        if provided_token != required_token:
            return JSONResponse(
                status_code=401,
                content={
                    'error': _with_request_id(request, 'Missing or invalid token header for POST request.'),
                    'required_header': settings.request_token_header,
                },
            )
    return await call_next(request)


def _path_requires_auth(path: str) -> bool:
    public_prefixes = (
        '/auth/',
        '/static/',
        '/docs',
        '/openapi.json',
        '/redoc',
        '/healthz',
        '/readyz',
    )
    return not path.startswith(public_prefixes)


def _expects_json_response(request: Request) -> bool:
    accept = (request.headers.get('accept') or '').lower()
    return request.url.path.startswith('/api/') or 'application/json' in accept


def _extract_bearer_token(request: Request) -> str:
    header = str(request.headers.get('authorization') or '').strip()
    if not header.lower().startswith('bearer '):
        return ''
    return header.split(' ', 1)[1].strip()


def _extension_api_user(request: Request) -> Optional[AuthUser]:
    if not request.url.path.startswith(EXTENSION_ROUTE_PREFIX):
        return None
    if not settings.enable_extension_api:
        return None
    token = _extract_bearer_token(request)
    user_id = auth_store.resolve_user_id_from_extension_api_key(token)
    if not user_id:
        return None
    return auth_store.get_user_by_id(user_id)


def _login_redirect(path: str, query: str) -> RedirectResponse:
    next_target = path
    if query:
        next_target = f'{path}?{query}'
    return RedirectResponse(url=f'/auth/login?next={quote(next_target, safe="/?=&")}', status_code=303)


@app.middleware('http')
async def require_auth_session(request: Request, call_next):
    if request.url.path.startswith(EXTENSION_ROUTE_PREFIX) and not settings.enable_extension_api:
        return JSONResponse(status_code=404, content={'error': 'Extension API is disabled.'})

    session_token = request.cookies.get(settings.session_cookie_name)
    user_id = session_manager.parse(session_token)
    current_user: Optional[AuthUser] = auth_store.get_user_by_id(user_id) if user_id else None
    if current_user is None:
        current_user = _extension_api_user(request)
    if current_user and not current_user.is_active:
        current_user = None

    request.state.current_user = current_user
    request.state.current_user_id = current_user.id if current_user else None
    token = set_current_user_id(request.state.current_user_id)
    try:
        if _path_requires_auth(request.url.path) and request.state.current_user_id is None:
            if _expects_json_response(request):
                return JSONResponse(status_code=401, content={'error': _with_request_id(request, 'Authentication required.')})
            return _login_redirect(request.url.path, request.url.query)
        return await call_next(request)
    finally:
        reset_current_user_id(token)


def _bootstrap_startup_user() -> None:
    try:
        user = auth_store.ensure_bootstrap_user(
            email=settings.bootstrap_user_email,
            password=settings.bootstrap_user_password,
        )
    except Exception as exc:
        logger.exception('Bootstrap user initialization failed: %s', exc)
        return
    if not user:
        return
    try:
        repository.migrate_legacy_data_to_user(user.id)
    except Exception as exc:
        logger.warning('Legacy migration skipped for bootstrap user %s: %s', user.id, exc)
    try:
        profile = auth_store.ensure_profile_for_user(
            user=user,
            seed_resume=repository.load_base_resume(user_id=user.id),
        )
        _sync_profile_progress(user.id, profile=profile)
    except Exception as exc:
        logger.warning('Profile bootstrap skipped for user %s: %s', user.id, exc)
    _safe_index_call(f'refresh_indexes:{user.id}', lambda: _refresh_user_indexes(user.id))


def _safe_next_path(value: Optional[str]) -> str:
    candidate = (value or '').strip()
    if not candidate.startswith('/'):
        return '/'
    if candidate.startswith('//'):
        return '/'
    return candidate


def _session_cookie_kwargs() -> Dict[str, Any]:
    return {
        'max_age': settings.session_ttl_seconds,
        'httponly': True,
        'secure': settings.session_cookie_secure,
        'samesite': 'lax',
        'path': '/',
    }


def _current_user(request: Request) -> Optional[AuthUser]:
    user = getattr(request.state, 'current_user', None)
    if isinstance(user, AuthUser):
        return user
    return None


def _active_user_id() -> Optional[str]:
    user_id = get_current_user_id()
    if user_id and user_id.strip():
        return user_id
    return None


def _max_upload_bytes() -> int:
    return settings.max_upload_mb * 1024 * 1024


def _upload_limit_message() -> str:
    return f'File exceeds the {settings.max_upload_mb} MB upload limit.'


def _build_setup_checklist() -> Dict[str, Any]:
    base_resume_exists = repository.load_base_resume() is not None
    vault_count = len(repository.list_vault_items())
    job_count = len(repository.list_jobs())
    llm_ready = bool(llm_service and llm_service.available)
    items = [
        {
            'label': 'Upload base resume',
            'done': base_resume_exists,
            'href': '/resume/upload',
            'detail': 'Parse your latest resume and store canonical profile data.',
        },
        {
            'label': 'Add vault evidence',
            'done': vault_count > 0,
            'href': '/vault',
            'detail': f'{vault_count} item{"s" if vault_count != 1 else ""} currently available.',
        },
        {
            'label': 'Enable AI tailoring',
            'done': llm_ready,
            'href': '/advance',
            'detail': 'Set OPENAI_API_KEY in environment config.',
        },
        {
            'label': 'Ingest at least one job',
            'done': job_count > 0,
            'href': '/jobs/new',
            'detail': f'{job_count} saved job{"s" if job_count != 1 else ""}.',
        },
    ]
    completed_count = sum(1 for item in items if item['done'])
    return {'items': items, 'completed_count': completed_count, 'total': len(items)}


def _load_or_create_profile(user: Optional[AuthUser]) -> Optional[UserProfile]:
    if user is None:
        return None
    seed_resume = repository.load_base_resume(user_id=user.id)
    profile = auth_store.ensure_profile_for_user(user=user, seed_resume=seed_resume)
    return _sync_profile_progress(user.id, profile=profile)


def _sync_profile_progress(user_id: str, *, profile: Optional[UserProfile] = None) -> Optional[UserProfile]:
    current = profile or auth_store.get_profile(user_id)
    if current is None:
        return None

    completed_set = {str(step).strip().lower() for step in current.completed_steps}
    if current.display_name.strip() and current.email.strip():
        completed_set.add('basics')
    if repository.load_base_resume(user_id=user_id):
        completed_set.add('resume_import')
    if len(repository.list_vault_items(user_id=user_id)) > 0:
        completed_set.add('evidence')
    if len(repository.list_jobs(user_id=user_id)) > 0:
        completed_set.add('first_job')

    ordered_steps = [row['id'] for row in GUIDED_PROFILE_STEPS if row['id'] in completed_set]
    current.completed_steps = ordered_steps
    if not ordered_steps:
        current.onboarding_state = 'not_started'
    elif REQUIRED_SETUP_STEP_IDS.issubset(set(ordered_steps)) and 'first_job' in ordered_steps:
        current.onboarding_state = 'completed'
    else:
        current.onboarding_state = 'in_progress'

    auth_store.upsert_profile(current)
    return auth_store.get_profile(user_id)


def _build_profile_steps(profile: Optional[UserProfile]) -> list[Dict[str, Any]]:
    base_resume_exists = repository.load_base_resume() is not None
    vault_count = len(repository.list_vault_items())
    job_count = len(repository.list_jobs())
    completed_set = set(profile.completed_steps if profile else [])
    llm_ready = bool(llm_service and llm_service.available)

    details: Dict[str, str] = {
        'basics': 'Save your name and contact details.',
        'resume_import': 'Upload and parse your latest resume.',
        'evidence': f'{vault_count} vault item{"s" if vault_count != 1 else ""} currently available.',
        'first_job': f'{job_count} job{"s" if job_count != 1 else ""} ingested.',
    }
    done_by_data = {
        'basics': bool(profile and profile.display_name.strip() and profile.email.strip()),
        'resume_import': base_resume_exists,
        'evidence': vault_count > 0,
        'first_job': job_count > 0,
    }

    steps: list[Dict[str, Any]] = []
    for raw in GUIDED_PROFILE_STEPS:
        step_id = raw['id']
        done = step_id in completed_set or done_by_data.get(step_id, False)
        step = dict(raw)
        step['done'] = done
        step['required'] = step_id in REQUIRED_SETUP_STEP_IDS
        step['detail'] = details[step_id]
        if step_id == 'resume_import' and base_resume_exists:
            step['detail'] = 'Canonical resume is available and synced.'
        if step_id == 'first_job' and not llm_ready:
            step['detail'] = 'Set OPENAI_API_KEY before running your first generation.'
        steps.append(step)
    return steps


def _next_incomplete_step(steps: list[Dict[str, Any]]) -> str:
    for step in steps:
        if step.get('required') and not step.get('done'):
            return str(step.get('id'))
    for step in steps:
        if not step.get('done'):
            return str(step.get('id'))
    return 'first_job'


def _build_guided_dashboard_context(user: Optional[AuthUser]) -> Dict[str, Any]:
    profile = _load_or_create_profile(user)
    steps = _build_profile_steps(profile)
    required_complete = all(bool(step.get('done')) for step in steps if step.get('required'))
    ready_to_generate = required_complete and bool(llm_service and llm_service.available)
    recent_jobs = repository.list_jobs()[:5]
    jd_signal = _latest_jd_signal()
    return {
        'profile': profile,
        'steps': steps,
        'required_complete': required_complete,
        'ready_to_generate': ready_to_generate,
        'next_step_id': _next_incomplete_step(steps),
        'vault_count': len(repository.list_vault_items()),
        'job_count': len(repository.list_jobs()),
        'base_resume_exists': repository.load_base_resume() is not None,
        'recent_jobs': recent_jobs,
        'jd_signal': jd_signal,
    }


def _display_skill_term(term: str) -> str:
    normalized = normalize_token(term)
    if not normalized:
        return ''
    if normalized in TERM_DISPLAY_OVERRIDES:
        return TERM_DISPLAY_OVERRIDES[normalized]
    if '_' in normalized:
        return ' '.join(chunk.capitalize() for chunk in normalized.split('_') if chunk)
    if normalized in UPPERCASE_DISPLAY_TERMS:
        return normalized.upper()
    return normalized.capitalize()


def _derive_jd_signal(jd_text: str, *, fallback_title: str = '') -> Dict[str, Any]:
    cleaned_text = (jd_text or '').strip()
    if not cleaned_text:
        return {'role_title': '', 'skills': [], 'years_required': None}

    try:
        parsed = parse_job_description(cleaned_text)
    except Exception:
        return {'role_title': '', 'skills': [], 'years_required': None}

    role_title = ''
    normalized_title = str(parsed.get('normalized_title') or '').strip()
    if normalized_title:
        role_title = ' '.join(chunk.capitalize() for chunk in normalized_title.split('_') if chunk)
    elif fallback_title.strip():
        role_title = fallback_title.strip()

    skill_ids: list[str] = []
    for row in parsed.get('required', {}).get('skills', []):
        skill = str(row.get('canonical_id') or '').strip()
        if skill:
            skill_ids.append(skill)
    for row in parsed.get('preferred', {}).get('skills', []):
        skill = str(row.get('canonical_id') or '').strip()
        if skill:
            skill_ids.append(skill)

    seen: set[str] = set()
    display_skills: list[str] = []
    for skill in skill_ids:
        token = normalize_token(skill)
        if not token or token in seen:
            continue
        seen.add(token)
        display = _display_skill_term(skill)
        if display:
            display_skills.append(display)

    return {
        'role_title': role_title,
        'skills': display_skills[:8],
        'years_required': parsed.get('years_required'),
    }


def _latest_jd_signal() -> Dict[str, Any]:
    jobs = repository.list_jobs()
    if not jobs:
        return {'role_title': '', 'skills': [], 'years_required': None, 'job_id': ''}
    latest = jobs[0]
    jd_text = repository.get_job_text(latest.job_id)
    signal = _derive_jd_signal(jd_text, fallback_title=latest.title or '')
    signal['job_id'] = latest.job_id
    return signal


def _update_profile_focus_from_jd(user_id: str, *, jd_text: str, fallback_title: str = '') -> None:
    profile = auth_store.get_profile(user_id)
    if profile is None:
        return
    signal = _derive_jd_signal(jd_text, fallback_title=fallback_title)
    role_title = str(signal.get('role_title') or '').strip()
    skills = [str(skill).strip() for skill in signal.get('skills', []) if str(skill).strip()]

    profile.target_roles = [role_title] if role_title else []
    if role_title and skills:
        profile.headline = f'{role_title} profile aligned to {", ".join(skills[:3])}.'
    elif role_title:
        profile.headline = f'{role_title} profile aligned to the current job description.'
    elif skills:
        profile.headline = f'Profile aligned to {", ".join(skills[:3])}.'
    else:
        profile.headline = ''

    auth_store.upsert_profile(profile)


def _step_lookup(step_id: str) -> Optional[Dict[str, str]]:
    normalized = str(step_id or '').strip().lower()
    for step in GUIDED_PROFILE_STEPS:
        if step['id'] == normalized:
            return step
    return None


def _build_profile_step_context(
    *,
    profile: Optional[UserProfile],
    step_id: str,
    error: Optional[str] = None,
    warnings: Optional[list[str]] = None,
    raw_text: str = '',
    job_url: str = '',
    jd_text: str = '',
) -> Dict[str, Any]:
    step = _step_lookup(step_id)
    if step is None:
        raise HTTPException(status_code=404, detail='Profile step not found')
    steps = _build_profile_steps(profile)
    return {
        'profile': profile,
        'steps': steps,
        'step': step,
        'error': error,
        'warnings': warnings or [],
        'raw_text': raw_text,
        'job_url': job_url,
        'jd_text': jd_text,
        'next_step_id': _next_incomplete_step(steps),
        'jd_signal': _latest_jd_signal(),
    }


def _safe_index_call(action: str, func) -> None:
    try:
        func()
    except Exception as exc:
        logger.warning('Index update skipped (%s): %s', action, exc)


def _refresh_vault_index(user_id: Optional[str]) -> None:
    if not user_id:
        return
    vault_dir = repository.vault_dir_for(user_id)
    for item_id, item in repository.list_vault_items(user_id=user_id):
        item_path = ensure_within(vault_dir, vault_dir / f'{item_id}.yaml')
        _safe_index_call(
            f'vault_item:{item_id}',
            lambda: auth_store.upsert_vault_item(
                user_id=user_id,
                item_id=item_id,
                title=item.title,
                item_type=item.type.value,
                path=str(item_path),
            ),
        )


def _refresh_user_indexes(user_id: Optional[str]) -> None:
    if not user_id:
        return
    resume_path = repository.base_resume_path_for(user_id)
    if resume_path.exists():
        _safe_index_call(
            'base_resume',
            lambda: auth_store.upsert_base_resume(user_id=user_id, path=str(resume_path)),
        )

    _refresh_vault_index(user_id)

    jobs_dir = repository.jobs_dir_for(user_id)
    for job in repository.list_jobs(user_id=user_id):
        job_path = ensure_within(jobs_dir, jobs_dir / job.job_id / 'job.yaml')
        _safe_index_call(
            f'job:{job.job_id}',
            lambda: auth_store.upsert_job(
                user_id=user_id,
                job_id=job.job_id,
                title=job.title,
                company=job.company,
                url=job.url,
                path=str(job_path),
            ),
        )

    outputs_dir = repository.outputs_dir_for(user_id)
    if outputs_dir.exists():
        for job_dir in sorted(path for path in outputs_dir.glob('*') if path.is_dir()):
            for run_dir in sorted(path for path in job_dir.glob('*') if path.is_dir()):
                _safe_index_call(
                    f'output:{job_dir.name}:{run_dir.name}',
                    lambda: auth_store.upsert_output(
                        user_id=user_id,
                        job_id=job_dir.name,
                        timestamp=run_dir.name,
                        path=str(run_dir),
                    ),
                )


@app.get('/auth/login', response_class=HTMLResponse)
async def auth_login_page(request: Request, next: str = '') -> HTMLResponse:
    if _current_user(request):
        return RedirectResponse(url='/', status_code=303)
    return render(
        request,
        'login.html',
        {
            'error': None,
            'email': '',
            'next_path': _safe_next_path(next),
        },
    )


@app.post('/auth/login', response_class=HTMLResponse)
async def auth_login(
    request: Request,
    email: str = Form(default=''),
    password: str = Form(default=''),
    next: str = Form(default=''),
):
    normalized_email = (email or '').strip().lower()
    next_path = _safe_next_path(next)
    try:
        user = auth_store.verify_credentials(email=normalized_email, password=password or '')
    except Exception as exc:
        logger.exception('Auth verification failed during login: %s', exc)
        return render(
            request,
            'login.html',
            {
                'error': 'Login is temporarily unavailable. Check server storage and retry.',
                'email': normalized_email,
                'next_path': next_path,
            },
        )
    if user is None:
        return render(
            request,
            'login.html',
            {
                'error': 'Invalid email or password.',
                'email': normalized_email,
                'next_path': next_path,
            },
        )

    try:
        auth_store.update_last_login(user.id)
    except Exception as exc:
        logger.warning('Failed to update last_login for user %s: %s', user.id, exc)
    if repository.has_legacy_data() and auth_store.count_users() == 1:
        _safe_index_call(f'migrate_legacy:{user.id}', lambda: repository.migrate_legacy_data_to_user(user.id))
    _safe_index_call(f'refresh_indexes:{user.id}', lambda: _refresh_user_indexes(user.id))
    _safe_index_call(
        f'profile_sync:{user.id}',
        lambda: _sync_profile_progress(
            user.id,
            profile=auth_store.ensure_profile_for_user(
                user=user,
                seed_resume=repository.load_base_resume(user_id=user.id),
            ),
        ),
    )

    response = RedirectResponse(url=next_path or '/', status_code=303)
    response.set_cookie(key=settings.session_cookie_name, value=session_manager.issue(user.id), **_session_cookie_kwargs())
    return response


@app.get('/auth/register', response_class=HTMLResponse)
async def auth_register_page(request: Request, next: str = '') -> HTMLResponse:
    next_path = _safe_next_path(next)
    if _current_user(request):
        return RedirectResponse(url='/', status_code=303)
    return render(
        request,
        'register.html',
        {
            'error': None,
            'email': '',
            'next_path': next_path,
        },
    )


@app.post('/auth/register', response_class=HTMLResponse)
async def auth_register(
    request: Request,
    email: str = Form(default=''),
    password: str = Form(default=''),
    confirm_password: str = Form(default=''),
    next: str = Form(default=''),
):
    normalized_email = (email or '').strip().lower()
    next_path = _safe_next_path(next)
    if (password or '') != (confirm_password or ''):
        return render(
            request,
            'register.html',
            {
                'error': 'Passwords do not match.',
                'email': normalized_email,
                'next_path': next_path,
            },
        )

    try:
        user = auth_store.create_user(email=normalized_email, password=password or '')
    except ValueError as exc:
        return render(
            request,
            'register.html',
            {
                'error': str(exc),
                'email': normalized_email,
                'next_path': next_path,
            },
        )
    except Exception as exc:
        logger.exception('Auth user creation failed: %s', exc)
        return render(
            request,
            'register.html',
            {
                'error': 'Account creation is temporarily unavailable. Check server storage and retry.',
                'email': normalized_email,
                'next_path': next_path,
            },
        )

    if repository.has_legacy_data() and auth_store.count_users() == 1:
        _safe_index_call(f'migrate_legacy:{user.id}', lambda: repository.migrate_legacy_data_to_user(user.id))
    _safe_index_call(f'refresh_indexes:{user.id}', lambda: _refresh_user_indexes(user.id))
    _safe_index_call(
        f'profile_sync:{user.id}',
        lambda: _sync_profile_progress(
            user.id,
            profile=auth_store.ensure_profile_for_user(
                user=user,
                seed_resume=repository.load_base_resume(user_id=user.id),
            ),
        ),
    )

    response = RedirectResponse(url=next_path or '/', status_code=303)
    response.set_cookie(key=settings.session_cookie_name, value=session_manager.issue(user.id), **_session_cookie_kwargs())
    return response


@app.post('/auth/logout')
async def auth_logout():
    response = RedirectResponse(url='/auth/login', status_code=303)
    response.delete_cookie(settings.session_cookie_name, path='/')
    return response


def _generate_page_context(*, jd_text: str = '', error: Optional[str] = None, warnings: Optional[list[str]] = None) -> Dict[str, Any]:
    setup_checklist = _build_setup_checklist()
    return {
        'jd_text': jd_text,
        'error': error,
        'warnings': warnings or [],
        'base_resume_exists': bool(setup_checklist['items'][0]['done']),
        'vault_count': len(repository.list_vault_items()),
        'setup_checklist': setup_checklist,
    }


@app.get('/', response_class=HTMLResponse)
async def generate_page(request: Request) -> HTMLResponse:
    if settings.enable_profile_rewrite:
        return render(request, 'home_guided.html', _build_guided_dashboard_context(_current_user(request)))
    return render(request, 'generate.html', _generate_page_context())


@app.post('/generate', response_class=HTMLResponse)
async def generate_tailored_resume(
    request: Request,
    resume_file: Optional[UploadFile] = File(default=None),
    jd_text: str = Form(default=''),
) -> HTMLResponse:
    pasted_jd_text = (jd_text or '').strip()
    if not pasted_jd_text:
        return render(
            request,
            'generate.html',
            _generate_page_context(jd_text=pasted_jd_text, error='Paste a job description before generating.'),
        )

    if not llm_service or not llm_service.available:
        return render(
            request,
            'generate.html',
            _generate_page_context(
                jd_text=pasted_jd_text,
                error='OPENAI_API_KEY is not configured. Tailoring is disabled until set.',
            ),
        )

    base_resume = repository.load_base_resume()
    extraction_warnings: list[str] = []
    has_resume_upload = bool(resume_file and resume_file.filename)
    if has_resume_upload:
        assert resume_file is not None
        suffix = Path(resume_file.filename).suffix.lower()
        if suffix not in {'.pdf', '.docx', '.txt'}:
            return render(
                request,
                'generate.html',
                _generate_page_context(
                    jd_text=pasted_jd_text,
                    error='Supported resume file types: PDF, DOCX, TXT.',
                ),
            )

        upload_path = settings.data_dir / 'uploads' / f'{uuid.uuid4().hex}{suffix}'
        content = await resume_file.read()
        if len(content) > _max_upload_bytes():
            return render(
                request,
                'generate.html',
                _generate_page_context(
                    jd_text=pasted_jd_text,
                    error=_upload_limit_message(),
                ),
            )
        upload_path.write_bytes(content)

        try:
            upload_result = public_upload_resume(upload_path, enable_ocr=settings.enable_ocr, llm=llm_service)
            canonical_payload = upload_result.get('canonical') or upload_result.get('parse_mirror', {}).get('canonical')
            canonical = CanonicalResume.model_validate(canonical_payload)
            extraction_warnings = [str(w).strip() for w in upload_result.get('extraction_warnings', []) if str(w).strip()]
        except Exception as exc:
            return render(
                request,
                'generate.html',
                _generate_page_context(
                    jd_text=pasted_jd_text,
                    error=_with_request_id(request, f'Resume processing failed: {exc}'),
                ),
            )
        finally:
            try:
                upload_path.unlink(missing_ok=True)
            except Exception:
                pass

        try:
            repository.save_base_resume(canonical)
            sync_base_resume_to_vault(repository, canonical)
            active_user_id = _active_user_id()
            if active_user_id:
                auth_store.upsert_base_resume(
                    user_id=active_user_id,
                    path=str(repository.base_resume_path_for(active_user_id)),
                )
                _refresh_vault_index(active_user_id)
            base_resume = canonical
        except Exception as exc:
            return render(
                request,
                'generate.html',
                _generate_page_context(
                    jd_text=pasted_jd_text,
                    error=_with_request_id(request, f'Failed to save base resume: {exc}'),
                ),
            )

    if not base_resume:
        return render(
            request,
            'generate.html',
            _generate_page_context(
                jd_text=pasted_jd_text,
                error='No base profile loaded. Upload once in Advanced > Resume Upload, then generate from vault only.',
            ),
        )

    vault_items = repository.list_vault_items()
    if not vault_items:
        return render(
            request,
            'generate.html',
            _generate_page_context(
                jd_text=pasted_jd_text,
                error='Vault is empty. Add vault evidence items before generating.',
            ),
        )

    job_id = uuid.uuid4().hex[:12]
    job = JobRecord(job_id=job_id, title='Pasted Job Description', company=None, url=None)
    repository.save_job(job, pasted_jd_text)
    active_user_id = _active_user_id()
    if active_user_id:
        jobs_dir = repository.jobs_dir_for(active_user_id)
        job_path = ensure_within(jobs_dir, jobs_dir / job_id / 'job.yaml')
        auth_store.upsert_job(
            user_id=active_user_id,
            job_id=job.job_id,
            title=job.title,
            company=job.company,
            url=job.url,
            path=str(job_path),
        )
        _update_profile_focus_from_jd(active_user_id, jd_text=pasted_jd_text, fallback_title=job.title or '')
        auth_store.mark_onboarding_step(user_id=active_user_id, step='first_job')
        _sync_profile_progress(active_user_id)

    workflow = _run_tailoring_workflow(
        base_resume=base_resume,
        job_id=job_id,
        jd_text=pasted_jd_text,
        mode=DEFAULT_TAILOR_MODE,
        job_title_hint=job.title,
    )

    return render(
        request,
        'tailor_result.html',
        _tailor_result_context(job=job, mode=DEFAULT_TAILOR_MODE, workflow=workflow, prepended_warnings=extraction_warnings),
    )


@app.get('/advance', response_class=HTMLResponse)
async def advance_page(request: Request) -> HTMLResponse:
    base_resume = repository.load_base_resume()
    vault_items = repository.list_vault_items()
    jobs = repository.list_jobs()
    setup_checklist = _build_setup_checklist()
    return render(
        request,
        'advanced.html',
        {
            'base_resume_exists': base_resume is not None,
            'vault_count': len(vault_items),
            'job_count': len(jobs),
            'setup_checklist': setup_checklist,
        },
    )


@app.get('/advanced')
async def advanced_redirect() -> RedirectResponse:
    return RedirectResponse(url='/advance', status_code=307)


@app.get('/advance/dashboard')
async def legacy_dashboard_redirect() -> RedirectResponse:
    return RedirectResponse(url='/advance', status_code=303)


@app.get('/advanced/dashboard')
async def advanced_dashboard_redirect() -> RedirectResponse:
    return RedirectResponse(url='/advance', status_code=303)


@app.get('/profile', response_class=HTMLResponse)
async def profile_overview_page(request: Request) -> HTMLResponse:
    if not settings.enable_profile_rewrite:
        return RedirectResponse(url='/resume/upload', status_code=303)
    profile = _load_or_create_profile(_current_user(request))
    steps = _build_profile_steps(profile)
    required_complete = all(bool(step.get('done')) for step in steps if step.get('required'))
    return render(
        request,
        'profile_overview.html',
        {
            'profile': profile,
            'steps': steps,
            'required_complete': required_complete,
            'next_step_id': _next_incomplete_step(steps),
            'base_resume_exists': repository.load_base_resume() is not None,
            'vault_count': len(repository.list_vault_items()),
            'job_count': len(repository.list_jobs()),
            'llm_available': bool(llm_service and llm_service.available),
            'jd_signal': _latest_jd_signal(),
        },
    )


@app.get('/profile/step/{step_id}', response_class=HTMLResponse)
async def profile_step_page(request: Request, step_id: str) -> HTMLResponse:
    if not settings.enable_profile_rewrite:
        return RedirectResponse(url='/resume/upload', status_code=303)
    profile = _load_or_create_profile(_current_user(request))
    return render(
        request,
        'profile_step.html',
        _build_profile_step_context(profile=profile, step_id=step_id),
    )


@app.post('/profile/step/{step_id}', response_class=HTMLResponse)
async def profile_step_submit(request: Request, step_id: str) -> HTMLResponse:
    if not settings.enable_profile_rewrite:
        return RedirectResponse(url='/resume/upload', status_code=303)
    step = _step_lookup(step_id)
    if step is None:
        raise HTTPException(status_code=404, detail='Profile step not found')

    user = _current_user(request)
    profile = _load_or_create_profile(user)
    if user is None or profile is None:
        raise HTTPException(status_code=401, detail='Authentication required.')

    form = await request.form()

    if step['id'] == 'basics':
        display_name = _to_text(form.get('display_name'))
        email = _to_text(form.get('email'))
        if not display_name or not email:
            return render(
                request,
                'profile_step.html',
                _build_profile_step_context(
                    profile=profile,
                    step_id=step_id,
                    error='Name and email are required.',
                ),
            )
        profile.display_name = display_name
        profile.email = email
        profile.phone = _to_text(form.get('phone'))
        profile.location = _to_text(form.get('location'))
        profile.years_experience = _to_text(form.get('years_experience'))
        profile.links = _split_multivalue(form.get('links_text'))
        # Role focus and headline are derived from JD, never user-entered.
        profile.headline = ''
        profile.target_roles = []
        profile = _sync_profile_progress(user.id, profile=profile) or profile
        auth_store.mark_onboarding_step(user_id=user.id, step='basics')
        return RedirectResponse(url='/profile?flash=profile_saved', status_code=303)

    if step['id'] == 'resume_import':
        upload = form.get('file')
        filename = str(getattr(upload, 'filename', '') or '').strip()
        if not filename:
            return render(
                request,
                'profile_step.html',
                _build_profile_step_context(
                    profile=profile,
                    step_id=step_id,
                    error='Upload a resume file before continuing.',
                ),
            )
        suffix = Path(filename).suffix.lower()
        if suffix not in {'.pdf', '.docx', '.txt'}:
            return render(
                request,
                'profile_step.html',
                _build_profile_step_context(
                    profile=profile,
                    step_id=step_id,
                    error='Supported resume file types: PDF, DOCX, TXT.',
                ),
            )

        upload_path = settings.data_dir / 'uploads' / f'{uuid.uuid4().hex}{suffix}'
        warnings: list[str] = []
        content = await upload.read()
        if len(content) > _max_upload_bytes():
            return render(
                request,
                'profile_step.html',
                _build_profile_step_context(
                    profile=profile,
                    step_id=step_id,
                    error=_upload_limit_message(),
                ),
            )
        upload_path.write_bytes(content)

        try:
            upload_result = public_upload_resume(upload_path, enable_ocr=settings.enable_ocr, llm=llm_service)
            canonical_payload = upload_result.get('canonical') or upload_result.get('parse_mirror', {}).get('canonical')
            canonical = CanonicalResume.model_validate(canonical_payload)
            warnings = [str(w).strip() for w in upload_result.get('extraction_warnings', []) if str(w).strip()]
            repository.save_base_resume(canonical)
            sync_base_resume_to_vault(repository, canonical)
            auth_store.upsert_base_resume(user_id=user.id, path=str(repository.base_resume_path_for(user.id)))
            _refresh_vault_index(user.id)
            profile.display_name = canonical.identity.name or profile.display_name
            profile.email = canonical.identity.email or profile.email
            profile.phone = canonical.identity.phone or profile.phone
            profile.location = canonical.identity.location or profile.location
            profile.links = list(canonical.identity.links or profile.links)
            auth_store.upsert_profile(profile)
            auth_store.mark_onboarding_step(user_id=user.id, step='resume_import')
            _sync_profile_progress(user.id)
        except Exception as exc:
            return render(
                request,
                'profile_step.html',
                _build_profile_step_context(
                    profile=profile,
                    step_id=step_id,
                    error=_with_request_id(request, f'Resume import failed: {exc}'),
                ),
            )
        finally:
            try:
                upload_path.unlink(missing_ok=True)
            except Exception:
                pass

        if warnings:
            return render(
                request,
                'profile_step.html',
                _build_profile_step_context(
                    profile=auth_store.get_profile(user.id),
                    step_id=step_id,
                    warnings=warnings,
                ),
            )
        return RedirectResponse(url='/profile?flash=profile_resume_imported', status_code=303)

    if step['id'] == 'evidence':
        title = _to_text(form.get('title'))
        item_type = _to_text(form.get('item_type') or 'project')
        bullet_text = _to_text(form.get('bullet_text'))
        if not title:
            return render(
                request,
                'profile_step.html',
                _build_profile_step_context(
                    profile=profile,
                    step_id=step_id,
                    error='Evidence title is required.',
                ),
            )
        payload: Dict[str, Any] = {
            'type': item_type,
            'title': title,
            'tags': _split_multivalue(form.get('tags_text')),
            'tech': _split_multivalue(form.get('tech_text')),
            'bullets': [{'text': bullet_text}] if bullet_text else [],
            'links': [],
            'source_artifacts': [],
        }
        try:
            item = VaultItem.model_validate(payload)
            item_id = uuid.uuid4().hex
            repository.save_vault_item(item_id, item)
            vault_dir = repository.vault_dir_for(user.id)
            item_path = ensure_within(vault_dir, vault_dir / f'{item_id}.yaml')
            auth_store.upsert_vault_item(
                user_id=user.id,
                item_id=item_id,
                title=item.title,
                item_type=item.type.value,
                path=str(item_path),
            )
            auth_store.mark_onboarding_step(user_id=user.id, step='evidence')
            _sync_profile_progress(user.id)
        except ValidationError as exc:
            return render(
                request,
                'profile_step.html',
                _build_profile_step_context(
                    profile=profile,
                    step_id=step_id,
                    error='Evidence validation failed: ' + '; '.join(_format_validation_errors(exc)),
                ),
            )
        except Exception as exc:
            return render(
                request,
                'profile_step.html',
                _build_profile_step_context(
                    profile=profile,
                    step_id=step_id,
                    error=_with_request_id(request, f'Failed to save evidence: {exc}'),
                ),
            )
        return RedirectResponse(url='/profile?flash=vault_saved', status_code=303)

    if step['id'] == 'first_job':
        url = _to_text(form.get('url'))
        jd_text = _to_text(form.get('jd_text'))
        if not url and not jd_text:
            return render(
                request,
                'profile_step.html',
                _build_profile_step_context(
                    profile=profile,
                    step_id=step_id,
                    error='Provide a URL or paste job description text.',
                    job_url=url,
                    jd_text=jd_text,
                ),
            )
        title: Optional[str] = None
        company: Optional[str] = None
        if not jd_text and url:
            try:
                scrape = await scrape_job_posting(url)
                title = scrape.title
                company = scrape.company
                jd_text = scrape.jd_text
            except ScrapeError as exc:
                return render(
                    request,
                    'profile_step.html',
                    _build_profile_step_context(
                        profile=profile,
                        step_id=step_id,
                        error=f'Scraping failed: {exc}. Paste JD text as fallback.',
                        job_url=url,
                        jd_text='',
                    ),
                )
        if not jd_text:
            return render(
                request,
                'profile_step.html',
                _build_profile_step_context(
                    profile=profile,
                    step_id=step_id,
                    error='Job description text is empty.',
                    job_url=url,
                    jd_text='',
                ),
            )
        job_id = uuid.uuid4().hex[:12]
        job = JobRecord(job_id=job_id, url=url or None, title=title, company=company)
        repository.save_job(job, jd_text)
        jobs_dir = repository.jobs_dir_for(user.id)
        job_path = ensure_within(jobs_dir, jobs_dir / job_id / 'job.yaml')
        auth_store.upsert_job(
            user_id=user.id,
            job_id=job.job_id,
            title=job.title,
            company=job.company,
            url=job.url,
            path=str(job_path),
        )
        _update_profile_focus_from_jd(user.id, jd_text=jd_text, fallback_title=job.title or '')
        auth_store.mark_onboarding_step(user_id=user.id, step='first_job')
        _sync_profile_progress(user.id)
        return RedirectResponse(url=f'/jobs/{job_id}?flash=job_ingested', status_code=303)

    raise HTTPException(status_code=400, detail='Unsupported profile step')


@app.get('/audit', response_class=HTMLResponse)
async def audit_page(request: Request) -> HTMLResponse:
    return render(
        request,
        'audit.html',
        {
            'jobs': repository.list_jobs(),
            'selected_job_id': '',
            'jd_text': '',
            'run_render_outputs': True,
            'error': None,
            'result': None,
        },
    )


@app.post('/audit/run', response_class=HTMLResponse)
async def audit_run(
    request: Request,
    resume_file: UploadFile = File(...),
    jd_text: str = Form(default=''),
    job_id: str = Form(default=''),
    render_outputs: str = Form(default=''),
) -> HTMLResponse:
    jobs = repository.list_jobs()
    selected_job_id = (job_id or '').strip()
    pasted_jd_text = (jd_text or '').strip()
    run_render_outputs = str(render_outputs or '').strip().lower() in {'1', 'true', 'yes', 'on'}

    if not resume_file.filename:
        return render(
            request,
            'audit.html',
            {
                'jobs': jobs,
                'selected_job_id': selected_job_id,
                'jd_text': pasted_jd_text,
                'run_render_outputs': run_render_outputs,
                'error': 'Resume file is required.',
                'result': None,
            },
        )

    suffix = Path(resume_file.filename).suffix.lower()
    if suffix not in {'.pdf', '.docx', '.txt'}:
        return render(
            request,
            'audit.html',
            {
                'jobs': jobs,
                'selected_job_id': selected_job_id,
                'jd_text': pasted_jd_text,
                'run_render_outputs': run_render_outputs,
                'error': 'Supported file types: PDF, DOCX, TXT.',
                'result': None,
            },
        )

    upload_path = settings.data_dir / 'uploads' / f'{uuid.uuid4().hex}{suffix}'
    content = await resume_file.read()
    if len(content) > _max_upload_bytes():
        return render(
            request,
            'audit.html',
            {
                'jobs': jobs,
                'selected_job_id': selected_job_id,
                'jd_text': pasted_jd_text,
                'run_render_outputs': run_render_outputs,
                'error': _upload_limit_message(),
                'result': None,
            },
        )
    upload_path.write_bytes(content)

    try:
        upload_result = public_upload_resume(upload_path, enable_ocr=settings.enable_ocr, llm=llm_service)
        canonical_payload = upload_result.get('canonical') or upload_result.get('parse_mirror', {}).get('canonical')
        canonical = CanonicalResume.model_validate(canonical_payload)
    except Exception as exc:
        return render(
            request,
            'audit.html',
            {
                'jobs': jobs,
                'selected_job_id': selected_job_id,
                'jd_text': pasted_jd_text,
                'run_render_outputs': run_render_outputs,
                'error': _with_request_id(request, f'Resume audit failed: {exc}'),
                'result': None,
            },
        )
    finally:
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            pass

    resolved_jd_text = pasted_jd_text
    jd_source = 'pasted'
    if not resolved_jd_text and selected_job_id:
        selected_job = repository.get_job(selected_job_id)
        if not selected_job:
            return render(
                request,
                'audit.html',
                {
                    'jobs': jobs,
                    'selected_job_id': selected_job_id,
                    'jd_text': pasted_jd_text,
                    'run_render_outputs': run_render_outputs,
                    'error': f'Job not found: {selected_job_id}',
                    'result': None,
                },
            )
        resolved_jd_text = repository.get_job_text(selected_job_id)
        jd_source = f'job:{selected_job_id}'

    parsed_job = None
    requirement_graph = None
    match_payload = None
    patch_payload = None
    if resolved_jd_text:
        parsed_job = parse_job_description(resolved_jd_text)
        requirement_graph = build_requirement_graph(parsed_job)
        match_payload = public_score_match(canonical, resolved_jd_text)
        patch_payload = public_generate_patches(canonical, resolved_jd_text)

    render_artifacts = None
    if run_render_outputs:
        output_job_id = normalize_token(selected_job_id or 'audit') or 'audit'
        outputs_dir = repository.outputs_dir_for()
        output_root = ensure_within(outputs_dir, outputs_dir / output_job_id)
        output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime_now_stamp()
        output_dir = output_root / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            ats_result = public_render_outputs(canonical, output_dir, filename_prefix='ats_')
            bundle_path = public_export_bundle(output_dir, bundle_path=output_dir / 'bundle.zip')
            render_artifacts = {
                'job_id': output_job_id,
                'timestamp': timestamp,
                'pdf_text_layer': ats_result.get('pdf_text_layer', {}),
                'ats_pdf_exists': Path(ats_result['pdf_path']).exists(),
                'ats_docx_exists': Path(ats_result['docx_path']).exists(),
                'ats_txt_exists': Path(ats_result['txt_path']).exists(),
                'bundle_exists': bundle_path.exists(),
            }
        except Exception as exc:
            render_artifacts = {'error': str(exc)}

    canonical_counts = {
        'education': len(canonical.education),
        'experience': len(canonical.experience),
        'projects': len(canonical.projects),
        'skill_categories': len(canonical.skills.categories),
        'certifications': len(canonical.certifications),
    }

    result = {
        'parse_quality': upload_result.get('parse_mirror', {}).get('quality', {}),
        'lint': upload_result.get('lint', {}),
        'contact_validation': upload_result.get('contact_validation', {}),
        'sensitive_data': upload_result.get('sensitive_data', {}),
        'timeline_overlaps': upload_result.get('timeline_overlaps', {}),
        'timeline_durations': upload_result.get('timeline_durations', {}),
        'extraction_warnings': upload_result.get('extraction_warnings', []),
        'raw_text_preview': str(upload_result.get('raw_text', ''))[:3000],
        'canonical_counts': canonical_counts,
        'jd_source': jd_source if resolved_jd_text else None,
        'jd_text': resolved_jd_text,
        'parsed_job': parsed_job,
        'requirement_graph': requirement_graph,
        'match': match_payload,
        'patches': patch_payload,
        'render_artifacts': render_artifacts,
    }

    return render(
        request,
        'audit.html',
        {
            'jobs': jobs,
            'selected_job_id': selected_job_id,
            'jd_text': pasted_jd_text,
            'run_render_outputs': run_render_outputs,
            'error': None,
            'result': result,
        },
    )


@app.get('/resume/upload', response_class=HTMLResponse)
async def resume_upload_page(request: Request) -> HTMLResponse:
    base_resume = repository.load_base_resume()
    resume_form = _resume_to_form_payload(base_resume) if base_resume else None
    return render(
        request,
        'resume_upload.html',
        {
            'warnings': [],
            'error': None,
            'field_errors': [],
            'raw_text': '',
            'resume_form': resume_form,
        },
    )


@app.post('/resume/upload', response_class=HTMLResponse)
async def resume_upload_extract(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    warnings = []
    if not file.filename:
        return render(
            request,
            'resume_upload.html',
            {'warnings': warnings, 'error': 'No file provided.', 'field_errors': [], 'raw_text': '', 'resume_form': None},
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {'.pdf', '.docx'}:
        return render(
            request,
            'resume_upload.html',
            {
                'warnings': warnings,
                'error': 'Unsupported file type. Upload PDF or DOCX.',
                'field_errors': [],
                'raw_text': '',
                'resume_form': None,
            },
        )

    upload_path = settings.data_dir / 'uploads' / f'{uuid.uuid4().hex}{suffix}'
    content = await file.read()
    if len(content) > _max_upload_bytes():
        return render(
            request,
            'resume_upload.html',
            {
                'warnings': warnings,
                'error': _upload_limit_message(),
                'field_errors': [],
                'raw_text': '',
                'resume_form': None,
            },
        )
    upload_path.write_bytes(content)

    try:
        raw_text, extract_warnings = extract_resume_text(upload_path, enable_ocr=settings.enable_ocr)
        warnings.extend(extract_warnings)
        canonical, canonical_warnings = canonicalize_resume_text(raw_text, llm_service)
        warnings.extend(canonical_warnings)
        resume_form = _resume_to_form_payload(canonical)
    except Exception as exc:
        return render(
            request,
            'resume_upload.html',
            {
                'warnings': warnings,
                'error': _with_request_id(request, f'Extraction failed: {exc}'),
                'field_errors': [],
                'raw_text': '',
                'resume_form': None,
            },
        )
    finally:
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            pass

    return render(
        request,
        'resume_upload.html',
        {'warnings': warnings, 'error': None, 'field_errors': [], 'raw_text': raw_text, 'resume_form': resume_form},
    )


@app.post('/resume/save')
async def resume_save(request: Request, canonical_yaml: Optional[str] = Form(default=None)):
    source_payload: Optional[Dict[str, Any]] = None
    try:
        if canonical_yaml and canonical_yaml.strip():
            source_payload = yaml.safe_load(canonical_yaml)
            if not isinstance(source_payload, dict):
                raise ValueError('YAML root must be an object.')
        else:
            form = await request.form()
            source_payload = _parse_resume_form_payload(form)
        canonical = CanonicalResume.model_validate(source_payload)
    except ValidationError as exc:
        return render(
            request,
            'resume_upload.html',
            {
                'warnings': [],
                'error': 'Resume validation failed. Fix the fields below and resubmit.',
                'field_errors': _format_validation_errors(exc),
                'raw_text': '',
                'resume_form': _resume_to_form_payload(source_payload),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid resume payload: {exc}') from exc

    try:
        repository.save_base_resume(canonical)
        sync_base_resume_to_vault(repository, canonical)
        active_user_id = _active_user_id()
        if active_user_id:
            auth_store.upsert_base_resume(
                user_id=active_user_id,
                path=str(repository.base_resume_path_for(active_user_id)),
            )
            _refresh_vault_index(active_user_id)
            existing_user = auth_store.get_user_by_id(active_user_id)
            if existing_user:
                profile = auth_store.ensure_profile_for_user(user=existing_user, seed_resume=canonical)
                profile.display_name = canonical.identity.name or profile.display_name
                profile.email = canonical.identity.email or profile.email
                profile.phone = canonical.identity.phone or profile.phone
                profile.location = canonical.identity.location or profile.location
                profile.links = list(canonical.identity.links or profile.links)
                auth_store.upsert_profile(profile)
                auth_store.mark_onboarding_step(user_id=active_user_id, step='resume_import')
                _sync_profile_progress(active_user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Failed to save/sync resume: {exc}') from exc

    return RedirectResponse(url='/?flash=resume_saved', status_code=303)


@app.post('/resume/sync-vault')
async def resume_sync_vault():
    base_resume = repository.load_base_resume()
    if not base_resume:
        raise HTTPException(status_code=400, detail='Base resume missing. Save base resume first.')
    try:
        sync_base_resume_to_vault(repository, base_resume)
        _refresh_vault_index(_active_user_id())
        active_user_id = _active_user_id()
        if active_user_id:
            auth_store.mark_onboarding_step(user_id=active_user_id, step='resume_import')
            _sync_profile_progress(active_user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Vault sync failed: {exc}') from exc
    return RedirectResponse(url='/vault?flash=vault_synced', status_code=303)


@app.get('/vault', response_class=HTMLResponse)
async def vault_list(request: Request) -> HTMLResponse:
    items = repository.list_vault_items()
    return render(request, 'vault_list.html', {'items': items})


@app.get('/vault/new', response_class=HTMLResponse)
async def vault_new_page(request: Request) -> HTMLResponse:
    return render(
        request,
        'vault_form.html',
        {'item_id': None, 'item_form': _default_vault_item_payload(), 'error': None, 'field_errors': []},
    )


@app.get('/vault/ingest', response_class=HTMLResponse)
async def vault_ingest_page(request: Request) -> HTMLResponse:
    return render(
        request,
        'vault_ingest.html',
        {
            'error': None,
            'warnings': [],
            'source_text': '',
            'type_hint': 'project',
            'parsed_item': None,
            'raw_text_preview': '',
        },
    )


@app.post('/vault/ingest/parse', response_class=HTMLResponse)
async def vault_ingest_parse(
    request: Request,
    source_text: str = Form(default=''),
    type_hint: str = Form(default='project'),
    file: Optional[UploadFile] = File(default=None),
) -> HTMLResponse:
    warnings = []
    parsed_item: Optional[Dict[str, Any]] = None
    raw_segments = []
    raw_text_preview = ''
    upload_path: Optional[Path] = None

    source_text = source_text.strip()
    if source_text:
        raw_segments.append(source_text)

    if file and file.filename:
        suffix = Path(file.filename).suffix.lower()
        upload_path = settings.data_dir / 'uploads' / f'{uuid.uuid4().hex}{suffix}'
        content = await file.read()
        if len(content) > _max_upload_bytes():
            return render(
                request,
                'vault_ingest.html',
                {
                    'error': _upload_limit_message(),
                    'warnings': warnings,
                    'source_text': source_text,
                    'type_hint': type_hint,
                    'parsed_item': parsed_item,
                    'raw_text_preview': raw_text_preview,
                },
            )
        upload_path.write_bytes(content)
        try:
            file_text, file_warnings = parse_uploaded_text(upload_path, enable_ocr=settings.enable_ocr)
            warnings.extend(file_warnings)
            if file_text:
                raw_segments.append(file_text)
        except VaultIngestError as exc:
            try:
                upload_path.unlink(missing_ok=True)
            except Exception:
                pass
            return render(
                request,
                'vault_ingest.html',
                {
                    'error': str(exc),
                    'warnings': warnings,
                    'source_text': source_text,
                    'type_hint': type_hint,
                    'parsed_item': parsed_item,
                    'raw_text_preview': raw_text_preview,
                },
            )

    if not raw_segments:
        if upload_path:
            try:
                upload_path.unlink(missing_ok=True)
            except Exception:
                pass
        return render(
            request,
            'vault_ingest.html',
            {
                'error': 'Provide notes/text or upload a file to parse.',
                'warnings': warnings,
                'source_text': source_text,
                'type_hint': type_hint,
                'parsed_item': parsed_item,
                'raw_text_preview': raw_text_preview,
            },
        )

    combined_text = '\n\n'.join(raw_segments).strip()
    raw_text_preview = combined_text[:6000]

    try:
        item, parse_warnings = parse_vault_source_text(combined_text, llm=llm_service, type_hint=type_hint)
        warnings.extend(parse_warnings)
        parsed_item = _vault_item_to_form_payload(item)
    except Exception as exc:
        return render(
            request,
            'vault_ingest.html',
            {
                'error': _with_request_id(request, f'Vault parsing failed: {exc}'),
                'warnings': warnings,
                'source_text': source_text,
                'type_hint': type_hint,
                'parsed_item': parsed_item,
                'raw_text_preview': raw_text_preview,
            },
        )
    finally:
        if upload_path:
            try:
                upload_path.unlink(missing_ok=True)
            except Exception:
                pass

    return render(
        request,
        'vault_ingest.html',
        {
            'error': None,
            'warnings': warnings,
            'source_text': source_text,
            'type_hint': type_hint,
            'parsed_item': parsed_item,
            'raw_text_preview': raw_text_preview,
        },
    )


@app.post('/vault/new', response_class=HTMLResponse)
async def vault_new(request: Request, item_yaml: Optional[str] = Form(default=None)) -> HTMLResponse:
    parsed: Optional[Dict[str, Any]] = None
    try:
        if item_yaml and item_yaml.strip():
            parsed = yaml.safe_load(item_yaml)
            if not isinstance(parsed, dict):
                raise ValueError('Vault YAML root must be an object.')
        else:
            form = await request.form()
            parsed = _parse_vault_form_payload(form)
        item = VaultItem.model_validate(parsed)
        item_id = uuid.uuid4().hex
        repository.save_vault_item(item_id, item)
        active_user_id = _active_user_id()
        if active_user_id:
            vault_dir = repository.vault_dir_for(active_user_id)
            item_path = ensure_within(vault_dir, vault_dir / f'{item_id}.yaml')
            auth_store.upsert_vault_item(
                user_id=active_user_id,
                item_id=item_id,
                title=item.title,
                item_type=item.type.value,
                path=str(item_path),
            )
            auth_store.mark_onboarding_step(user_id=active_user_id, step='evidence')
            _sync_profile_progress(active_user_id)
        return RedirectResponse(url='/vault?flash=vault_saved', status_code=303)
    except ValidationError as exc:
        return render(
            request,
            'vault_form.html',
            {
                'item_id': None,
                'item_form': _vault_item_to_form_payload(parsed),
                'error': 'Vault item validation failed. Fix the fields below and resubmit.',
                'field_errors': _format_validation_errors(exc),
            },
        )
    except Exception as exc:
        return render(
            request,
            'vault_form.html',
            {'item_id': None, 'item_form': _vault_item_to_form_payload(parsed), 'error': str(exc), 'field_errors': []},
        )


@app.get('/vault/{item_id}', response_class=HTMLResponse)
async def vault_edit_page(request: Request, item_id: str) -> HTMLResponse:
    item = repository.get_vault_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail='Vault item not found')
    return render(
        request,
        'vault_form.html',
        {'item_id': item_id, 'item_form': _vault_item_to_form_payload(item), 'error': None, 'field_errors': []},
    )


@app.post('/vault/{item_id}', response_class=HTMLResponse)
async def vault_edit(request: Request, item_id: str, item_yaml: Optional[str] = Form(default=None)) -> HTMLResponse:
    parsed: Optional[Dict[str, Any]] = None
    try:
        if item_yaml and item_yaml.strip():
            parsed = yaml.safe_load(item_yaml)
            if not isinstance(parsed, dict):
                raise ValueError('Vault YAML root must be an object.')
        else:
            form = await request.form()
            parsed = _parse_vault_form_payload(form)
        item = VaultItem.model_validate(parsed)
        repository.save_vault_item(item_id, item)
        active_user_id = _active_user_id()
        if active_user_id:
            vault_dir = repository.vault_dir_for(active_user_id)
            item_path = ensure_within(vault_dir, vault_dir / f'{item_id}.yaml')
            auth_store.upsert_vault_item(
                user_id=active_user_id,
                item_id=item_id,
                title=item.title,
                item_type=item.type.value,
                path=str(item_path),
            )
            auth_store.mark_onboarding_step(user_id=active_user_id, step='evidence')
            _sync_profile_progress(active_user_id)
        return RedirectResponse(url='/vault?flash=vault_saved', status_code=303)
    except ValidationError as exc:
        return render(
            request,
            'vault_form.html',
            {
                'item_id': item_id,
                'item_form': _vault_item_to_form_payload(parsed),
                'error': 'Vault item validation failed. Fix the fields below and resubmit.',
                'field_errors': _format_validation_errors(exc),
            },
        )
    except Exception as exc:
        return render(
            request,
            'vault_form.html',
            {'item_id': item_id, 'item_form': _vault_item_to_form_payload(parsed), 'error': str(exc), 'field_errors': []},
        )


@app.post('/vault/{item_id}/delete')
async def vault_delete(item_id: str):
    item = repository.get_vault_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail='Vault item not found')
    repository.delete_vault_item(item_id)
    active_user_id = _active_user_id()
    if active_user_id:
        auth_store.delete_vault_item(user_id=active_user_id, item_id=item_id)
        _sync_profile_progress(active_user_id)
    return RedirectResponse(url='/vault?flash=vault_deleted', status_code=303)


@app.get('/jobs', response_class=HTMLResponse)
async def jobs_list(request: Request) -> HTMLResponse:
    jobs = repository.list_jobs()
    return render(request, 'jobs_list.html', {'jobs': jobs})


@app.get('/jobs/new', response_class=HTMLResponse)
async def jobs_new_page(request: Request) -> HTMLResponse:
    return render(request, 'jobs_new.html', {'error': None, 'url': '', 'jd_text': ''})


@app.post('/jobs/ingest', response_class=HTMLResponse)
async def jobs_ingest(
    request: Request,
    url: Optional[str] = Form(default=None),
    jd_text: Optional[str] = Form(default=None),
) -> HTMLResponse:
    url = (url or '').strip()
    jd_text = (jd_text or '').strip()

    if not url and not jd_text:
        return render(
            request,
            'jobs_new.html',
            {'error': 'Provide a URL or paste job description text.', 'url': url, 'jd_text': jd_text},
        )

    title: Optional[str] = None
    company: Optional[str] = None
    warnings = []

    if not jd_text and url:
        try:
            scrape = await scrape_job_posting(url)
            title = scrape.title
            company = scrape.company
            jd_text = scrape.jd_text
            warnings.extend(scrape.warnings)
        except ScrapeError as exc:
            return render(
                request,
                'jobs_new.html',
                {
                    'error': f'Scraping failed: {exc}. Paste JD text as fallback.',
                    'url': url,
                    'jd_text': jd_text,
                },
            )

    if not jd_text:
        return render(
            request,
            'jobs_new.html',
            {'error': 'Job description text is empty.', 'url': url, 'jd_text': jd_text},
        )

    job_id = uuid.uuid4().hex[:12]
    job = JobRecord(job_id=job_id, url=url or None, title=title, company=company)
    repository.save_job(job, jd_text)
    active_user_id = _active_user_id()
    if active_user_id:
        jobs_dir = repository.jobs_dir_for(active_user_id)
        job_path = ensure_within(jobs_dir, jobs_dir / job_id / 'job.yaml')
        auth_store.upsert_job(
            user_id=active_user_id,
            job_id=job.job_id,
            title=job.title,
            company=job.company,
            url=job.url,
            path=str(job_path),
        )
        _update_profile_focus_from_jd(active_user_id, jd_text=jd_text, fallback_title=job.title or '')
        auth_store.mark_onboarding_step(user_id=active_user_id, step='first_job')
        _sync_profile_progress(active_user_id)

    return RedirectResponse(url=f'/jobs/{job_id}?flash=job_ingested', status_code=303)


@app.get('/jobs/{job_id}', response_class=HTMLResponse)
async def jobs_detail(request: Request, job_id: str) -> HTMLResponse:
    job = repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    jd_text = repository.get_job_text(job_id)

    output_root = repository.outputs_dir_for() / job_id
    output_runs = sorted([p for p in output_root.glob('*') if p.is_dir()], reverse=True) if output_root.exists() else []
    latest_output = output_runs[0].name if output_runs else None
    latest_report = None
    if latest_output:
        report_path = output_root / latest_output / 'report.json'
        if report_path.exists():
            latest_report = load_json(report_path)

    return render(
        request,
        'job_detail.html',
        {
            'job': job,
            'jd_text': jd_text,
            'error': None,
            'latest_output': latest_output,
            'latest_report': latest_report,
        },
    )


@app.post('/jobs/{job_id}/jd')
async def jobs_update_jd(job_id: str, jd_text: str = Form(...)):
    job = repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    repository.update_job_text(job_id, jd_text)
    active_user_id = _active_user_id()
    if active_user_id:
        jobs_dir = repository.jobs_dir_for(active_user_id)
        job_path = ensure_within(jobs_dir, jobs_dir / job_id / 'job.yaml')
        auth_store.upsert_job(
            user_id=active_user_id,
            job_id=job.job_id,
            title=job.title,
            company=job.company,
            url=job.url,
            path=str(job_path),
        )
        _update_profile_focus_from_jd(active_user_id, jd_text=jd_text, fallback_title=job.title or '')
        auth_store.mark_onboarding_step(user_id=active_user_id, step='first_job')
        _sync_profile_progress(active_user_id)
    return RedirectResponse(url=f'/jobs/{job_id}?flash=jd_saved', status_code=303)


def _run_tailoring_workflow(
    *,
    base_resume: CanonicalResume,
    job_id: str,
    jd_text: str,
    mode: TailorMode,
    job_title_hint: Optional[str],
) -> Dict[str, Any]:
    assert llm_service and llm_service.available
    vault_items = repository.list_vault_items()

    tailored = tailor_resume(
        base_resume=base_resume,
        vault_items=vault_items,
        jd_text=jd_text,
        mode=mode,
        llm=llm_service,
        job_title_hint=job_title_hint,
    )
    match_payload = compute_match_score(tailored.tailored_resume, jd_text)
    match_score = float(match_payload.get('overall_score', 0.0) or 0.0)

    output_dir = repository.create_output_dir(job_id)
    active_user_id = _active_user_id()
    if active_user_id:
        auth_store.upsert_output(
            user_id=active_user_id,
            job_id=job_id,
            timestamp=output_dir.name,
            path=str(output_dir),
        )
    compile_error: Optional[str] = None
    pdf_exists = False
    ats_pdf_exists = False
    ats_docx_exists = False
    ats_txt_exists = False
    resume_to_render = tailored.tailored_resume
    optimization_match_score = match_score
    score_lookup = {
        normalize_token(item.title): item.score
        for item in tailored.report.chosen_items
        if normalize_token(item.title)
    }
    used_expansions: set[str] = set()
    tighten_level = 0
    expand_attempts = 0
    max_expand_attempts = 6
    project_target_min = MIN_PROJECT_ITEMS
    project_target_max = MAX_PROJECT_ITEMS
    last_stable_resume = resume_to_render

    while True:
        latex_service.render_resume(resume_to_render, output_dir)
        try:
            pdf_path = latex_service.compile_resume(output_dir)
        except LatexRenderError as exc:
            compile_error = str(exc)
            tailored.report.warnings.append(f'PDF compile failed: {exc}')
            break

        pdf_exists = True
        page_count = latex_service.count_pdf_pages(pdf_path)
        if page_count > 1:
            if tighten_level >= 3:
                tailored.report.warnings.append(f'Output is still {page_count} pages after aggressive one-page pruning.')
                break
            tailored.report.warnings.append(
                f'Output exceeded one page ({page_count} pages). Tightening and recompiling.'
            )
            resume_to_render = tighten_resume_for_one_page(
                resume=resume_to_render,
                score_lookup=score_lookup,
                warnings=tailored.report.warnings,
                level=tighten_level + 1,
            )
            tighten_level += 1
            continue

        last_stable_resume = resume_to_render
        if len(resume_to_render.projects) >= project_target_max:
            break
        if expand_attempts >= max_expand_attempts:
            break

        expanded_resume, changed, expansion_marker = expand_resume_with_projects(
            resume=resume_to_render,
            base_resume=base_resume,
            vault_items=vault_items,
            score_lookup=score_lookup,
            used_expansions=used_expansions,
        )
        if not changed or not expansion_marker:
            break

        used_expansions.add(expansion_marker)
        expand_attempts += 1
        resume_to_render = expanded_resume
        latex_service.render_resume(resume_to_render, output_dir)
        try:
            trial_pdf_path = latex_service.compile_resume(output_dir)
        except LatexRenderError:
            resume_to_render = last_stable_resume
            break

        trial_pages = latex_service.count_pdf_pages(trial_pdf_path)
        if trial_pages <= 1:
            if len(resume_to_render.projects) <= project_target_min:
                tailored.report.warnings.append('Added project content to hit the opinionated minimum project count.')
            else:
                tailored.report.warnings.append('Added project content to use remaining page space.')
            last_stable_resume = resume_to_render
            continue

        resume_to_render = last_stable_resume
        latex_service.render_resume(resume_to_render, output_dir)
        try:
            latex_service.compile_resume(output_dir)
        except LatexRenderError as exc:
            compile_error = str(exc)
            tailored.report.warnings.append(f'PDF compile failed after expansion rollback: {exc}')
            pdf_exists = False
        break

    if len(resume_to_render.projects) < project_target_min:
        tailored.report.warnings.append(
            f'Could only fit {len(resume_to_render.projects)} project(s) on one page; preferred range is '
            f'{project_target_min}-{project_target_max}.'
        )

    report_path = output_dir / 'report.json'
    try:
        ats_render_result = public_render_outputs(resume_to_render, output_dir, filename_prefix='ats_')
        ats_pdf_exists = Path(ats_render_result['pdf_path']).exists()
        ats_docx_exists = Path(ats_render_result['docx_path']).exists()
        ats_txt_exists = Path(ats_render_result['txt_path']).exists()
        if not ats_render_result['pdf_text_layer']['ok']:
            tailored.report.warnings.append('ATS PDF text-layer verification failed.')
    except Exception as exc:
        tailored.report.warnings.append(f'ATS output render failed: {exc}')
    save_json(report_path, tailored.report.model_dump(mode='json'))

    return {
        'warnings': list(tailored.report.warnings),
        'keywords_covered': list(tailored.report.keywords_covered),
        'keywords_missed': list(tailored.report.keywords_missed),
        'chosen_items': [item.model_dump(mode='json') for item in tailored.report.chosen_items],
        'vault_relevance': [item.model_dump(mode='json') for item in tailored.report.vault_relevance],
        'missing_required_evidence': list(tailored.report.missing_required_evidence),
        'required_skill_evidence_map': [
            item.model_dump(mode='json') for item in tailored.report.required_skill_evidence_map
        ],
        'high_confidence_exclusions': [
            item.model_dump(mode='json') for item in tailored.report.high_confidence_exclusions
        ],
        'job_id': job_id,
        'timestamp': output_dir.name,
        'pdf_exists': pdf_exists,
        'ats_pdf_exists': ats_pdf_exists,
        'ats_docx_exists': ats_docx_exists,
        'ats_txt_exists': ats_txt_exists,
        'compile_error': compile_error,
        'match_score': optimization_match_score,
    }


def _require_authenticated_user_id(request: Request) -> str:
    user_id = str(getattr(request.state, 'current_user_id', '') or '').strip()
    if not user_id:
        raise HTTPException(status_code=401, detail='Authentication required.')
    return user_id


async def _execute_extension_tailor_run(
    *,
    run_id: str,
    user_id: str,
    job_id: str,
    jd_text: str,
    job_title_hint: str,
) -> None:
    auth_store.update_extension_run(run_id=run_id, status='running', error=None, output_timestamp=None)
    token = set_current_user_id(user_id)
    try:
        if not llm_service or not llm_service.available:
            raise RuntimeError('OPENAI_API_KEY is not configured. Tailoring is disabled.')

        base_resume = repository.load_base_resume(user_id=user_id)
        if not base_resume:
            raise RuntimeError('Base resume is missing. Upload and save a base resume first.')

        vault_items = repository.list_vault_items(user_id=user_id)
        if not vault_items:
            raise RuntimeError('Vault is empty. Add evidence items before tailoring.')

        workflow = await run_in_threadpool(
            _run_tailoring_workflow,
            base_resume=base_resume,
            job_id=job_id,
            jd_text=jd_text,
            mode=DEFAULT_TAILOR_MODE,
            job_title_hint=job_title_hint or None,
        )
        auth_store.update_extension_run(
            run_id=run_id,
            status='succeeded',
            error=None,
            output_timestamp=str(workflow['timestamp']),
        )
    except Exception as exc:
        auth_store.update_extension_run(
            run_id=run_id,
            status='failed',
            error=str(exc),
            output_timestamp=None,
        )
    finally:
        reset_current_user_id(token)
        EXTENSION_RUN_TASKS.pop(run_id, None)


def _extension_run_download_url(run_id: str) -> str:
    return f'/api/ext/v1/tailor-runs/{run_id}/resume.pdf'


def _tailor_result_context(
    *,
    job: JobRecord,
    mode: TailorMode,
    workflow: Dict[str, Any],
    prepended_warnings: Optional[list[str]] = None,
) -> Dict[str, Any]:
    def _format_skill_term(term: str) -> str:
        cleaned = str(term or '').strip()
        if not cleaned:
            return ''
        normalized = normalize_token(cleaned)
        if not normalized:
            return cleaned
        if normalized in TERM_DISPLAY_OVERRIDES:
            return TERM_DISPLAY_OVERRIDES[normalized]
        original_parts = [normalize_token(part) for part in re.split(r'[\s_-]+', cleaned) if normalize_token(part)]
        parts = original_parts if len(original_parts) > 1 else [normalized]
        if not parts:
            return cleaned
        display_parts = [part.upper() if part in UPPERCASE_DISPLAY_TERMS else part.capitalize() for part in parts]
        return ' '.join(display_parts)

    def _format_skill_terms(terms: Any) -> list[str]:
        formatted: list[str] = []
        for term in terms or []:
            display = _format_skill_term(str(term))
            if display and display not in formatted:
                formatted.append(display)
        return formatted

    def _format_source_type(source_type: str) -> str:
        normalized = normalize_token(source_type)
        if not normalized:
            return ''
        if normalized in {'experience', 'job', 'vaultjob'}:
            return 'Work'
        if normalized in {'coursework', 'vaultcoursework'}:
            return 'Coursework'
        if normalized in {'award', 'vaultaward'}:
            return 'Award'
        return 'Project'

    warnings: list[str] = []
    seen: set[str] = set()
    for warning in [*(prepended_warnings or []), *workflow['warnings']]:
        cleaned = str(warning).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        warnings.append(cleaned)

    vault_relevance: list[Dict[str, Any]] = []
    for raw_item in workflow['vault_relevance']:
        item = dict(raw_item)
        item['matched_required_terms_display'] = _format_skill_terms(item.get('matched_required_terms', []))
        item['missing_required_terms_display'] = _format_skill_terms(item.get('missing_required_terms', []))
        vault_relevance.append(item)

    missing_required_evidence_display = _format_skill_terms(workflow['missing_required_evidence'])
    required_skill_evidence_map: list[Dict[str, Any]] = []
    for raw_item in workflow.get('required_skill_evidence_map', []):
        item = dict(raw_item)
        item['required_term_display'] = _format_skill_term(str(item.get('required_term', '')))
        item['source_type_display'] = _format_source_type(str(item.get('source_type', '')))
        required_skill_evidence_map.append(item)

    return {
        'job': job,
        'mode': mode,
        'warnings': warnings,
        'job_id': workflow['job_id'],
        'timestamp': workflow['timestamp'],
        'pdf_exists': workflow['pdf_exists'],
        'ats_pdf_exists': workflow['ats_pdf_exists'],
        'ats_docx_exists': workflow['ats_docx_exists'],
        'ats_txt_exists': workflow['ats_txt_exists'],
        'compile_error': workflow['compile_error'],
        'match_score': workflow['match_score'],
        'chosen_items': workflow['chosen_items'],
        'vault_relevance': vault_relevance,
        'missing_required_evidence': workflow['missing_required_evidence'],
        'missing_required_evidence_display': missing_required_evidence_display,
        'required_skill_evidence_map': required_skill_evidence_map,
        'high_confidence_exclusions': workflow.get('high_confidence_exclusions', []),
        'keywords_covered': workflow['keywords_covered'],
        'keywords_missed': workflow['keywords_missed'],
    }


@app.post('/jobs/{job_id}/tailor', response_class=HTMLResponse)
async def jobs_tailor(
    request: Request,
    job_id: str,
) -> HTMLResponse:
    job = repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')

    base_resume = repository.load_base_resume()
    if not base_resume:
        return render(
            request,
            'job_detail.html',
            {
                'job': job,
                'jd_text': repository.get_job_text(job_id),
                'error': 'Base resume is missing. Upload and save canonical resume first.',
                'latest_output': None,
                'latest_report': None,
            },
        )

    if not llm_service or not llm_service.available:
        return render(
            request,
            'job_detail.html',
            {
                'job': job,
                'jd_text': repository.get_job_text(job_id),
                'error': 'OPENAI_API_KEY is not configured. Tailoring is disabled until set.',
                'latest_output': None,
                'latest_report': None,
            },
        )

    vault_items = repository.list_vault_items()
    if not vault_items:
        return render(
            request,
            'job_detail.html',
            {
                'job': job,
                'jd_text': repository.get_job_text(job_id),
                'error': 'Vault is empty. Add vault evidence items before tailoring.',
                'latest_output': None,
                'latest_report': None,
            },
        )

    jd_text = repository.get_job_text(job_id)
    workflow = _run_tailoring_workflow(
        base_resume=base_resume,
        job_id=job_id,
        jd_text=jd_text,
        mode=DEFAULT_TAILOR_MODE,
        job_title_hint=job.title,
    )

    return render(
        request,
        'tailor_result.html',
        _tailor_result_context(job=job, mode=DEFAULT_TAILOR_MODE, workflow=workflow),
    )


def _resolve_output_paths(job_id: str, timestamp: str) -> tuple[Path, Path, Path]:
    output_root = repository.outputs_dir_for()
    output_dir = ensure_within(output_root, output_root / job_id / timestamp)
    if not output_dir.exists() or not output_dir.is_dir():
        raise HTTPException(status_code=404, detail='Output run not found')
    tex_path = ensure_within(output_dir, output_dir / 'resume.tex')
    pdf_path = ensure_within(output_dir, output_dir / 'resume.pdf')
    return output_dir, tex_path, pdf_path


@app.get('/outputs/{job_id}/{timestamp}/latex', response_class=HTMLResponse)
async def output_latex_editor(request: Request, job_id: str, timestamp: str) -> HTMLResponse:
    _, tex_path, pdf_path = _resolve_output_paths(job_id, timestamp)
    if not tex_path.exists():
        raise HTTPException(status_code=404, detail='LaTeX source not found')
    return render(
        request,
        'latex_editor.html',
        {
            'job_id': job_id,
            'timestamp': timestamp,
            'tex_content': tex_path.read_text(encoding='utf-8'),
            'pdf_exists': pdf_path.exists(),
            'save_message': None,
            'compile_error': None,
        },
    )


@app.post('/outputs/{job_id}/{timestamp}/latex', response_class=HTMLResponse)
async def output_latex_editor_save(
    request: Request,
    job_id: str,
    timestamp: str,
    tex_content: str = Form(...),
    action: str = Form(default='save_compile'),
) -> HTMLResponse:
    output_dir, tex_path, pdf_path = _resolve_output_paths(job_id, timestamp)
    if not tex_path.exists():
        raise HTTPException(status_code=404, detail='LaTeX source not found')

    compile_error: Optional[str] = None
    save_message: Optional[str] = None
    cleaned_action = (action or '').strip().lower()

    if not tex_content.strip():
        compile_error = 'LaTeX source cannot be empty.'
    else:
        tex_path.write_text(tex_content, encoding='utf-8')
        save_message = 'Saved LaTeX source.'
        if cleaned_action == 'save_compile':
            try:
                latex_service.compile_resume(output_dir)
                save_message = 'Saved and recompiled LaTeX source.'
            except LatexRenderError as exc:
                compile_error = str(exc)

    return render(
        request,
        'latex_editor.html',
        {
            'job_id': job_id,
            'timestamp': timestamp,
            'tex_content': tex_path.read_text(encoding='utf-8'),
            'pdf_exists': pdf_path.exists(),
            'save_message': save_message,
            'compile_error': compile_error,
        },
    )


@app.get('/outputs/{job_id}/{timestamp}/resume.pdf')
async def download_pdf(job_id: str, timestamp: str):
    _, _, pdf_path = _resolve_output_paths(job_id, timestamp)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail='Output PDF not found')
    return FileResponse(path=pdf_path, media_type='application/pdf', filename='resume.pdf')


@app.get('/outputs/{job_id}/{timestamp}/{artifact}')
async def download_output_artifact(job_id: str, timestamp: str, artifact: str):
    allowed = {
        'resume.pdf': 'application/pdf',
        'resume.tex': 'text/plain; charset=utf-8',
        'report.json': 'application/json',
        'ats_resume.pdf': 'application/pdf',
        'ats_resume.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'ats_resume.txt': 'text/plain; charset=utf-8',
        'bundle.zip': 'application/zip',
    }
    if artifact not in allowed:
        raise HTTPException(status_code=404, detail='Artifact not available')
    output_dir, _, _ = _resolve_output_paths(job_id, timestamp)
    target = ensure_within(output_dir, output_dir / artifact)
    if not target.exists():
        raise HTTPException(status_code=404, detail='Artifact not found')
    return FileResponse(path=target, media_type=allowed[artifact], filename=artifact)


@app.get('/api/ext/v1/key/status')
async def extension_key_status(request: Request):
    user_id = _require_authenticated_user_id(request)
    status = auth_store.get_extension_api_key_status(user_id=user_id)
    if status is None:
        return JSONResponse(content={'has_key': False, 'key_id': None, 'created_at': None, 'rotated_at': None})
    return JSONResponse(
        content={
            'has_key': bool(status.is_active),
            'key_id': status.key_id,
            'created_at': status.created_at,
            'rotated_at': status.rotated_at,
        }
    )


@app.post('/api/ext/v1/key/regenerate')
async def extension_key_regenerate(request: Request):
    user_id = _require_authenticated_user_id(request)
    api_key = auth_store.regenerate_extension_api_key(user_id=user_id)
    status = auth_store.get_extension_api_key_status(user_id=user_id)
    return JSONResponse(
        content={
            'api_key': api_key,
            'key_id': status.key_id if status else None,
            'rotated_at': status.rotated_at if status else None,
        }
    )


@app.post('/api/ext/v1/tailor-runs')
async def extension_create_tailor_run(request: Request):
    user_id = _require_authenticated_user_id(request)
    payload = await request.json()
    jd_text = _clean_payload_text(payload.get('jd_text'))
    source_url = _clean_payload_text(payload.get('source_url'))
    job_title = _clean_payload_text(payload.get('job_title'))
    company = _clean_payload_text(payload.get('company'))

    if not jd_text:
        raise HTTPException(status_code=400, detail='jd_text is required.')

    job_id = uuid.uuid4().hex[:12]
    title = job_title or 'Extension Captured Job'
    job = JobRecord(job_id=job_id, title=title, company=company or None, url=source_url or None)
    repository.save_job(job, jd_text, user_id=user_id)

    jobs_dir = repository.jobs_dir_for(user_id)
    job_path = ensure_within(jobs_dir, jobs_dir / job_id / 'job.yaml')
    auth_store.upsert_job(
        user_id=user_id,
        job_id=job.job_id,
        title=job.title,
        company=job.company,
        url=job.url,
        path=str(job_path),
    )
    _update_profile_focus_from_jd(user_id, jd_text=jd_text, fallback_title=job.title or '')
    auth_store.mark_onboarding_step(user_id=user_id, step='first_job')
    _sync_profile_progress(user_id)

    run = auth_store.create_extension_run(user_id=user_id, job_id=job_id)
    task = asyncio.create_task(
        _execute_extension_tailor_run(
            run_id=run.run_id,
            user_id=user_id,
            job_id=job_id,
            jd_text=jd_text,
            job_title_hint=job.title or '',
        )
    )
    EXTENSION_RUN_TASKS[run.run_id] = task

    return JSONResponse(
        status_code=202,
        content={
            'run_id': run.run_id,
            'job_id': job_id,
            'status': run.status,
        },
    )


@app.get('/api/ext/v1/tailor-runs/{run_id}')
async def extension_tailor_run_status(request: Request, run_id: str):
    user_id = _require_authenticated_user_id(request)
    run = auth_store.get_extension_run(run_id=run_id, user_id=user_id)
    if run is None:
        raise HTTPException(status_code=404, detail='Run not found.')

    payload: Dict[str, Any] = {
        'run_id': run.run_id,
        'job_id': run.job_id,
        'status': run.status,
        'error': run.error,
        'created_at': run.created_at,
        'updated_at': run.updated_at,
    }
    if run.status == 'succeeded' and run.output_timestamp:
        payload['timestamp'] = run.output_timestamp
        payload['pdf_download_url'] = _extension_run_download_url(run.run_id)
    return JSONResponse(content=payload)


@app.get('/api/ext/v1/tailor-runs/{run_id}/resume.pdf')
async def extension_tailor_run_pdf(request: Request, run_id: str):
    user_id = _require_authenticated_user_id(request)
    run = auth_store.get_extension_run(run_id=run_id, user_id=user_id)
    if run is None:
        raise HTTPException(status_code=404, detail='Run not found.')
    if run.status != 'succeeded' or not run.output_timestamp:
        raise HTTPException(status_code=409, detail='Run is not complete yet.')

    outputs_root = repository.outputs_dir_for(user_id)
    pdf_path = ensure_within(outputs_root, outputs_root / run.job_id / run.output_timestamp / 'resume.pdf')
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail='Output PDF not found.')
    return FileResponse(path=pdf_path, media_type='application/pdf', filename=f'resume-{run.job_id}.pdf')


@app.post('/api/upload_resume')
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail='Missing resume file.')
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {'.pdf', '.docx', '.txt'}:
        raise HTTPException(status_code=400, detail='Supported file types: PDF, DOCX, TXT.')

    upload_path = settings.data_dir / 'uploads' / f'{uuid.uuid4().hex}{suffix}'
    upload_path.write_bytes(await file.read())
    result = public_upload_resume(upload_path, enable_ocr=settings.enable_ocr, llm=llm_service)
    return JSONResponse(content=result)


@app.post('/api/upload_job_description')
async def upload_job_description(
    url: Optional[str] = Form(default=None),
    jd_text: Optional[str] = Form(default=None),
):
    url = (url or '').strip()
    jd_text = (jd_text or '').strip()
    if not jd_text and not url:
        raise HTTPException(status_code=400, detail='Provide jd_text or url.')

    scrape_warnings = []
    if not jd_text and url:
        try:
            scraped = await scrape_job_posting(url)
            jd_text = scraped.jd_text
            scrape_warnings.extend(scraped.warnings)
        except ScrapeError as exc:
            raise HTTPException(status_code=400, detail=f'Job scraping failed: {exc}') from exc

    assert jd_text
    parsed = public_upload_job_description(jd_text)
    parsed['warnings'] = scrape_warnings
    return JSONResponse(content=parsed)


@app.post('/api/lint_resume')
async def lint_resume(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail='Missing resume file.')
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {'.pdf', '.docx', '.txt'}:
        raise HTTPException(status_code=400, detail='Supported file types: PDF, DOCX, TXT.')
    upload_path = settings.data_dir / 'uploads' / f'{uuid.uuid4().hex}{suffix}'
    upload_path.write_bytes(await file.read())
    lint = public_lint_resume(upload_path)
    return JSONResponse(content=lint)


@app.post('/api/parse_mirror')
async def parse_mirror(raw_text: str = Form(...)):
    result = public_parse_mirror(raw_text, llm=llm_service)
    return JSONResponse(content=result)


@app.post('/api/build_canonical')
async def build_canonical(request: Request):
    payload = await request.json()
    canonical = public_build_canonical(payload)
    return JSONResponse(content=canonical.model_dump(mode='json'))


@app.post('/api/score_parse_quality')
async def score_parse_quality(request: Request):
    payload = await request.json()
    return JSONResponse(content=public_score_parse_quality(payload))


@app.post('/api/score_match')
async def score_match(request: Request):
    payload = await request.json()
    jd_text = _clean_payload_text(payload.get('jd_text'))
    if not jd_text and payload.get('job_id'):
        job_id = str(payload.get('job_id'))
        job = repository.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail='Job not found')
        jd_text = repository.get_job_text(job_id)
    if not jd_text:
        raise HTTPException(status_code=400, detail='Missing jd_text or valid job_id.')

    resume_payload = payload.get('resume')
    if resume_payload:
        resume = CanonicalResume.model_validate(resume_payload)
    else:
        resume = repository.load_base_resume()
        if not resume:
            raise HTTPException(status_code=400, detail='Base resume not found.')
    return JSONResponse(content=public_score_match(resume, jd_text))


@app.post('/api/generate_patches')
async def generate_patches(request: Request):
    payload = await request.json()
    jd_text = _clean_payload_text(payload.get('jd_text'))
    if not jd_text and payload.get('job_id'):
        job_id = str(payload.get('job_id'))
        job = repository.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail='Job not found')
        jd_text = repository.get_job_text(job_id)
    if not jd_text:
        raise HTTPException(status_code=400, detail='Missing jd_text or valid job_id.')

    resume_payload = payload.get('resume')
    if resume_payload:
        resume = CanonicalResume.model_validate(resume_payload)
    else:
        resume = repository.load_base_resume()
        if not resume:
            raise HTTPException(status_code=400, detail='Base resume not found.')

    response = public_generate_patches(resume, jd_text)
    return JSONResponse(content=response)


@app.post('/api/apply_patches')
async def apply_patches(request: Request):
    payload = await request.json()
    resume_payload = payload.get('resume')
    patch_payload = payload.get('patches')
    if not isinstance(patch_payload, list):
        raise HTTPException(status_code=400, detail='patches must be a list.')
    if resume_payload:
        resume = CanonicalResume.model_validate(resume_payload)
    else:
        resume = repository.load_base_resume()
        if not resume:
            raise HTTPException(status_code=400, detail='Base resume not found.')

    allow_unconfirmed = bool(payload.get('allow_requires_confirmation', False))
    result = public_apply_patches(resume, patch_payload, allow_requires_confirmation=allow_unconfirmed)
    updated_resume = result['resume']
    result['resume'] = updated_resume.model_dump(mode='json')

    job_id = _clean_payload_text(payload.get('job_id')) or 'manual'
    jd_text = _clean_payload_text(payload.get('jd_text'))
    score_before = None
    score_after = None
    if jd_text:
        score_before = compute_match_score(resume, jd_text).get('overall_score')
        score_after = compute_match_score(updated_resume, jd_text).get('overall_score')
    version_payload = version_resume(
        updated_resume,
        data_dir=settings.data_dir,
        job_id=job_id,
        match_score=score_after,
        metadata={'score_before': score_before, 'score_after': score_after},
    )
    result['version'] = version_payload
    return JSONResponse(content=result)


@app.post('/api/render_outputs')
async def render_outputs(request: Request):
    payload = await request.json()
    resume_payload = payload.get('resume')
    if resume_payload:
        resume = CanonicalResume.model_validate(resume_payload)
    else:
        resume = repository.load_base_resume()
        if not resume:
            raise HTTPException(status_code=400, detail='Base resume not found.')

    job_id = _clean_payload_text(payload.get('job_id')) or 'manual'
    outputs_dir = repository.outputs_dir_for()
    output_root = ensure_within(outputs_dir, outputs_dir / normalize_token(job_id or 'manual'))
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime_now_stamp()
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    active_user_id = _active_user_id()
    if active_user_id:
        auth_store.upsert_output(
            user_id=active_user_id,
            job_id=normalize_token(job_id or 'manual') or 'manual',
            timestamp=timestamp,
            path=str(output_dir),
        )

    prefix = _clean_payload_text(payload.get('filename_prefix'))
    render_result = public_render_outputs(resume, output_dir, filename_prefix=prefix)
    if not render_result['pdf_text_layer']['ok']:
        raise HTTPException(status_code=500, detail='Rendered PDF does not contain enough extractable text.')
    bundle_path = public_export_bundle(output_dir, bundle_path=output_dir / 'bundle.zip')
    return JSONResponse(content={'output_dir': str(output_dir), 'bundle_path': str(bundle_path), **render_result})


@app.get('/api/export_bundle/{job_id}/{timestamp}')
async def export_bundle(job_id: str, timestamp: str):
    output_root = repository.outputs_dir_for()
    target_dir = ensure_within(output_root, output_root / job_id / timestamp)
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail='Output directory not found')
    bundle_path = public_export_bundle(target_dir, bundle_path=target_dir / 'bundle.zip')
    return FileResponse(path=bundle_path, media_type='application/zip', filename='resume_bundle.zip')


@app.post('/api/compare_versions')
async def compare_versions_endpoint(request: Request):
    payload = await request.json()
    left = payload.get('left')
    right = payload.get('right')
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise HTTPException(status_code=400, detail='left and right version payloads are required.')
    return JSONResponse(content=compare_versions(left, right))


def _format_validation_errors(exc: ValidationError) -> list[str]:
    errors: list[str] = []
    for row in exc.errors():
        loc = '.'.join(str(part) for part in row.get('loc', []))
        msg = str(row.get('msg', 'Invalid value'))
        errors.append(f'{loc}: {msg}' if loc else msg)
    return errors


def _form_value(form: Any, key: str) -> str:
    value = form.get(key, '')
    if value is None:
        return ''
    return str(value).strip()


def _to_text(value: Any) -> str:
    if value is None:
        return ''
    return str(value).strip()


def _split_multivalue(value: Any) -> list[str]:
    raw = str(value or '').replace(',', '\n').replace(';', '\n')
    seen: set[str] = set()
    output: list[str] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        token = normalize_token(cleaned)
        if not cleaned or not token or token in seen:
            continue
        seen.add(token)
        output.append(cleaned)
    return output


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            cleaned = _to_text(item)
            if cleaned:
                output.append(cleaned)
        return output
    if isinstance(value, tuple):
        output: list[str] = []
        for item in value:
            cleaned = _to_text(item)
            if cleaned:
                output.append(cleaned)
        return output
    return _split_multivalue(value)


def _extract_indexes(keys: Any, pattern: str) -> list[int]:
    regex = re.compile(pattern)
    indexes: set[int] = set()
    for key in keys:
        match = regex.match(str(key))
        if match:
            indexes.add(int(match.group(1)))
    return sorted(indexes)


def _has_nonempty(*values: Any) -> bool:
    for value in values:
        if isinstance(value, str):
            if value.strip():
                return True
            continue
        if isinstance(value, (list, tuple, set, dict)):
            if value:
                return True
            continue
        if value is not None:
            return True
    return False


def _resume_to_form_payload(resume: Optional[Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if isinstance(resume, CanonicalResume):
        payload = resume.model_dump(exclude_none=True, mode='json')
    elif isinstance(resume, dict):
        payload = resume

    identity = payload.get('identity') if isinstance(payload.get('identity'), dict) else {}
    identity_links = _coerce_string_list(identity.get('links'))
    education_rows: list[Dict[str, Any]] = []
    for row in payload.get('education') or []:
        if not isinstance(row, dict):
            continue
        dates = row.get('dates') if isinstance(row.get('dates'), dict) else {}
        education_rows.append(
            {
                'school': _to_text(row.get('school')),
                'degree': _to_text(row.get('degree')),
                'major': _to_text(row.get('major')),
                'minors': _coerce_string_list(row.get('minors')),
                'gpa': _to_text(row.get('gpa')),
                'start': _to_text(dates.get('start')),
                'end': _to_text(dates.get('end')),
                'coursework': _coerce_string_list(row.get('coursework')),
            }
        )

    experience_rows: list[Dict[str, Any]] = []
    for row in payload.get('experience') or []:
        if not isinstance(row, dict):
            continue
        dates = row.get('dates') if isinstance(row.get('dates'), dict) else {}
        bullets = _coerce_string_list(row.get('bullets'))
        experience_rows.append(
            {
                'company': _to_text(row.get('company')),
                'title': _to_text(row.get('title')),
                'location': _to_text(row.get('location')),
                'start': _to_text(dates.get('start')),
                'end': _to_text(dates.get('end')),
                'bullets': bullets,
            }
        )

    project_rows: list[Dict[str, Any]] = []
    for row in payload.get('projects') or []:
        if not isinstance(row, dict):
            continue
        dates = row.get('dates') if isinstance(row.get('dates'), dict) else {}
        bullets = _coerce_string_list(row.get('bullets'))
        project_rows.append(
            {
                'name': _to_text(row.get('name')),
                'link': _to_text(row.get('link')),
                'start': _to_text(dates.get('start')),
                'end': _to_text(dates.get('end')),
                'section': _to_text(row.get('section') or 'projects'),
                'tech': _coerce_string_list(row.get('tech')),
                'bullets': bullets,
            }
        )

    categories_raw: Dict[str, Any] = {}
    skills = payload.get('skills')
    if isinstance(skills, dict) and isinstance(skills.get('categories'), dict):
        categories_raw = skills.get('categories', {})
    skill_categories = [
        {'name': _to_text(name), 'values': _coerce_string_list(values)}
        for name, values in categories_raw.items()
        if _to_text(name)
    ]

    return {
        'schema_version': _to_text(payload.get('schema_version') or '1.1.0') or '1.1.0',
        'identity': {
            'name': _to_text(identity.get('name')),
            'email': _to_text(identity.get('email')),
            'phone': _to_text(identity.get('phone')),
            'location': _to_text(identity.get('location')),
            'links': identity_links,
        },
        'summary': _to_text(payload.get('summary')),
        'education': education_rows,
        'experience': experience_rows,
        'projects': project_rows,
        'skill_categories': skill_categories,
        'certifications': _coerce_string_list(payload.get('certifications')),
        'awards': _coerce_string_list(payload.get('awards')),
    }


def _parse_resume_form_payload(form: Any) -> Dict[str, Any]:
    identity = {
        'name': _form_value(form, 'identity_name'),
        'email': _form_value(form, 'identity_email'),
        'phone': _form_value(form, 'identity_phone'),
        'location': _form_value(form, 'identity_location'),
        'links': [],
    }
    for index in _extract_indexes(form.keys(), r'^identity_link_(\d+)$'):
        value = _form_value(form, f'identity_link_{index}')
        if value:
            identity['links'].append(value)

    education_rows: list[Dict[str, Any]] = []
    for index in _extract_indexes(form.keys(), r'^education_(\d+)_'):
        school = _form_value(form, f'education_{index}_school')
        degree = _form_value(form, f'education_{index}_degree')
        major = _form_value(form, f'education_{index}_major')
        gpa = _form_value(form, f'education_{index}_gpa')
        start = _form_value(form, f'education_{index}_start')
        end = _form_value(form, f'education_{index}_end')
        minors = _split_multivalue(_form_value(form, f'education_{index}_minors'))
        coursework = _split_multivalue(_form_value(form, f'education_{index}_coursework'))
        if not _has_nonempty(school, degree, major, gpa, start, end, minors, coursework):
            continue
        education_rows.append(
            {
                'school': school,
                'degree': degree,
                'major': major,
                'minors': minors,
                'gpa': gpa,
                'dates': {'start': start, 'end': end},
                'coursework': coursework,
            }
        )

    experience_rows: list[Dict[str, Any]] = []
    for index in _extract_indexes(form.keys(), r'^experience_(\d+)_'):
        company = _form_value(form, f'experience_{index}_company')
        title = _form_value(form, f'experience_{index}_title')
        location = _form_value(form, f'experience_{index}_location')
        start = _form_value(form, f'experience_{index}_start')
        end = _form_value(form, f'experience_{index}_end')
        bullets: list[str] = []
        bullet_indexes = _extract_indexes(form.keys(), rf'^experience_{index}_bullet_(\d+)$')
        for bullet_index in bullet_indexes:
            bullet = _form_value(form, f'experience_{index}_bullet_{bullet_index}')
            if bullet:
                bullets.append(bullet)
        if not _has_nonempty(company, title, location, start, end, bullets):
            continue
        experience_rows.append(
            {
                'company': company,
                'title': title,
                'location': location,
                'dates': {'start': start, 'end': end},
                'bullets': bullets,
            }
        )

    project_rows: list[Dict[str, Any]] = []
    for index in _extract_indexes(form.keys(), r'^project_(\d+)_'):
        name = _form_value(form, f'project_{index}_name')
        link = _form_value(form, f'project_{index}_link')
        start = _form_value(form, f'project_{index}_start')
        end = _form_value(form, f'project_{index}_end')
        section = _form_value(form, f'project_{index}_section') or 'projects'
        tech = _split_multivalue(_form_value(form, f'project_{index}_tech'))
        bullets: list[str] = []
        bullet_indexes = _extract_indexes(form.keys(), rf'^project_{index}_bullet_(\d+)$')
        for bullet_index in bullet_indexes:
            bullet = _form_value(form, f'project_{index}_bullet_{bullet_index}')
            if bullet:
                bullets.append(bullet)
        if not _has_nonempty(name, link, start, end, tech, bullets):
            continue
        project_payload: Dict[str, Any] = {
            'name': name,
            'link': link or None,
            'tech': tech,
            'bullets': bullets,
            'section': section,
        }
        if _has_nonempty(start, end):
            project_payload['dates'] = {'start': start, 'end': end}
        project_rows.append(project_payload)

    categories: Dict[str, list[str]] = {}
    for index in _extract_indexes(form.keys(), r'^skill_category_(\d+)_'):
        name = _form_value(form, f'skill_category_{index}_name')
        values = _split_multivalue(_form_value(form, f'skill_category_{index}_values'))
        if not name:
            continue
        categories[name] = values

    return {
        'schema_version': _form_value(form, 'schema_version') or '1.1.0',
        'identity': identity,
        'summary': _form_value(form, 'summary') or None,
        'education': education_rows,
        'experience': experience_rows,
        'projects': project_rows,
        'skills': {'categories': categories},
        'certifications': _split_multivalue(_form_value(form, 'certifications_text')),
        'awards': _split_multivalue(_form_value(form, 'awards_text')),
    }


def _default_vault_item_payload() -> Dict[str, Any]:
    return _vault_item_to_form_payload(
        {
            'type': 'project',
            'title': '',
            'dates': {'start': '', 'end': ''},
            'tags': [],
            'tech': [],
            'bullets': [{'text': ''}],
            'links': [],
            'source_artifacts': [],
        }
    )


def _vault_item_to_form_payload(item: Optional[Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if isinstance(item, VaultItem):
        payload = item.model_dump(exclude_none=True, mode='json')
    elif isinstance(item, dict):
        payload = item

    dates = payload.get('dates') if isinstance(payload.get('dates'), dict) else {}
    bullets: list[Dict[str, Any]] = []
    for row in payload.get('bullets') or []:
        if isinstance(row, str):
            row = {'text': row}
        if not isinstance(row, dict):
            continue
        bullets.append(
            {
                'text': _to_text(row.get('text')),
                'situation': _to_text(row.get('situation')),
                'task': _to_text(row.get('task')),
                'action': _to_text(row.get('action')),
                'outcome': _to_text(row.get('outcome')),
                'impact': _to_text(row.get('impact')),
            }
        )
    if not bullets:
        bullets = [{'text': '', 'situation': '', 'task': '', 'action': '', 'outcome': '', 'impact': ''}]

    return {
        'type': _to_text(payload.get('type') or 'project'),
        'title': _to_text(payload.get('title')),
        'start': _to_text(dates.get('start')),
        'end': _to_text(dates.get('end')),
        'tags': _coerce_string_list(payload.get('tags')),
        'tech': _coerce_string_list(payload.get('tech')),
        'bullets': bullets,
        'links': _coerce_string_list(payload.get('links')),
        'source_artifacts': _coerce_string_list(payload.get('source_artifacts')),
    }


def _parse_vault_form_payload(form: Any) -> Dict[str, Any]:
    item_type = _form_value(form, 'type') or 'project'
    title = _form_value(form, 'title')
    start = _form_value(form, 'start')
    end = _form_value(form, 'end')
    tags = _split_multivalue(_form_value(form, 'tags_text'))
    tech = _split_multivalue(_form_value(form, 'tech_text'))
    links = _split_multivalue(_form_value(form, 'links_text'))
    source_artifacts = _split_multivalue(_form_value(form, 'source_artifacts_text'))

    bullets: list[Dict[str, Any]] = []
    for index in _extract_indexes(form.keys(), r'^bullet_(\d+)_'):
        text = _form_value(form, f'bullet_{index}_text')
        situation = _form_value(form, f'bullet_{index}_situation')
        task = _form_value(form, f'bullet_{index}_task')
        action = _form_value(form, f'bullet_{index}_action')
        outcome = _form_value(form, f'bullet_{index}_outcome')
        impact = _form_value(form, f'bullet_{index}_impact')
        if not _has_nonempty(text, situation, task, action, outcome, impact):
            continue
        bullet_payload: Dict[str, Any] = {'text': text}
        if situation:
            bullet_payload['situation'] = situation
        if task:
            bullet_payload['task'] = task
        if action:
            bullet_payload['action'] = action
        if outcome:
            bullet_payload['outcome'] = outcome
        if impact:
            bullet_payload['impact'] = impact
        bullets.append(bullet_payload)

    payload: Dict[str, Any] = {
        'type': item_type,
        'title': title,
        'tags': tags,
        'tech': tech,
        'bullets': bullets,
        'links': links,
        'source_artifacts': source_artifacts,
    }
    if _has_nonempty(start, end):
        payload['dates'] = {'start': start, 'end': end}
    return payload


def _clean_payload_text(value: Any) -> str:
    if value is None:
        return ''
    return str(value).strip()


def datetime_now_stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')


@app.get('/healthz')
async def healthcheck() -> Dict[str, str]:
    return {'status': 'ok'}


def _writable_dir_check(path: Path) -> tuple[bool, Optional[str]]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f'.ready-{uuid.uuid4().hex}'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink(missing_ok=True)
        return True, None
    except Exception as exc:
        return False, str(exc)


@app.get('/readyz')
async def readiness() -> JSONResponse:
    checks: Dict[str, Dict[str, Any]] = {}
    status_ok = True

    writable_targets = {
        'data_dir': settings.data_dir,
        'uploads_dir': settings.data_dir / 'uploads',
        'outputs_dir': settings.data_dir / 'outputs',
    }
    for name, path in writable_targets.items():
        ok, error = _writable_dir_check(path)
        checks[name] = {'ok': ok}
        if error:
            checks[name]['error'] = error
            status_ok = False

    checks['templates_dir'] = {'ok': settings.templates_dir.exists()}
    if not checks['templates_dir']['ok']:
        checks['templates_dir']['error'] = f'Missing path: {settings.templates_dir}'
        status_ok = False

    static_dir = Path('app/static')
    checks['static_dir'] = {'ok': static_dir.exists()}
    if not checks['static_dir']['ok']:
        checks['static_dir']['error'] = f'Missing path: {static_dir}'
        status_ok = False

    sqlite_ok = True
    sqlite_error: Optional[str] = None
    try:
        connection = sqlite3.connect(settings.resolved_sqlite_path)
        try:
            connection.execute('SELECT 1')
        finally:
            connection.close()
    except Exception as exc:
        sqlite_ok = False
        sqlite_error = str(exc)
        status_ok = False
    checks['sqlite'] = {'ok': sqlite_ok}
    if sqlite_error:
        checks['sqlite']['error'] = sqlite_error

    payload = {'status': 'ok' if status_ok else 'degraded', 'checks': checks}
    return JSONResponse(status_code=200 if status_ok else 503, content=payload)
