from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import Settings, get_settings
from app.models import CanonicalResume, JobRecord, JobSelectionFeedback, TailorMode, VaultItem
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
from app.services.latex import LatexRenderError, LatexService
from app.services.llm import LLMService, LLMUnavailableError
from app.services.repository import DataRepository
from app.services.scraper import ScrapeError, scrape_job_posting
from app.services.tailoring import expand_resume_with_projects, tailor_resume, tighten_resume_for_one_page
from app.services.vault_sync import sync_base_resume_to_vault
from app.services.vault_ingest import VaultIngestError, parse_uploaded_text, parse_vault_source_text
from app.storage import load_json, save_json
from app.utils import ensure_within, normalize_token

settings: Settings = get_settings()
repository = DataRepository(settings)
latex_service = LatexService(settings.templates_dir)

llm_init_error: Optional[str] = None
try:
    llm_service = LLMService(settings.openai_api_key, settings.openai_model)
except LLMUnavailableError as exc:
    llm_service = None
    llm_init_error = str(exc)

app = FastAPI(title='Local Resume Tailor', version='1.0.0')
app.mount('/static', StaticFiles(directory='app/static'), name='static')
templates = Jinja2Templates(directory='app/templates')


def render(request: Request, template_name: str, context: Dict[str, Any]) -> HTMLResponse:
    base_context = {
        'request': request,
        'llm_available': bool(llm_service and llm_service.available),
        'token_header': settings.request_token_header,
        'llm_init_error': llm_init_error,
    }
    base_context.update(context)
    return templates.TemplateResponse(template_name, base_context)


@app.middleware('http')
async def require_token_header(request: Request, call_next):
    required_token = settings.resume_app_token
    if required_token and request.method.upper() == 'POST':
        provided_token = request.headers.get(settings.request_token_header)
        if provided_token != required_token:
            return JSONResponse(
                status_code=401,
                content={
                    'error': 'Missing or invalid token header for POST request.',
                    'required_header': settings.request_token_header,
                },
            )
    return await call_next(request)


def _generate_page_context(*, jd_text: str = '', error: Optional[str] = None, warnings: Optional[list[str]] = None) -> Dict[str, Any]:
    return {
        'jd_text': jd_text,
        'error': error,
        'warnings': warnings or [],
        'base_resume_exists': repository.load_base_resume() is not None,
        'vault_count': len(repository.list_vault_items()),
    }


@app.get('/', response_class=HTMLResponse)
async def generate_page(request: Request) -> HTMLResponse:
    return render(request, 'generate.html', _generate_page_context())


@app.post('/generate', response_class=HTMLResponse)
async def generate_tailored_resume(
    request: Request,
    resume_file: UploadFile = File(...),
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

    if not resume_file.filename:
        return render(
            request,
            'generate.html',
            _generate_page_context(jd_text=pasted_jd_text, error='Resume file is required.'),
        )

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
    upload_path.write_bytes(await resume_file.read())

    try:
        upload_result = public_upload_resume(upload_path, enable_ocr=settings.enable_ocr, llm=llm_service)
        canonical_payload = upload_result.get('canonical') or upload_result.get('parse_mirror', {}).get('canonical')
        canonical = CanonicalResume.model_validate(canonical_payload)
    except Exception as exc:
        return render(
            request,
            'generate.html',
            _generate_page_context(jd_text=pasted_jd_text, error=f'Resume processing failed: {exc}'),
        )

    try:
        repository.save_base_resume(canonical)
        sync_base_resume_to_vault(repository, canonical)
    except Exception as exc:
        return render(
            request,
            'generate.html',
            _generate_page_context(jd_text=pasted_jd_text, error=f'Failed to save base resume: {exc}'),
        )

    job_id = uuid.uuid4().hex[:12]
    job = JobRecord(job_id=job_id, title='Pasted Job Description', company=None, url=None)
    repository.save_job(job, pasted_jd_text)

    workflow = _run_tailoring_workflow(
        base_resume=canonical,
        job_id=job_id,
        jd_text=pasted_jd_text,
        mode=TailorMode.HARD_TRUTH,
        target_score=82.0,
        max_passes=5,
        job_title_hint=job.title,
        feedback_payload=_empty_feedback_payload(),
    )

    extraction_warnings = [str(w).strip() for w in upload_result.get('extraction_warnings', []) if str(w).strip()]
    return render(
        request,
        'tailor_result.html',
        _tailor_result_context(job=job, mode=TailorMode.HARD_TRUTH, workflow=workflow, prepended_warnings=extraction_warnings),
    )


@app.get('/advance', response_class=HTMLResponse)
async def advance_page(request: Request) -> HTMLResponse:
    base_resume = repository.load_base_resume()
    vault_items = repository.list_vault_items()
    jobs = repository.list_jobs()
    return render(
        request,
        'advanced.html',
        {
            'base_resume_exists': base_resume is not None,
            'vault_count': len(vault_items),
            'job_count': len(jobs),
        },
    )


@app.get('/advanced')
async def advanced_redirect() -> RedirectResponse:
    return RedirectResponse(url='/advance', status_code=307)


@app.get('/advance/dashboard', response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    base_resume = repository.load_base_resume()
    vault_items = repository.list_vault_items()
    jobs = repository.list_jobs()
    return render(
        request,
        'dashboard.html',
        {
            'base_resume_exists': base_resume is not None,
            'vault_count': len(vault_items),
            'job_count': len(jobs),
        },
    )


@app.get('/advanced/dashboard')
async def advanced_dashboard_redirect() -> RedirectResponse:
    return RedirectResponse(url='/advance/dashboard', status_code=307)


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
    upload_path.write_bytes(await resume_file.read())

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
                'error': f'Resume audit failed: {exc}',
                'result': None,
            },
        )

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
        output_root = ensure_within(settings.data_dir / 'outputs', settings.data_dir / 'outputs' / output_job_id)
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
    existing_yaml = None
    if base_resume:
        existing_yaml = yaml.safe_dump(base_resume.model_dump(exclude_none=True, mode='json'), sort_keys=False)
    return render(
        request,
        'resume_upload.html',
        {
            'warnings': [],
            'error': None,
            'raw_text': '',
            'canonical_yaml': existing_yaml,
        },
    )


@app.post('/resume/upload', response_class=HTMLResponse)
async def resume_upload_extract(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    warnings = []
    if not file.filename:
        return render(
            request,
            'resume_upload.html',
            {'warnings': warnings, 'error': 'No file provided.', 'raw_text': '', 'canonical_yaml': ''},
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {'.pdf', '.docx'}:
        return render(
            request,
            'resume_upload.html',
            {
                'warnings': warnings,
                'error': 'Unsupported file type. Upload PDF or DOCX.',
                'raw_text': '',
                'canonical_yaml': '',
            },
        )

    upload_path = settings.data_dir / 'uploads' / f'{uuid.uuid4().hex}{suffix}'
    content = await file.read()
    upload_path.write_bytes(content)

    try:
        raw_text, extract_warnings = extract_resume_text(upload_path, enable_ocr=settings.enable_ocr)
        warnings.extend(extract_warnings)
        canonical, canonical_warnings = canonicalize_resume_text(raw_text, llm_service)
        warnings.extend(canonical_warnings)
        canonical_yaml = yaml.safe_dump(canonical.model_dump(exclude_none=True, mode='json'), sort_keys=False)
    except Exception as exc:
        return render(
            request,
            'resume_upload.html',
            {
                'warnings': warnings,
                'error': f'Extraction failed: {exc}',
                'raw_text': '',
                'canonical_yaml': '',
            },
        )

    return render(
        request,
        'resume_upload.html',
        {'warnings': warnings, 'error': None, 'raw_text': raw_text, 'canonical_yaml': canonical_yaml},
    )


@app.post('/resume/save')
async def resume_save(canonical_yaml: str = Form(...)):
    try:
        parsed = yaml.safe_load(canonical_yaml)
        if not isinstance(parsed, dict):
            raise ValueError('YAML root must be an object.')
        canonical = CanonicalResume.model_validate(parsed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid canonical YAML: {exc}') from exc

    try:
        repository.save_base_resume(canonical)
        sync_base_resume_to_vault(repository, canonical)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Failed to save/sync resume: {exc}') from exc

    return RedirectResponse(url='/', status_code=303)


@app.post('/resume/sync-vault')
async def resume_sync_vault():
    base_resume = repository.load_base_resume()
    if not base_resume:
        raise HTTPException(status_code=400, detail='Base resume missing. Save base resume first.')
    try:
        sync_base_resume_to_vault(repository, base_resume)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Vault sync failed: {exc}') from exc
    return RedirectResponse(url='/vault', status_code=303)


@app.get('/vault', response_class=HTMLResponse)
async def vault_list(request: Request) -> HTMLResponse:
    items = repository.list_vault_items()
    return render(request, 'vault_list.html', {'items': items})


def _default_vault_yaml() -> str:
    payload = {
        'type': 'project',
        'title': 'New Vault Item',
        'dates': {'start': '', 'end': ''},
        'tags': [],
        'tech': [],
        'bullets': [{'text': ''}],
        'links': [],
        'source_artifacts': [],
    }
    return yaml.safe_dump(payload, sort_keys=False)


@app.get('/vault/new', response_class=HTMLResponse)
async def vault_new_page(request: Request) -> HTMLResponse:
    return render(
        request,
        'vault_form.html',
        {'item_id': None, 'item_yaml': _default_vault_yaml(), 'error': None},
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
            'parsed_yaml': None,
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
    parsed_yaml: Optional[str] = None
    raw_segments = []
    raw_text_preview = ''
    upload_path: Optional[Path] = None

    source_text = source_text.strip()
    if source_text:
        raw_segments.append(source_text)

    if file and file.filename:
        suffix = Path(file.filename).suffix.lower()
        upload_path = settings.data_dir / 'uploads' / f'{uuid.uuid4().hex}{suffix}'
        upload_path.write_bytes(await file.read())
        try:
            file_text, file_warnings = parse_uploaded_text(upload_path, enable_ocr=settings.enable_ocr)
            warnings.extend(file_warnings)
            if file_text:
                raw_segments.append(file_text)
        except VaultIngestError as exc:
            return render(
                request,
                'vault_ingest.html',
                {
                    'error': str(exc),
                    'warnings': warnings,
                    'source_text': source_text,
                    'type_hint': type_hint,
                    'parsed_yaml': parsed_yaml,
                    'raw_text_preview': raw_text_preview,
                },
            )

    if not raw_segments:
        return render(
            request,
            'vault_ingest.html',
            {
                'error': 'Provide notes/text or upload a file to parse.',
                'warnings': warnings,
                'source_text': source_text,
                'type_hint': type_hint,
                'parsed_yaml': parsed_yaml,
                'raw_text_preview': raw_text_preview,
            },
        )

    combined_text = '\n\n'.join(raw_segments).strip()
    raw_text_preview = combined_text[:6000]

    try:
        item, parse_warnings = parse_vault_source_text(combined_text, llm=llm_service, type_hint=type_hint)
        warnings.extend(parse_warnings)
        if upload_path:
            artifacts = list(item.source_artifacts)
            path_text = str(upload_path)
            if path_text not in artifacts:
                artifacts.append(path_text)
            item = item.model_copy(update={'source_artifacts': artifacts})
        parsed_yaml = yaml.safe_dump(item.model_dump(exclude_none=True, mode='json'), sort_keys=False)
    except Exception as exc:
        return render(
            request,
            'vault_ingest.html',
            {
                'error': f'Vault parsing failed: {exc}',
                'warnings': warnings,
                'source_text': source_text,
                'type_hint': type_hint,
                'parsed_yaml': parsed_yaml,
                'raw_text_preview': raw_text_preview,
            },
        )

    return render(
        request,
        'vault_ingest.html',
        {
            'error': None,
            'warnings': warnings,
            'source_text': source_text,
            'type_hint': type_hint,
            'parsed_yaml': parsed_yaml,
            'raw_text_preview': raw_text_preview,
        },
    )


@app.post('/vault/new', response_class=HTMLResponse)
async def vault_new(request: Request, item_yaml: str = Form(...)) -> HTMLResponse:
    try:
        parsed = yaml.safe_load(item_yaml)
        if not isinstance(parsed, dict):
            raise ValueError('Vault YAML root must be an object.')
        item = VaultItem.model_validate(parsed)
        item_id = uuid.uuid4().hex
        repository.save_vault_item(item_id, item)
        return RedirectResponse(url='/vault', status_code=303)
    except Exception as exc:
        return render(request, 'vault_form.html', {'item_id': None, 'item_yaml': item_yaml, 'error': str(exc)})


@app.get('/vault/{item_id}', response_class=HTMLResponse)
async def vault_edit_page(request: Request, item_id: str) -> HTMLResponse:
    item = repository.get_vault_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail='Vault item not found')
    item_yaml = yaml.safe_dump(item.model_dump(exclude_none=True, mode='json'), sort_keys=False)
    return render(request, 'vault_form.html', {'item_id': item_id, 'item_yaml': item_yaml, 'error': None})


@app.post('/vault/{item_id}', response_class=HTMLResponse)
async def vault_edit(request: Request, item_id: str, item_yaml: str = Form(...)) -> HTMLResponse:
    try:
        parsed = yaml.safe_load(item_yaml)
        if not isinstance(parsed, dict):
            raise ValueError('Vault YAML root must be an object.')
        item = VaultItem.model_validate(parsed)
        repository.save_vault_item(item_id, item)
        return RedirectResponse(url='/vault', status_code=303)
    except Exception as exc:
        return render(request, 'vault_form.html', {'item_id': item_id, 'item_yaml': item_yaml, 'error': str(exc)})


@app.post('/vault/{item_id}/delete')
async def vault_delete(item_id: str):
    item = repository.get_vault_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail='Vault item not found')
    repository.delete_vault_item(item_id)
    return RedirectResponse(url='/vault', status_code=303)


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

    return RedirectResponse(url=f'/jobs/{job_id}', status_code=303)


@app.get('/jobs/{job_id}', response_class=HTMLResponse)
async def jobs_detail(request: Request, job_id: str) -> HTMLResponse:
    job = repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    jd_text = repository.get_job_text(job_id)
    feedback = repository.get_job_feedback(job_id)
    feedback_payload = feedback.model_dump(mode='json') if feedback else {'preferred_titles': [], 'blocked_titles': []}

    output_root = settings.data_dir / 'outputs' / job_id
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
            'feedback': feedback_payload,
        },
    )


@app.post('/jobs/{job_id}/jd')
async def jobs_update_jd(job_id: str, jd_text: str = Form(...)):
    if not repository.get_job(job_id):
        raise HTTPException(status_code=404, detail='Job not found')
    repository.update_job_text(job_id, jd_text)
    return RedirectResponse(url=f'/jobs/{job_id}', status_code=303)


@app.post('/jobs/{job_id}/feedback')
async def jobs_update_feedback(
    job_id: str,
    preferred_titles: str = Form(default=''),
    blocked_titles: str = Form(default=''),
):
    if not repository.get_job(job_id):
        raise HTTPException(status_code=404, detail='Job not found')

    preferred = _parse_feedback_titles(preferred_titles)
    blocked = _parse_feedback_titles(blocked_titles)
    blocked_ids = {normalize_token(value) for value in blocked if normalize_token(value)}
    preferred = [value for value in preferred if normalize_token(value) not in blocked_ids]

    feedback = JobSelectionFeedback(preferred_titles=preferred, blocked_titles=blocked)
    repository.save_job_feedback(job_id, feedback)
    return RedirectResponse(url=f'/jobs/{job_id}', status_code=303)


def _empty_feedback_payload() -> Dict[str, list[str]]:
    return {'preferred_titles': [], 'blocked_titles': []}


def _run_tailoring_workflow(
    *,
    base_resume: CanonicalResume,
    job_id: str,
    jd_text: str,
    mode: TailorMode,
    target_score: float,
    max_passes: int,
    job_title_hint: Optional[str],
    feedback_payload: Dict[str, Any],
) -> Dict[str, Any]:
    assert llm_service and llm_service.available
    vault_items = repository.list_vault_items()

    clamped_target_score = max(0.0, min(100.0, float(target_score)))
    clamped_max_passes = max(1, min(5, int(max_passes)))

    best_tailored = None
    best_score = -1.0
    best_pass = 1
    passes_executed = 0

    for optimization_pass in range(1, clamped_max_passes + 1):
        pass_tailored = tailor_resume(
            base_resume=base_resume,
            vault_items=vault_items,
            jd_text=jd_text,
            mode=mode,
            llm=llm_service,
            job_title_hint=job_title_hint,
            selection_feedback=feedback_payload,
            optimization_level=optimization_pass,
        )
        pass_match = compute_match_score(pass_tailored.tailored_resume, jd_text)
        pass_score = float(pass_match.get('overall_score', 0.0) or 0.0)
        pass_tailored.report.warnings.append(
            f'Optimization pass {optimization_pass}/{clamped_max_passes} match score: {pass_score:.2f}.'
        )
        passes_executed = optimization_pass

        if pass_score > best_score:
            best_tailored = pass_tailored
            best_score = pass_score
            best_pass = optimization_pass

        if pass_score >= clamped_target_score:
            break

    assert best_tailored is not None
    tailored = best_tailored
    optimization_reached_target = best_score >= clamped_target_score
    if not optimization_reached_target:
        tailored.report.warnings.append(
            f'Target match score {clamped_target_score:.2f} not reached after {passes_executed} passes. Best score: {best_score:.2f}.'
        )
    if best_pass < passes_executed:
        tailored.report.warnings.append(
            f'Using best resume from pass {best_pass}/{passes_executed} with score {best_score:.2f}.'
        )

    output_dir = repository.create_output_dir(job_id)
    compile_error: Optional[str] = None
    pdf_exists = False
    ats_pdf_exists = False
    ats_docx_exists = False
    ats_txt_exists = False
    resume_to_render = tailored.tailored_resume
    optimization_match_score = best_score
    score_lookup = {
        normalize_token(item.title): item.score
        for item in tailored.report.chosen_items
        if normalize_token(item.title)
    }
    used_expansions: set[str] = set()
    tighten_level = 0
    expand_attempts = 0
    max_expand_attempts = 8
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
        'job_id': job_id,
        'timestamp': output_dir.name,
        'pdf_exists': pdf_exists,
        'ats_pdf_exists': ats_pdf_exists,
        'ats_docx_exists': ats_docx_exists,
        'ats_txt_exists': ats_txt_exists,
        'compile_error': compile_error,
        'match_score': optimization_match_score,
        'target_score': clamped_target_score,
        'passes_used': passes_executed,
        'max_passes': clamped_max_passes,
    }


def _tailor_result_context(
    *,
    job: JobRecord,
    mode: TailorMode,
    workflow: Dict[str, Any],
    prepended_warnings: Optional[list[str]] = None,
) -> Dict[str, Any]:
    warnings: list[str] = []
    seen: set[str] = set()
    for warning in [*(prepended_warnings or []), *workflow['warnings']]:
        cleaned = str(warning).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        warnings.append(cleaned)
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
        'target_score': workflow['target_score'],
        'passes_used': workflow['passes_used'],
        'max_passes': workflow['max_passes'],
        'chosen_items': workflow['chosen_items'],
        'keywords_covered': workflow['keywords_covered'],
        'keywords_missed': workflow['keywords_missed'],
    }


@app.post('/jobs/{job_id}/tailor', response_class=HTMLResponse)
async def jobs_tailor(
    request: Request,
    job_id: str,
    mode: TailorMode = Form(...),
    target_score: float = Form(default=82.0),
    max_passes: int = Form(default=5),
) -> HTMLResponse:
    job = repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')

    feedback = repository.get_job_feedback(job_id)
    feedback_payload = feedback.model_dump(mode='json') if feedback else _empty_feedback_payload()

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
                'feedback': feedback_payload,
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
                'feedback': feedback_payload,
            },
        )

    jd_text = repository.get_job_text(job_id)
    workflow = _run_tailoring_workflow(
        base_resume=base_resume,
        job_id=job_id,
        jd_text=jd_text,
        mode=mode,
        target_score=target_score,
        max_passes=max_passes,
        job_title_hint=job.title,
        feedback_payload=feedback_payload,
    )

    return render(
        request,
        'tailor_result.html',
        _tailor_result_context(job=job, mode=mode, workflow=workflow),
    )


@app.get('/outputs/{job_id}/{timestamp}/resume.pdf')
async def download_pdf(job_id: str, timestamp: str):
    output_root = settings.data_dir / 'outputs'
    target = ensure_within(output_root, output_root / job_id / timestamp / 'resume.pdf')
    if not target.exists():
        raise HTTPException(status_code=404, detail='Output PDF not found')
    return FileResponse(path=target, media_type='application/pdf', filename='resume.pdf')


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
    output_root = settings.data_dir / 'outputs'
    target = ensure_within(output_root, output_root / job_id / timestamp / artifact)
    if not target.exists():
        raise HTTPException(status_code=404, detail='Artifact not found')
    return FileResponse(path=target, media_type=allowed[artifact], filename=artifact)


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
    output_root = ensure_within(settings.data_dir / 'outputs', settings.data_dir / 'outputs' / normalize_token(job_id or 'manual'))
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime_now_stamp()
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = _clean_payload_text(payload.get('filename_prefix'))
    render_result = public_render_outputs(resume, output_dir, filename_prefix=prefix)
    if not render_result['pdf_text_layer']['ok']:
        raise HTTPException(status_code=500, detail='Rendered PDF does not contain enough extractable text.')
    bundle_path = public_export_bundle(output_dir, bundle_path=output_dir / 'bundle.zip')
    return JSONResponse(content={'output_dir': str(output_dir), 'bundle_path': str(bundle_path), **render_result})


@app.get('/api/export_bundle/{job_id}/{timestamp}')
async def export_bundle(job_id: str, timestamp: str):
    output_root = settings.data_dir / 'outputs'
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


def _clean_payload_text(value: Any) -> str:
    if value is None:
        return ''
    return str(value).strip()


def _parse_feedback_titles(value: str) -> list[str]:
    raw = (value or '').replace(',', '\n').replace(';', '\n')
    seen: set[str] = set()
    output: list[str] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        token = normalize_token(cleaned)
        if not token or token in seen:
            continue
        seen.add(token)
        output.append(cleaned)
    return output


def datetime_now_stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')


@app.get('/healthz')
async def healthcheck() -> Dict[str, str]:
    return {'status': 'ok'}
