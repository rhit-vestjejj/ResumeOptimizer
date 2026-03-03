from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from jinja2 import Environment, FileSystemLoader

from app.models import CanonicalResume
from app.utils import latex_escape


class LatexRenderError(RuntimeError):
    pass


class LatexService:
    def __init__(self, templates_dir: Path) -> None:
        self.env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.env.filters['latex_escape'] = latex_escape

    def render_resume(self, resume: CanonicalResume, output_dir: Path, template_name: str = 'resume.tex.j2') -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        template = self.env.get_template(template_name)
        tex_content = template.render(resume=_sanitize_resume_for_render(resume))
        tex_path = output_dir / 'resume.tex'
        tex_path.write_text(tex_content, encoding='utf-8')
        return tex_path

    def compile_resume(self, output_dir: Path, mock_compile: bool = False) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        tex_path = output_dir / 'resume.tex'
        pdf_path = output_dir / 'resume.pdf'

        if mock_compile or os.getenv('RESUME_MOCK_COMPILE') == '1':
            pdf_path.write_bytes(b'%PDF-1.4\n% mock pdf\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n')
            return pdf_path

        renderer_url = (os.getenv('RENDERER_URL') or '').strip()
        if renderer_url:
            return _compile_resume_remote(renderer_url=renderer_url, tex_path=tex_path, pdf_path=pdf_path)

        latexmk = shutil.which('latexmk')
        if not latexmk:
            raise LatexRenderError('latexmk is not installed or not on PATH.')

        command = [
            latexmk,
            '-pdf',
            '-interaction=nonstopmode',
            '-halt-on-error',
            '-file-line-error',
            str(tex_path.name),
        ]

        result = subprocess.run(
            command,
            cwd=output_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            log_excerpt = _extract_latex_error_excerpt(output_dir / 'resume.log')
            process_excerpt = (result.stderr or result.stdout)[-2000:]
            details = log_excerpt or process_excerpt
            raise LatexRenderError(f'LaTeX compile failed:\n{details}')

        if not pdf_path.exists():
            raise LatexRenderError('LaTeX compile did not produce resume.pdf')
        return pdf_path

    def count_pdf_pages(self, pdf_path: Path) -> int:
        if not pdf_path.exists():
            return 0
        try:
            import pdfplumber
        except Exception:
            return 1
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except Exception:
            return 1


def _extract_latex_error_excerpt(log_path: Path) -> str:
    if not log_path.exists():
        return ''

    lines = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    matches = []
    error_patterns = [
        re.compile(r'^!'),
        re.compile(r'LaTeX Error'),
        re.compile(r'Undefined control sequence'),
        re.compile(r'Misplaced alignment tab character'),
        re.compile(r'Fatal error'),
    ]

    for index, line in enumerate(lines):
        if any(pattern.search(line) for pattern in error_patterns):
            start = max(0, index - 2)
            end = min(len(lines), index + 4)
            excerpt = '\n'.join(lines[start:end])
            matches.append(excerpt)

    if matches:
        return '\n---\n'.join(matches[-3:])

    return '\n'.join(lines[-60:])


def _normalize_space(value: str) -> str:
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def _split_identity_links(raw_link: str) -> list[str]:
    raw = _normalize_space(raw_link)
    if not raw:
        return []
    parts = [segment.strip() for segment in raw.split('|')]
    cleaned_parts = [segment for segment in parts if segment]
    return cleaned_parts or [raw]


def _normalize_link_value(link: str) -> str:
    cleaned = _normalize_space(link)
    if not cleaned:
        return ''
    if cleaned.startswith(('http://', 'https://', 'mailto:')):
        return cleaned
    if cleaned.startswith('www.'):
        return f'https://{cleaned}'
    if re.fullmatch(r'[a-z0-9][a-z0-9.-]+\.[a-z]{2,}(/[^\s]*)?', cleaned.lower()):
        return f'https://{cleaned}'
    return cleaned


def _balanced_parentheses(text: str) -> bool:
    depth = 0
    for char in text:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _dedupe_skill_entries(entries: list[str]) -> list[str]:
    cleaned = [_normalize_space(entry) for entry in entries if _normalize_space(entry)]
    unique: list[str] = []
    seen: set[str] = set()
    for entry in cleaned:
        key = entry.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)

    filtered: list[str] = []
    for entry in unique:
        lowered = entry.lower()
        if not _balanced_parentheses(entry):
            has_balanced_container = any(
                lowered != other.lower()
                and lowered in other.lower()
                and _balanced_parentheses(other)
                for other in unique
            )
            if has_balanced_container:
                continue
        if len(entry) >= 12:
            is_fragment_of_longer = any(
                lowered != other.lower() and lowered in other.lower() and len(other) > len(entry) + 8
                for other in unique
            )
            if is_fragment_of_longer:
                continue
        filtered.append(entry)
    return filtered


def _sanitize_resume_for_render(resume: CanonicalResume) -> CanonicalResume:
    cloned = CanonicalResume.model_validate(resume.model_dump())

    normalized_links: list[str] = []
    seen_links: set[str] = set()
    for raw_link in cloned.identity.links:
        for part in _split_identity_links(raw_link):
            normalized = _normalize_link_value(part)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen_links:
                continue
            seen_links.add(key)
            normalized_links.append(normalized)
    cloned.identity.links = normalized_links[:5]

    sanitized_categories: dict[str, list[str]] = {}
    for category, entries in cloned.skills.categories.items():
        sanitized_categories[category] = _dedupe_skill_entries(list(entries))
    cloned.skills.categories = sanitized_categories
    return cloned


def _compile_resume_remote(*, renderer_url: str, tex_path: Path, pdf_path: Path) -> Path:
    if not tex_path.exists():
        raise LatexRenderError('LaTeX source file not found for remote render.')

    endpoint = f'{renderer_url.rstrip("/")}/render/pdf'
    payload = {'tex': tex_path.read_text(encoding='utf-8'), 'assets': {}}
    request = urllib_request.Request(
        endpoint,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Accept': 'application/pdf',
        },
        method='POST',
    )

    try:
        with urllib_request.urlopen(request, timeout=90) as response:
            pdf_bytes = response.read()
    except urllib_error.HTTPError as exc:
        body = exc.read().decode('utf-8', errors='ignore')
        raise LatexRenderError(f'Remote renderer returned HTTP {exc.code}: {body[-2000:]}')
    except urllib_error.URLError as exc:
        raise LatexRenderError(f'Failed to reach remote renderer {endpoint}: {exc.reason}')
    except Exception as exc:
        raise LatexRenderError(f'Remote renderer request failed: {exc}')

    if not pdf_bytes.startswith(b'%PDF'):
        raise LatexRenderError('Remote renderer response was not a valid PDF payload.')

    pdf_path.write_bytes(pdf_bytes)
    if not pdf_path.exists():
        raise LatexRenderError('Remote renderer did not produce resume.pdf')
    return pdf_path
