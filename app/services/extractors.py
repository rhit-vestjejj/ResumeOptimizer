from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
from docx import Document

from app.models import CanonicalResume, Identity, Skills
from app.services.llm import LLMService


def extract_text_from_pdf(path: Path, enable_ocr: bool = False, min_chars: int = 120) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    pages: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append((page.extract_text() or '').strip())

    text = '\n'.join(page for page in pages if page).strip()
    if len(text) >= min_chars:
        return text, warnings

    if not enable_ocr:
        warnings.append('PDF text extraction was short and OCR is disabled (ENABLE_OCR=1 to enable).')
        return text, warnings

    ocr_text, ocr_warning = _ocr_pdf(path)
    if ocr_warning:
        warnings.append(ocr_warning)
    if len(ocr_text) > len(text):
        text = ocr_text
    return text, warnings


def _ocr_pdf(path: Path) -> Tuple[str, Optional[str]]:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception as exc:  # pragma: no cover - dependency availability varies
        return '', f'OCR dependencies unavailable: {exc}'

    try:
        images = convert_from_path(str(path))
        chunks = [pytesseract.image_to_string(image) for image in images]
        return '\n'.join(chunks).strip(), None
    except Exception as exc:  # pragma: no cover - external binaries vary
        return '', f'OCR failed: {exc}'


def extract_text_from_docx(path: Path) -> str:
    doc = Document(str(path))
    return '\n'.join(paragraph.text for paragraph in doc.paragraphs).strip()


def extract_resume_text(path: Path, enable_ocr: bool = False) -> Tuple[str, List[str]]:
    suffix = path.suffix.lower()
    if suffix == '.pdf':
        return extract_text_from_pdf(path, enable_ocr=enable_ocr)
    if suffix == '.docx':
        return extract_text_from_docx(path), []
    raise ValueError('Unsupported file type. Use PDF or DOCX.')


def canonicalize_resume_text(raw_text: str, llm: Optional[LLMService]) -> Tuple[CanonicalResume, List[str]]:
    warnings: List[str] = []
    if llm and llm.available:
        try:
            return llm.extract_canonical_resume(raw_text), warnings
        except Exception as exc:
            warnings.append(f'LLM extraction failed; using heuristic parser: {exc}')

    warnings.append('Using heuristic parser; review YAML carefully before saving.')
    return heuristic_resume_to_canonical(raw_text), warnings


def heuristic_resume_to_canonical(raw_text: str) -> CanonicalResume:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    text_blob = '\n'.join(lines)

    email_match = re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text_blob)
    phone_match = re.search(r'(\+?\d[\d\s\-\(\)]{7,}\d)', text_blob)
    link_matches = re.findall(r'https?://[^\s)]+', text_blob)

    name = lines[0] if lines else 'Unknown Name'
    location = ''
    for line in lines[1:6]:
        if ',' in line and len(line) < 80:
            location = line
            break

    identity = Identity(
        name=name,
        email=email_match.group(0) if email_match else '',
        phone=phone_match.group(0) if phone_match else '',
        location=location,
        links=link_matches,
    )

    summary = None
    if len(lines) > 2:
        summary_candidates = [line for line in lines[1:8] if len(line.split()) > 5]
        if summary_candidates:
            summary = summary_candidates[0]

    return CanonicalResume(
        identity=identity,
        summary=summary,
        education=[],
        experience=[],
        projects=[],
        skills=Skills(categories={}),
        certifications=[],
        awards=[],
    )
