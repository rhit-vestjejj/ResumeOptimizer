from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from app.models import DateRange, VaultItem, VaultItemType
from app.services.extractors import extract_resume_text
from app.services.llm import LLMService
from app.utils import tokenize, unique_preserve_order

TECH_CATALOG = {
    'python', 'java', 'c', 'c++', 'c#', 'javascript', 'typescript', 'go', 'rust', 'sql', 'postgresql',
    'mysql', 'mongodb', 'redis', 'fastapi', 'flask', 'django', 'react', 'docker', 'kubernetes', 'aws',
    'gcp', 'azure', 'linux', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'git', 'github',
    'ci', 'cd', 'airflow', 'spark', 'hadoop', 'node', 'nodejs', 'graphql', 'rest', 'grpc', 'html', 'css',
}

STOPWORDS = {
    'and', 'or', 'the', 'a', 'an', 'to', 'for', 'with', 'on', 'in', 'of', 'at', 'as', 'is', 'was', 'were',
    'be', 'been', 'being', 'it', 'this', 'that', 'from', 'by', 'into', 'over', 'under', 'about', 'project',
    'experience', 'work', 'worked', 'built', 'created', 'developed', 'used', 'using', 'team', 'role',
}


class VaultIngestError(RuntimeError):
    pass


def parse_vault_source_text(
    raw_text: str,
    *,
    llm: Optional[LLMService],
    type_hint: Optional[str] = None,
) -> Tuple[VaultItem, List[str]]:
    warnings: List[str] = []
    normalized_hint = _normalize_type_hint(type_hint)

    if llm and llm.available:
        try:
            hint_text = normalized_hint.value if normalized_hint else None
            return llm.extract_vault_item(raw_text=raw_text, type_hint=hint_text), warnings
        except Exception as exc:
            warnings.append(f'LLM vault parsing failed; using heuristic parser ({exc}).')

    warnings.append('Heuristic vault parsing was used; review YAML before saving.')
    return heuristic_parse_vault_text(raw_text, type_hint=normalized_hint), warnings


def parse_uploaded_text(path: Path, *, enable_ocr: bool) -> Tuple[str, List[str]]:
    suffix = path.suffix.lower()
    if suffix in {'.pdf', '.docx'}:
        return extract_resume_text(path, enable_ocr=enable_ocr)

    if suffix in {'.txt', '.md', '.rst', '.rtf', '.yaml', '.yml', '.json'}:
        return path.read_text(encoding='utf-8', errors='ignore').strip(), []

    raise VaultIngestError('Unsupported file type for vault ingest. Use PDF, DOCX, or text-based files.')


def heuristic_parse_vault_text(raw_text: str, *, type_hint: Optional[VaultItemType] = None) -> VaultItem:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        raise VaultIngestError('No text provided for vault parsing.')

    links = re.findall(r'https?://[^\s)]+', raw_text)
    bullets = _extract_bullets(lines)
    title = _extract_title(lines)
    inferred_type = type_hint or _infer_type(raw_text)

    tags = _extract_tags(raw_text, title=title)
    tech = _extract_tech(raw_text)
    dates = _extract_dates(raw_text)

    return VaultItem(
        type=inferred_type,
        title=title,
        dates=dates,
        tags=tags,
        tech=tech,
        bullets=[{'text': bullet} for bullet in bullets],
        links=links,
        source_artifacts=[],
    )


def _normalize_type_hint(type_hint: Optional[str]) -> Optional[VaultItemType]:
    if not type_hint:
        return None
    value = type_hint.strip()
    if not value:
        return None
    try:
        return VaultItemType(value)
    except ValueError as exc:
        raise VaultIngestError(f'Invalid type hint: {value}') from exc


def _extract_title(lines: Sequence[str]) -> str:
    for line in lines[:8]:
        cleaned = re.sub(r'^[\-\*•\d\.)\s]+', '', line).strip()
        if cleaned and len(cleaned) <= 100 and not cleaned.lower().startswith(('responsibilities', 'summary', 'details')):
            return cleaned[:100]
    return 'Vault Item'


def _extract_bullets(lines: Sequence[str]) -> List[str]:
    bullets: List[str] = []

    for line in lines:
        if re.match(r'^[\-\*•]\s+', line) or re.match(r'^\d+[\.)]\s+', line):
            bullet = re.sub(r'^[\-\*•\d\.)\s]+', '', line).strip()
            if bullet:
                bullets.append(bullet)

    if bullets:
        return bullets[:6]

    text = ' '.join(lines)
    sentence_candidates = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentence_candidates:
        clean = sentence.strip(' -;')
        if len(clean) >= 24:
            bullets.append(clean)
    if bullets:
        return bullets[:6]

    return [lines[0][:160]]


def _infer_type(raw_text: str) -> VaultItemType:
    text = raw_text.lower()
    if any(word in text for word in ['award', 'winner', 'scholarship', 'dean\'s list']):
        return VaultItemType.award
    if any(word in text for word in ['coursework', 'course', 'class', 'semester']):
        return VaultItemType.coursework
    if any(word in text for word in ['club', 'chapter', 'society', 'organization', 'committee']):
        return VaultItemType.club
    if any(word in text for word in ['intern', 'engineer', 'manager', 'analyst', 'worked at', 'employer']):
        return VaultItemType.job
    if any(word in text for word in ['skillset', 'stack', 'tooling', 'technologies']):
        return VaultItemType.skillset
    if any(word in text for word in ['project', 'built', 'developed', 'implemented', 'created']):
        return VaultItemType.project
    return VaultItemType.other


def _extract_tags(raw_text: str, *, title: str) -> List[str]:
    counts = {}
    title_tokens = set(tokenize(title))
    for token in tokenize(raw_text):
        if token in STOPWORDS or len(token) < 3:
            continue
        counts[token] = counts.get(token, 0) + 1

    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    tags = [term for term, _ in ranked if term not in title_tokens][:8]
    return unique_preserve_order(tags)


def _extract_tech(raw_text: str) -> List[str]:
    tokens = set(tokenize(raw_text))
    tech = []
    for term in sorted(TECH_CATALOG):
        token = term.replace('nodejs', 'node')
        if token in tokens or term in tokens:
            tech.append(term if term != 'nodejs' else 'node')
    return unique_preserve_order(tech)


def _extract_dates(raw_text: str) -> Optional[DateRange]:
    month_year = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}'
    year_only = r'\b\d{4}\b'

    range_match = re.search(rf'({month_year}|{year_only})\s*(?:-|to|–|—)\s*(Present|{month_year}|{year_only})', raw_text, flags=re.IGNORECASE)
    if range_match:
        return DateRange(start=range_match.group(1), end=range_match.group(2))

    single_years = re.findall(year_only, raw_text)
    if len(single_years) >= 2:
        return DateRange(start=single_years[0], end=single_years[1])

    return None
