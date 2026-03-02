from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from app.models import (
    CanonicalResume,
    CandidateItem,
    DateRange,
    EducationEntry,
    ExperienceEntry,
    Identity,
    JDAnalysis,
    ProjectEntry,
    SelectedItem,
    Skills,
    TailorMode,
    TailorReport,
    VaultItem,
    VaultItemType,
)
from app.services.ats_engine import compute_match_score
from app.services.llm import LLMService
from app.utils import normalize_token, tokenize, unique_preserve_order

STOPWORDS = {
    'and', 'or', 'with', 'for', 'the', 'you', 'your', 'our', 'are', 'will', 'from', 'that', 'have', 'has',
    'this', 'their', 'they', 'job', 'role', 'about', 'using', 'work', 'team', 'years', 'year', 'plus',
    'required', 'preferred', 'responsibilities', 'experience', 'ability', 'strong', 'skills', 'skill',
}

GENERIC_TERMS = {
    'system', 'systems', 'software', 'engineering', 'engineer', 'intern', 'project', 'projects', 'application',
    'applications', 'service', 'services', 'product', 'products', 'feature', 'features', 'build', 'design',
    'develop', 'development', 'maintain', 'maintainable', 'scalable', 'distributed', 'collaborate', 'team',
    'documentation', 'code', 'testing', 'tools', 'tooling', 'api', 'apis',
}

LOW_SIGNAL_TERMS = {
    'ai', 'genai', 'llm', 'llms', 'assistant', 'assistants', 'agent', 'agents', 'chatbot', 'chatbots', 'prompt',
    'prompts', 'automation', 'automations',
}

TOKEN_ALIASES = {
    'machinelearning': 'ml',
    'deeplearning': 'ml',
    'models': 'model',
    'pipelines': 'pipeline',
    'agents': 'agent',
    'assistants': 'assistant',
    'genai': 'ai',
    'llms': 'llm',
    'scikitlearn': 'sklearn',
}

PROFILE_TERMS: Dict[str, Set[str]] = {
    'ml_core': {
        'ml', 'machine', 'learning', 'model', 'inference', 'training', 'feature', 'features', 'xgboost',
        'lightgbm', 'tensorflow', 'pytorch', 'sklearn', 'classification', 'regression', 'anomaly', 'fraud', 'risk',
        'evaluation', 'experimentation',
    },
    'agentic': {
        'agent', 'assistant', 'ai', 'llm', 'rag', 'langchain', 'chatbot', 'prompt', 'tool', 'tooling',
    },
    'data_platform': {
        'etl', 'pipeline', 'kafka', 'airflow', 'spark', 'warehouse', 'dbt', 'batch', 'streaming',
    },
    'backend': {
        'backend', 'api', 'microservice', 'microservices', 'distributed', 'scalable', 'kubernetes', 'docker',
        'postgresql', 'sql', 'service', 'services',
    },
}

TITLE_MARKER_TERMS = {
    'engineer', 'scientist', 'developer', 'analyst', 'manager', 'architect', 'researcher', 'intern', 'specialist',
}

ROLE_DESCRIPTOR_TERMS = {
    'machine', 'learning', 'software', 'data', 'backend', 'frontend', 'fullstack', 'full', 'stack', 'platform',
    'security', 'cloud', 'devops', 'site', 'reliability', 'ml', 'ai', 'applied', 'research', 'product', 'analytics',
    'quant', 'computer', 'vision', 'nlp', 'infra', 'infrastructure', 'embedded', 'mobile', 'web',
}

TITLE_REJECT_TERMS = {
    'we', 'our', 'you', 'your', 'looking', 'hire', 'hiring', 'candidate', 'creative', 'join', 'team',
    'responsibilities', 'qualification', 'qualifications', 'must', 'should', 'need', 'seeking',
}

SUMMARY_TERM_DISPLAY = {
    'ml': 'machine learning',
    'ai': 'AI',
    'llm': 'LLM',
    'sql': 'SQL',
    'xgboost': 'XGBoost',
    'pytorch': 'PyTorch',
    'tensorflow': 'TensorFlow',
    'sklearn': 'scikit-learn',
    'aws': 'AWS',
    'gcp': 'GCP',
    'api': 'API',
    'kafka': 'Kafka',
    'docker': 'Docker',
    'kubernetes': 'Kubernetes',
    'postgresql': 'PostgreSQL',
    'fraud': 'fraud detection',
    'inference': 'model inference',
}

SUMMARY_BANNED_TERMS = {
    'looking', 'hire', 'hiring', 'candidate', 'creative', 'team', 'teams', 'process', 'processes', 'result',
    'results', 'immediately', 'deliver', 'delivered', 'building', 'built', 'work', 'works', 'working', 'experience',
    'role', 'roles', 'company', 'strong', 'ability', 'responsibility', 'responsibilities', 'requirement',
    'requirements', 'people', 'communication', 'collaboration', 'automate', 'automation', 'automations',
    'data', 'analytics', 'analysis',
}

SUMMARY_ALLOWED_GENERIC_TERMS = {
    'backend', 'distributed', 'pipeline', 'inference', 'fraud', 'risk', 'model', 'ml', 'sql', 'docker', 'kafka',
    'postgresql', 'pytorch', 'tensorflow', 'sklearn', 'xgboost', 'aws', 'gcp', 'api', 'etl', 'spark', 'airflow',
    'dbt', 'nlp', 'vision',
}

MAX_EXP_ITEMS = 3
MAX_PROJECT_ITEMS = 6
MIN_PROJECT_ITEMS = 2
MAX_BULLETS_PER_ITEM = 3
MAX_TOTAL_BULLETS = 12
MAX_PAGE_UNITS = 41
MIN_PROJECT_BULLETS = 2
MUST_HAVE_MIN_TERMS = 3
MUST_HAVE_MIN_COVERAGE_BASE = 0.45
MUST_HAVE_MIN_COVERAGE_STEP = 0.08


@dataclass
class TailorResult:
    tailored_resume: CanonicalResume
    report: TailorReport


@dataclass
class CandidateSource:
    candidate: CandidateItem
    origin: Dict[str, object]


@dataclass
class ScoringContext:
    required_tokens: Set[str]
    role_tokens: Set[str]
    nice_tokens: Set[str]
    resp_tokens: Set[str]
    all_terms: Set[str]
    phrases: List[Tuple[str, ...]]
    strict_required_tokens: Set[str]
    token_weights: Dict[str, float]
    jd_profile_hits: Dict[str, int]


def _normalize_spacing(text: str) -> str:
    cleaned = re.sub(r'\s+', ' ', text or '').strip()
    cleaned = re.sub(r'\s+([,.;:!?])', r'\1', cleaned)
    cleaned = re.sub(r'([(\[]) +', r'\1', cleaned)
    cleaned = re.sub(r' +([)\]])', r'\1', cleaned)
    cleaned = re.sub(r',\s*,+', ', ', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned.strip()


def _clean_title_hint(title: str) -> str:
    cleaned = _normalize_spacing(re.sub(r'\s*\([^)]*\)', '', title))
    lowered = cleaned.lower()
    for separator in (' at ', ' | ', ' — ', ' – ', ' - ', '/'):
        idx = lowered.find(separator)
        if idx <= 0:
            continue
        left = cleaned[:idx].strip(' ,')
        left_tokens = tokenize(left)
        if 2 <= len(left_tokens) <= 8:
            cleaned = left
            lowered = cleaned.lower()
            break
    return cleaned


def _looks_like_role_title(candidate: str) -> bool:
    lowered = candidate.lower()
    blocked_patterns = (
        'we are looking',
        'looking to hire',
        'join our team',
        'what you will',
        'you will',
        'responsibilities',
        'qualifications',
    )
    if any(pattern in lowered for pattern in blocked_patterns):
        return False

    tokens = _canonicalize_terms(tokenize(candidate))
    if len(tokens) < 2 or len(tokens) > 8:
        return False
    marker_hits = sum(1 for token in tokens if token in TITLE_MARKER_TERMS)
    if marker_hits <= 0:
        return False

    reject_hits = sum(1 for token in tokens if token in TITLE_REJECT_TERMS)
    if reject_hits >= 2:
        return False
    if tokens and tokens[0] in TITLE_REJECT_TERMS:
        return False

    descriptor_or_marker = sum(1 for token in tokens if token in ROLE_DESCRIPTOR_TERMS or token in TITLE_MARKER_TERMS)
    return descriptor_or_marker >= max(2, len(tokens) // 2)


def _title_case_tokens(tokens: Sequence[str]) -> str:
    words: List[str] = []
    for token in tokens:
        if token == 'ml':
            words.extend(['Machine', 'Learning'])
            continue
        if token in {'ai', 'llm'}:
            words.append(token.upper())
            continue
        if token in {'sre'}:
            words.append(token.upper())
            continue
        words.append(token.capitalize())
    return ' '.join(words)


def _derive_target_title(jd: JDAnalysis, jd_text: str, job_title_hint: Optional[str]) -> Optional[str]:
    if job_title_hint:
        cleaned_hint = _clean_title_hint(job_title_hint)
        if _looks_like_role_title(cleaned_hint):
            hint_tokens = _canonicalize_terms(tokenize(cleaned_hint))
            return _title_case_tokens(hint_tokens[:6]) if hint_tokens else cleaned_hint

    candidates = list(jd.target_role_keywords) + list(jd.required_skills) + list(jd.responsibilities)

    best_phrase: Optional[str] = None
    best_score = float('-inf')
    for phrase in candidates:
        cleaned_phrase = _clean_title_hint(phrase)
        if not _looks_like_role_title(cleaned_phrase):
            continue
        phrase_tokens = _canonicalize_terms(tokenize(cleaned_phrase))
        marker_hits = sum(1 for token in phrase_tokens if token in TITLE_MARKER_TERMS)
        if marker_hits <= 0:
            continue

        score = marker_hits * 3.0
        score += sum(1.0 for token in phrase_tokens if token in {'ml', 'fraud', 'inference', 'backend', 'data'})
        score += min(2.0, len(phrase_tokens) * 0.25)
        if len(phrase_tokens) > 8:
            score -= 2.0
        if any(token in {'intern', 'internship'} for token in phrase_tokens):
            score += 0.5

        normalized_phrase = _title_case_tokens(phrase_tokens[:6])
        if score > best_score:
            best_score = score
            best_phrase = normalized_phrase

    if best_phrase:
        return best_phrase

    jd_tokens = set(_canonicalize_terms(tokenize(' '.join(jd.target_role_keywords + jd.required_skills))))
    if (('ml' in jd_tokens) or ('machine' in jd_tokens and 'learning' in jd_tokens)) and 'engineer' in jd_tokens:
        return 'Machine Learning Engineer'
    if 'data' in jd_tokens and 'scientist' in jd_tokens:
        return 'Data Scientist'
    if 'software' in jd_tokens and 'engineer' in jd_tokens:
        return 'Software Engineer'
    return None


def _format_summary_term(term: str) -> str:
    if term in SUMMARY_TERM_DISPLAY:
        return SUMMARY_TERM_DISPLAY[term]
    if term in {'etl', 'rag', 'nlp'}:
        return term.upper()
    return term.replace('-', ' ')


def _is_summary_term(term: str) -> bool:
    if not term:
        return False
    if term in SUMMARY_BANNED_TERMS:
        return False
    if term in TITLE_REJECT_TERMS:
        return False
    if _is_high_signal_term(term):
        return True
    return term in SUMMARY_ALLOWED_GENERIC_TERMS


def _structured_resume_terms(resume: CanonicalResume) -> List[str]:
    terms: List[str] = []

    for entry in resume.experience:
        terms.extend(_canonicalize_terms(tokenize(entry.title)))

    for project in resume.projects:
        terms.extend(_canonicalize_terms(tokenize(' '.join(project.tech))))

    for skills in resume.skills.categories.values():
        for skill in skills:
            terms.extend(_canonicalize_terms(tokenize(skill)))

    return unique_preserve_order(terms)


def _collect_summary_terms(resume: CanonicalResume, jd: JDAnalysis, limit: int = 5) -> List[str]:
    resume_terms: Set[str] = set(_structured_resume_terms(resume))
    if not resume_terms:
        resume_terms = _resume_evidence_terms(resume)

    jd_priority = _canonicalize_terms(
        tokenize(' '.join(jd.required_skills + jd.responsibilities + jd.target_role_keywords + jd.nice_to_haves))
    )
    selected: List[str] = []
    for term in jd_priority:
        if term not in resume_terms:
            continue
        if not _is_summary_term(term):
            continue
        if term in selected:
            continue
        selected.append(term)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for term in _structured_resume_terms(resume):
            if term in selected:
                continue
            if not _is_summary_term(term):
                continue
            selected.append(term)
            if len(selected) >= limit:
                break

    return selected


def _join_terms_for_summary(terms: Sequence[str]) -> str:
    if not terms:
        return ''
    formatted = [_format_summary_term(term) for term in terms]
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f'{formatted[0]} and {formatted[1]}'
    return f"{', '.join(formatted[:-1])}, and {formatted[-1]}"


def _resume_evidence_terms(resume: CanonicalResume) -> Set[str]:
    terms: Set[str] = set()
    for entry in resume.experience:
        terms.update(_canonicalize_terms(tokenize(' '.join([entry.company, entry.title, *entry.bullets]))))
    for project in resume.projects:
        terms.update(_canonicalize_terms(tokenize(' '.join([project.name, *project.tech, *project.bullets]))))
    for skills in resume.skills.categories.values():
        terms.update(_canonicalize_terms(tokenize(' '.join(skills))))
    return terms


def _scope_phrase(experience_count: int, project_count: int) -> str:
    parts: List[str] = []
    if experience_count > 0:
        label = 'work experience' if experience_count == 1 else 'work experiences'
        parts.append(f'{experience_count} {label}')
    if project_count > 0:
        label = 'project' if project_count == 1 else 'projects'
        parts.append(f'{project_count} selected {label}')
    if not parts:
        return 'my selected resume experience'
    if len(parts) == 1:
        return parts[0]
    return f'{parts[0]} and {parts[1]}'


def _ensure_sentence(text: str) -> str:
    cleaned = _normalize_spacing(text).rstrip()
    if not cleaned:
        return ''
    if cleaned.endswith(('.', '!', '?')):
        return cleaned
    return cleaned + '.'


def _sentence_count(text: str) -> int:
    return len([part for part in re.split(r'(?<=[.!?])\s+', text.strip()) if part.strip()])


def _resume_role_title(resume: CanonicalResume) -> Optional[str]:
    for entry in resume.experience:
        cleaned = _clean_title_hint(entry.title)
        if _looks_like_role_title(cleaned):
            tokens = _canonicalize_terms(tokenize(cleaned))
            if tokens:
                return _title_case_tokens(tokens[:6])
            return cleaned

    for project in resume.projects:
        tokens = _canonicalize_terms(tokenize(project.name))
        if not tokens:
            continue
        if 'ml' in tokens and 'engineer' in tokens:
            return 'Machine Learning Engineer'
    return None


def _ensure_target_summary(
    resume: CanonicalResume,
    jd: JDAnalysis,
    jd_text: str,
    job_title_hint: Optional[str],
) -> CanonicalResume:
    cloned = CanonicalResume.model_validate(deepcopy(resume.model_dump()))
    target_title = _derive_target_title(jd, jd_text, job_title_hint)
    resume_role_title = _resume_role_title(cloned)
    headline_title = target_title or resume_role_title or 'Engineer'

    aligned_terms = _collect_summary_terms(cloned, jd, limit=6)
    primary_terms = aligned_terms[:3]
    secondary_terms = aligned_terms[3:6]
    evidence_terms = _resume_evidence_terms(cloned)

    sentences: List[str] = []
    if primary_terms:
        sentences.append(
            _ensure_sentence(
                f'{headline_title} with hands-on experience in {_join_terms_for_summary(primary_terms)}'
            )
        )
    else:
        sentences.append(
            _ensure_sentence(
                f'{headline_title} with hands-on experience across software and data-driven projects'
            )
        )

    support_terms = secondary_terms if secondary_terms else primary_terms[:2]
    if support_terms:
        sentences.append(
            _ensure_sentence(
                f'Background includes building and shipping practical solutions using {_join_terms_for_summary(support_terms)}'
            )
        )
    else:
        sentences.append(
            _ensure_sentence(
                'Background includes professional and project-based work focused on reliable implementation'
            )
        )

    role_tokens = _canonicalize_terms(tokenize(' '.join(jd.target_role_keywords + jd.required_skills + jd.responsibilities)))
    role_aligned_terms: List[str] = []
    for token in role_tokens:
        if token not in evidence_terms:
            continue
        if not _is_summary_term(token):
            continue
        if token in role_aligned_terms:
            continue
        role_aligned_terms.append(token)
        if len(role_aligned_terms) >= 3:
            break
    if target_title and role_aligned_terms:
        sentences.append(
            _ensure_sentence(
                f'Well aligned with {target_title} roles that emphasize {_join_terms_for_summary(role_aligned_terms[:2])}'
            )
        )

    if len(sentences) < 2 and cloned.summary:
        cleaned_existing = _normalize_spacing(cloned.summary)
        if cleaned_existing:
            sentences.append(_ensure_sentence(cleaned_existing))

    summary = _normalize_spacing(' '.join(sentences[:3]))
    if _sentence_count(summary) < 2:
        summary = _ensure_sentence(f'{summary} Prepared to contribute quickly with this background').strip()

    words = summary.split()
    if len(words) > 52:
        summary = ' '.join(words[:52]).rstrip(' ,;') + '.'
    cloned.summary = summary
    return cloned


def _canonical_token(token: str) -> str:
    normalized = normalize_token(token)
    if not normalized:
        return ''
    return TOKEN_ALIASES.get(normalized, normalized)


def _canonicalize_terms(tokens: Sequence[str]) -> List[str]:
    canonical: List[str] = []
    for token in tokens:
        mapped = _canonical_token(token)
        if mapped:
            canonical.append(mapped)
    return canonical


def _is_high_signal_term(term: str) -> bool:
    return (
        len(term) > 2
        and term not in STOPWORDS
        and term not in GENERIC_TERMS
        and term not in LOW_SIGNAL_TERMS
    )


def _profile_hit_counts(tokens: Set[str]) -> Dict[str, int]:
    hits: Dict[str, int] = {}
    for profile, profile_terms in PROFILE_TERMS.items():
        hits[profile] = len(tokens & profile_terms)
    return hits


def analyze_jd_text(jd_text: str, llm: Optional[LLMService] = None) -> JDAnalysis:
    if llm and llm.available:
        try:
            return llm.analyze_jd(jd_text)
        except Exception:
            pass

    lines = [line.strip() for line in jd_text.splitlines() if line.strip()]
    required: List[str] = []
    nice: List[str] = []
    responsibilities: List[str] = []

    for line in lines:
        lowered = line.lower()
        if any(marker in lowered for marker in ['required', 'must have', 'minimum qualifications', 'qualifications']):
            required.extend(_extract_phrases(line))
        elif any(marker in lowered for marker in ['preferred', 'nice to have', 'bonus']):
            nice.extend(_extract_phrases(line))
        elif any(marker in lowered for marker in ['responsibilit', 'you will', 'what you', 'day-to-day']):
            responsibilities.extend(_extract_phrases(line))

    role_keywords = _extract_top_keywords(jd_text, limit=20)
    required = unique_preserve_order(required)[:20]
    nice = unique_preserve_order(nice)[:20]
    responsibilities = unique_preserve_order(responsibilities)[:20]

    if not required:
        required = _extract_top_keywords(jd_text, limit=15)

    return JDAnalysis(
        target_role_keywords=role_keywords,
        required_skills=required,
        nice_to_haves=nice,
        responsibilities=responsibilities,
    )


def _extract_phrases(line: str) -> List[str]:
    parts = re.split(r'[,:;\-•]', line)
    phrases = []
    for part in parts:
        phrase = part.strip()
        if not phrase:
            continue
        tokens = [
            token for token in _canonicalize_terms(tokenize(phrase))
            if token not in STOPWORDS and len(token) > 1
        ]
        if tokens:
            phrases.append(' '.join(tokens[:4]))
    return phrases


def _extract_top_keywords(text: str, limit: int = 20) -> List[str]:
    counts: Dict[str, int] = {}
    for token in _canonicalize_terms(tokenize(text)):
        if token in STOPWORDS or len(token) < 2:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return [term for term, _ in ranked[:limit]]


def build_candidate_pool(base: CanonicalResume, vault_items: Sequence[Tuple[str, VaultItem]]) -> List[CandidateSource]:
    vault_candidates = _build_vault_candidates(vault_items)
    vault_content = [
        candidate
        for candidate in vault_candidates
        if candidate.candidate.source_type not in {'vault:award', 'vault:skillset'}
    ]
    if vault_content:
        return vault_candidates

    # Fallback for legacy behavior when vault is empty.
    candidates: List[CandidateSource] = []

    for index, entry in enumerate(base.experience):
        bullets = [bullet for bullet in entry.bullets if bullet.strip()]
        candidate = CandidateItem(
            source_type='experience',
            source_id=f'base-experience-{index}',
            title=f'{entry.title} at {entry.company}',
            dates=entry.dates,
            tags=[entry.company, entry.title, entry.location],
            tech=_extract_inline_terms(' '.join(entry.bullets)),
            bullets=bullets,
            location=entry.location,
            company=entry.company,
            role=entry.title,
        )
        candidates.append(CandidateSource(candidate=candidate, origin={'kind': 'base_experience', 'index': index}))

    for index, entry in enumerate(base.projects):
        bullets = [bullet for bullet in entry.bullets if bullet.strip()]
        if len(bullets) < MIN_PROJECT_BULLETS:
            continue
        section = _project_section_from_entry(entry)
        candidate = CandidateItem(
            source_type='project',
            source_id=f'base-project-{index}',
            title=entry.name,
            dates=entry.dates,
            tags=[entry.name, f'section:{section}'],
            tech=entry.tech,
            bullets=bullets,
        )
        candidates.append(CandidateSource(candidate=candidate, origin={'kind': 'base_project', 'index': index}))

    for index, entry in enumerate(base.education):
        if not entry.coursework:
            continue
        candidate = CandidateItem(
            source_type='coursework',
            source_id=f'base-education-{index}',
            title=f'{entry.school} coursework',
            dates=entry.dates,
            tags=[entry.school, entry.major],
            tech=entry.coursework,
            bullets=[f'Coursework: {", ".join(entry.coursework[:6])}'],
        )
        candidates.append(CandidateSource(candidate=candidate, origin={'kind': 'base_education', 'index': index}))

    return candidates


def _build_vault_candidates(vault_items: Sequence[Tuple[str, VaultItem]]) -> List[CandidateSource]:
    candidates: List[CandidateSource] = []
    for item_id, item in vault_items:
        if item.type == VaultItemType.skillset:
            continue

        company = _extract_tag_value(item.tags, 'company:')
        role = _extract_tag_value(item.tags, 'role:') or item.title
        location = _extract_tag_value(item.tags, 'location:')
        bullets = [bullet.text for bullet in item.bullets if bullet.text.strip()]
        if item.type in {VaultItemType.project, VaultItemType.club, VaultItemType.other, VaultItemType.coursework}:
            if len(bullets) < MIN_PROJECT_BULLETS:
                continue
        section = _project_section_from_tags(item.tags, item.title)
        tags = list(item.tags)
        if not any(tag.lower().startswith('section:') for tag in tags):
            tags.append(f'section:{section}')

        candidate = CandidateItem(
            source_type=f'vault:{item.type.value}',
            source_id=f'vault-{item_id}',
            title=item.title,
            dates=item.dates,
            tags=tags,
            tech=item.tech,
            bullets=bullets,
            location=location,
            company=company,
            role=role,
        )
        candidates.append(CandidateSource(candidate=candidate, origin={'kind': 'vault', 'item_id': item_id, 'item': item}))

    return candidates


def _extract_tag_value(tags: Sequence[str], prefix: str) -> Optional[str]:
    lowered_prefix = prefix.lower()
    for tag in tags:
        if tag.lower().startswith(lowered_prefix):
            value = tag[len(prefix):].strip()
            if value:
                return value
    return None


def _project_section_from_tags(tags: Sequence[str], title: str) -> str:
    for tag in tags:
        lowered = tag.strip().lower()
        if lowered in {'minor_project', 'minor', 'section:minor', 'section:minor_projects'}:
            return 'minor_projects'
        if lowered in {'section:projects', 'section:project'}:
            return 'projects'
    if 'minor project' in title.lower():
        return 'minor_projects'
    return 'projects'


def _project_section_from_entry(project: ProjectEntry) -> str:
    section = (project.section or '').strip().lower()
    if section in {'minor', 'minor_projects', 'minor-projects', 'minorprojects'}:
        return 'minor_projects'
    if section in {'projects', ''}:
        if 'minor project' in project.name.lower():
            return 'minor_projects'
        return 'projects'
    return 'projects'


def _extract_inline_terms(text: str) -> List[str]:
    raw_tokens = re.findall(r'[A-Za-z][A-Za-z0-9\+#\.]{1,}', text)
    filtered = [token for token in raw_tokens if any(ch.isupper() for ch in token) or '+' in token or '#' in token]
    return unique_preserve_order(_canonicalize_terms([token.lower() for token in filtered]))


def _candidate_token_set(candidate: CandidateItem) -> Set[str]:
    raw_tokens = tokenize(' '.join(candidate.bullets + candidate.tags + [candidate.title] + candidate.tech))
    return set(_canonicalize_terms(raw_tokens))


def _candidate_focus_terms(candidate: CandidateItem) -> Set[str]:
    focus_tokens = set(_canonicalize_terms(tokenize(' '.join(candidate.tech + candidate.tags + [candidate.title]))))
    return {token for token in focus_tokens if len(token) > 2 and token not in STOPWORDS and token not in GENERIC_TERMS}


def _extract_year(date_text: Optional[str]) -> Optional[int]:
    if not date_text:
        return None
    match = re.search(r'\b(19|20)\d{2}\b', date_text)
    if not match:
        return None
    return int(match.group(0))


def _recency_bonus(dates: Optional[DateRange]) -> float:
    if dates is None:
        return 0.0
    end = (dates.end or '').strip().lower()
    if end == 'present':
        return 1.2

    end_year = _extract_year(dates.end)
    if end_year is None:
        return 0.0
    if end_year >= 2025:
        return 1.0
    if end_year >= 2023:
        return 0.6
    return 0.0


def _extract_jd_phrases(jd: JDAnalysis, limit: int = 40) -> List[Tuple[str, ...]]:
    phrases: List[Tuple[str, ...]] = []
    raw_chunks = jd.required_skills + jd.responsibilities + jd.target_role_keywords + jd.nice_to_haves

    for chunk in raw_chunks:
        tokens = [
            token for token in _canonicalize_terms(tokenize(chunk))
            if token not in STOPWORDS and len(token) > 2
        ]
        if len(tokens) < 2:
            continue
        for n in (3, 2):
            for index in range(0, len(tokens) - n + 1):
                phrase = tuple(tokens[index:index + n])
                if phrase not in phrases:
                    phrases.append(phrase)
                if len(phrases) >= limit:
                    return phrases
    return phrases


def _jd_term_sets(jd: JDAnalysis) -> Tuple[Set[str], Set[str], Set[str], Set[str], Set[str], List[Tuple[str, ...]]]:
    required_tokens = set(_canonicalize_terms(tokenize(' '.join(jd.required_skills))))
    role_tokens = set(_canonicalize_terms(tokenize(' '.join(jd.target_role_keywords))))
    nice_tokens = set(_canonicalize_terms(tokenize(' '.join(jd.nice_to_haves))))
    responsibility_tokens = set(_canonicalize_terms(tokenize(' '.join(jd.responsibilities))))
    all_terms = required_tokens | role_tokens | nice_tokens | responsibility_tokens
    phrases = _extract_jd_phrases(jd)
    return required_tokens, role_tokens, nice_tokens, responsibility_tokens, all_terms, phrases


def _required_must_have_terms(jd: JDAnalysis) -> Set[str]:
    required_tokens = set(_canonicalize_terms(tokenize(' '.join(jd.required_skills))))
    strict_required = {term for term in required_tokens if _is_high_signal_term(term)}
    if len(strict_required) >= MUST_HAVE_MIN_TERMS:
        return strict_required
    return {
        term for term in required_tokens
        if term not in STOPWORDS and term not in GENERIC_TERMS and len(term) > 2
    }


def _selected_required_coverage(selected: Sequence[Tuple[CandidateSource, float]], required_terms: Set[str]) -> float:
    if not required_terms:
        return 1.0
    covered: Set[str] = set()
    for candidate_source, _ in selected:
        covered.update(_candidate_token_set(candidate_source.candidate))
    return len(covered & required_terms) / max(1, len(required_terms))


def _parse_feedback_terms(values: Sequence[str]) -> Tuple[Set[str], Set[str]]:
    normalized_titles: Set[str] = set()
    tokens: Set[str] = set()
    for value in values:
        cleaned = (value or '').strip()
        normalized = normalize_token(cleaned)
        if normalized:
            normalized_titles.add(normalized)
        tokens.update(_canonicalize_terms(tokenize(cleaned)))
    tokens = {token for token in tokens if len(token) > 1}
    return normalized_titles, tokens


def _feedback_adjustment(
    candidate: CandidateItem,
    preferred_titles: Set[str],
    blocked_titles: Set[str],
    preferred_terms: Set[str],
    blocked_terms: Set[str],
) -> float:
    adjustment = 0.0
    title_id = normalize_token(candidate.title)
    candidate_tokens = _candidate_token_set(candidate)

    if title_id and title_id in preferred_titles:
        adjustment += 7.0
    if title_id and title_id in blocked_titles:
        adjustment -= 10.0

    preferred_hits = len(candidate_tokens & preferred_terms)
    blocked_hits = len(candidate_tokens & blocked_terms)
    adjustment += min(4.0, preferred_hits * 1.2)
    adjustment -= min(6.0, blocked_hits * 1.4)
    return adjustment


def _optimization_adjustment(candidate: CandidateItem, jd: JDAnalysis, optimization_level: int) -> float:
    if optimization_level <= 1:
        return 0.0
    required_terms = _required_must_have_terms(jd)
    if not required_terms:
        return 0.0
    candidate_terms = _candidate_token_set(candidate)
    hits = len(candidate_terms & required_terms)
    if hits <= 0:
        if len(required_terms) >= MUST_HAVE_MIN_TERMS:
            return -0.8 * float(optimization_level - 1)
        return 0.0
    return min(5.5, hits * (0.45 + (0.2 * float(optimization_level - 1))))


def _build_scoring_context(jd: JDAnalysis, candidates: Sequence[CandidateItem]) -> ScoringContext:
    required_tokens, role_tokens, nice_tokens, resp_tokens, all_terms, phrases = _jd_term_sets(jd)
    strict_required_tokens = {term for term in required_tokens if _is_high_signal_term(term)}
    if len(strict_required_tokens) < 2:
        strict_required_tokens = {
            term for term in required_tokens
            if term not in STOPWORDS and term not in GENERIC_TERMS
        }

    candidate_count = max(1, len(candidates))
    token_frequencies: Dict[str, int] = {}
    for candidate in candidates:
        candidate_terms = _candidate_token_set(candidate)
        for token in candidate_terms & all_terms:
            token_frequencies[token] = token_frequencies.get(token, 0) + 1

    token_weights: Dict[str, float] = {}
    for term in all_terms:
        base_weight = 0.0
        if term in required_tokens:
            base_weight += 2.8
        if term in resp_tokens:
            base_weight += 2.0
        if term in role_tokens:
            base_weight += 1.4
        if term in nice_tokens:
            base_weight += 0.8
        if base_weight <= 0:
            base_weight = 0.7
        if term in strict_required_tokens:
            base_weight *= 1.35
        if term in LOW_SIGNAL_TERMS:
            base_weight *= 0.45
        elif term in GENERIC_TERMS:
            base_weight *= 0.6

        frequency = token_frequencies.get(term, 0)
        idf = 1.0 + math.log((candidate_count + 1) / (frequency + 1))
        token_weights[term] = round(base_weight * idf, 4)

    jd_profile_hits = _profile_hit_counts(required_tokens | resp_tokens | role_tokens)

    return ScoringContext(
        required_tokens=required_tokens,
        role_tokens=role_tokens,
        nice_tokens=nice_tokens,
        resp_tokens=resp_tokens,
        all_terms=all_terms,
        phrases=phrases,
        strict_required_tokens=strict_required_tokens,
        token_weights=token_weights,
        jd_profile_hits=jd_profile_hits,
    )


def _weighted_overlap(tokens: Set[str], targets: Set[str], weights: Dict[str, float]) -> float:
    return sum(weights.get(token, 1.0) for token in tokens & targets)


def _weighted_phrase_score(
    candidate_tokens: Set[str],
    phrases: Sequence[Tuple[str, ...]],
    weights: Dict[str, float],
) -> float:
    total = 0.0
    for phrase in phrases:
        if all(token in candidate_tokens for token in phrase):
            phrase_weight = sum(weights.get(token, 1.0) for token in phrase) / max(1, len(phrase))
            total += min(4.0, phrase_weight)
    return total


def _jd_analysis_to_text(jd: JDAnalysis) -> str:
    sections: List[str] = []
    if jd.target_role_keywords:
        sections.append(f"Target role keywords: {', '.join(jd.target_role_keywords)}")
    if jd.required_skills:
        sections.append(f"Required skills: {', '.join(jd.required_skills)}")
    if jd.nice_to_haves:
        sections.append(f"Preferred skills: {', '.join(jd.nice_to_haves)}")
    if jd.responsibilities:
        sections.append(f"Responsibilities: {', '.join(jd.responsibilities)}")
    return '\n'.join(sections)


def _scoring_seed_resume(scoring_resume: Optional[CanonicalResume]) -> CanonicalResume:
    if scoring_resume is None:
        return CanonicalResume(
            identity=Identity(name='Candidate', email='', phone='', location='', links=[]),
            summary=None,
            education=[],
            experience=[],
            projects=[],
            skills=Skills(categories={}),
            certifications=[],
            awards=[],
        )

    seed = CanonicalResume.model_validate(deepcopy(scoring_resume.model_dump()))
    seed.summary = None
    seed.experience = []
    seed.projects = []
    seed.skills = Skills(categories={})
    seed.certifications = []
    seed.awards = []
    return seed


def _candidate_section(candidate: CandidateItem, source_kind: Optional[str]) -> str:
    normalized_source_type = (candidate.source_type or '').strip().lower()
    normalized_source_kind = (source_kind or '').strip().lower()

    if normalized_source_kind == 'base_experience':
        return 'experience'
    if normalized_source_kind == 'base_education':
        return 'coursework'
    if normalized_source_type in {'experience', 'job', 'vault:job'}:
        return 'experience'
    if normalized_source_type in {'coursework', 'vault:coursework'}:
        return 'coursework'
    return 'project'


def _add_coursework_evidence(resume: CanonicalResume, candidate: CandidateItem) -> None:
    coursework = unique_preserve_order([term.strip() for term in candidate.tech if term.strip()])
    if not coursework:
        coursework = unique_preserve_order(
            _canonicalize_terms(tokenize(' '.join([candidate.title, *candidate.bullets])))
        )

    if resume.education:
        primary_entry = resume.education[0]
        primary_entry.coursework = unique_preserve_order(primary_entry.coursework + coursework)[:12]
        return

    resume.education.append(
        EducationEntry(
            school=candidate.title or 'Coursework',
            degree='',
            major='',
            minors=[],
            gpa='',
            dates=candidate.dates or DateRange(),
            coursework=coursework[:12],
        )
    )


def _candidate_resume_for_ats(
    candidate: CandidateItem,
    scoring_seed: CanonicalResume,
    source_kind: Optional[str],
) -> CanonicalResume:
    candidate_resume = CanonicalResume.model_validate(deepcopy(scoring_seed.model_dump()))
    section = _candidate_section(candidate, source_kind=source_kind)

    if section == 'experience':
        candidate_resume.experience.append(
            ExperienceEntry(
                company=(candidate.company or candidate.title),
                title=(candidate.role or candidate.title),
                location=(candidate.location or candidate_resume.identity.location),
                dates=candidate.dates or DateRange(),
                bullets=list(candidate.bullets),
            )
        )
        return candidate_resume

    if section == 'coursework':
        _add_coursework_evidence(candidate_resume, candidate)
        return candidate_resume

    candidate_resume.projects.append(
        ProjectEntry(
            name=candidate.title,
            link=None,
            dates=candidate.dates,
            tech=list(candidate.tech),
            bullets=list(candidate.bullets),
            section=_project_section_from_tags(candidate.tags, candidate.title),
        )
    )
    return candidate_resume


def _ats_candidate_base_score(
    *,
    candidate: CandidateItem,
    jd_text: str,
    scoring_seed: CanonicalResume,
    source_kind: Optional[str],
) -> float:
    candidate_resume = _candidate_resume_for_ats(candidate, scoring_seed=scoring_seed, source_kind=source_kind)
    match_payload = compute_match_score(candidate_resume, jd_text)
    return float(match_payload.get('overall_score', 0.0) or 0.0)


def score_candidate_item(
    candidate: CandidateItem,
    jd: JDAnalysis,
    context: Optional[ScoringContext] = None,
    *,
    jd_text: Optional[str] = None,
    scoring_resume: Optional[CanonicalResume] = None,
    source_kind: Optional[str] = None,
) -> float:
    _ = context  # Backward-compatible arg; ATS scoring is now canonical.
    active_jd_text = (jd_text or _jd_analysis_to_text(jd)).strip()
    if not active_jd_text:
        return 0.0
    scoring_seed = _scoring_seed_resume(scoring_resume)
    score = _ats_candidate_base_score(
        candidate=candidate,
        jd_text=active_jd_text,
        scoring_seed=scoring_seed,
        source_kind=source_kind,
    )
    return round(max(0.0, score), 3)


def score_candidates(
    candidates: Sequence[CandidateSource],
    jd: JDAnalysis,
    selection_feedback: Optional[Dict[str, List[str]]] = None,
    optimization_level: int = 1,
    *,
    jd_text: Optional[str] = None,
    scoring_resume: Optional[CanonicalResume] = None,
) -> List[Tuple[CandidateSource, float]]:
    active_jd_text = (jd_text or _jd_analysis_to_text(jd)).strip()
    if not active_jd_text:
        return []
    scoring_seed = _scoring_seed_resume(scoring_resume)
    feedback_payload = selection_feedback or {}
    preferred_titles, preferred_terms = _parse_feedback_terms(feedback_payload.get('preferred_titles', []))
    blocked_titles, blocked_terms = _parse_feedback_terms(feedback_payload.get('blocked_titles', []))

    scored: List[Tuple[CandidateSource, float]] = []
    for candidate_source in candidates:
        score = _ats_candidate_base_score(
            candidate=candidate_source.candidate,
            jd_text=active_jd_text,
            scoring_seed=scoring_seed,
            source_kind=str(candidate_source.origin.get('kind', '')),
        )
        score += _feedback_adjustment(
            candidate_source.candidate,
            preferred_titles=preferred_titles,
            blocked_titles=blocked_titles,
            preferred_terms=preferred_terms,
            blocked_terms=blocked_terms,
        )
        score += _optimization_adjustment(candidate_source.candidate, jd, optimization_level=optimization_level)
        scored.append((candidate_source, round(max(0.0, score), 3)))
    return sorted(scored, key=lambda pair: (-pair[1], pair[0].candidate.title.lower()))


def extract_metric_tokens(text: str) -> Set[str]:
    return set(re.findall(r'\b\d+(?:\.\d+)?(?:%|x|k|m|\+)?\b', text.lower()))


def detect_terms_present(text: str, term_pool: Set[str]) -> Set[str]:
    lowered = text.lower()
    found: Set[str] = set()
    for term in sorted(term_pool, key=len, reverse=True):
        pattern = re.compile(rf'(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])', flags=re.IGNORECASE)
        if pattern.search(lowered):
            found.add(term)
    return found


def enforce_bullet_constraints(
    *,
    source_bullet: str,
    rewritten_bullet: str,
    allowed_terms: Set[str],
    known_terms: Set[str],
    mode: TailorMode,
) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    source_metrics = extract_metric_tokens(source_bullet)
    rewritten_metrics = extract_metric_tokens(rewritten_bullet)

    if rewritten_metrics - source_metrics:
        warnings.append('Removed unsupported numeric claim from bullet rewrite.')
        if mode == TailorMode.HARD_TRUTH:
            return source_bullet, warnings
        rewritten_bullet = _strip_new_metrics(rewritten_bullet, source_metrics)

    normalized_allowed = {normalize_token(term) for term in allowed_terms if normalize_token(term)}
    normalized_known = {normalize_token(term) for term in known_terms if normalize_token(term)}
    present_terms = detect_terms_present(rewritten_bullet, normalized_known)
    disallowed = {term for term in present_terms if term not in normalized_allowed}

    if disallowed:
        warnings.append('Removed unsupported technology/term from bullet rewrite.')
        if mode == TailorMode.HARD_TRUTH:
            return source_bullet, warnings
        rewritten_bullet = _strip_terms(rewritten_bullet, disallowed)

    cleaned = _normalize_spacing(rewritten_bullet).strip(' ;,-')
    if not cleaned:
        cleaned = source_bullet
    elif _is_low_quality_rewrite(source_bullet, cleaned):
        warnings.append('Kept original bullet due to low-quality rewrite.')
        cleaned = source_bullet
    return cleaned, warnings


def _is_low_quality_rewrite(source: str, rewritten: str) -> bool:
    source_words = source.split()
    rewritten_words = rewritten.split()
    if len(rewritten_words) < 6:
        return True
    if len(rewritten) < max(24, int(len(source) * 0.58)):
        return True

    trailing_stop = {'to', 'for', 'with', 'and', 'or', 'of', 'in', 'on', 'via', 'using', 'by'}
    last_word = rewritten_words[-1].strip('.,;:').lower()
    if last_word in trailing_stop:
        return True
    return False


def _strip_terms(text: str, terms: Set[str]) -> str:
    cleaned = text
    for term in sorted(terms, key=len, reverse=True):
        cleaned = re.sub(rf'(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b(and|or)\s+(,|\.|;)', r'\2', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r',\s*(and|or)\b', ',', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+-\s+', ' ', cleaned)
    return _normalize_spacing(cleaned)


def _strip_new_metrics(text: str, allowed_metrics: Set[str]) -> str:
    def _replace(match: re.Match[str]) -> str:
        token = match.group(0).lower()
        return match.group(0) if token in allowed_metrics else ''

    cleaned = re.sub(r'\b\d+(?:\.\d+)?(?:%|x|k|m|\+)?\b', _replace, text)
    cleaned = re.sub(r'\b(and|or)\s+(,|\.|;)', r'\2', cleaned, flags=re.IGNORECASE)
    return _normalize_spacing(cleaned)


def _aggressive_rephrase(bullet: str, jd_keywords: List[str]) -> str:
    replacements = {
        'worked on': 'delivered',
        'helped': 'drove',
        'responsible for': 'owned',
        'assisted with': 'executed',
        'involved in': 'led',
    }
    rewritten = bullet
    lowered = rewritten.lower()
    for old, new in replacements.items():
        if old in lowered:
            rewritten = re.sub(old, new, rewritten, flags=re.IGNORECASE)
            break

    if rewritten and rewritten[0].islower():
        rewritten = rewritten[0].upper() + rewritten[1:]

    tokens = set(tokenize(rewritten))
    for keyword in jd_keywords:
        norm = normalize_token(keyword)
        if norm and norm in tokens:
            return rewritten
    return rewritten


def rewrite_candidate_bullets(
    candidate: CandidateItem,
    jd: JDAnalysis,
    mode: TailorMode,
    llm: Optional[LLMService],
    known_terms: Set[str],
    warnings: List[str],
) -> List[str]:
    source_bullets = candidate.bullets[:]
    if not source_bullets:
        return []

    allowed_terms: Set[str] = set(candidate.tech + candidate.tags)
    allowed_terms.update(tokenize(' '.join(source_bullets)))

    rewritten = source_bullets
    if llm and llm.available:
        try:
            rewritten = llm.rewrite_bullets(
                item_title=candidate.title,
                source_bullets=source_bullets,
                jd_keywords=jd.required_skills + jd.target_role_keywords,
                allowed_tech=list(allowed_terms),
                mode=mode,
            )
        except Exception as exc:
            warnings.append(f'LLM rewrite failed for {candidate.title}; using deterministic fallback ({exc}).')

    if rewritten is source_bullets:
        if mode == TailorMode.FUCK_IT:
            rewritten = [_aggressive_rephrase(bullet, jd.required_skills + jd.target_role_keywords) for bullet in source_bullets]

    final_bullets: List[str] = []
    for source_bullet, rewritten_bullet in zip(source_bullets, rewritten):
        final_bullet, bullet_warnings = enforce_bullet_constraints(
            source_bullet=source_bullet,
            rewritten_bullet=rewritten_bullet,
            allowed_terms=allowed_terms,
            known_terms=known_terms,
            mode=mode,
        )
        final_bullets.append(final_bullet)
        warnings.extend(bullet_warnings)

    return final_bullets


def _collect_known_terms(candidates: Sequence[CandidateSource], jd: JDAnalysis) -> Set[str]:
    terms: Set[str] = set()
    for candidate_source in candidates:
        candidate = candidate_source.candidate
        terms.update(tokenize(' '.join(candidate.tech + candidate.tags + candidate.bullets + [candidate.title])))
    terms.update(tokenize(' '.join(jd.required_skills + jd.nice_to_haves + jd.target_role_keywords)))
    return {term for term in terms if len(term) > 1}


def tailor_resume(
    *,
    base_resume: CanonicalResume,
    vault_items: Sequence[Tuple[str, VaultItem]],
    jd_text: str,
    mode: TailorMode,
    llm: Optional[LLMService],
    job_title_hint: Optional[str] = None,
    selection_feedback: Optional[Dict[str, List[str]]] = None,
    optimization_level: int = 1,
) -> TailorResult:
    jd = analyze_jd_text(jd_text, llm=llm)
    candidates = build_candidate_pool(base_resume, vault_items)
    scored = score_candidates(
        candidates,
        jd,
        selection_feedback=selection_feedback,
        optimization_level=optimization_level,
        jd_text=jd_text,
        scoring_resume=base_resume,
    )

    known_terms = _collect_known_terms(candidates, jd)
    warnings: List[str] = []

    selected = _select_top_candidates(scored, jd, optimization_level=optimization_level)
    score_lookup: Dict[str, float] = {}
    for candidate_source, score in selected:
        candidate = candidate_source.candidate
        title_key = normalize_token(candidate.title)
        if title_key:
            score_lookup[title_key] = max(score, score_lookup.get(title_key, score))
        if candidate.company and candidate.role:
            role_company = normalize_token(f'{candidate.role}-{candidate.company}')
            if role_company:
                score_lookup[role_company] = max(score, score_lookup.get(role_company, score))

    selected_items: List[SelectedItem] = []
    tailored = CanonicalResume.model_validate(deepcopy(base_resume.model_dump()))
    tailored.experience = []
    tailored.projects = []

    for candidate_source, score in selected:
        candidate = candidate_source.candidate
        rewritten_bullets = rewrite_candidate_bullets(candidate, jd, mode, llm, known_terms, warnings)

        if candidate_source.origin['kind'] == 'base_experience':
            index = int(candidate_source.origin['index'])
            source_entry = base_resume.experience[index]
            tailored.experience.append(
                ExperienceEntry(
                    company=source_entry.company,
                    title=source_entry.title,
                    location=source_entry.location,
                    dates=source_entry.dates,
                    bullets=rewritten_bullets,
                )
            )
        elif candidate_source.origin['kind'] == 'base_project':
            index = int(candidate_source.origin['index'])
            source_entry = base_resume.projects[index]
            tailored.projects.append(
                ProjectEntry(
                    name=source_entry.name,
                    link=source_entry.link,
                    dates=source_entry.dates,
                    tech=source_entry.tech,
                    bullets=rewritten_bullets,
                    section=_project_section_from_entry(source_entry),
                )
            )
        elif candidate_source.origin['kind'] == 'base_education':
            # Education entries are preserved from base resume; coursework candidate is only used for scoring.
            pass
        else:
            item: VaultItem = candidate_source.origin['item']  # type: ignore[assignment]
            if item.type.value == 'job':
                company = candidate.company or item.title
                role = candidate.role or item.title
                location = candidate.location or _extract_tag_value(item.tags, 'location:') or ''
                tailored.experience.append(
                    ExperienceEntry(
                        company=company,
                        title=role,
                        location=location,
                        dates=item.dates or DateRange(),
                        bullets=rewritten_bullets,
                    )
                )
            elif item.type.value == 'award':
                awards = list(tailored.awards or [])
                if item.title not in awards:
                    awards.append(item.title)
                tailored.awards = awards
            else:
                section = _project_section_from_tags(item.tags, item.title)
                tailored.projects.append(
                    ProjectEntry(
                        name=item.title,
                        link=item.links[0] if item.links else None,
                        dates=item.dates,
                        tech=item.tech,
                        bullets=rewritten_bullets,
                        section=section,
                    )
                )

        selected_items.append(
            SelectedItem(
                source_type=candidate.source_type,
                source_id=candidate.source_id,
                title=candidate.title,
                score=score,
            )
        )

    tailored = prune_resume_for_one_page(tailored, score_lookup, warnings)
    tailored = _ensure_target_summary(tailored, jd, jd_text, job_title_hint)

    required_terms = _required_must_have_terms(jd)
    required_coverage = _selected_required_coverage(selected, required_terms) if required_terms else 1.0
    if required_terms:
        warnings.append(
            f'Must-have coverage after selection: {required_coverage * 100:.1f}% ({len(required_terms)} required terms).'
        )

    covered = _covered_keywords(jd, tailored)
    required_keywords = unique_preserve_order(tokenize(' '.join(jd.required_skills + jd.target_role_keywords)))
    missed = [term for term in required_keywords if term not in covered]

    report = TailorReport(
        chosen_items=selected_items,
        keywords_covered=sorted(covered),
        keywords_missed=missed,
        warnings=unique_preserve_order(warnings),
        mode=mode,
    )
    return TailorResult(tailored_resume=tailored, report=report)


def _is_experience_candidate(candidate_source: CandidateSource) -> bool:
    kind = str(candidate_source.origin.get('kind', ''))
    return kind == 'base_experience' or (kind == 'vault' and candidate_source.candidate.source_type == 'vault:job')


def _is_coursework_candidate(candidate_source: CandidateSource) -> bool:
    kind = str(candidate_source.origin.get('kind', ''))
    return kind == 'base_education' or candidate_source.candidate.source_type == 'vault:coursework'


def _is_project_candidate(candidate_source: CandidateSource) -> bool:
    if _is_experience_candidate(candidate_source) or _is_coursework_candidate(candidate_source):
        return False

    source_type = candidate_source.candidate.source_type
    if source_type in {'vault:award', 'vault:skillset'}:
        return False
    return True


def _selection_limits(jd: JDAnalysis, available_projects: int) -> Tuple[int, int]:
    jd_tokens = set(tokenize(' '.join(jd.target_role_keywords + jd.required_skills + jd.responsibilities)))
    internship_like = any(token in jd_tokens for token in {'intern', 'internship', 'entry', 'newgrad', 'student'})
    jd_specificity = len({
        token for token in _canonicalize_terms(jd_tokens)
        if _is_high_signal_term(token)
    })

    exp_limit = 2 if internship_like else MAX_EXP_ITEMS
    if internship_like:
        project_limit = 4
    else:
        project_limit = 3 if exp_limit >= 3 else 4
        if jd_specificity >= 16:
            project_limit += 1
    project_limit = min(project_limit, MAX_PROJECT_ITEMS, available_projects)
    if available_projects > 0:
        project_limit = max(project_limit, min(MIN_PROJECT_ITEMS, available_projects))

    return exp_limit, project_limit


def _candidate_similarity(left: CandidateSource, right: CandidateSource) -> float:
    left_terms = _candidate_focus_terms(left.candidate) or _candidate_token_set(left.candidate)
    right_terms = _candidate_focus_terms(right.candidate) or _candidate_token_set(right.candidate)
    union = left_terms | right_terms
    if not union:
        return 0.0
    return len(left_terms & right_terms) / len(union)


def _select_diverse_candidates(
    pool: Sequence[Tuple[CandidateSource, float]],
    limit: int,
) -> List[Tuple[CandidateSource, float]]:
    if limit <= 0 or not pool:
        return []

    remaining = sorted(pool, key=lambda pair: (-pair[1], pair[0].candidate.title.lower()))
    selected: List[Tuple[CandidateSource, float]] = [remaining.pop(0)]
    best_relevance = selected[0][1]
    relevance_floor = max(0.0, best_relevance * 0.65)

    while remaining and len(selected) < limit:
        strong_indices = [index for index, (_, score) in enumerate(remaining) if score >= relevance_floor]
        scan_indices = strong_indices if strong_indices else list(range(len(remaining)))

        best_index = scan_indices[0]
        best_score = float('-inf')
        for index in scan_indices:
            candidate_source, score = remaining[index]
            max_similarity = max(_candidate_similarity(candidate_source, chosen[0]) for chosen in selected)
            diversity_penalty = 0.9 if score >= best_relevance * 0.85 else 0.6
            mmr_score = score - (max_similarity * diversity_penalty)
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index
        selected.append(remaining.pop(best_index))

    return selected


def _coverage_swap_pass(
    selected: List[Tuple[CandidateSource, float]],
    scored: Sequence[Tuple[CandidateSource, float]],
    required_terms: Set[str],
    total_limit: int,
    optimization_level: int,
) -> List[Tuple[CandidateSource, float]]:
    selected_ids = {candidate_source.candidate.source_id for candidate_source, _ in selected}
    current_coverage = _selected_required_coverage(selected, required_terms)
    current_terms: Set[str] = set()
    for candidate_source, _ in selected:
        current_terms.update(_candidate_token_set(candidate_source.candidate))
    missing = required_terms - current_terms
    if not missing:
        return selected

    candidates = [pair for pair in scored if pair[0].candidate.source_id not in selected_ids]
    if not candidates:
        return selected

    def _candidate_gain(pair: Tuple[CandidateSource, float]) -> Tuple[int, float, str]:
        candidate_source, score = pair
        hits = len(_candidate_token_set(candidate_source.candidate) & missing)
        return (hits, score, candidate_source.candidate.title.lower())

    best_candidate = sorted(candidates, key=_candidate_gain, reverse=True)[0]
    best_hits = len(_candidate_token_set(best_candidate[0].candidate) & missing)
    if best_hits <= 0:
        return selected

    if len(selected) < total_limit:
        selected.append(best_candidate)
        return selected

    replacement_order = []
    for idx, (candidate_source, score) in enumerate(selected):
        replace_priority = 2
        if _is_project_candidate(candidate_source):
            replace_priority = 0
        elif _is_coursework_candidate(candidate_source):
            replace_priority = 1
        elif _is_experience_candidate(candidate_source):
            replace_priority = 3

        if replace_priority >= 3 and optimization_level < 4:
            continue

        contribution = len(_candidate_token_set(candidate_source.candidate) & required_terms)
        replacement_order.append((replace_priority, contribution, score, idx))

    replacement_order = sorted(replacement_order, key=lambda row: (row[0], row[1], row[2], row[3]))
    for _, _, _, idx in replacement_order:
        current_source, _ = selected[idx]
        current_source_terms = _candidate_token_set(current_source.candidate)
        candidate_terms = _candidate_token_set(best_candidate[0].candidate)
        new_terms = (current_terms - current_source_terms) | candidate_terms
        new_coverage = len(new_terms & required_terms) / max(1, len(required_terms))
        if new_coverage > current_coverage:
            selected[idx] = best_candidate
            return selected

    return selected


def _enforce_must_have_coverage(
    selected: List[Tuple[CandidateSource, float]],
    scored: Sequence[Tuple[CandidateSource, float]],
    jd: JDAnalysis,
    total_limit: int,
    optimization_level: int,
) -> List[Tuple[CandidateSource, float]]:
    required_terms = _required_must_have_terms(jd)
    if len(required_terms) < MUST_HAVE_MIN_TERMS:
        return selected

    target_coverage = min(0.85, MUST_HAVE_MIN_COVERAGE_BASE + (max(0, optimization_level - 1) * MUST_HAVE_MIN_COVERAGE_STEP))
    attempts = 0
    max_attempts = max(1, len(scored))
    while _selected_required_coverage(selected, required_terms) < target_coverage and attempts < max_attempts:
        previous_ids = [candidate_source.candidate.source_id for candidate_source, _ in selected]
        selected = _coverage_swap_pass(
            selected=selected,
            scored=scored,
            required_terms=required_terms,
            total_limit=total_limit,
            optimization_level=optimization_level,
        )
        attempts += 1
        updated_ids = [candidate_source.candidate.source_id for candidate_source, _ in selected]
        if updated_ids == previous_ids:
            break
    return selected


def _select_top_candidates(
    scored: Sequence[Tuple[CandidateSource, float]],
    jd: JDAnalysis,
    optimization_level: int = 1,
) -> List[Tuple[CandidateSource, float]]:
    if not scored:
        return []

    experience_pool = [pair for pair in scored if _is_experience_candidate(pair[0])]
    project_pool = [pair for pair in scored if _is_project_candidate(pair[0]) and len(pair[0].candidate.bullets) >= MIN_PROJECT_BULLETS]
    coursework_pool = [pair for pair in scored if _is_coursework_candidate(pair[0])]

    exp_limit, project_limit = _selection_limits(jd, available_projects=len(project_pool))
    selected_experience = experience_pool[:exp_limit]
    selected_projects = _select_diverse_candidates(project_pool, project_limit)

    min_projects = min(MIN_PROJECT_ITEMS, len(project_pool))
    if len(selected_projects) < min_projects:
        selected_ids = {candidate_source.candidate.source_id for candidate_source, _ in selected_projects}
        top_project_score = project_pool[0][1] if project_pool else 0.0
        supplemental_floor = max(0.0, top_project_score * 0.45)
        for candidate_source, score in project_pool:
            if candidate_source.candidate.source_id in selected_ids:
                continue
            if score < supplemental_floor:
                continue
            selected_projects.append((candidate_source, score))
            selected_ids.add(candidate_source.candidate.source_id)
            if len(selected_projects) >= min_projects:
                break

    selected: List[Tuple[CandidateSource, float]] = []
    selected.extend(selected_experience)
    selected.extend(selected_projects)

    selected_ids = {candidate_source.candidate.source_id for candidate_source, _ in selected}
    if coursework_pool:
        top_coursework = coursework_pool[0]
        if top_coursework[1] > 0 and top_coursework[0].candidate.source_id not in selected_ids:
            selected.append(top_coursework)
            selected_ids.add(top_coursework[0].candidate.source_id)

    total_limit = exp_limit + project_limit + 1
    for candidate_source, score in scored:
        if len(selected) >= total_limit:
            break
        source_id = candidate_source.candidate.source_id
        if source_id in selected_ids:
            continue
        if score <= 0:
            continue
        selected.append((candidate_source, score))
        selected_ids.add(source_id)

    selected = _enforce_must_have_coverage(
        selected=list(selected),
        scored=scored,
        jd=jd,
        total_limit=total_limit,
        optimization_level=optimization_level,
    )

    if not selected:
        return list(scored[:3])

    return sorted(selected, key=lambda pair: (-pair[1], pair[0].candidate.title.lower()))


def prune_resume_for_one_page(
    resume: CanonicalResume,
    score_lookup: Dict[str, float],
    warnings: List[str],
    max_total_bullets: int = MAX_TOTAL_BULLETS,
    max_bullets_per_item: int = MAX_BULLETS_PER_ITEM,
    max_page_units: int = MAX_PAGE_UNITS,
) -> CanonicalResume:
    pruned = CanonicalResume.model_validate(deepcopy(resume.model_dump()))
    pruned.education = pruned.education[:2]

    experience_cap = max(1, max_bullets_per_item)
    project_cap = max(MIN_PROJECT_BULLETS, max_bullets_per_item - 1)

    for item in pruned.experience:
        if len(item.bullets) > experience_cap:
            warnings.append(f'Pruned bullets for experience: {item.title}')
            item.bullets = item.bullets[:experience_cap]

    for item in pruned.projects:
        if len(item.bullets) > project_cap:
            warnings.append(f'Pruned bullets for project: {item.name}')
            item.bullets = item.bullets[:project_cap]

    _compact_skills_for_space(pruned, warnings, max_per_category=9)

    while True:
        total_bullets = _count_bullets(pruned)
        page_units = _estimate_page_units(pruned)
        if total_bullets <= max_total_bullets and page_units <= max_page_units:
            break

        # Priority: preserve work experience + skills, trim projects first.
        if pruned.projects:
            project_index = _lowest_scoring_project_index(pruned, score_lookup)
            if project_index is not None:
                project = pruned.projects[project_index]
                if len(project.bullets) > MIN_PROJECT_BULLETS:
                    project.bullets = project.bullets[:-1]
                    warnings.append(f'Trimmed project bullet for one-page constraint: {project.name}')
                else:
                    removed = pruned.projects.pop(project_index)
                    warnings.append(f'Removed low-priority project for one-page constraint: {removed.name}')
                continue

        exp_index = _experience_with_extra_bullets(pruned, min_bullets=1)
        if exp_index is not None:
            pruned.experience[exp_index].bullets = pruned.experience[exp_index].bullets[:-1]
            warnings.append(f'Trimmed experience bullet for one-page constraint: {pruned.experience[exp_index].title}')
            continue

        changed = _compact_skills_for_space(pruned, warnings, max_per_category=6)
        if changed:
            continue

        break

    return pruned


def tighten_resume_for_one_page(
    resume: CanonicalResume,
    score_lookup: Dict[str, float],
    warnings: List[str],
    level: int = 1,
) -> CanonicalResume:
    level = max(1, min(level, 3))
    tightened_bullets = max(1, MAX_BULLETS_PER_ITEM - level)
    tightened_total = max(7, MAX_TOTAL_BULLETS - (level * 2))
    tightened_units = max(30, MAX_PAGE_UNITS - (level * 4))

    return prune_resume_for_one_page(
        resume=resume,
        score_lookup=score_lookup,
        warnings=warnings,
        max_total_bullets=tightened_total,
        max_bullets_per_item=tightened_bullets,
        max_page_units=tightened_units,
    )


def _count_bullets(resume: CanonicalResume) -> int:
    return sum(len(item.bullets) for item in resume.experience) + sum(len(item.bullets) for item in resume.projects)


def _estimate_page_units(resume: CanonicalResume) -> int:
    units = 6  # header/contact block
    if resume.summary:
        units += 2

    if resume.education:
        units += 2
        for entry in resume.education:
            units += 2
            if entry.coursework:
                units += 1

    if resume.experience:
        units += 2
        for entry in resume.experience:
            units += 2 + len(entry.bullets)

    if resume.skills and resume.skills.categories:
        units += 2 + len(resume.skills.categories)

    if resume.projects:
        units += 2
        for project in resume.projects:
            units += 2 + len(project.bullets)

    if resume.awards:
        units += 1 + len(resume.awards)

    return units


def _lowest_scoring_project_index(resume: CanonicalResume, score_lookup: Dict[str, float]) -> Optional[int]:
    if not resume.projects:
        return None
    candidates: List[Tuple[int, float]] = []
    for idx, entry in enumerate(resume.projects):
        key = normalize_token(entry.name)
        score = score_lookup.get(key, 0.0)
        candidates.append((idx, score))
    if not candidates:
        return None
    index, _ = sorted(candidates, key=lambda row: (row[1], row[0]))[0]
    return index


def _experience_with_extra_bullets(resume: CanonicalResume, min_bullets: int = 1) -> Optional[int]:
    candidates = [(idx, len(entry.bullets)) for idx, entry in enumerate(resume.experience) if len(entry.bullets) > min_bullets]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (-row[1], row[0]))[0][0]


def _compact_skills_for_space(resume: CanonicalResume, warnings: List[str], max_per_category: int) -> bool:
    if not resume.skills or not resume.skills.categories:
        return False

    changed = False
    compacted: Dict[str, List[str]] = {}
    for category, entries in resume.skills.categories.items():
        if len(entries) > max_per_category:
            compacted[category] = entries[:max_per_category]
            warnings.append(f'Compacted skills in category for one-page constraint: {category}')
            changed = True
        else:
            compacted[category] = entries

    if changed:
        resume.skills.categories = compacted
    return changed


def expand_resume_with_projects(
    *,
    resume: CanonicalResume,
    base_resume: CanonicalResume,
    vault_items: Sequence[Tuple[str, VaultItem]],
    score_lookup: Dict[str, float],
    used_expansions: Set[str],
) -> Tuple[CanonicalResume, bool, Optional[str]]:
    current_names = {normalize_token(project.name) for project in resume.projects}

    candidates: List[Tuple[str, ProjectEntry, float]] = []
    for project in base_resume.projects:
        if len([bullet for bullet in project.bullets if bullet.strip()]) < MIN_PROJECT_BULLETS:
            continue
        key = normalize_token(project.name)
        marker = f'project:{key}'
        if not key or key in current_names or marker in used_expansions:
            continue
        score = score_lookup.get(key, 0.0)
        candidates.append((marker, project, score))

    for item_id, item in vault_items:
        if item.type.value in {'job', 'award'}:
            continue
        key = normalize_token(item.title)
        marker = f'vault-project:{item_id}'
        if not key or key in current_names or marker in used_expansions:
            continue
        bullets = [bullet.text for bullet in item.bullets if bullet.text.strip()]
        if len(bullets) < MIN_PROJECT_BULLETS:
            continue
        section = _project_section_from_tags(item.tags, item.title)
        project = ProjectEntry(
            name=item.title,
            link=item.links[0] if item.links else None,
            dates=item.dates,
            tech=item.tech,
            bullets=bullets,
            section=section,
        )
        score = score_lookup.get(key, 0.0)
        candidates.append((marker, project, score))

    candidates = sorted(candidates, key=lambda row: (-row[2], row[1].name.lower()))
    if candidates:
        marker, source_project, _ = candidates[0]
        cloned = CanonicalResume.model_validate(deepcopy(resume.model_dump()))
        section = _project_section_from_entry(source_project)
        cloned.projects.append(
            ProjectEntry(
                name=source_project.name,
                link=source_project.link,
                dates=source_project.dates,
                tech=source_project.tech[:5],
                bullets=source_project.bullets[:max(MIN_PROJECT_BULLETS, 2)],
                section=section,
            )
        )
        return cloned, True, marker

    return resume, False, None


def _covered_keywords(jd: JDAnalysis, resume: CanonicalResume) -> Set[str]:
    text_parts = []
    text_parts.extend(jd.target_role_keywords)
    text_parts.extend(jd.required_skills)
    text_parts.extend(jd.nice_to_haves)

    resume_text = [resume.summary or '']
    for entry in resume.experience:
        resume_text.append(entry.company)
        resume_text.append(entry.title)
        resume_text.extend(entry.bullets)
    for project in resume.projects:
        resume_text.append(project.name)
        resume_text.extend(project.tech)
        resume_text.extend(project.bullets)
    for skills in resume.skills.categories.values():
        resume_text.extend(skills)

    jd_tokens = set(_canonicalize_terms(tokenize(' '.join(text_parts))))
    resume_tokens = set(_canonicalize_terms(tokenize(' '.join(resume_text))))
    return jd_tokens & resume_tokens
