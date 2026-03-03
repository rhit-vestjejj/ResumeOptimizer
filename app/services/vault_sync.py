from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from app.models import CanonicalResume, ProjectEntry, VaultItem, VaultItemType
from app.services.repository import DataRepository
from app.utils import slugify

BASE_MARKER_PREFIX = 'base_resume:'
MAX_BASE_PROJECT_ITEMS = 4


@dataclass
class VaultSyncResult:
    created: int
    updated: int
    deleted: int


def sync_base_resume_to_vault(repository: DataRepository, resume: CanonicalResume) -> VaultSyncResult:
    generated = _generate_base_items(resume)
    existing = {item_id: item for item_id, item in repository.list_vault_items()}

    created = 0
    updated = 0
    deleted = 0

    existing_base_ids = {
        item_id
        for item_id, item in existing.items()
        if any(artifact.startswith(BASE_MARKER_PREFIX) for artifact in item.source_artifacts)
    }

    for item_id in sorted(existing_base_ids - set(generated.keys())):
        repository.delete_vault_item(item_id)
        deleted += 1

    for item_id, item in generated.items():
        if item_id in existing:
            repository.save_vault_item(item_id, item)
            updated += 1
        else:
            repository.save_vault_item(item_id, item)
            created += 1

    return VaultSyncResult(created=created, updated=updated, deleted=deleted)


def _generate_base_items(resume: CanonicalResume) -> Dict[str, VaultItem]:
    output: Dict[str, VaultItem] = {}

    for idx, entry in enumerate(resume.experience):
        item_id = _stable_item_id('exp', idx, f'{entry.title}-{entry.company}')
        bullets = [bullet for bullet in entry.bullets if bullet.strip()]
        tags = [
            f'role:{entry.title}',
            f'company:{entry.company}',
            f'location:{entry.location}',
            entry.title,
            entry.company,
            entry.location,
        ]
        output[item_id] = VaultItem(
            type=VaultItemType.job,
            title=entry.title,
            dates=entry.dates,
            tags=[tag for tag in tags if tag],
            tech=_extract_inline_terms(' '.join(bullets)),
            bullets=[{'text': bullet} for bullet in bullets],
            links=[],
            source_artifacts=[f'{BASE_MARKER_PREFIX}experience:{idx}'],
        )

    selected_projects = _select_projects_for_vault(resume.projects, limit=MAX_BASE_PROJECT_ITEMS)
    for idx, project in enumerate(selected_projects):
        item_id = _stable_item_id('proj', idx, project.name)
        bullets = [bullet for bullet in project.bullets if bullet.strip()]
        section = _project_section(project)
        tags = [project.name, f'section:{section}']
        if section == 'minor_projects':
            tags.append('minor_project')
        output[item_id] = VaultItem(
            type=VaultItemType.project,
            title=project.name,
            dates=project.dates,
            tags=[tag for tag in tags if tag],
            tech=project.tech,
            bullets=[{'text': bullet} for bullet in bullets],
            links=[project.link] if project.link else [],
            source_artifacts=[f'{BASE_MARKER_PREFIX}project:{idx}'],
        )

    for idx, education in enumerate(resume.education):
        coursework = [course for course in education.coursework if course.strip()]
        if coursework:
            item_id = _stable_item_id('course', idx, f'{education.school}-coursework')
            output[item_id] = VaultItem(
                type=VaultItemType.coursework,
                title=f'{education.school} Coursework',
                dates=education.dates,
                tags=[education.school, education.major],
                tech=coursework,
                bullets=[{'text': f'Coursework: {", ".join(coursework)}'}],
                links=[],
                source_artifacts=[f'{BASE_MARKER_PREFIX}education:{idx}:coursework'],
            )

    for idx, award in enumerate(resume.awards or []):
        if not award.strip():
            continue
        item_id = _stable_item_id('award', idx, award)
        output[item_id] = VaultItem(
            type=VaultItemType.award,
            title=award,
            tags=['award'],
            tech=[],
            bullets=[{'text': award}],
            links=[],
            source_artifacts=[f'{BASE_MARKER_PREFIX}award:{idx}'],
        )

    return output


def _stable_item_id(prefix: str, index: int, text: str) -> str:
    slug = slugify(text)[:36]
    return f'base_{prefix}_{index}_{slug}'


def _extract_inline_terms(text: str) -> List[str]:
    raw = re.findall(r'[A-Za-z][A-Za-z0-9\+#\.]{1,}', text)
    keep = [token for token in raw if any(ch.isupper() for ch in token) or '+' in token or '#' in token]
    # normalized lowercase with stable order
    seen = set()
    output: List[str] = []
    for token in keep:
        lowered = token.lower()
        if lowered not in seen:
            seen.add(lowered)
            output.append(lowered)
    return output


def _project_section(project: ProjectEntry) -> str:
    normalized = (project.section or '').strip().lower()
    if normalized in {'minor', 'minor_projects', 'minor-projects', 'minorprojects'}:
        return 'minor_projects'
    if 'minor project' in project.name.lower():
        return 'minor_projects'
    return 'projects'


def _select_projects_for_vault(projects: Sequence[ProjectEntry], *, limit: int) -> List[ProjectEntry]:
    if limit <= 0:
        return []
    ranked: List[Tuple[float, int, ProjectEntry]] = []
    for index, project in enumerate(projects):
        score = _project_quality_score(project, index=index)
        ranked.append((score, index, project))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in ranked[:limit]]


def _project_quality_score(project: ProjectEntry, *, index: int) -> float:
    name = (project.name or '').strip()
    if not name:
        return -1_000.0 + index

    score = 0.0
    lowered = name.lower()
    words = re.findall(r'\w+', name)
    has_year_hint = bool(re.search(r'\b(?:19|20)\d{2}\b', name))
    has_title_separator = any(token in name for token in (' - ', ' – ', ' — '))
    bullet_count = len([bullet for bullet in project.bullets if bullet.strip()])

    if has_year_hint:
        score += 90.0
    if has_title_separator:
        score += 20.0
    if bullet_count >= 2:
        score += 20.0
    elif bullet_count == 1:
        score += 10.0
    if project.tech:
        score += 12.0
    if lowered.startswith('tech:'):
        score -= 120.0
    if lowered.endswith('.'):
        score -= 20.0
    if len(words) < 3:
        score -= 35.0
    if name[0].islower():
        score -= 30.0
    if len(name) < 18:
        score -= 15.0

    # Stable tie-breaker keeps earlier canonical order.
    score -= index * 0.001
    return score
