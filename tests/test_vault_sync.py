from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.models import CanonicalResume, DateRange, EducationEntry, ExperienceEntry, Identity, ProjectEntry, Skills, VaultItem, VaultItemType
from app.services.repository import DataRepository
from app.services.tailoring import build_candidate_pool
from app.services.vault_sync import sync_base_resume_to_vault


def _sample_resume() -> CanonicalResume:
    return CanonicalResume(
        identity=Identity(name='User', email='u@example.com', phone='555', location='TX', links=[]),
        summary='Builder',
        education=[
            EducationEntry(
                school='State U',
                degree='BS',
                major='CS',
                minors=[],
                gpa='3.8',
                dates=DateRange(start='2019', end='2023'),
                coursework=['ML', 'Distributed Systems'],
            )
        ],
        experience=[
            ExperienceEntry(
                company='Acme',
                title='SWE Intern',
                location='Austin, TX',
                dates=DateRange(start='2024', end='2024'),
                bullets=['Built Python APIs for ingestion.'],
            )
        ],
        projects=[
            ProjectEntry(
                name='Fraud ML Pipeline',
                link=None,
                dates=DateRange(start='2025', end='2025'),
                tech=['Python', 'XGBoost'],
                bullets=['Trained models for fraud detection.', 'Built evaluation pipelines for model validation.'],
            )
        ],
        skills=Skills(categories={'Languages': ['Python']}),
        awards=['Dean List'],
    )


def test_sync_base_resume_to_vault_creates_and_prunes(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    settings.ensure_directories()
    repo = DataRepository(settings)

    resume = _sample_resume()
    first = sync_base_resume_to_vault(repo, resume)
    assert first.created >= 3

    items = repo.list_vault_items()
    ids = {item_id for item_id, _ in items}
    assert any(item_id.startswith('base_exp_') for item_id in ids)
    assert any(item_id.startswith('base_proj_') for item_id in ids)

    resume.projects = []
    second = sync_base_resume_to_vault(repo, resume)
    assert second.deleted >= 1


def test_sync_base_resume_to_vault_limits_projects_to_four(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    settings.ensure_directories()
    repo = DataRepository(settings)

    resume = _sample_resume()
    resume.projects = [
        ProjectEntry(name='Project One Jan 2025 – Mar 2025', tech=[], bullets=[], section='projects'),
        ProjectEntry(name='Tech: Python, FastAPI', tech=[], bullets=['Fragment'], section='projects'),
        ProjectEntry(name='project fragment with no metadata.', tech=[], bullets=['Fragment'], section='projects'),
        ProjectEntry(name='Project Two Apr 2025 – Jun 2025', tech=[], bullets=[], section='projects'),
        ProjectEntry(name='Project Three Jul 2025 – Sep 2025', tech=[], bullets=[], section='projects'),
        ProjectEntry(name='Project Four Oct 2025 – Dec 2025', tech=[], bullets=[], section='projects'),
        ProjectEntry(name='Project Five Jan 2026 – Feb 2026', tech=[], bullets=[], section='projects'),
    ]

    sync_base_resume_to_vault(repo, resume)
    project_items = [(item_id, item) for item_id, item in repo.list_vault_items() if item_id.startswith('base_proj_')]
    assert len(project_items) == 4
    titles = {item.title for _, item in project_items}
    assert 'Tech: Python, FastAPI' not in titles
    assert 'project fragment with no metadata.' not in titles


def test_build_candidate_pool_prefers_vault_items() -> None:
    base = _sample_resume()
    vault_item = VaultItem(
        type=VaultItemType.project,
        title='Vault Project',
        dates=DateRange(start='2025', end='2025'),
        tags=['vault'],
        tech=['Python'],
        bullets=[{'text': 'Built vault-first project.'}, {'text': 'Added monitoring for model behavior.'}],
        links=[],
        source_artifacts=['base_resume:project:0'],
    )

    candidates = build_candidate_pool(base, [('id1', vault_item)])
    assert candidates
    assert all(candidate.candidate.source_type.startswith('vault:') for candidate in candidates)


def test_build_candidate_pool_is_vault_only_when_vault_is_empty() -> None:
    base = _sample_resume()
    candidates = build_candidate_pool(base, [])
    assert candidates == []
