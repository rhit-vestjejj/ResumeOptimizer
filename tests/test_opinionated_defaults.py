from __future__ import annotations

from app.models import CanonicalResume, DateRange, ExperienceEntry, Identity, JDAnalysis, ProjectEntry, Skills
from app.services.tailoring import MAX_PROJECT_ITEMS, _selection_limits, prune_resume_for_one_page


def test_selection_limits_keep_project_range_opinionated() -> None:
    jd = JDAnalysis(
        target_role_keywords=['Software Engineer'],
        required_skills=['Python', 'FastAPI', 'PostgreSQL'],
        responsibilities=['Build backend APIs', 'Ship reliable services'],
    )

    _, project_limit = _selection_limits(jd, available_projects=8)
    assert project_limit == MAX_PROJECT_ITEMS

    _, three_project_limit = _selection_limits(jd, available_projects=3)
    assert three_project_limit == 3


def test_prune_enforces_project_cap_of_four() -> None:
    projects = [
        ProjectEntry(
            name=f'Project {idx}',
            dates=DateRange(start='2024', end='2024'),
            tech=['Python'],
            bullets=['Delivered backend feature.', 'Improved reliability metrics.'],
        )
        for idx in range(1, 6)
    ]
    resume = CanonicalResume(
        identity=Identity(name='Opinionated User', email='a@b.com', phone='555', location='TX', links=[]),
        experience=[
            ExperienceEntry(
                company='Acme',
                title='Engineer',
                location='TX',
                dates=DateRange(start='2024', end='Present'),
                bullets=['Built APIs.', 'Improved service latency.'],
            )
        ],
        projects=projects,
        skills=Skills(categories={'Languages': ['Python', 'SQL']}),
        awards=[],
    )

    warnings: list[str] = []
    pruned = prune_resume_for_one_page(
        resume=resume,
        score_lookup={f'project{idx}': float(6 - idx) for idx in range(1, 6)},
        warnings=warnings,
        max_total_bullets=30,
        max_page_units=200,
    )

    assert len(pruned.projects) == MAX_PROJECT_ITEMS
    assert any('project cap' in warning.lower() for warning in warnings)
