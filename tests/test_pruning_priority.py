from __future__ import annotations

from app.models import CanonicalResume, DateRange, EducationEntry, ExperienceEntry, Identity, ProjectEntry, Skills
from app.services.tailoring import prune_resume_for_one_page


def test_pruning_prioritizes_projects_before_experience() -> None:
    resume = CanonicalResume(
        identity=Identity(name='Test User', email='a@b.com', phone='555', location='TX', links=[]),
        education=[
            EducationEntry(
                school='School',
                degree='BS',
                major='CS',
                minors=[],
                gpa='',
                dates=DateRange(start='2019', end='2023'),
                coursework=['ML', 'Systems'],
            )
        ],
        experience=[
            ExperienceEntry(
                company='Acme',
                title='Engineer',
                location='TX',
                dates=DateRange(start='2024', end='Present'),
                bullets=['e1', 'e2', 'e3'],
            )
        ],
        projects=[
            ProjectEntry(name='Project A', dates=DateRange(start='2024', end='2024'), tech=['Python'], bullets=['a1', 'a2', 'a3']),
            ProjectEntry(name='Project B', dates=DateRange(start='2024', end='2024'), tech=['Python'], bullets=['b1', 'b2', 'b3']),
        ],
        skills=Skills(categories={'Languages': ['Python', 'Java'], 'ML': ['XGBoost', 'PyTorch']}),
        awards=[],
    )

    warnings: list[str] = []
    pruned = prune_resume_for_one_page(
        resume=resume,
        score_lookup={'projecta': 1.0, 'projectb': 0.2},
        warnings=warnings,
        max_total_bullets=4,
        max_bullets_per_item=2,
        max_page_units=20,
    )

    assert len(pruned.experience) == 1
    assert len(pruned.experience[0].bullets) >= 1
    assert len(pruned.projects) <= 2
    assert any('project' in warning.lower() for warning in warnings)
