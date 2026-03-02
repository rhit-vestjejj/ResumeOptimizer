from __future__ import annotations

from app.models import CanonicalResume, DateRange, EducationEntry, ExperienceEntry, Identity, ProjectEntry, Skills
from app.services.tailoring import prune_resume_for_one_page


def test_prune_to_one_page_constraints_by_bullet_count() -> None:
    resume = CanonicalResume(
        identity=Identity(name='Test User', email='a@b.com', phone='555', location='TX', links=[]),
        education=[
            EducationEntry(
                school='School',
                degree='BS',
                major='CS',
                minors=[],
                gpa='3.8',
                dates=DateRange(start='2019', end='2023'),
                coursework=['A', 'B'],
            )
        ],
        experience=[
            ExperienceEntry(
                company='A',
                title='Engineer',
                location='TX',
                dates=DateRange(start='2023', end='Present'),
                bullets=['b1', 'b2', 'b3', 'b4'],
            ),
            ExperienceEntry(
                company='B',
                title='Intern',
                location='TX',
                dates=DateRange(start='2022', end='2022'),
                bullets=['c1', 'c2', 'c3'],
            ),
        ],
        projects=[
            ProjectEntry(name='P1', dates=DateRange(start='2021', end='2022'), tech=['Python'], bullets=['p1', 'p2', 'p3'])
        ],
        skills=Skills(categories={'Languages': ['Python']}),
        awards=[],
    )

    warnings: list[str] = []
    pruned = prune_resume_for_one_page(
        resume,
        score_lookup={},
        warnings=warnings,
        max_total_bullets=5,
        max_bullets_per_item=2,
    )

    total_bullets = sum(len(e.bullets) for e in pruned.experience) + sum(len(p.bullets) for p in pruned.projects)
    assert total_bullets <= 5
    assert all(len(e.bullets) <= 2 for e in pruned.experience)
    assert all(len(p.bullets) <= 2 for p in pruned.projects)
