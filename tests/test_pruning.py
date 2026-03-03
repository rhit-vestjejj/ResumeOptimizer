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


def test_prune_keeps_three_bullets_per_project_by_default() -> None:
    resume = CanonicalResume(
        identity=Identity(name='Taylor User', email='t@example.com', phone='555-222-1111', location='TX', links=[]),
        education=[],
        experience=[],
        projects=[
            ProjectEntry(
                name='Project Keep Three',
                dates=DateRange(start='2024', end='2025'),
                tech=['Python', 'SQL'],
                bullets=['b1', 'b2', 'b3', 'b4'],
            )
        ],
        skills=Skills(categories={'Languages': ['Python']}),
        certifications=[],
        awards=[],
    )

    warnings: list[str] = []
    pruned = prune_resume_for_one_page(
        resume=resume,
        score_lookup={'projectkeepthree': 10.0},
        warnings=warnings,
        max_total_bullets=100,
        max_page_units=200,
    )

    assert len(pruned.projects) == 1
    assert len(pruned.projects[0].bullets) == 3


def test_prune_does_not_stall_when_projects_are_at_enforced_minimum() -> None:
    resume = CanonicalResume(
        identity=Identity(name='Loop Guard', email='l@example.com', phone='555', location='TX', links=[]),
        education=[],
        experience=[
            ExperienceEntry(
                company='Acme',
                title='Engineer',
                location='TX',
                dates=DateRange(start='2024', end='Present'),
                bullets=['e1', 'e2'],
            )
        ],
        projects=[
            ProjectEntry(
                name='P1',
                dates=DateRange(start='2024', end='2024'),
                tech=['Python'],
                bullets=['p1', 'p2', 'p3'],
            ),
            ProjectEntry(
                name='P2',
                dates=DateRange(start='2024', end='2024'),
                tech=['Python'],
                bullets=['p1', 'p2', 'p3'],
            ),
            ProjectEntry(
                name='P3',
                dates=DateRange(start='2024', end='2024'),
                tech=['Python'],
                bullets=['p1', 'p2', 'p3'],
            ),
        ],
        skills=Skills(categories={}),
        certifications=[],
        awards=[],
    )

    warnings: list[str] = []
    pruned = prune_resume_for_one_page(
        resume=resume,
        score_lookup={'p1': 3.0, 'p2': 2.0, 'p3': 1.0, 'engineer': 10.0},
        warnings=warnings,
        max_total_bullets=9,
        max_page_units=200,
    )

    total_bullets = sum(len(entry.bullets) for entry in pruned.experience) + sum(len(project.bullets) for project in pruned.projects)
    assert total_bullets <= 9
