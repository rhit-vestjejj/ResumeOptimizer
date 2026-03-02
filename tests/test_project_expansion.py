from __future__ import annotations

from app.models import CanonicalResume, DateRange, EducationEntry, ExperienceEntry, Identity, ProjectEntry, Skills, VaultItem
from app.services.tailoring import expand_resume_with_projects


def test_expand_resume_adds_unselected_project() -> None:
    base = CanonicalResume(
        identity=Identity(name='A', email='a@b.com', phone='1', location='X', links=[]),
        education=[
            EducationEntry(
                school='S',
                degree='BS',
                major='CS',
                minors=[],
                gpa='',
                dates=DateRange(start='2019', end='2023'),
                coursework=[],
            )
        ],
        experience=[
            ExperienceEntry(company='C', title='E', location='L', dates=DateRange(start='2024', end='Present'), bullets=['b1'])
        ],
        projects=[
            ProjectEntry(
                name='Project One',
                link=None,
                dates=DateRange(start='2024', end='2024'),
                tech=['Python'],
                bullets=['did one', 'did one more'],
            ),
            ProjectEntry(
                name='Project Two',
                link=None,
                dates=DateRange(start='2025', end='2025'),
                tech=['SQL'],
                bullets=['did two', 'did two more'],
            ),
        ],
        skills=Skills(categories={'Languages': ['Python']}),
        awards=[],
    )

    current = CanonicalResume.model_validate(base.model_dump())
    current.projects = [base.projects[0]]

    expanded, changed, marker = expand_resume_with_projects(
        resume=current,
        base_resume=base,
        vault_items=[],
        score_lookup={'projecttwo': 10.0},
        used_expansions=set(),
    )

    assert changed is True
    assert marker is not None
    assert len(expanded.projects) == 2
    assert any(p.name == 'Project Two' for p in expanded.projects)


def test_expand_resume_skips_when_no_eligible_two_bullet_project_left() -> None:
    base = CanonicalResume(
        identity=Identity(name='A', email='a@b.com', phone='1', location='X', links=[]),
        education=[],
        experience=[],
        projects=[ProjectEntry(name='Spare Project', link=None, dates=None, tech=['Python', 'SQL'], bullets=['x'])],
        skills=Skills(categories={}),
        awards=[],
    )

    current = CanonicalResume.model_validate(base.model_dump())
    current.projects = [ProjectEntry(name='Spare Project', link=None, dates=None, tech=['Python'], bullets=['x'])]

    expanded, changed, marker = expand_resume_with_projects(
        resume=current,
        base_resume=base,
        vault_items=[],
        score_lookup={},
        used_expansions={'project:spareproject'},
    )

    assert changed is False
    assert marker is None
