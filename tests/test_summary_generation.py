from __future__ import annotations

from app.models import JDAnalysis
from app.services.tailoring import _derive_target_title


def test_derive_target_title_rejects_recruiter_phrase() -> None:
    jd = JDAnalysis(
        target_role_keywords=['we are looking to hire a highly creative machine learning engineer'],
        required_skills=['python', 'sql'],
        nice_to_haves=[],
        responsibilities=['build fraud models'],
    )

    title = _derive_target_title(jd, jd_text='We are looking to hire a highly creative machine learning engineer', job_title_hint=None)
    assert title == 'Machine Learning Engineer'
