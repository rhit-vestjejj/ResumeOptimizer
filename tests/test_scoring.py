from __future__ import annotations

from app.models import CandidateItem, DateRange, JDAnalysis
from app.services.tailoring import score_candidate_item


def test_keyword_scoring_prefers_higher_overlap() -> None:
    jd = JDAnalysis(
        target_role_keywords=['backend', 'api', 'reliability'],
        required_skills=['python', 'fastapi', 'postgresql', 'docker'],
        nice_to_haves=['redis'],
        responsibilities=['build backend services', 'optimize sql'],
    )

    strong = CandidateItem(
        source_type='project',
        source_id='strong',
        title='Backend API Platform',
        dates=DateRange(start='Jan 2024', end='Present'),
        tags=['backend', 'api'],
        tech=['python', 'fastapi', 'postgresql', 'docker'],
        bullets=['Built reliable backend services and optimized SQL performance.'],
    )
    weak = CandidateItem(
        source_type='project',
        source_id='weak',
        title='Graphic Design Website',
        dates=DateRange(start='Jan 2023', end='Dec 2023'),
        tags=['design'],
        tech=['figma'],
        bullets=['Created visual assets and styling patterns.'],
    )

    assert score_candidate_item(strong, jd) > score_candidate_item(weak, jd)
