from __future__ import annotations

from app.models import CandidateItem, DateRange, JDAnalysis, TailorMode
from app.services.tailoring import rewrite_candidate_bullets


def test_project_bullets_are_forced_to_three_with_intro_first() -> None:
    candidate = CandidateItem(
        source_type='vault:project',
        source_id='vault-p1',
        title='Fraud Detection Platform',
        dates=DateRange(start='2025', end='Present'),
        tags=['fraud', 'ml'],
        tech=['Python', 'XGBoost', 'SQL'],
        bullets=[
            'Built real-time fraud scoring APIs for card transactions.',
            'Improved precision by 18% and reduced false positives by 22%.',
            'Processed 1.2M events/day with p95 latency under 140ms.',
        ],
    )
    jd = JDAnalysis(
        target_role_keywords=['machine learning engineer'],
        required_skills=['python', 'sql', 'xgboost'],
        nice_to_haves=[],
        responsibilities=['build model inference systems'],
    )

    warnings: list[str] = []
    bullets = rewrite_candidate_bullets(
        candidate=candidate,
        jd=jd,
        mode=TailorMode.HARD_TRUTH,
        llm=None,
        known_terms={'python', 'sql', 'xgboost', 'fraud'},
        warnings=warnings,
    )

    assert len(bullets) == 3
    assert bullets[0].startswith('Built Fraud Detection Platform')
    assert all(bullet.endswith('.') for bullet in bullets)
    assert any('%' in bullet or '1.2M' in bullet for bullet in bullets[1:])

