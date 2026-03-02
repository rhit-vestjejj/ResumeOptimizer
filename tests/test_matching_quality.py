from __future__ import annotations

from app.models import CandidateItem, DateRange, JDAnalysis
from app.services.tailoring import CandidateSource, score_candidate_item, score_candidates


def test_score_penalizes_off_topic_items_with_same_generic_overlap() -> None:
    jd = JDAnalysis(
        target_role_keywords=['machine learning', 'distributed systems', 'backend'],
        required_skills=['python', 'xgboost', 'sql', 'docker'],
        nice_to_haves=['aws', 'kafka'],
        responsibilities=['design scalable systems', 'build inference pipelines'],
    )

    relevant = CandidateItem(
        source_type='project',
        source_id='p1',
        title='Fraud Detection ML Pipeline',
        dates=DateRange(start='2025', end='2026'),
        tags=['machine-learning', 'inference', 'backend'],
        tech=['python', 'xgboost', 'sql', 'docker', 'kafka'],
        bullets=['Built scalable inference pipelines for fraud scoring and model monitoring.'],
    )

    off_topic = CandidateItem(
        source_type='project',
        source_id='p2',
        title='Voice Productivity Assistant',
        dates=DateRange(start='2025', end='2026'),
        tags=['assistant', 'voice', 'calendar', 'email'],
        tech=['python', 'gmail', 'calendar-api', 'speech'],
        bullets=['Built a voice assistant that automates calendar and email workflows.'],
    )

    assert score_candidate_item(relevant, jd) > score_candidate_item(off_topic, jd)


def test_score_prefers_ml_systems_over_agentic_apps_for_ml_engineer_jd() -> None:
    jd = JDAnalysis(
        target_role_keywords=['machine learning engineer', 'model inference', 'fraud detection'],
        required_skills=['python', 'xgboost', 'feature engineering', 'model evaluation', 'sql'],
        nice_to_haves=['aws', 'spark'],
        responsibilities=['build fraud models', 'ship inference pipelines', 'monitor model performance'],
    )

    fraud_ml = CandidateItem(
        source_type='project',
        source_id='fraud-ml',
        title='Real-Time Fraud Detection Platform',
        dates=DateRange(start='2025', end='Present'),
        tags=['fraud', 'ml', 'inference', 'risk-modeling'],
        tech=['python', 'xgboost', 'sql', 'docker', 'kafka'],
        bullets=[
            'Built fraud detection models with feature engineering and model evaluation workflows.',
            'Shipped inference pipelines and monitoring for model drift in production.',
        ],
    )

    agentic_app = CandidateItem(
        source_type='project',
        source_id='agentic',
        title='AI Scheduling Assistant',
        dates=DateRange(start='2025', end='Present'),
        tags=['ai', 'assistant', 'agentic', 'productivity'],
        tech=['python', 'llm', 'langchain', 'gmail-api'],
        bullets=[
            'Built an AI assistant that automates email and calendar workflows.',
            'Implemented prompt orchestration and tool-calling flows for user tasks.',
        ],
    )

    assert score_candidate_item(fraud_ml, jd) > score_candidate_item(agentic_app, jd)


def test_feedback_boosts_preferred_and_penalizes_blocked_titles() -> None:
    jd = JDAnalysis(
        target_role_keywords=['machine learning engineer'],
        required_skills=['python', 'sql', 'xgboost', 'fraud'],
        nice_to_haves=[],
        responsibilities=['build models'],
    )

    preferred_candidate = CandidateSource(
        candidate=CandidateItem(
            source_type='project',
            source_id='preferred',
            title='Fraud Detection Platform',
            dates=DateRange(start='2025', end='Present'),
            tags=['fraud', 'ml'],
            tech=['python', 'sql', 'xgboost'],
            bullets=['Built fraud models and inference pipelines.'],
        ),
        origin={'kind': 'vault', 'item_id': 'preferred'},
    )
    blocked_candidate = CandidateSource(
        candidate=CandidateItem(
            source_type='project',
            source_id='blocked',
            title='AI Scheduling Assistant',
            dates=DateRange(start='2025', end='Present'),
            tags=['assistant', 'agentic'],
            tech=['python', 'llm'],
            bullets=['Built a scheduling assistant workflow.'],
        ),
        origin={'kind': 'vault', 'item_id': 'blocked'},
    )

    baseline = score_candidates([preferred_candidate, blocked_candidate], jd)
    adjusted = score_candidates(
        [preferred_candidate, blocked_candidate],
        jd,
        selection_feedback={
            'preferred_titles': ['Fraud Detection Platform'],
            'blocked_titles': ['AI Scheduling Assistant'],
        },
    )

    baseline_order = [candidate_source.candidate.source_id for candidate_source, _ in baseline]
    adjusted_order = [candidate_source.candidate.source_id for candidate_source, _ in adjusted]
    assert baseline_order[0] == 'preferred'
    assert adjusted_order[0] == 'preferred'
    assert adjusted[0][1] > baseline[0][1]
    assert adjusted[1][1] < baseline[1][1]
