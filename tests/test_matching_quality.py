from __future__ import annotations

from app.models import CandidateItem, DateRange, JDAnalysis
from app.services.tailoring import score_candidate_item


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
