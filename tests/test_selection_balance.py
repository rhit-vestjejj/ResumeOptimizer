from __future__ import annotations

from app.models import CandidateItem, DateRange
from app.services.tailoring import CandidateSource, _select_diverse_candidates


def _candidate(source_id: str, title: str, tags: list[str], tech: list[str], bullets: list[str]) -> CandidateSource:
    return CandidateSource(
        candidate=CandidateItem(
            source_type='project',
            source_id=source_id,
            title=title,
            dates=DateRange(start='2025', end='Present'),
            tags=tags,
            tech=tech,
            bullets=bullets,
        ),
        origin={'kind': 'vault', 'item_id': source_id},
    )


def test_diversity_selection_keeps_relevance_floor() -> None:
    fraud_1 = _candidate(
        'fraud-1',
        'Fraud Detection Pipeline A',
        ['fraud', 'ml', 'inference'],
        ['python', 'xgboost', 'sql'],
        ['Built fraud inference pipeline with model monitoring.'],
    )
    fraud_2 = _candidate(
        'fraud-2',
        'Fraud Detection Pipeline B',
        ['fraud', 'ml', 'feature-engineering'],
        ['python', 'xgboost', 'sql'],
        ['Built feature engineering and model evaluation for fraud models.'],
    )
    agentic = _candidate(
        'agentic',
        'Personal AI Assistant',
        ['assistant', 'agentic'],
        ['python', 'llm', 'langchain'],
        ['Built an assistant to automate calendar and email tasks.'],
    )

    pool = [
        (fraud_1, 20.0),
        (fraud_2, 19.0),
        (agentic, 11.0),
    ]
    selected = _select_diverse_candidates(pool, limit=2)
    selected_ids = {candidate_source.candidate.source_id for candidate_source, _ in selected}

    assert 'fraud-1' in selected_ids
    assert 'fraud-2' in selected_ids
    assert 'agentic' not in selected_ids
