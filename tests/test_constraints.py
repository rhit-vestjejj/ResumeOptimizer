from __future__ import annotations

from app.models import TailorMode
from app.services.tailoring import enforce_bullet_constraints


def test_hard_truth_reverts_new_tech_and_metrics() -> None:
    source = 'Built Python ETL jobs and reduced failures by 40%.'
    rewritten = 'Built Python and Kubernetes ETL jobs and reduced failures by 70%.'

    final_bullet, warnings = enforce_bullet_constraints(
        source_bullet=source,
        rewritten_bullet=rewritten,
        allowed_terms={'python', 'etl'},
        known_terms={'python', 'etl', 'kubernetes'},
        mode=TailorMode.HARD_TRUTH,
    )

    assert final_bullet == source
    assert warnings


def test_fuck_it_strips_unsupported_claims_without_fabrication() -> None:
    source = 'Implemented SQL reports for internal stakeholders.'
    rewritten = 'Implemented SQL and Kubernetes reports for internal stakeholders and improved speed by 300%.'

    final_bullet, warnings = enforce_bullet_constraints(
        source_bullet=source,
        rewritten_bullet=rewritten,
        allowed_terms={'sql', 'reports', 'internal', 'stakeholders'},
        known_terms={'sql', 'reports', 'kubernetes', 'stakeholders'},
        mode=TailorMode.FUCK_IT,
    )

    assert 'kubernetes' not in final_bullet.lower()
    assert '300%' not in final_bullet.lower()
    assert warnings


def test_fuck_it_spacing_cleanup_after_term_and_metric_strip() -> None:
    source = 'Built fraud pipeline using Python and SQL for model scoring.'
    rewritten = 'Built fraud pipeline using Python, Kubernetes and SQL, and improved throughput by 300%.'

    final_bullet, warnings = enforce_bullet_constraints(
        source_bullet=source,
        rewritten_bullet=rewritten,
        allowed_terms={'built', 'fraud', 'pipeline', 'python', 'sql', 'model', 'scoring'},
        known_terms={'built', 'fraud', 'pipeline', 'python', 'sql', 'kubernetes', 'throughput', 'model', 'scoring'},
        mode=TailorMode.FUCK_IT,
    )

    assert 'kubernetes' not in final_bullet.lower()
    assert '300%' not in final_bullet.lower()
    assert '  ' not in final_bullet
    assert ', ,' not in final_bullet
    assert warnings
