from __future__ import annotations

from pathlib import Path

from app.services.evaluation import run_selection_benchmark


def test_selection_benchmark_gate_and_artifact(tmp_path: Path) -> None:
    benchmark = run_selection_benchmark(
        cases_dir=Path('data/eval/cases'),
        results_dir=tmp_path,
    )

    output_path = Path(benchmark['output_path'])
    assert output_path.exists()
    assert output_path.suffix == '.json'

    aggregate = benchmark['aggregate']
    assert aggregate['case_count'] >= 6
    assert 0.0 <= aggregate['precision_mean'] <= 1.0
    assert 0.0 <= aggregate['recall_mean'] <= 1.0
    assert 0.0 <= aggregate['f1_mean'] <= 1.0
    assert 0.0 <= aggregate['required_term_coverage_mean'] <= 1.0
    assert aggregate['unsupported_claim_rate_max'] <= 1.0

    assert benchmark['aggregate_passes'] is True
