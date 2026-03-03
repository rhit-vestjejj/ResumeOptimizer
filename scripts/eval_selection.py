#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.evaluation import DEFAULT_THRESHOLDS, run_selection_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run offline selection-quality benchmark.')
    parser.add_argument(
        '--cases-dir',
        type=Path,
        default=Path('data/eval/cases'),
        help='Directory containing benchmark case YAML files.',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('data/eval/results'),
        help='Directory where timestamped benchmark result JSON files are written.',
    )
    parser.add_argument('--min-precision', type=float, default=DEFAULT_THRESHOLDS['precision'])
    parser.add_argument('--min-recall', type=float, default=DEFAULT_THRESHOLDS['recall'])
    parser.add_argument('--min-f1', type=float, default=DEFAULT_THRESHOLDS['f1'])
    parser.add_argument(
        '--min-required-coverage',
        type=float,
        default=DEFAULT_THRESHOLDS['required_term_coverage'],
    )
    parser.add_argument(
        '--max-unsupported-claim-rate',
        type=float,
        default=DEFAULT_THRESHOLDS['unsupported_claim_rate'],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = {
        'precision': args.min_precision,
        'recall': args.min_recall,
        'f1': args.min_f1,
        'required_term_coverage': args.min_required_coverage,
        'unsupported_claim_rate': args.max_unsupported_claim_rate,
    }
    benchmark = run_selection_benchmark(
        cases_dir=args.cases_dir,
        results_dir=args.results_dir,
        thresholds=thresholds,
    )
    aggregate = benchmark['aggregate']
    print('Selection benchmark summary')
    print(f"- cases: {aggregate['case_count']}")
    print(
        f"- precision mean: {aggregate['precision_mean']:.4f} (threshold {thresholds['precision']:.4f})"
    )
    print(f"- recall mean: {aggregate['recall_mean']:.4f} (threshold {thresholds['recall']:.4f})")
    print(f"- f1 mean: {aggregate['f1_mean']:.4f} (threshold {thresholds['f1']:.4f})")
    print(
        '- required-term coverage mean: '
        f"{aggregate['required_term_coverage_mean']:.4f} "
        f"(threshold {thresholds['required_term_coverage']:.4f})"
    )
    print(
        '- unsupported-claim rate max: '
        f"{aggregate['unsupported_claim_rate_max']:.6f} "
        f"(threshold {thresholds['unsupported_claim_rate']:.6f})"
    )
    print(f"- result file: {benchmark['output_path']}")

    failing_cases = [case for case in benchmark['cases'] if not case.get('passes_thresholds')]
    if failing_cases:
        print('- failing cases:')
        for case in failing_cases:
            print(
                f"  - {case['case_id']}: "
                f"precision={case['precision']:.4f}, "
                f"recall={case['recall']:.4f}, "
                f"f1={case['f1']:.4f}, "
                f"coverage={case['required_term_coverage']:.4f}, "
                f"unsupported_rate={case['unsupported_claim_rate']:.6f}"
            )

    return 0 if benchmark['aggregate_passes'] else 1


if __name__ == '__main__':
    sys.exit(main())
