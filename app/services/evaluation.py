from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models import CanonicalResume, TailorMode, VaultItem
from app.services.tailoring import tailor_resume
from app.utils import normalize_token, tokenize


class SelectionBenchmarkCase(BaseModel):
    model_config = ConfigDict(extra='forbid')

    case_id: str
    description: str = ''
    base_resume_path: str
    vault_items_dir: str
    jd_text_path: str
    expected_selected_ids: List[str] = Field(default_factory=list)
    must_cover_terms: List[str] = Field(default_factory=list)
    must_not_claim_terms: List[str] = Field(default_factory=list)
    selection_window: int = 4
    mode: TailorMode = TailorMode.HARD_TRUTH

    @field_validator(
        'expected_selected_ids',
        'must_cover_terms',
        'must_not_claim_terms',
        mode='before',
    )
    @classmethod
    def _normalize_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError('value must be a list')
        output: List[str] = []
        for item in value:
            cleaned = str(item).strip()
            if cleaned and cleaned not in output:
                output.append(cleaned)
        return output


DEFAULT_THRESHOLDS: Dict[str, float] = {
    'precision': 0.72,
    'recall': 0.82,
    'f1': 0.76,
    'required_term_coverage': 0.90,
    'unsupported_claim_rate': 0.0,
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(payload, dict):
        raise ValueError(f'Expected mapping in YAML file: {path}')
    return payload


def _resolve_path(base_dir: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    return (base_dir / raw).resolve()


def _load_vault_items(vault_dir: Path) -> List[Tuple[str, VaultItem]]:
    items: List[Tuple[str, VaultItem]] = []
    for item_path in sorted(vault_dir.glob('*.yaml')):
        payload = _load_yaml(item_path)
        item = VaultItem.model_validate(payload)
        items.append((item_path.stem, item))
    return items


def _selected_vault_ids(chosen_items: Sequence[Dict[str, Any]], limit: Optional[int] = None) -> List[str]:
    selected: List[str] = []
    for item in chosen_items:
        source_id = str(item.get('source_id', '')).strip()
        if not source_id.startswith('vault-'):
            continue
        vault_id = source_id[len('vault-'):]
        if vault_id and vault_id not in selected:
            selected.append(vault_id)
        if limit is not None and len(selected) >= max(1, limit):
            break
    return selected


def _resume_token_set(resume: CanonicalResume) -> Set[str]:
    chunks: List[str] = []
    if resume.summary:
        chunks.append(resume.summary)

    for entry in resume.experience:
        chunks.append(entry.company)
        chunks.append(entry.title)
        chunks.extend(entry.bullets)
    for entry in resume.projects:
        chunks.append(entry.name)
        chunks.extend(entry.tech)
        chunks.extend(entry.bullets)
    for skills in resume.skills.categories.values():
        chunks.extend(skills)

    return {normalize_token(token) for token in tokenize(' '.join(chunks)) if normalize_token(token)}


def _token_coverage(term_pool: Set[str], required_terms: Sequence[str]) -> float:
    normalized_required = [normalize_token(term) for term in required_terms if normalize_token(term)]
    if not normalized_required:
        return 1.0
    hits = sum(1 for term in normalized_required if term in term_pool)
    return hits / len(normalized_required)


def _unsupported_claim_rate(warnings: Sequence[str], bullet_count: int) -> float:
    unsupported_count = 0
    for warning in warnings:
        lowered = str(warning).lower()
        if 'unsupported numeric claim' in lowered:
            unsupported_count += 1
    if bullet_count <= 0:
        return 0.0
    return unsupported_count / bullet_count


def _f1_score(precision: float, recall: float) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _case_passes(case_result: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    # Case-level checks are intentionally looser than aggregate gate checks.
    # This keeps individual-case diagnostics useful without overfitting to any one JD.
    case_precision = max(0.5, thresholds['precision'] - 0.08)
    case_recall = max(0.5, thresholds['recall'] - 0.12)
    case_f1 = max(0.5, thresholds['f1'] - 0.08)
    case_coverage = max(0.5, thresholds['required_term_coverage'] - 0.15)
    return (
        case_result['precision'] >= case_precision
        and case_result['recall'] >= case_recall
        and case_result['f1'] >= case_f1
        and case_result['required_term_coverage'] >= case_coverage
        and case_result['unsupported_claim_rate'] <= thresholds['unsupported_claim_rate']
    )


def _run_case(case_path: Path) -> Dict[str, Any]:
    payload = _load_yaml(case_path)
    case = SelectionBenchmarkCase.model_validate(payload)
    case_dir = case_path.parent

    base_resume_path = _resolve_path(case_dir, case.base_resume_path)
    vault_items_dir = _resolve_path(case_dir, case.vault_items_dir)
    jd_text_path = _resolve_path(case_dir, case.jd_text_path)

    base_resume = CanonicalResume.model_validate(_load_yaml(base_resume_path))
    vault_items = _load_vault_items(vault_items_dir)
    jd_text = jd_text_path.read_text(encoding='utf-8')

    result = tailor_resume(
        base_resume=base_resume,
        vault_items=vault_items,
        jd_text=jd_text,
        mode=case.mode,
        llm=None,
        job_title_hint=None,
    )

    chosen_items = [item.model_dump(mode='json') for item in result.report.chosen_items]
    selected_ids = _selected_vault_ids(chosen_items, limit=case.selection_window)
    selected_set = set(selected_ids)
    expected_set = set(case.expected_selected_ids)

    true_positives = selected_set & expected_set
    false_positives = selected_set - expected_set
    false_negatives = expected_set - selected_set

    precision = len(true_positives) / max(1, len(selected_set))
    recall = len(true_positives) / max(1, len(expected_set))
    f1 = _f1_score(precision, recall)

    resume_terms = _resume_token_set(result.tailored_resume)
    required_term_coverage = _token_coverage(resume_terms, case.must_cover_terms)
    must_not_claim_hits = sorted(
        {
            normalize_token(term)
            for term in case.must_not_claim_terms
            if normalize_token(term) in resume_terms
        }
    )

    bullet_count = sum(len(entry.bullets) for entry in result.tailored_resume.experience) + sum(
        len(entry.bullets) for entry in result.tailored_resume.projects
    )
    unsupported_claim_rate = _unsupported_claim_rate(result.report.warnings, bullet_count)

    return {
        'case_id': case.case_id,
        'description': case.description,
        'mode': case.mode.value,
        'selected_ids': selected_ids,
        'expected_selected_ids': list(case.expected_selected_ids),
        'matched_expected_ids': sorted(true_positives),
        'unexpected_selected_ids': sorted(false_positives),
        'missing_expected_ids': sorted(false_negatives),
        'must_cover_terms': list(case.must_cover_terms),
        'must_not_claim_terms': list(case.must_not_claim_terms),
        'must_not_claim_hits': must_not_claim_hits,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'required_term_coverage': round(required_term_coverage, 4),
        'unsupported_claim_rate': round(unsupported_claim_rate, 6),
        'warnings': list(result.report.warnings),
    }


def _aggregate_results(case_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not case_results:
        return {
            'case_count': 0,
            'precision_mean': 0.0,
            'recall_mean': 0.0,
            'f1_mean': 0.0,
            'required_term_coverage_mean': 0.0,
            'unsupported_claim_rate_mean': 0.0,
            'unsupported_claim_rate_max': 0.0,
            'top_false_positives': [],
            'top_false_negatives': [],
        }

    false_positive_counts: Dict[str, int] = {}
    false_negative_counts: Dict[str, int] = {}
    for case in case_results:
        for source_id in case.get('unexpected_selected_ids', []):
            false_positive_counts[source_id] = false_positive_counts.get(source_id, 0) + 1
        for source_id in case.get('missing_expected_ids', []):
            false_negative_counts[source_id] = false_negative_counts.get(source_id, 0) + 1

    top_false_positives = sorted(false_positive_counts.items(), key=lambda row: (-row[1], row[0]))[:5]
    top_false_negatives = sorted(false_negative_counts.items(), key=lambda row: (-row[1], row[0]))[:5]

    unsupported_claim_rates = [float(case['unsupported_claim_rate']) for case in case_results]

    return {
        'case_count': len(case_results),
        'precision_mean': round(mean(float(case['precision']) for case in case_results), 4),
        'recall_mean': round(mean(float(case['recall']) for case in case_results), 4),
        'f1_mean': round(mean(float(case['f1']) for case in case_results), 4),
        'required_term_coverage_mean': round(
            mean(float(case['required_term_coverage']) for case in case_results),
            4,
        ),
        'unsupported_claim_rate_mean': round(mean(unsupported_claim_rates), 6),
        'unsupported_claim_rate_max': round(max(unsupported_claim_rates), 6),
        'top_false_positives': [
            {'source_id': source_id, 'count': count}
            for source_id, count in top_false_positives
        ],
        'top_false_negatives': [
            {'source_id': source_id, 'count': count}
            for source_id, count in top_false_negatives
        ],
    }


def run_selection_benchmark(
    *,
    cases_dir: Path,
    results_dir: Path,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    active_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        for key, value in thresholds.items():
            if key in active_thresholds and value is not None:
                active_thresholds[key] = float(value)

    case_paths = sorted(cases_dir.glob('*.yaml'))
    if not case_paths:
        raise ValueError(f'No evaluation cases found in {cases_dir}')

    case_results = [_run_case(case_path) for case_path in case_paths]
    for case_result in case_results:
        case_result['passes_thresholds'] = _case_passes(case_result, active_thresholds)

    aggregate = _aggregate_results(case_results)
    aggregate_passes = (
        aggregate['precision_mean'] >= active_thresholds['precision']
        and aggregate['recall_mean'] >= active_thresholds['recall']
        and aggregate['f1_mean'] >= active_thresholds['f1']
        and aggregate['required_term_coverage_mean'] >= active_thresholds['required_term_coverage']
        and aggregate['unsupported_claim_rate_max'] <= active_thresholds['unsupported_claim_rate']
    )

    benchmark = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'thresholds': active_thresholds,
        'aggregate': aggregate,
        'cases': case_results,
        'aggregate_passes': aggregate_passes,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    output_path = results_dir / f'{timestamp}.json'
    output_path.write_text(json.dumps(benchmark, indent=2, sort_keys=False), encoding='utf-8')
    benchmark['output_path'] = str(output_path)
    return benchmark
