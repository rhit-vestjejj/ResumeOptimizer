from __future__ import annotations

from app.main import _tailor_result_context
from app.models import JobRecord, TailorMode


def test_tailor_result_context_formats_missing_evidence_terms_for_display() -> None:
    job = JobRecord(job_id='job123', title='ML Engineer', company='Example', url=None)
    workflow = {
        'warnings': [],
        'job_id': 'job123',
        'timestamp': 'run1',
        'pdf_exists': True,
        'ats_pdf_exists': True,
        'ats_docx_exists': True,
        'ats_txt_exists': True,
        'compile_error': None,
        'match_score': 84.2,
        'target_score': 82.0,
        'passes_used': 2,
        'max_passes': 5,
        'chosen_items': [],
        'vault_relevance': [
            {
                'item_id': 'v1',
                'title': 'Fraud Detection Project',
                'item_type': 'project',
                'relevance_score': 90.1,
                'selected': True,
                'why_selected': 'Selected as project evidence.',
                'why_not_selected': '',
                'matched_required_terms': ['ml', 'sql', 'aws'],
                'missing_required_terms': ['xgboost'],
            }
        ],
        'missing_required_evidence': ['ml', 'sql', 'xgboost', 'computer-vision', 'ml'],
        'keywords_covered': [],
        'keywords_missed': [],
    }

    context = _tailor_result_context(job=job, mode=TailorMode.HARD_TRUTH, workflow=workflow)

    assert context['missing_required_evidence_display'] == ['Machine Learning', 'SQL', 'XGBoost', 'Computer Vision']
    assert context['vault_relevance'][0]['matched_required_terms_display'] == ['Machine Learning', 'SQL', 'AWS']
    assert context['vault_relevance'][0]['missing_required_terms_display'] == ['XGBoost']
