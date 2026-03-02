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
        'required_skill_evidence_map': [
            {
                'required_term': 'postgresql',
                'has_evidence': True,
                'source_title': 'Backend Intern at Example',
                'source_type': 'experience',
                'evidence_bullet': 'Built PostgreSQL-backed APIs with Python.',
            },
            {
                'required_term': 'kubernetes',
                'has_evidence': False,
                'source_title': '',
                'source_type': '',
                'evidence_bullet': '',
            },
        ],
        'keywords_covered': [],
        'keywords_missed': [],
    }

    context = _tailor_result_context(job=job, mode=TailorMode.HARD_TRUTH, workflow=workflow)

    assert context['missing_required_evidence_display'] == ['Machine Learning', 'SQL', 'XGBoost', 'Computer Vision']
    assert context['vault_relevance'][0]['matched_required_terms_display'] == ['Machine Learning', 'SQL', 'AWS']
    assert context['vault_relevance'][0]['missing_required_terms_display'] == ['XGBoost']
    assert context['required_skill_evidence_map'][0]['required_term_display'] == 'PostgreSQL'
    assert context['required_skill_evidence_map'][0]['source_type_display'] == 'Work'
    assert context['required_skill_evidence_map'][1]['required_term_display'] == 'Kubernetes'
