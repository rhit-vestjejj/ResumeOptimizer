from __future__ import annotations

from pathlib import Path

from app.models import CanonicalResume, DateRange, EducationEntry, ExperienceEntry, Identity, ProjectEntry, Skills
from app.services.ats_engine import (
    ISSUE_NONSTANDARD_BULLETS,
    apply_patches,
    build_requirement_graph,
    compare_versions,
    compute_match_score,
    detect_overlaps,
    detect_sensitive_data,
    export_bundle,
    extract_skills,
    generate_patches,
    lint_resume,
    map_to_canonical_skills,
    normalize_dates,
    parse_job_description,
    parse_mirror,
    render_outputs,
    validate_contact,
    version_resume,
)


def _sample_resume() -> CanonicalResume:
    return CanonicalResume(
        identity=Identity(
            name='Alex Morgan',
            email='alex@example.com',
            phone='555-123-4567',
            location='Austin, TX',
            links=['https://github.com/alexm'],
        ),
        summary='Software engineer focused on backend and ML systems.',
        education=[
            EducationEntry(
                school='State University',
                degree='Bachelor of Science',
                major='Computer Science',
                minors=[],
                gpa='3.8',
                dates=DateRange(start='Aug 2019', end='May 2023'),
                coursework=['Machine Learning'],
            )
        ],
        experience=[
            ExperienceEntry(
                company='BlueRiver',
                title='Software Engineer',
                location='Austin, TX',
                dates=DateRange(start='Jul 2023', end='Present'),
                bullets=['Built Python APIs and SQL data pipelines.'],
            )
        ],
        projects=[
            ProjectEntry(
                name='Fraud Detection Pipeline',
                dates=DateRange(start='Jan 2024', end='Apr 2024'),
                tech=['Python', 'XGBoost', 'SQL'],
                bullets=[
                    'Built fraud model inference service.',
                    'Designed feature engineering and evaluation pipeline.',
                ],
            )
        ],
        skills=Skills(categories={'Languages': ['Python', 'SQL'], 'Cloud': ['AWS']}),
        certifications=['AWS CCP'],
        awards=['Dean List'],
    )


def test_parse_mirror_returns_quality_and_diagnostics() -> None:
    raw_text = '\n'.join(
        [
            'Alex Morgan',
            'alex@example.com',
            '555-123-4567',
            'Austin, TX',
            'Skills',
            'Languages: Python, SQL',
            'Projects',
            'Fraud Detection Pipeline',
            '- Built fraud model inference service.',
            '- Designed feature engineering pipeline.',
        ]
    )

    result = parse_mirror(raw_text, llm=None)
    assert len(result['parsers']) >= 2
    assert 0.0 <= result['quality']['parse_quality'] <= 100.0
    assert 'completeness_score' in result['quality']
    assert 'agreement_score' in result['quality']


def test_lint_and_sensitive_validation_detect_issues(tmp_path: Path) -> None:
    text_path = tmp_path / 'resume.txt'
    text_path.write_text('Alex Morgan\n▪ Built systems\nDOB: 01/01/2000\n123-45-6789\n', encoding='utf-8')

    lint = lint_resume(text_path)
    assert any(issue['code'] == ISSUE_NONSTANDARD_BULLETS for issue in lint['issues'])

    sensitive = detect_sensitive_data(text_path.read_text(encoding='utf-8'))
    assert sensitive['count'] >= 2


def test_render_outputs_produces_docx_pdf_txt_with_text_layer(tmp_path: Path) -> None:
    resume = _sample_resume()
    result = render_outputs(resume, tmp_path)
    assert Path(result['txt_path']).exists()
    assert Path(result['docx_path']).exists()
    assert Path(result['pdf_path']).exists()
    assert result['pdf_text_layer']['ok'] is True


def test_render_outputs_supports_filename_prefix(tmp_path: Path) -> None:
    resume = _sample_resume()
    result = render_outputs(resume, tmp_path, filename_prefix='ats_')
    assert Path(result['txt_path']).name == 'ats_resume.txt'
    assert Path(result['docx_path']).name == 'ats_resume.docx'
    assert Path(result['pdf_path']).name == 'ats_resume.pdf'


def test_export_bundle_excludes_bundle_file_itself(tmp_path: Path) -> None:
    (tmp_path / 'resume.txt').write_text('resume', encoding='utf-8')
    bundle = export_bundle(tmp_path, tmp_path / 'bundle.zip')
    assert bundle.exists()

    import zipfile

    with zipfile.ZipFile(bundle, 'r') as archive:
        names = set(archive.namelist())
    assert 'resume.txt' in names
    assert 'bundle.zip' not in names


def test_contact_timeline_and_overlap_detection() -> None:
    resume = _sample_resume()
    resume.experience.append(
        ExperienceEntry(
            company='Delta',
            title='ML Intern',
            location='Remote',
            dates=DateRange(start='Jan 2023', end='Aug 2023'),
            bullets=['Built XGBoost model for risk scoring.'],
        )
    )
    contact = validate_contact(resume)
    assert contact['all_valid'] is True

    timeline = normalize_dates(resume)
    overlaps = detect_overlaps(timeline)
    assert overlaps['count'] >= 1


def test_skill_jd_graph_and_match_score_are_explainable() -> None:
    resume = _sample_resume()
    jd_text = """
    We are hiring a Machine Learning Engineer.
    Required: Python, SQL, XGBoost, 2+ years experience.
    Preferred: AWS, Spark.
    """

    raw = extract_skills(jd_text)
    mapped = map_to_canonical_skills(raw['skills'])
    assert any(row['canonical_id'] == 'python' for row in mapped)

    parsed = parse_job_description(jd_text)
    graph = build_requirement_graph(parsed)
    assert graph['nodes']
    assert graph['edges']

    score = compute_match_score(resume, jd_text)
    assert 0.0 <= score['overall_score'] <= 100.0
    assert 'subscores' in score
    assert 'top_drivers' in score
    assert 'ranked_gaps' in score


def test_custom_alias_graph_keeps_canonical_ids_normalized() -> None:
    resume = _sample_resume()
    resume.skills.categories['Custom'] = ['Machine Learning']
    jd_text = 'Required: Machine Learning.'
    alias_graph = {'Machine Learning': ['machine learning']}

    parsed = parse_job_description(jd_text, alias_graph=alias_graph)
    required = {row['canonical_id'] for row in parsed['required']['skills']}
    assert 'machine_learning' in required

    score = compute_match_score(resume, jd_text, alias_graph=alias_graph)
    metadata = score['metadata']
    assert 'machine_learning' in metadata['resume_hard_skill_ids']
    assert 'machine_learning' in metadata['matched_required_hard_skills']


def test_match_score_applies_must_have_gate_when_required_hard_coverage_is_low() -> None:
    resume = _sample_resume()
    jd_text = """
    Required: Kubernetes, Spark, Kafka, Airflow, TensorFlow.
    Preferred: communication, collaboration.
    """
    score = compute_match_score(resume, jd_text)
    metadata = score['metadata']
    assert metadata['must_have_gate_applied'] is True
    assert score['overall_score'] <= metadata['must_have_gate_cap']
    assert any(gap['type'] == 'must_have_coverage_shortfall' for gap in score['ranked_gaps'])


def test_section_aware_skill_evidence_weights_experience_above_summary() -> None:
    summary_only = CanonicalResume(
        identity=Identity(
            name='Case A',
            email='a@example.com',
            phone='5551234567',
            location='Remote',
            links=[],
        ),
        summary='Python engineer focused on backend systems.',
        education=[],
        experience=[],
        projects=[],
        skills=Skills(categories={}),
        certifications=[],
        awards=[],
    )
    experience_evidence = CanonicalResume(
        identity=Identity(
            name='Case B',
            email='b@example.com',
            phone='5551234567',
            location='Remote',
            links=[],
        ),
        summary='Engineer focused on backend systems.',
        education=[],
        experience=[
            ExperienceEntry(
                company='Blue',
                title='Engineer',
                location='Remote',
                dates=DateRange(start='Jan 2024', end='Present'),
                bullets=['Built services with Python and SQL.'],
            )
        ],
        projects=[],
        skills=Skills(categories={}),
        certifications=[],
        awards=[],
    )
    jd_text = 'Required: Python.'
    summary_score = compute_match_score(summary_only, jd_text)
    experience_score = compute_match_score(experience_evidence, jd_text)
    assert experience_score['subscores']['hard_skill_alignment'] > summary_score['subscores']['hard_skill_alignment']


def test_parse_job_description_extracts_hard_and_soft_requirements() -> None:
    jd_text = """
    Required: Python, SQL, XGBoost, and strong communication with cross-functional stakeholders.
    Preferred: AWS and leadership or mentoring experience.
    """
    parsed = parse_job_description(jd_text)
    required_hard = {row['canonical_id'] for row in parsed['required']['skills']}
    required_soft = {row['canonical_id'] for row in parsed['required']['soft_skills']}
    preferred_soft = {row['canonical_id'] for row in parsed['preferred']['soft_skills']}

    assert {'python', 'sql', 'xgboost'}.issubset(required_hard)
    assert 'communication' in required_soft
    assert 'stakeholder_management' in required_soft
    assert 'leadership' in preferred_soft


def test_match_score_includes_soft_skill_alignment_and_soft_gap_flags() -> None:
    resume = _sample_resume()
    resume.summary = 'Collaborative engineer with strong communication and leadership.'
    resume.experience[0].bullets.append('Collaborated with cross-functional stakeholders and presented technical updates.')

    jd_text = """
    Required: Python, SQL, communication, collaboration, adaptability.
    Preferred: leadership.
    """
    score = compute_match_score(resume, jd_text)
    assert score['subscores']['soft_skill_alignment'] > 0
    assert 'required_soft_skills' in score['metadata']
    assert any(driver['type'] == 'required_soft_skill_match' for driver in score['top_drivers'])
    assert any(gap['type'] == 'missing_required_soft_skill' for gap in score['ranked_gaps'])

    patches = generate_patches(resume, jd_text)['patches']
    assert any(
        patch.get('op') == 'flag_missing_requirement' and patch.get('requirement_kind') == 'soft_skill'
        for patch in patches
    )


def test_generate_and_apply_patches_are_grounded() -> None:
    resume = _sample_resume()
    jd_text = 'Required: Python, SQL, Kubernetes. Preferred: AWS.'
    generated = generate_patches(resume, jd_text)
    patches = generated['patches']
    assert patches
    assert all('grounded' in patch and 'requires_user_confirmation' in patch for patch in patches)
    assert all(patch['status'] in {'GROUNDED', 'REQUIRES_USER_CONFIRMATION'} for patch in patches)

    applied = apply_patches(resume, patches, allow_requires_confirmation=False)
    assert 'resume' in applied
    assert any(entry['reason'] == 'requires_user_confirmation' for entry in applied['skipped'])


def test_versioning_and_compare_versions(tmp_path: Path) -> None:
    resume = _sample_resume()
    v1 = version_resume(resume, data_dir=tmp_path, job_id='job-1', match_score=50.0)
    resume.summary = 'Updated summary text.'
    v2 = version_resume(resume, data_dir=tmp_path, job_id='job-1', match_score=62.0)
    diff = compare_versions(v1, v2)
    assert diff['diff_count'] >= 1
    assert diff['score_delta'] == 12.0
