from __future__ import annotations

from pathlib import Path
import re

import yaml

from app.models import CanonicalResume, JDAnalysis, TailorMode, VaultItem
from app.services.latex import LatexService
from app.services.tailoring import MAX_PROJECT_ITEMS, tailor_resume


class FakeLLM:
    available = True

    def analyze_jd(self, jd_text: str) -> JDAnalysis:
        return JDAnalysis(
            target_role_keywords=['backend', 'api'],
            required_skills=['python', 'fastapi', 'postgresql', 'docker'],
            nice_to_haves=['redis'],
            responsibilities=['build backend services', 'optimize sql'],
        )

    def rewrite_bullets(self, *, item_title, source_bullets, jd_keywords, allowed_tech, mode):
        if mode == TailorMode.HARD_TRUTH:
            return source_bullets
        return [bullet.replace('Built', 'Delivered').replace('Implemented', 'Executed') for bullet in source_bullets]


def load_sample_inputs() -> tuple[CanonicalResume, list[tuple[str, VaultItem]], str]:
    base_payload = yaml.safe_load(Path('data/sample/base.yaml').read_text(encoding='utf-8'))
    base = CanonicalResume.model_validate(base_payload)

    vault_items: list[tuple[str, VaultItem]] = []
    vault_dir = Path('data/sample/vault/items')
    for file in sorted(vault_dir.glob('*.yaml')):
        item_payload = yaml.safe_load(file.read_text(encoding='utf-8'))
        vault_items.append((file.stem, VaultItem.model_validate(item_payload)))

    jd_text = Path('data/sample/jobs/sample-backend-role/jd.txt').read_text(encoding='utf-8')
    return base, vault_items, jd_text


def test_tailor_integration_both_modes_and_render(tmp_path: Path) -> None:
    base, vault_items, jd_text = load_sample_inputs()
    llm = FakeLLM()
    latex = LatexService(Path('app/templates'))

    for mode in [TailorMode.HARD_TRUTH, TailorMode.FUCK_IT]:
        result = tailor_resume(
            base_resume=base,
            vault_items=vault_items,
            jd_text=jd_text,
            mode=mode,
            llm=llm,
            job_title_hint='Machine Learning Engineer - Amazon',
        )

        run_dir = tmp_path / mode.value.lower()
        tex_path = latex.render_resume(result.tailored_resume, run_dir)
        pdf_path = latex.compile_resume(run_dir, mock_compile=True)

        assert tex_path.exists()
        assert pdf_path.exists()
        assert result.report.mode == mode
        assert result.report.chosen_items
        assert all(item.why_included for item in result.report.chosen_items)
        assert all(item.why_included.startswith('Selected as ') for item in result.report.chosen_items)
        assert all(isinstance(item.score_breakdown, dict) for item in result.report.chosen_items)
        assert all('ats_score' in item.score_breakdown for item in result.report.chosen_items)
        assert result.report.vault_relevance
        assert all(item.item_id for item in result.report.vault_relevance)
        assert all(item.relevance_score >= 0 for item in result.report.vault_relevance)
        selected_vault_items = [item for item in result.report.vault_relevance if item.selected]
        assert selected_vault_items
        assert all(item.why_selected.startswith('Selected as ') for item in selected_vault_items)
        unselected_vault_items = [item for item in result.report.vault_relevance if not item.selected]
        if unselected_vault_items:
            assert all(item.why_not_selected for item in unselected_vault_items)
        assert isinstance(result.report.missing_required_evidence, list)
        required_terms = {item.required_term for item in result.report.required_skill_evidence_map}
        assert required_terms == {'python', 'fastapi', 'postgresql', 'docker'}
        for evidence in result.report.required_skill_evidence_map:
            assert evidence.required_term
            if evidence.has_evidence:
                assert evidence.source_title
                assert evidence.evidence_bullet
        assert isinstance(result.report.high_confidence_exclusions, list)
        assert len(result.tailored_resume.projects) <= MAX_PROJECT_ITEMS
        assert result.tailored_resume.summary is not None
        assert 'Machine Learning Engineer' in result.tailored_resume.summary
        summary = result.tailored_resume.summary
        sentence_count = len([part for part in re.split(r'(?<=[.!?])\s+', summary.strip()) if part.strip()])
        assert 2 <= sentence_count <= 3
        assert 'Target role:' not in summary
        assert 'I am a' not in summary
        assert 'For this role, I can contribute immediately' not in summary
