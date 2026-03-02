from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models import CanonicalResume
from app.services.ats_engine import (
    apply_patches as ats_apply_patches,
    build_canonical as ats_build_canonical,
    export_bundle as ats_export_bundle,
    generate_patches as ats_generate_patches,
    lint_resume as ats_lint_resume,
    parse_mirror as ats_parse_mirror,
    render_outputs as ats_render_outputs,
    score_match as ats_score_match,
    score_parse_quality as ats_score_parse_quality,
    upload_job_description as ats_upload_job_description,
    upload_resume as ats_upload_resume,
)
from app.services.llm import LLMService


def upload_resume(file_path: Path, *, enable_ocr: bool, llm: Optional[LLMService] = None) -> Dict[str, Any]:
    return ats_upload_resume(file_path=file_path, enable_ocr=enable_ocr, llm=llm)


def upload_job_description(jd_text: str) -> Dict[str, Any]:
    return ats_upload_job_description(jd_text=jd_text)


def lint_resume(file_path: Path) -> Dict[str, Any]:
    return ats_lint_resume(file_path)


def parse_mirror(raw_text: str, llm: Optional[LLMService] = None) -> Dict[str, Any]:
    return ats_parse_mirror(raw_text, llm=llm)


def build_canonical(parse_mirror_result: Dict[str, Any]) -> CanonicalResume:
    return ats_build_canonical(parse_mirror_result)


def score_parse_quality(parse_mirror_result: Dict[str, Any]) -> Dict[str, Any]:
    return ats_score_parse_quality(parse_mirror_result)


def score_match(resume: CanonicalResume, jd_text: str) -> Dict[str, Any]:
    return ats_score_match(resume=resume, jd_text=jd_text)


def generate_patches(resume: CanonicalResume, jd_text: str) -> Dict[str, Any]:
    return ats_generate_patches(resume, jd_text)


def apply_patches(
    resume: CanonicalResume,
    patches: List[Dict[str, Any]],
    *,
    allow_requires_confirmation: bool = False,
) -> Dict[str, Any]:
    return ats_apply_patches(resume, patches, allow_requires_confirmation=allow_requires_confirmation)


def render_outputs(resume: CanonicalResume, output_dir: Path, *, filename_prefix: str = '') -> Dict[str, Any]:
    return ats_render_outputs(resume, output_dir, filename_prefix=filename_prefix)


def export_bundle(output_dir: Path, bundle_path: Optional[Path] = None) -> Path:
    return ats_export_bundle(output_dir, bundle_path=bundle_path)
