from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from app.models import CanonicalResume, JDAnalysis, LLMResumeExtraction, LLMRewriteResponse, TailorMode, VaultItem


class LLMUnavailableError(RuntimeError):
    pass


class LLMValidationError(RuntimeError):
    pass


class LLMService:
    def __init__(self, api_key: Optional[str], model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.client = None

        if api_key:
            try:
                from openai import OpenAI
            except Exception as exc:  # pragma: no cover - import environment specific
                raise LLMUnavailableError(f'openai package unavailable: {exc}') from exc
            self.client = OpenAI(api_key=api_key)

    @property
    def available(self) -> bool:
        return self.client is not None

    def _require(self) -> None:
        if not self.available:
            raise LLMUnavailableError('OPENAI_API_KEY not configured; tailoring is disabled.')

    def _json_completion(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> Dict[str, Any]:
        self._require()
        assert self.client is not None
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMValidationError('LLM returned empty content.')
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMValidationError(f'LLM returned invalid JSON: {exc}') from exc

    def extract_canonical_resume(self, raw_text: str) -> CanonicalResume:
        payload = self._json_completion(
            system_prompt=(
                'You convert resume text into strict JSON matching a canonical resume schema. '
                'Do not invent details. Use empty arrays/strings when unknown. Return only JSON '
                'with the root key "resume".'
            ),
            user_prompt=(
                'Extract this resume into canonical fields:\n\n'
                '{'
                '"resume": {'
                '"schema_version": "1.1.0", '
                '"identity": {"name":"", "email":"", "phone":"", "location":"", "links":[]}, '
                '"summary": "", '
                '"education": [{"school":"", "degree":"", "major":"", "minors":[], "gpa":"", '
                '"dates":{"start":"", "end":""}, "coursework":[]}], '
                '"experience": [{"company":"", "title":"", "location":"", "dates":{"start":"", "end":""}, "bullets":[]}], '
                '"projects": [{"name":"", "link":"", "dates":{"start":"", "end":""}, "tech":[], "bullets":[]}], '
                '"skills": {"categories": {}}, '
                '"certifications": [], '
                '"awards": []'
                '}'
                '}\n\n'
                f'Resume text:\n{raw_text[:20000]}'
            ),
            max_tokens=3500,
        )
        try:
            parsed = LLMResumeExtraction.model_validate(payload)
        except ValidationError as exc:
            raise LLMValidationError(f'Canonical resume schema validation failed: {exc}') from exc
        return parsed.resume

    def analyze_jd(self, jd_text: str) -> JDAnalysis:
        payload = self._json_completion(
            system_prompt=(
                'Extract structured hiring signal from job descriptions. '
                'Return strict JSON only, no prose.'
            ),
            user_prompt=(
                'Parse this job description and return JSON with keys: '
                'target_role_keywords, required_skills, nice_to_haves, responsibilities. '
                'Each key must be an array of short phrases (max 20 each).\n\n'
                f'{jd_text[:20000]}'
            ),
            max_tokens=1200,
        )
        try:
            return JDAnalysis.model_validate(payload)
        except ValidationError as exc:
            raise LLMValidationError(f'JD analysis schema validation failed: {exc}') from exc

    def rewrite_bullets(
        self,
        *,
        item_title: str,
        source_bullets: List[str],
        jd_keywords: List[str],
        allowed_tech: List[str],
        mode: TailorMode,
    ) -> List[str]:
        payload = self._json_completion(
            system_prompt=(
                'You rewrite resume bullets for ATS alignment. Never fabricate achievements, metrics, '
                'roles, dates, titles, or technologies. Preserve factual meaning. '
                'Preserve original writing style and sentence completeness; avoid fragmentary rewrites. '
                'Keep each bullet close in length to the original (about 80%-120%). '
                f'Mode is {mode.value}. In HARD_TRUTH be conservative. In FUCK_IT be assertive but factual.'
            ),
            user_prompt=(
                'Return JSON: {"rewritten_bullets": ["..."]}. Keep same count as source bullets.\n\n'
                f'Item title: {item_title}\n'
                f'Source bullets: {json.dumps(source_bullets)}\n'
                f'JD keywords: {json.dumps(jd_keywords)}\n'
                f'Allowed technologies and terms: {json.dumps(allowed_tech)}\n'
            ),
            max_tokens=1200,
        )
        try:
            validated = LLMRewriteResponse.model_validate(payload)
        except ValidationError as exc:
            raise LLMValidationError(f'Bullet rewrite schema validation failed: {exc}') from exc
        if len(validated.rewritten_bullets) != len(source_bullets):
            raise LLMValidationError('Rewritten bullet count mismatch.')
        return validated.rewritten_bullets

    def extract_vault_item(self, *, raw_text: str, type_hint: Optional[str] = None) -> VaultItem:
        type_hint_text = type_hint or 'project'
        payload = self._json_completion(
            system_prompt=(
                'You extract a single experience vault item from user notes/transcripts/examples. '
                'Never invent details, metrics, dates, links, or technologies. '
                'Use empty arrays/nulls if unknown. Return strict JSON only with root key "item".'
            ),
            user_prompt=(
                'Convert this text into one vault item using the schema:\n'
                '{'
                '"item": {'
                '"type":"project|job|club|coursework|award|skillset|other", '
                '"title":"", '
                '"dates":{"start":"", "end":""}, '
                '"tags":[], '
                '"tech":[], '
                '"bullets":[{"text":"", "situation":"", "task":"", "action":"", "outcome":"", "impact":""}], '
                '"links":[], '
                '"source_artifacts":[]'
                '}'
                '}\n\n'
                f'Type hint (use if consistent with text): {type_hint_text}\n'
                f'Input text:\n{raw_text[:20000]}'
            ),
            max_tokens=1800,
        )
        candidate = payload.get('item', payload)
        try:
            return VaultItem.model_validate(candidate)
        except ValidationError as exc:
            raise LLMValidationError(f'Vault item schema validation failed: {exc}') from exc
