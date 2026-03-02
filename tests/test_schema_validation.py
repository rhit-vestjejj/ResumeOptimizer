from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.models import CanonicalResume, VaultItem


def test_canonical_schema_accepts_sample() -> None:
    payload = yaml.safe_load(Path('data/sample/base.yaml').read_text(encoding='utf-8'))
    model = CanonicalResume.model_validate(payload)
    assert model.identity.name == 'Alex Morgan'
    assert len(model.experience) >= 1


def test_canonical_schema_rejects_missing_identity() -> None:
    payload = yaml.safe_load(Path('data/sample/base.yaml').read_text(encoding='utf-8'))
    del payload['identity']['name']
    with pytest.raises(ValidationError):
        CanonicalResume.model_validate(payload)


def test_vault_schema_coerces_string_bullets() -> None:
    payload = {
        'type': 'project',
        'title': 'Sample Item',
        'tags': ['backend'],
        'tech': ['Python'],
        'bullets': ['Built ingestion flow'],
        'links': [],
        'source_artifacts': [],
    }
    model = VaultItem.model_validate(payload)
    assert model.bullets[0].text == 'Built ingestion flow'


def test_vault_schema_rejects_invalid_type() -> None:
    payload = {
        'type': 'invalid-kind',
        'title': 'Bad Item',
        'tags': [],
        'tech': [],
        'bullets': [],
        'links': [],
        'source_artifacts': [],
    }
    with pytest.raises(ValidationError):
        VaultItem.model_validate(payload)
