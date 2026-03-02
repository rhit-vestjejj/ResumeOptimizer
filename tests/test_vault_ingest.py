from __future__ import annotations

from pathlib import Path

from app.models import VaultItem, VaultItemType
from app.services.vault_ingest import heuristic_parse_vault_text, parse_uploaded_text, parse_vault_source_text


def test_heuristic_parse_project_notes_extracts_structure() -> None:
    raw = (
        'Realtime Telemetry Dashboard\n'
        '- Built backend APIs in Python and FastAPI for device metrics.\n'
        '- Stored processed events in PostgreSQL and Redis caches.\n'
        '- Deployed services with Docker and CI workflows.\n'
        'Project link: https://github.com/example/telemetry-dashboard\n'
        'Jan 2024 - May 2024\n'
    )

    item = heuristic_parse_vault_text(raw, type_hint=VaultItemType.project)

    assert item.type == VaultItemType.project
    assert item.title == 'Realtime Telemetry Dashboard'
    assert len(item.bullets) >= 3
    assert 'python' in [tech.lower() for tech in item.tech]
    assert any('github.com/example/telemetry-dashboard' in link for link in item.links)
    assert item.dates is not None


def test_heuristic_parse_infers_job_when_hint_missing() -> None:
    raw = (
        'Software Engineering Intern\n'
        'Worked at Acme Data Labs and implemented internal ETL pipelines.\n'
        'Collaborated with analytics team to improve data quality checks.\n'
    )

    item = heuristic_parse_vault_text(raw)
    assert item.type == VaultItemType.job


def test_parse_vault_source_uses_llm_when_available() -> None:
    class FakeLLM:
        available = True

        def extract_vault_item(self, *, raw_text: str, type_hint: str | None = None) -> VaultItem:
            return VaultItem(
                type=VaultItemType.project,
                title='Parsed by LLM',
                tags=['api'],
                tech=['python'],
                bullets=[{'text': 'Built API service.'}],
                links=[],
                source_artifacts=[],
            )

    item, warnings = parse_vault_source_text('notes', llm=FakeLLM(), type_hint='project')
    assert item.title == 'Parsed by LLM'
    assert warnings == []


def test_parse_uploaded_text_reads_plain_text(tmp_path: Path) -> None:
    file_path = tmp_path / 'example.txt'
    file_path.write_text('sample project notes', encoding='utf-8')

    text, warnings = parse_uploaded_text(file_path, enable_ocr=False)
    assert text == 'sample project notes'
    assert warnings == []
