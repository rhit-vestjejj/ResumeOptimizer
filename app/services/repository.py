from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from app.config import Settings
from app.models import CanonicalResume, JobRecord, VaultItem
from app.storage import list_yaml_files, load_model, maybe_load_model, save_model, safe_read_text, safe_write_text
from app.utils import ensure_within

ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


class DataRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = settings.data_dir

    @property
    def base_resume_path(self) -> Path:
        return self.data_dir / 'resume' / 'base.yaml'

    @property
    def vault_dir(self) -> Path:
        return self.data_dir / 'vault' / 'items'

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / 'jobs'

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / 'outputs'

    def _validate_id(self, value: str) -> str:
        if not ID_PATTERN.fullmatch(value):
            raise ValueError(f'Invalid identifier: {value}')
        return value

    def load_base_resume(self) -> Optional[CanonicalResume]:
        return maybe_load_model(self.base_resume_path, CanonicalResume)

    def save_base_resume(self, resume: CanonicalResume) -> None:
        save_model(self.base_resume_path, resume)

    def list_vault_items(self) -> List[Tuple[str, VaultItem]]:
        items: List[Tuple[str, VaultItem]] = []
        for path in list_yaml_files(self.vault_dir):
            item_id = path.stem
            item = load_model(path, VaultItem)
            items.append((item_id, item))
        return sorted(items, key=lambda pair: pair[1].title.lower())

    def get_vault_item(self, item_id: str) -> Optional[VaultItem]:
        item_id = self._validate_id(item_id)
        path = ensure_within(self.vault_dir, self.vault_dir / f'{item_id}.yaml')
        return maybe_load_model(path, VaultItem)

    def save_vault_item(self, item_id: str, item: VaultItem) -> None:
        item_id = self._validate_id(item_id)
        path = ensure_within(self.vault_dir, self.vault_dir / f'{item_id}.yaml')
        save_model(path, item)

    def delete_vault_item(self, item_id: str) -> None:
        item_id = self._validate_id(item_id)
        path = ensure_within(self.vault_dir, self.vault_dir / f'{item_id}.yaml')
        if path.exists():
            path.unlink()

    def list_jobs(self) -> List[JobRecord]:
        jobs: List[JobRecord] = []
        for directory in sorted(self.jobs_dir.glob('*')):
            if not directory.is_dir():
                continue
            job_file = directory / 'job.yaml'
            if job_file.exists():
                jobs.append(load_model(job_file, JobRecord))
        return sorted(jobs, key=lambda item: item.scraped_at, reverse=True)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        job_id = self._validate_id(job_id)
        path = ensure_within(self.jobs_dir, self.jobs_dir / job_id / 'job.yaml')
        return maybe_load_model(path, JobRecord)

    def save_job(self, job: JobRecord, jd_text: str) -> None:
        job_id = self._validate_id(job.job_id)
        root = ensure_within(self.jobs_dir, self.jobs_dir / job_id)
        root.mkdir(parents=True, exist_ok=True)
        save_model(root / 'job.yaml', job)
        safe_write_text(root / 'jd.txt', jd_text)

    def get_job_text(self, job_id: str) -> str:
        job_id = self._validate_id(job_id)
        path = ensure_within(self.jobs_dir, self.jobs_dir / job_id / 'jd.txt')
        return safe_read_text(path)

    def update_job_text(self, job_id: str, jd_text: str) -> None:
        job_id = self._validate_id(job_id)
        path = ensure_within(self.jobs_dir, self.jobs_dir / job_id / 'jd.txt')
        safe_write_text(path, jd_text)

    def create_output_dir(self, job_id: str) -> Path:
        job_id = self._validate_id(job_id)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        output = ensure_within(self.outputs_dir, self.outputs_dir / job_id / timestamp)
        output.mkdir(parents=True, exist_ok=True)
        return output
