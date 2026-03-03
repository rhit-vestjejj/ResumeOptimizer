from __future__ import annotations

from contextvars import ContextVar, Token
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from app.config import Settings
from app.models import CanonicalResume, JobRecord, VaultItem
from app.storage import list_yaml_files, load_model, maybe_load_model, save_model, safe_read_text, safe_write_text
from app.utils import ensure_within

ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
_CURRENT_USER_ID: ContextVar[Optional[str]] = ContextVar('resume_current_user_id', default=None)


def set_current_user_id(user_id: Optional[str]) -> Token:
    return _CURRENT_USER_ID.set(user_id)


def reset_current_user_id(token: Token) -> None:
    _CURRENT_USER_ID.reset(token)


def get_current_user_id() -> Optional[str]:
    return _CURRENT_USER_ID.get()


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

    @property
    def users_dir(self) -> Path:
        return self.data_dir / 'users'

    def _validate_id(self, value: str) -> str:
        if not ID_PATTERN.fullmatch(value):
            raise ValueError(f'Invalid identifier: {value}')
        return value

    def _resolve_user_id(self, user_id: Optional[str]) -> Optional[str]:
        if user_id is None:
            user_id = get_current_user_id()
        if user_id is None:
            return None
        return self._validate_id(user_id)

    def _user_root(self, user_id: Optional[str]) -> Optional[Path]:
        resolved_user_id = self._resolve_user_id(user_id)
        if resolved_user_id is None:
            return None
        return ensure_within(self.users_dir, self.users_dir / resolved_user_id)

    def base_resume_path_for(self, user_id: Optional[str] = None) -> Path:
        user_root = self._user_root(user_id)
        if user_root is None:
            return self.base_resume_path
        return ensure_within(user_root, user_root / 'resume' / 'base.yaml')

    def vault_dir_for(self, user_id: Optional[str] = None) -> Path:
        user_root = self._user_root(user_id)
        if user_root is None:
            return self.vault_dir
        return ensure_within(user_root, user_root / 'vault' / 'items')

    def jobs_dir_for(self, user_id: Optional[str] = None) -> Path:
        user_root = self._user_root(user_id)
        if user_root is None:
            return self.jobs_dir
        return ensure_within(user_root, user_root / 'jobs')

    def outputs_dir_for(self, user_id: Optional[str] = None) -> Path:
        user_root = self._user_root(user_id)
        if user_root is None:
            return self.outputs_dir
        return ensure_within(user_root, user_root / 'outputs')

    def has_user_data(self, user_id: str) -> bool:
        root = self._user_root(user_id)
        if root is None or not root.exists():
            return False
        targets = [root / 'resume', root / 'vault', root / 'jobs', root / 'outputs']
        return any(path.exists() and any(path.iterdir()) for path in targets if path.exists())

    def has_legacy_data(self) -> bool:
        candidates = [
            self.base_resume_path,
            self.vault_dir,
            self.jobs_dir,
            self.outputs_dir,
        ]
        for path in candidates:
            if not path.exists():
                continue
            if path.is_file():
                return True
            if any(path.iterdir()):
                return True
        return False

    def migrate_legacy_data_to_user(self, user_id: str) -> None:
        resolved = self._validate_id(user_id)
        user_root = ensure_within(self.users_dir, self.users_dir / resolved)
        user_root.mkdir(parents=True, exist_ok=True)
        if not self.has_legacy_data():
            return

        # Migrate each domain independently to support safe retries and partial copies.
        if self.base_resume_path.exists():
            target = ensure_within(user_root, user_root / 'resume' / 'base.yaml')
            if target.exists():
                pass
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.base_resume_path, target)

        for source, target in [
            (self.vault_dir, ensure_within(user_root, user_root / 'vault' / 'items')),
            (self.jobs_dir, ensure_within(user_root, user_root / 'jobs')),
            (self.outputs_dir, ensure_within(user_root, user_root / 'outputs')),
        ]:
            if not source.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            for child in source.iterdir():
                destination = ensure_within(target, target / child.name)
                if destination.exists():
                    continue
                if child.is_dir():
                    shutil.copytree(child, destination)
                else:
                    shutil.copy2(child, destination)

    def load_base_resume(self, *, user_id: Optional[str] = None) -> Optional[CanonicalResume]:
        return maybe_load_model(self.base_resume_path_for(user_id), CanonicalResume)

    def save_base_resume(self, resume: CanonicalResume, *, user_id: Optional[str] = None) -> None:
        save_model(self.base_resume_path_for(user_id), resume)

    def list_vault_items(self, *, user_id: Optional[str] = None) -> List[Tuple[str, VaultItem]]:
        items: List[Tuple[str, VaultItem]] = []
        for path in list_yaml_files(self.vault_dir_for(user_id)):
            item_id = path.stem
            item = load_model(path, VaultItem)
            items.append((item_id, item))
        return sorted(items, key=lambda pair: pair[1].title.lower())

    def get_vault_item(self, item_id: str, *, user_id: Optional[str] = None) -> Optional[VaultItem]:
        item_id = self._validate_id(item_id)
        vault_dir = self.vault_dir_for(user_id)
        path = ensure_within(vault_dir, vault_dir / f'{item_id}.yaml')
        return maybe_load_model(path, VaultItem)

    def save_vault_item(self, item_id: str, item: VaultItem, *, user_id: Optional[str] = None) -> None:
        item_id = self._validate_id(item_id)
        vault_dir = self.vault_dir_for(user_id)
        path = ensure_within(vault_dir, vault_dir / f'{item_id}.yaml')
        save_model(path, item)

    def delete_vault_item(self, item_id: str, *, user_id: Optional[str] = None) -> None:
        item_id = self._validate_id(item_id)
        vault_dir = self.vault_dir_for(user_id)
        path = ensure_within(vault_dir, vault_dir / f'{item_id}.yaml')
        if path.exists():
            path.unlink()

    def list_jobs(self, *, user_id: Optional[str] = None) -> List[JobRecord]:
        jobs: List[JobRecord] = []
        jobs_dir = self.jobs_dir_for(user_id)
        for directory in sorted(jobs_dir.glob('*')):
            if not directory.is_dir():
                continue
            job_file = directory / 'job.yaml'
            if job_file.exists():
                jobs.append(load_model(job_file, JobRecord))
        return sorted(jobs, key=lambda item: item.scraped_at, reverse=True)

    def get_job(self, job_id: str, *, user_id: Optional[str] = None) -> Optional[JobRecord]:
        job_id = self._validate_id(job_id)
        jobs_dir = self.jobs_dir_for(user_id)
        path = ensure_within(jobs_dir, jobs_dir / job_id / 'job.yaml')
        return maybe_load_model(path, JobRecord)

    def save_job(self, job: JobRecord, jd_text: str, *, user_id: Optional[str] = None) -> None:
        job_id = self._validate_id(job.job_id)
        jobs_dir = self.jobs_dir_for(user_id)
        root = ensure_within(jobs_dir, jobs_dir / job_id)
        root.mkdir(parents=True, exist_ok=True)
        save_model(root / 'job.yaml', job)
        safe_write_text(root / 'jd.txt', jd_text)

    def get_job_text(self, job_id: str, *, user_id: Optional[str] = None) -> str:
        job_id = self._validate_id(job_id)
        jobs_dir = self.jobs_dir_for(user_id)
        path = ensure_within(jobs_dir, jobs_dir / job_id / 'jd.txt')
        return safe_read_text(path)

    def update_job_text(self, job_id: str, jd_text: str, *, user_id: Optional[str] = None) -> None:
        job_id = self._validate_id(job_id)
        jobs_dir = self.jobs_dir_for(user_id)
        path = ensure_within(jobs_dir, jobs_dir / job_id / 'jd.txt')
        safe_write_text(path, jd_text)

    def create_output_dir(self, job_id: str, *, user_id: Optional[str] = None) -> Path:
        job_id = self._validate_id(job_id)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        outputs_dir = self.outputs_dir_for(user_id)
        output = ensure_within(outputs_dir, outputs_dir / job_id / timestamp)
        output.mkdir(parents=True, exist_ok=True)
        return output
