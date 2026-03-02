from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)


class StorageError(RuntimeError):
    pass


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise StorageError(f'File does not exist: {path}')
    with path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise StorageError(f'Expected YAML object at root: {path}')
    return data


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise StorageError(f'File does not exist: {path}')
    with path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise StorageError(f'Expected JSON object at root: {path}')
    return data


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def load_model(path: Path, model_cls: Type[T]) -> T:
    data = load_yaml(path)
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise StorageError(f'Invalid data in {path}: {exc}') from exc


def save_model(path: Path, model: BaseModel) -> None:
    save_yaml(path, model.model_dump(exclude_none=True, mode='json'))


def list_yaml_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(path.glob('*.yaml'))


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return ''
    return path.read_text(encoding='utf-8')


def safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def maybe_load_model(path: Path, model_cls: Type[T]) -> Optional[T]:
    if not path.exists():
        return None
    return load_model(path, model_cls)
