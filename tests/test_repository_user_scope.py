from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.models import CanonicalResume, Identity, Skills, VaultBullet, VaultItem, VaultItemType
from app.services.repository import DataRepository, reset_current_user_id, set_current_user_id


def _minimal_resume(name: str) -> CanonicalResume:
    return CanonicalResume(
        identity=Identity(name=name, email=f'{name}@example.com', phone='555', location='TX', links=[]),
        summary='',
        education=[],
        experience=[],
        projects=[],
        skills=Skills(categories={}),
        awards=[],
    )


def _vault_item(title: str) -> VaultItem:
    return VaultItem(type=VaultItemType.project, title=title, bullets=[VaultBullet(text='One'), VaultBullet(text='Two')])


def test_repository_scopes_data_by_current_user(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    settings.ensure_directories()
    repository = DataRepository(settings)

    token_a = set_current_user_id('user_a')
    try:
        repository.save_base_resume(_minimal_resume('UserA'))
        repository.save_vault_item(
            'item1',
            VaultItem(type=VaultItemType.project, title='A', bullets=[VaultBullet(text='A bullet'), VaultBullet(text='B bullet')]),
        )
    finally:
        reset_current_user_id(token_a)

    token_b = set_current_user_id('user_b')
    try:
        repository.save_base_resume(_minimal_resume('UserB'))
        assert repository.get_vault_item('item1') is None
    finally:
        reset_current_user_id(token_b)

    token_a_again = set_current_user_id('user_a')
    try:
        resume = repository.load_base_resume()
        assert resume is not None
        assert resume.identity.name == 'UserA'
        assert repository.get_vault_item('item1') is not None
    finally:
        reset_current_user_id(token_a_again)


def test_repository_legacy_migration_to_user(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    settings.ensure_directories()
    repository = DataRepository(settings)

    repository.save_base_resume(_minimal_resume('Legacy'))
    repository.save_vault_item('legacy_item', _vault_item('Legacy Project'))

    repository.migrate_legacy_data_to_user('user_legacy')
    migrated_resume = repository.load_base_resume(user_id='user_legacy')
    assert migrated_resume is not None
    assert migrated_resume.identity.name == 'Legacy'
    migrated_items = repository.list_vault_items(user_id='user_legacy')
    assert any(item_id == 'legacy_item' for item_id, _ in migrated_items)


def test_repository_migration_does_not_overwrite_existing_user_files(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    settings.ensure_directories()
    repository = DataRepository(settings)

    repository.save_base_resume(_minimal_resume('Legacy'))
    repository.save_vault_item('shared_item', _vault_item('Legacy Shared'))
    repository.save_vault_item('legacy_only', _vault_item('Legacy Only'))

    repository.save_base_resume(_minimal_resume('User Existing'), user_id='user_legacy')
    repository.save_vault_item('shared_item', _vault_item('User Shared'), user_id='user_legacy')

    repository.migrate_legacy_data_to_user('user_legacy')

    migrated_resume = repository.load_base_resume(user_id='user_legacy')
    assert migrated_resume is not None
    assert migrated_resume.identity.name == 'User Existing'

    shared_item = repository.get_vault_item('shared_item', user_id='user_legacy')
    assert shared_item is not None
    assert shared_item.title == 'User Shared'

    copied_item = repository.get_vault_item('legacy_only', user_id='user_legacy')
    assert copied_item is not None
    assert copied_item.title == 'Legacy Only'


def test_repository_migration_fills_missing_domains_for_existing_user(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    settings.ensure_directories()
    repository = DataRepository(settings)

    repository.save_base_resume(_minimal_resume('Legacy Base'))
    repository.save_vault_item('legacy_item', _vault_item('Legacy Item'))

    repository.save_base_resume(_minimal_resume('User Base'), user_id='user_legacy')
    assert repository.get_vault_item('legacy_item', user_id='user_legacy') is None

    repository.migrate_legacy_data_to_user('user_legacy')

    migrated_resume = repository.load_base_resume(user_id='user_legacy')
    assert migrated_resume is not None
    assert migrated_resume.identity.name == 'User Base'

    migrated_item = repository.get_vault_item('legacy_item', user_id='user_legacy')
    assert migrated_item is not None
    assert migrated_item.title == 'Legacy Item'
