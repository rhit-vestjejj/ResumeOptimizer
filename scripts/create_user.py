#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import sys

from app.config import get_settings
from app.services.auth import AuthStore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create an invite-only user account.')
    parser.add_argument('--email', required=True, help='Email address for the new account')
    parser.add_argument(
        '--password',
        default='',
        help='Password for the account. If omitted, prompt securely.',
    )
    return parser.parse_args()


def _password_from_args(raw_password: str) -> str:
    if raw_password:
        return raw_password
    first = getpass.getpass('Password: ')
    second = getpass.getpass('Confirm password: ')
    if first != second:
        raise ValueError('Passwords do not match.')
    return first


def main() -> int:
    args = _parse_args()
    try:
        password = _password_from_args(args.password or '')
        settings = get_settings()
        store = AuthStore(settings.resolved_sqlite_path)
        user = store.create_user(email=args.email, password=password)
    except Exception as exc:
        print(f'Failed to create user: {exc}', file=sys.stderr)
        return 1

    print(f'Created user {user.email} (id={user.id})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
