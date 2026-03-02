from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Set

LATEX_ESCAPE_MAP = {
    '\\': r'\textbackslash{}',
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
}

UNICODE_LATEX_NORMALIZE_MAP = {
    '\u00a0': ' ',   # non-breaking space
    '\u200b': '',    # zero-width space
    '\ufeff': '',    # byte-order mark
    '\u2010': '-',   # hyphen
    '\u2011': '-',   # non-breaking hyphen
    '\u2012': '-',   # figure dash
    '\u2013': '-',   # en dash
    '\u2014': '-',   # em dash
    '\u2015': '-',   # horizontal bar
    '\u2018': "'",
    '\u2019': "'",
    '\u201c': '\"',
    '\u201d': '\"',
    '\u2026': '...',
}


def latex_escape(value: str) -> str:
    preescaped_replacements = {
        r'\&': '&',
        r'\%': '%',
        r'\$': '$',
        r'\#': '#',
        r'\_': '_',
        r'\{': '{',
        r'\}': '}',
        r'\~': '~',
        r'\^': '^',
    }
    for escaped_token, plain_token in preescaped_replacements.items():
        value = value.replace(escaped_token, plain_token)

    value = ''.join(UNICODE_LATEX_NORMALIZE_MAP.get(ch, ch) for ch in value)
    escaped = ''.join(LATEX_ESCAPE_MAP.get(ch, ch) for ch in value)
    escaped = escaped.replace('\n', ' ')
    return escaped


def normalize_token(token: str) -> str:
    return re.sub(r'[^a-z0-9\+#\.]', '', token.lower()).strip()


def tokenize(text: str) -> List[str]:
    raw = re.findall(r"[A-Za-z0-9\+#\.\-/']+", text.lower())
    return [normalize_token(piece) for piece in raw if normalize_token(piece)]


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    output: List[str] = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def slugify(value: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9]+', '-', value.lower()).strip('-')
    return cleaned or 'job'


def ensure_within(base: Path, candidate: Path) -> Path:
    base_resolved = base.resolve()
    candidate_resolved = candidate.resolve()
    if base_resolved not in candidate_resolved.parents and candidate_resolved != base_resolved:
        raise ValueError('Path escape detected')
    return candidate_resolved
