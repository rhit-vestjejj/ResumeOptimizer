from __future__ import annotations

from app.utils import latex_escape


def test_latex_escape_uses_single_backslash_sequences() -> None:
    escaped = latex_escape('A & B #1')
    assert escaped == r'A \& B \#1'
    assert '\\\\&' not in escaped


def test_latex_escape_normalizes_unicode_punctuation() -> None:
    escaped = latex_escape('Rose\u2011Hulman \u2014 Build\u2026')
    assert 'Rose-Hulman - Build...' in escaped
