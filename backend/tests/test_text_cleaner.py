"""
Unit tests for TextCleaner service.
"""

import pytest
from backend.app.services.embedding.text_cleaner import TextCleaner


def test_normalize_unicode():
    # Test ligature normalization (e.g. "ﬁ" -> "fi")
    text = "The efﬁcient algorithm is ﬁne."
    normalized = TextCleaner.normalize_unicode(text)
    assert normalized == "The efficient algorithm is fine."


def test_clean_control_characters():
    text = "Hello\x00World\x08Test\x1FDone"
    cleaned = TextCleaner.clean_control_characters(text)
    assert cleaned == "HelloWorldTestDone"


def test_dehyphenate():
    text = "The docu-\n  mentation is ready."
    dehyphenated = TextCleaner.dehyphenate(text)
    assert dehyphenated == "The documentation is ready."


def test_normalize_whitespace():
    text = "  Multiple    spaces   and   \t  tabs. \n\n\n\nToo many   newlines.  "
    cleaned = TextCleaner.normalize_whitespace(text)
    assert cleaned == "Multiple spaces and tabs.\n\nToo many newlines."


def test_clean_table_markdown():
    raw_table = """
    | Header 1   |   Header 2 |
    |---|---|
    |  Val A   | Val B  |
    """
    cleaned = TextCleaner.clean_table_markdown(raw_table)
    assert "| Header 1 | Header 2 |" in cleaned
    assert "| Val A | Val B |" in cleaned


def test_clean_full_pipeline():
    dirty = "  The efﬁcient sys-\n  tem is \x00ready.\n\n\nNext line.  "
    res = TextCleaner.clean(dirty)
    assert res == "The efficient system is ready.\n\nNext line."
    assert TextCleaner.clean(None) == ""
    assert TextCleaner.clean("") == ""
