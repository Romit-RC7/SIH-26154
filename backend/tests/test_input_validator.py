"""
Unit tests for SourceInputValidator.
Tests length constraints, greeting and typo matching, gibberish filtering,
and mode detection (RAW_ARTICLE vs FREEFORM_PROMPT).
"""

import pytest
from backend.app.services.input_validator import SourceInputValidator, source_input_validator


def test_validator_empty_text():
    res = source_input_validator.validate_source_text("")
    assert not res.is_valid
    assert any("empty" in err.lower() for err in res.errors)


def test_validator_too_short():
    # Less than 5 words or less than 30 characters
    res = source_input_validator.validate_source_text("Quick test")
    assert not res.is_valid
    assert any("too short" in err.lower() for err in res.errors)


def test_validator_conversational_greetings():
    # Pure greetings should be rejected
    greetings = [
        "Hello everyone here today",
        "Good morning to all of you",
        "Hey there how are you doing",
        "Good afternoon folks",
    ]
    for g in greetings:
        res = source_input_validator.validate_source_text(g)
        assert not res.is_valid, f"Expected greeting to be rejected: {g}"
        assert any("greeting" in err.lower() for err in res.errors)


def test_validator_greeting_typos():
    # Common typo variations like "good gorming"
    typos = [
        "good gorming everyone today",
        "goood morning to all team",
        "gud morning everyone today",
    ]
    for typo in typos:
        res = source_input_validator.validate_source_text(typo)
        assert not res.is_valid, f"Expected typo to be rejected: {typo}"
        assert any("greeting" in err.lower() for err in res.errors)


def test_validator_repetitive_words_spam():
    spam = "critical critical critical critical critical critical alert"
    res = source_input_validator.validate_source_text(spam)
    assert not res.is_valid
    assert any("repetitive" in err.lower() for err in res.errors)


def test_validator_repeated_character_gibberish():
    gibberish = "zzzzzzzzzzzzzzzz something something text here"
    res = source_input_validator.validate_source_text(gibberish)
    assert not res.is_valid
    assert any("repeated character" in err.lower() for err in res.errors)


def test_validator_consonant_keyboard_mash():
    mash = "The system encountered asdfghjkl issue during deployment"
    res = source_input_validator.validate_source_text(mash)
    assert not res.is_valid
    assert any("unpronounceable" in err.lower() for err in res.errors)


def test_validator_valid_raw_article():
    article = (
        "The National Cyber Security Centre has detected an active campaign targeting "
        "unpatched VPN gateways across financial services. Immediate patching to version 4.2 "
        "is required to mitigate potential remote code execution vulnerabilities."
    )
    res = source_input_validator.validate_source_text(article)
    assert res.is_valid
    assert len(res.errors) == 0
    assert res.detected_mode == "RAW_ARTICLE"
    assert res.word_count > 20
    assert res.char_count > 100


def test_validator_valid_freeform_prompt():
    prompt = (
        "Write a detailed executive summary about the recent zero-day vulnerability in Apache Log4j "
        "and provide actionable remediation steps for cloud infrastructure engineers."
    )
    res = source_input_validator.validate_source_text(prompt)
    assert res.is_valid
    assert len(res.errors) == 0
    assert res.detected_mode == "FREEFORM_PROMPT"


def test_validator_prompt_with_please():
    prompt = (
        "Please generate a professional LinkedIn post highlighting our recent 40% growth "
        "in AI model efficiency and enterprise security compliance."
    )
    res = source_input_validator.validate_source_text(prompt)
    assert res.is_valid
    assert res.detected_mode == "FREEFORM_PROMPT"
