"""
Input Validator Service.
Enforces validation guardrails for direct text input and free-form prompts:
- Minimum length and word count checks
- Conversational greeting and typo fuzzy matching (e.g. "good gorming")
- Low-entropy repetition and gibberish detection
- Ingestion mode detection (RAW_ARTICLE vs. FREEFORM_PROMPT)
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class ValidationResult:
    """Outcome of source text validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    detected_mode: str = "RAW_ARTICLE"  # "RAW_ARTICLE" or "FREEFORM_PROMPT"
    cleaned_text: str = ""
    word_count: int = 0
    char_count: int = 0


class SourceInputValidator:
    """Validates source text and prompt submissions against quality and noise guardrails."""

    MIN_WORDS: int = 5
    MIN_CHARS: int = 30

    GREETING_VOCABULARY: Set[str] = {
        "hello",
        "hi",
        "hey",
        "howdy",
        "hola",
        "greetings",
        "wassup",
        "sup",
        "yo",
        "dear",
        "bye",
        "goodbye",
        "welcome",
        "morning",
        "afternoon",
        "evening",
    }

    MULTIWORD_GREETINGS: List[str] = [
        "good morning",
        "good afternoon",
        "good evening",
        "good day",
        "good night",
        "hey there",
        "hello there",
        "hi there",
        "hope this finds you well",
        "how are you",
        "how do you do",
        "nice to meet you",
    ]

    PROMPT_DIRECTIVES: List[str] = [
        "write",
        "create",
        "generate",
        "draft",
        "summarize",
        "produce",
        "explain",
        "tell me",
        "make",
        "prepare",
        "outline",
        "compose",
        "give me",
        "convert",
        "transform",
        "develop",
        "synthesize",
        "analyze",
    ]

    def validate_source_text(self, text: str) -> ValidationResult:
        """
        Validates text input against length, greeting typos, and gibberish guardrails.
        """
        if not text:
            return ValidationResult(
                is_valid=False,
                errors=["Source text cannot be empty."],
                cleaned_text="",
                word_count=0,
                char_count=0,
            )

        cleaned = re.sub(r"\s+", " ", text).strip()
        words = re.findall(r"\b\w+\b", cleaned)
        word_count = len(words)
        char_count = len(cleaned)

        errors: List[str] = []

        # 1. Length bounds
        if word_count < self.MIN_WORDS or char_count < self.MIN_CHARS:
            errors.append(
                f"Input text is too short ({word_count} words, {char_count} characters). "
                f"Minimum required is {self.MIN_WORDS} words and {self.MIN_CHARS} characters."
            )

        # 2. Conversational greetings and greeting typos
        greeting_error = self._check_greeting_or_typo(cleaned, words)
        if greeting_error:
            errors.append(greeting_error)

        # 3. Gibberish and repetition checks
        gibberish_error = self._check_gibberish_and_repetition(cleaned, words)
        if gibberish_error:
            errors.append(gibberish_error)

        # 4. Ingestion mode classification
        detected_mode = self._detect_mode(cleaned)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            detected_mode=detected_mode,
            cleaned_text=cleaned,
            word_count=word_count,
            char_count=char_count,
        )

    def _check_greeting_or_typo(self, text: str, words: List[str]) -> str | None:
        """
        Detects if input is purely a greeting, pleasantry, or misspelled greeting (e.g. 'good gorming').
        """
        lower_text = text.lower().strip()
        lower_words = [w.lower() for w in words]

        # Check multiword greeting matches and fuzzy typos
        for greeting in self.MULTIWORD_GREETINGS:
            if lower_text == greeting or (lower_text.startswith(greeting) and len(words) <= 7):
                return (
                    "Input appears to be only a conversational greeting without substantive source content. "
                    "Please provide an informative document, article, or task prompt."
                )

        # Fuzzy check for multi-word greetings like "good gorming", "goood morning", "gud morning"
        # Check two-word combinations
        if len(words) <= 14:
            for i in range(len(lower_words) - 1):
                pair = f"{lower_words[i]} {lower_words[i+1]}"
                close_pairs = difflib.get_close_matches(pair, self.MULTIWORD_GREETINGS, n=1, cutoff=0.70)
                if close_pairs:
                    return (
                        f"Input appears to contain a conversational greeting/typo ('{pair}' ~ '{close_pairs[0]}') "
                        "without substantive content. Please provide an informative document, report, or descriptive prompt."
                    )

            # Check single word greetings / typos for short inputs
            greeting_hits = 0
            for w in lower_words:
                if w in self.GREETING_VOCABULARY:
                    greeting_hits += 1
                else:
                    close = difflib.get_close_matches(w, list(self.GREETING_VOCABULARY), n=1, cutoff=0.78)
                    if close:
                        greeting_hits += 1

            if greeting_hits >= max(1, len(lower_words) // 2):
                return (
                    "Input appears to be a greeting or pleasantry without sufficient source content. "
                    "Please provide an informative article or prompt."
                )

        return None

    def _check_gibberish_and_repetition(self, text: str, words: List[str]) -> str | None:
        """Detects low-entropy word spam, character runs, and vowel-deficient strings."""
        if not words:
            return "Input contains no recognizable words."

        # Repeated characters: e.g. "aaaaaa", "zzzzzz"
        if re.search(r"(.)\1{4,}", text):
            return "Input contains excessive repeated character patterns (gibberish detected)."

        # Unique word ratio check for repetition spam (e.g. "test test test test test")
        if len(words) >= 5:
            unique_words = {w.lower() for w in words}
            unique_ratio = len(unique_words) / len(words)
            if unique_ratio < 0.4:
                return (
                    f"Input contains excessive repetitive words (uniqueness ratio {unique_ratio:.2f}). "
                    "Please provide coherent informational content."
                )

        # Consonant cluster / keyboard mash (e.g. "asdfghjkl", "qwrtypsdfg")
        vowels = set("aeiouy")
        for word in words:
            clean_w = word.lower()
            if len(clean_w) >= 6:
                # 5+ consecutive consonants (e.g. asdfghjkl has sdfghjkl)
                if re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", clean_w):
                    return f"Input contains unpronounceable token '{word}' (gibberish detected)."
                # Vowel deficient
                if not any(ch in vowels for ch in clean_w):
                    return f"Input contains unpronounceable token '{word}' (gibberish detected)."

        return None

    def _detect_mode(self, text: str) -> str:
        """Classifies text into RAW_ARTICLE or FREEFORM_PROMPT."""
        lower = text.lower().strip()
        first_words = re.findall(r"\b\w+\b", lower)[:4]

        if not first_words:
            return "RAW_ARTICLE"

        # Check imperative starter
        first_word = first_words[0]
        if first_word in self.PROMPT_DIRECTIVES:
            return "FREEFORM_PROMPT"

        if len(first_words) >= 2 and first_words[0] == "please" and first_words[1] in self.PROMPT_DIRECTIVES:
            return "FREEFORM_PROMPT"

        # Check prompt phrasing
        prompt_phrases = [
            "generate a", "create a", "write a", "draft an", "draft a",
            "summarize this", "summarize the", "explain how", "outline the",
            "tell me about", "convert this", "transform this"
        ]
        for phrase in prompt_phrases:
            if phrase in lower[:60]:
                return "FREEFORM_PROMPT"

        return "RAW_ARTICLE"


source_input_validator = SourceInputValidator()

__all__ = ["SourceInputValidator", "ValidationResult", "source_input_validator"]
