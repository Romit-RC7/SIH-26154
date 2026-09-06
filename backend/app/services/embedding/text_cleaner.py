"""
Text Cleaning and Normalization Service for Document Embedding and Knowledge Extraction.
Prepares raw extracted text, markdown tables, and captions for dense vector embedding.
"""

import re
import unicodedata
from typing import Optional


class TextCleaner:
    """
    Cleans, sanitizes, and normalizes text extracted from multi-modal documents.
    """

    # Non-printable and control characters (excluding newline and tab)
    CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
    
    # Repeated whitespace / horizontal whitespace
    MULTI_SPACE_REGEX = re.compile(r"[^\S\r\n]+")
    
    # Excessive consecutive newlines (3 or more -> 2)
    MULTI_NEWLINE_REGEX = re.compile(r"\n{3,}")
    
    # Common OCR hyphenation at line breaks (e.g. "transfor-\nmation" -> "transformation")
    HYPHENATED_LINEBREAK_REGEX = re.compile(r"(\b\w+)-\s*\n\s*(\w+\b)")

    # Markdown table separator cleaning
    MD_TABLE_DIVIDER_REGEX = re.compile(r"\|(?:\s*:?-+:?\s*\|)+")

    @classmethod
    def normalize_unicode(cls, text: str) -> str:
        """Applies NFKC Unicode normalization to standardize accents, ligatures, and symbols."""
        if not text:
            return ""
        # Normalizes ligatures (like fi, fl) and fullwidth/halfwidth forms
        normalized = unicodedata.normalize("NFKC", text)
        return normalized

    @classmethod
    def clean_control_characters(cls, text: str) -> str:
        """Removes non-printable ASCII and Unicode control characters."""
        if not text:
            return ""
        return cls.CONTROL_CHAR_REGEX.sub("", text)

    @classmethod
    def dehyphenate(cls, text: str) -> str:
        """Fixes hyphenated line breaks common in multi-column PDF extractions."""
        if not text:
            return ""
        return cls.HYPHENATED_LINEBREAK_REGEX.sub(r"\1\2", text)

    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """
        Collapses excessive spaces and tabs while preserving paragraph structure.
        """
        if not text:
            return ""
        # Replace non-breaking spaces with standard space
        text = text.replace("\u00a0", " ").replace("\u200b", "")
        # Collapse multiple horizontal whitespace characters to a single space
        lines = [cls.MULTI_SPACE_REGEX.sub(" ", line).strip() for line in text.splitlines()]
        cleaned_text = "\n".join(lines)
        # Collapse 3+ newlines into max 2
        cleaned_text = cls.MULTI_NEWLINE_REGEX.sub("\n\n", cleaned_text)
        return cleaned_text.strip()

    @classmethod
    def clean_table_markdown(cls, table_md: str) -> str:
        """
        Normalizes markdown tables for embedding and LLM context ingestion.
        Ensures consistent pipe formatting and trimmed cells.
        """
        if not table_md:
            return ""
        lines = table_md.strip().splitlines()
        cleaned_lines = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("|") and line_str.endswith("|"):
                cells = [c.strip() for c in line_str.split("|")[1:-1]]
                cleaned_line = "| " + " | ".join(cells) + " |"
                cleaned_lines.append(cleaned_line)
            else:
                cleaned_lines.append(line_str)
        return "\n".join(cleaned_lines)

    @classmethod
    def clean(cls, text: Optional[str]) -> str:
        """
        Full cleaning pipeline for standard prose and element text.
        """
        if not text:
            return ""
        
        # Step 1: Unicode normalization
        res = cls.normalize_unicode(text)
        # Step 2: Remove control characters
        res = cls.clean_control_characters(res)
        # Step 3: Dehyphenate broken words across line breaks
        res = cls.dehyphenate(res)
        # Step 4: Normalize whitespace and paragraph breaks
        res = cls.normalize_whitespace(res)
        
        return res
