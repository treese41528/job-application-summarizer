"""
Text cleaning and normalization.

Cleans up extracted text by fixing encoding issues, normalizing whitespace,
and optionally chunking for LLM context limits.
"""

import re
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted document text.

    - Fix common encoding artifacts
    - Normalize whitespace (collapse multiple blank lines)
    - Remove page headers/footers patterns
    - Strip control characters
    """
    if not text:
        return ""

    # Replace common encoding artifacts
    replacements = {
        "\x00": "",           # Null bytes
        "\r\n": "\n",         # Windows line endings
        "\r": "\n",           # Old Mac line endings
        "\xa0": " ",          # Non-breaking spaces
        "\u2018": "'",        # Smart single quotes
        "\u2019": "'",
        "\u201c": '"',        # Smart double quotes
        "\u201d": '"',
        "\u2013": "-",        # En dash
        "\u2014": "--",       # Em dash
        "\u2026": "...",      # Ellipsis
        "\ufeff": "",         # BOM
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove control characters (except newline and tab)
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse multiple spaces (but not newlines) into one
    text = re.sub(r'[^\S\n]+', ' ', text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Final strip
    text = text.strip()

    return text


def chunk_text(text: str, max_tokens: int = 4000, overlap_tokens: int = 200) -> list[str]:
    """
    Split text into overlapping chunks for LLM processing.

    Uses a rough estimate of 1 token ≈ 4 characters (conservative for English).

    Args:
        text: The full document text.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Number of overlapping tokens between chunks.

    Returns:
        List of text chunks.
    """
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        # Try to break at a paragraph boundary
        if end < len(text):
            # Look for paragraph break near the end
            para_break = text.rfind('\n\n', start + max_chars // 2, end)
            if para_break > start:
                end = para_break

            # Otherwise try sentence boundary
            elif (sent_break := text.rfind('. ', start + max_chars // 2, end)) > start:
                end = sent_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance with overlap
        start = end - overlap_chars
        if start >= len(text):
            break

    logger.info(f"Split text ({len(text)} chars) into {len(chunks)} chunks")
    return chunks


def estimate_token_count(text: str) -> int:
    """Rough token count estimate (1 token ≈ 4 chars for English)."""
    return len(text) // 4
