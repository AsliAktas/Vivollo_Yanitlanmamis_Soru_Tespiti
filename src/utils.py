"""
utils.py
--------
Helper utilities for text cleaning.

Functions:
- clean_text(text): Lowercase, remove HTML tags, punctuation, normalize whitespace.
"""
import re

def clean_text(text: str) -> str:
    """Simple text cleaning: lowercase, strip HTML, remove punctuation, collapse spaces."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove punctuation
    text = re.sub(r"[\.,;:!\?\"\(\)\[\]{}]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
