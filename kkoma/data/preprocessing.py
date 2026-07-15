"""Minimal document cleaning (spec section 4.3 / 14.3).

Only light filters are applied: drop empty / too-short docs, strip NUL and
abnormal control characters, and reject documents that fail UTF-8 round trip.
No heavy Unicode normalization and no Korean morphological analysis.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

# Control chars except tab/newline/carriage-return.
_CONTROL_RE = re.compile(
    "[" + "".join(map(re.escape, [chr(c) for c in range(32) if c not in (9, 10, 13)])) + "\x7f]"
)


def strip_control_chars(text: str) -> str:
    text = text.replace("\x00", "")
    return _CONTROL_RE.sub("", text)


def is_valid_utf8(text: str) -> bool:
    try:
        text.encode("utf-8").decode("utf-8")
        return True
    except (UnicodeDecodeError, UnicodeEncodeError):
        return False


@dataclass
class CleanConfig:
    min_chars: int = 16
    strip_controls: bool = True
    require_utf8: bool = True


def clean_document(text: str, config: Optional[CleanConfig] = None) -> Optional[str]:
    """Return a cleaned document, or ``None`` if it should be dropped."""

    config = config or CleanConfig()
    if text is None:
        return None
    if config.strip_controls:
        text = strip_control_chars(text)
    text = text.strip()
    if not text:
        return None
    if len(text) < config.min_chars:
        return None
    if config.require_utf8 and not is_valid_utf8(text):
        return None
    return text


def normalize_minimal(text: str) -> str:
    """NFC normalization only (kept minimal, optional helper)."""

    return unicodedata.normalize("NFC", text)


__all__ = [
    "CleanConfig",
    "clean_document",
    "strip_control_chars",
    "is_valid_utf8",
    "normalize_minimal",
]
