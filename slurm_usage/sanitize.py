"""Make untrusted text safe to show in a terminal.

Job names, comments, paths, reasons and log lines are chosen by other users.

``clean(text)``
    One printable line: no ANSI/OSC/DCS sequences, C0/C1 controls, invisible or
    bidi formatting characters. Undecodable bytes become U+FFFD.
``clean_multiline(text)``
    The same, keeping line breaks.
``clean_id(text)``
    For values used as keys (user, account, ...). If cleaning changed the value
    U+FFFD is appended, so a disguised value cannot merge with a real one.

All parsers and collectors in this package store cleaned text in display fields.
Fields holding raw text for ``open()`` say so and are never to be displayed.
"""

import re
import unicodedata
from typing import Optional

__all__ = ["clean", "clean_multiline", "clean_id"]

REPLACEMENT = "�"

# Every alternative is linear: terminators are optional, so an unterminated
# sequence is eaten up to the end of its line and never backtracks.
_ESCAPES = re.compile(
    r"(?:\x1b[\]PX^_]|[\x90\x98\x9d\x9e\x9f])[^\x07\x1b\x9c\n]*(?:\x07|\x1b\\|\x9c)?"
    r"|(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]?"
    r"|\x1b[ -/]*[0-~]?"
)


def _safe_char(char: str, keep_newline: bool) -> str:
    if char == "\n":
        return "\n" if keep_newline else " "
    if char in "\t\r  ":
        return " "
    category = unicodedata.category(char)
    if category == "Cs":  # bytes that were not UTF-8 (surrogateescape)
        return REPLACEMENT
    if category in ("Cc", "Cf"):
        return ""
    return char


def _scrub(text: str, keep_newline: bool) -> str:
    if text.isascii() and text.isprintable():
        return text
    text = _ESCAPES.sub("", text)
    return "".join(_safe_char(char, keep_newline) for char in text)


def clean(text: Optional[object], max_len: Optional[int] = None) -> str:
    """Return one safe line; None becomes the empty string."""
    if text is None:
        return ""
    result = _scrub(str(text), False)
    if max_len is not None and len(result) > max_len:
        result = result[: max(0, max_len - 1)] + "…" if max_len > 0 else ""
    return result


def clean_multiline(text: Optional[object]) -> str:
    if text is None:
        return ""
    return _scrub(str(text).replace("\r\n", "\n"), True)


def clean_id(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    result = clean(text)
    return result if result == text else result + REPLACEMENT
