"""Scalar values: counts, memory, durations, times, exit codes, states."""

import math
import re
import time
from datetime import datetime
from typing import Optional, Tuple

KIB = 1024
_POWERS = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5, "E": 6}
_UNSET = frozenset(("", "(null)", "none", "n/a", "unknown", "invalid", "(none)", "not_set",
                    "notset", "null"))
_UNLIMITED = frozenset(("unlimited", "infinite"))
# NO_VAL and INFINITE leak into text output as numbers.
_SENTINELS = frozenset((2**32 - 1, 2**32 - 2, 2**64 - 1, 2**64 - 2))

_MAX_LEN = 48  # no real value is longer; int() refuses absurd digit strings
_NUMBER = re.compile(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)", re.I)
_DURATION = re.compile(r"(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?(\.\d+)?")


def optional_text(text: Optional[str]) -> Optional[str]:
    """None for the ways Slurm spells "no value", otherwise the stripped text."""
    if text is None:
        return None
    value = text.strip()
    return None if value.lower() in _UNSET else value


def is_unlimited(text: Optional[str]) -> bool:
    return text is not None and text.strip().lower() in _UNLIMITED


def parse_int(text: Optional[str]) -> Optional[int]:
    value = (text or "").strip()
    try:
        number = int(value)
    except ValueError:
        return None
    return None if number in _SENTINELS else number


def parse_float(text: Optional[str]) -> Optional[float]:
    try:
        number = float((text or "").strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _scaled(value: str, default_unit: str) -> Optional[int]:
    match = _NUMBER.fullmatch(value) if len(value) <= _MAX_LEN else None
    if not match:
        return None
    number, unit = match.groups()
    if not unit and "." not in number and int(number) in _SENTINELS:
        return None
    factor = KIB ** _POWERS[(unit or default_unit).upper()]
    # int arithmetic where possible: these values can exceed float precision
    return int(number) * factor if "." not in number else int(round(float(number) * factor))


def parse_count(text: Optional[str]) -> Optional[int]:
    """A count that Slurm may abbreviate with a 1024-based suffix ("2K")."""
    return _scaled((text or "").strip(), "")


def parse_memory(text: Optional[str], default_unit: str = "M") -> Optional[int]:
    """Bytes. A bare number is MiB in scheduling output; pass "" for bytes."""
    value = (text or "").strip()
    if value[-1:] in ("c", "n"):  # old sacct ReqMem: per cpu / per node
        value = value[:-1]
    if value[-1:] in ("B", "b"):
        value = value[:-1]
    return _scaled(value, default_unit)


def memory_basis(text: Optional[str]) -> Optional[str]:
    """"cpu" or "node" for old-style ReqMem suffixes (4000Mc, 16Gn)."""
    return {"c": "cpu", "n": "node"}.get((text or "").strip()[-1:])


def parse_duration(text: Optional[str]) -> Optional[float]:
    """Seconds from D-HH[:MM[:SS]], HH:MM:SS, MM:SS[.mmm] or bare minutes.

    UNLIMITED, N/A and implausible values (over a century, seen as garbage CPU
    times) are None. Use ``is_unlimited`` to tell unlimited from unknown.
    """
    value = (text or "").strip()
    match = _DURATION.fullmatch(value) if len(value) <= _MAX_LEN else None
    if not match:
        return None
    days, first, second, third, fraction = match.groups()
    a, b, c = int(first), int(second or 0), int(third or 0)
    if days is not None:
        seconds = int(days) * 86400 + a * 3600 + b * 60 + c
    elif third is not None:
        seconds = a * 3600 + b * 60 + c
    elif second is not None:
        seconds = a * 60 + b
    else:
        if a in _SENTINELS:
            return None
        seconds = a * 60
    if seconds > 100 * 366 * 86400:
        return None
    return seconds + float(fraction) if fraction else seconds


def parse_time(text: Optional[str]) -> Optional[int]:
    """Epoch seconds from epoch digits or an ISO timestamp.

    The runner asks Slurm for epoch seconds. ISO values without an offset are
    read in the process's timezone, which is how Slurm prints them.
    """
    value = (text or "").strip()
    if len(value) > _MAX_LEN:
        return None
    if value.isascii() and value.isdigit():
        number = int(value)
        return number if 0 < number < 2**32 - 2 else None
    if len(value) < 19 or value[10] != "T":
        return None
    try:
        when = datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S")
        return int(time.mktime(when.timetuple()))
    except (ValueError, OverflowError):
        return None


def parse_exit_code(text: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """(exit status, signal) from "1:0"."""
    status, separator, signal = (text or "").strip().partition(":")
    return parse_int(status), parse_int(signal) if separator else None


def parse_state(text: Optional[str]) -> Optional[str]:
    """"CANCELLED by 123" and "CANCELLED+" become "CANCELLED"."""
    value = optional_text(text)
    return value.split()[0].rstrip("+").upper() if value else None
