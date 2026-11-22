"""Hostlists and integer range lists."""

import re
from typing import Iterable, List, Optional

from .units import optional_text

_RANGE = re.compile(r"([0-9]{1,18})(?:-([0-9]{1,18})(?::([0-9]{1,18}))?)?")
DEFAULT_LIMIT = 1 << 20


def split_outside(text: str, separator: str = ",") -> List[str]:
    """Split on a separator that is not inside () or []."""
    parts, depth, begin = [], 0, 0
    for position, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        elif char == separator and depth == 0:
            parts.append(text[begin:position])
            begin = position + 1
    parts.append(text[begin:])
    return [part for part in parts if part]


def _expand_ranges(body: str, limit: int) -> Optional[List[str]]:
    out = []  # type: List[str]
    for part in body.split(","):
        match = _RANGE.fullmatch(part.strip())
        if not match:
            return None
        low, high, step = match.groups()
        if high is None:
            out.append(low)
        else:
            first, last, stride = int(low), int(high), int(step or 1)
            if last < first or stride < 1 or (last - first) // stride + len(out) >= limit:
                return None
            width = len(low)
            out.extend(str(number).zfill(width) for number in range(first, last + 1, stride))
        if len(out) > limit:
            return None
    return out


def parse_ranges(text: Optional[str], limit: int = DEFAULT_LIMIT) -> Optional[List[int]]:
    """"0-2,5" to [0, 1, 2, 5]; None if unknown or not a range list."""
    value = optional_text(text)
    expanded = _expand_ranges(value, limit) if value else None
    return None if expanded is None else [int(item) for item in expanded]


def compress_ranges(numbers: Iterable[int]) -> str:
    ordered, parts = sorted(set(numbers)), []
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1] == ordered[end] + 1:
            end += 1
        low, high = ordered[index], ordered[end]
        parts.append(str(low) if low == high else "%d-%d" % (low, high))
        index = end + 1
    return ",".join(parts)


def expand_hostlist(text: Optional[str], limit: int = DEFAULT_LIMIT) -> List[str]:
    """Expand "gpu[01-03],cpu1" and multi-bracket names; order kept, duplicates removed.

    A malformed or over-limit term is kept literally instead of being guessed at.
    """
    value = optional_text(text)
    if not value:
        return []
    hosts = []  # type: List[str]
    for term in split_outside(value):
        expanded, rest = [""], term.strip()
        while "[" in rest:
            head, _, tail = rest.partition("[")
            body, closed, rest = tail.partition("]")
            numbers = _expand_ranges(body, limit) if closed else None
            if numbers is None or len(expanded) * len(numbers) > limit:
                expanded, rest = [term.strip()], ""
                break
            expanded = [prefix + head + number for prefix in expanded for number in numbers]
        hosts.extend(prefix + rest for prefix in expanded)
    return list(dict.fromkeys(hosts))
