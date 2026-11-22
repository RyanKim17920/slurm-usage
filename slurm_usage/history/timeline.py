"""Time windows and calendar buckets.

Arithmetic is on epoch seconds; only the bucket edges follow the viewer's
timezone, so a DST change gives a 23 or 25 hour day instead of lost or doubled usage.
"""

import datetime
import time
from typing import List, NamedTuple, Optional

HOUR = 3600
DAY = 86400


class Window(NamedTuple):
    """[start, end) in epoch seconds. None means "since the first record" / "until now"."""

    start: Optional[int] = None
    end: Optional[int] = None

    @classmethod
    def last(cls, seconds: float, now: Optional[float] = None) -> "Window":
        end = int(time.time() if now is None else now)
        return cls(end - int(seconds), None if now is None else end)

    @classmethod
    def days(cls, first: datetime.date, last: Optional[datetime.date] = None) -> "Window":
        """Local calendar days, both included."""
        after = (last or first) + datetime.timedelta(days=1)
        return cls(local_midnight(first), local_midnight(after))

    @classmethod
    def all_time(cls) -> "Window":
        return cls(None, None)


def local_midnight(day: datetime.date) -> int:
    return int(time.mktime((day.year, day.month, day.day, 0, 0, 0, 0, 1, -1)))


def local_date(when: float) -> datetime.date:
    parts = time.localtime(when)
    return datetime.date(parts.tm_year, parts.tm_mon, parts.tm_mday)


def hour_edges(start: int, end: int) -> List[int]:
    """Edges of local clock hours covering [start, end], first and last clipped."""
    if end <= start:
        return [start]
    edges = [start]
    current = start
    while True:
        after = current + HOUR
        parts = time.localtime(after)
        # Aligning after the step keeps zones with fractional offsets or shifts on the hour.
        aligned = after - parts.tm_min * 60 - parts.tm_sec
        current = aligned if aligned > current else after
        if current >= end:
            break
        edges.append(current)
    edges.append(end)
    return edges


class Bucket(NamedTuple):
    start: int
    end: int
    value: int
