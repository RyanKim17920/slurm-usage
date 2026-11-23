"""Breakdowns of a list of attempts, for the run-type screen and the Me/Person page:
GPU-seconds per time bin (a chart, a mini trend per kind of run), outcome counts,
runtime and wait quantiles, and the largest groups inside one kind of run.

Pure functions of rows (``history.JobRow``-shaped: start, end, gpus, gpu_seconds,
elapsed, wait, outcome, name, user, job_id); no clock, no I/O.
"""

import heapq
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .data import FAIL_OUTCOMES, OUTCOME_ORDER, name_family


def held_per_bin(rows: Sequence[Any], edges: Sequence[int], now: int,
                 key_of: Optional[Callable[[Any], str]] = None) -> Dict[Optional[str], List[float]]:
    """GPU-seconds held in each bin of ``edges`` (a running attempt counts up to
    ``now``), per key when ``key_of`` is given, else under None."""
    from bisect import bisect_right
    count = max(0, len(edges) - 1)
    out = {}  # type: Dict[Optional[str], List[float]]
    if not count:
        return out
    low, high = edges[0], edges[-1]
    for row in rows:
        gpus, start = row.gpus, row.start
        if not gpus or start is None:
            continue
        end = row.end if row.end is not None else now
        a, b = max(start, low), min(end, high)
        if b <= a:
            continue
        key = key_of(row) if key_of is not None else None
        bins = out.get(key)
        if bins is None:
            bins = out[key] = [0.0] * count
        at = max(0, bisect_right(edges, a) - 1)
        while at < count and edges[at] < b:
            overlap = min(b, edges[at + 1]) - max(a, edges[at])
            if overlap > 0:
                bins[at] += gpus * overlap
            at += 1
    return out


def quantile(values: Sequence[float], fraction: float) -> Optional[float]:
    """Nearest rank; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


class Group:
    """One part of a kind of run: a job-name family, or a person."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.gpu_seconds = 0.0
        self.attempts = 0
        self.failed = 0
        self.ended = 0

    @property
    def failure(self) -> Optional[float]:
        return self.failed / float(self.ended) if self.ended else None


class Detail:
    """What one kind of run looks like, from its attempts."""

    def __init__(self) -> None:
        self.gpu_seconds = 0.0
        self.attempts = 0
        self.jobs = 0
        self.outcomes = {}  # type: Dict[str, int]
        self.runtime = (None, None)  # type: Tuple[Optional[float], Optional[float]]
        self.wait = (None, None)  # type: Tuple[Optional[float], Optional[float]]
        self.groups = []  # type: List[Group]

    @property
    def failure(self) -> Optional[float]:
        ended = sum(n for outcome, n in self.outcomes.items() if outcome != "running")
        if not ended:
            return None
        return sum(self.outcomes.get(o, 0) for o in FAIL_OUTCOMES) / float(ended)


def detail(rows: Sequence[Any], by: str = "family") -> Detail:
    """Outcomes, runtime and wait (median, p90), and the groups (job-name families, or
    people with ``by="user"``) largest first."""
    out = Detail()
    groups = {}  # type: Dict[str, Group]
    runtimes, waits, jobs = [], [], set()
    for row in rows:
        out.attempts += 1
        jobs.add(row.job_id)
        out.gpu_seconds += row.gpu_seconds or 0
        out.outcomes[row.outcome] = out.outcomes.get(row.outcome, 0) + 1
        if row.elapsed is not None and row.outcome != "running":
            runtimes.append(row.elapsed)
        if row.wait is not None:
            waits.append(row.wait)
        key = row.user if by == "user" else name_family(row.name or "")
        group = groups.get(key)
        if group is None:
            group = groups[key] = Group(key)
        group.gpu_seconds += row.gpu_seconds or 0
        group.attempts += 1
        if row.outcome not in ("running", "other"):
            group.ended += 1
            group.failed += row.outcome in FAIL_OUTCOMES
    out.jobs = len(jobs)
    out.runtime = (quantile(runtimes, 0.5), quantile(runtimes, 0.9))
    out.wait = (quantile(waits, 0.5), quantile(waits, 0.9))
    out.groups = sorted(groups.values(), key=lambda g: (-g.gpu_seconds, -g.attempts, g.key))
    return out


def outcome_split(outcomes: Dict[str, int]) -> List[Tuple[str, int]]:
    """(outcome, attempts) in the usual order, empty ones left out."""
    order = list(OUTCOME_ORDER) + sorted(k for k in outcomes if k not in OUTCOME_ORDER)
    return [(name, outcomes[name]) for name in order if outcomes.get(name)]


def proportional(counts: Sequence[float], size: int) -> List[int]:
    """Cells per count, summing to ``size`` (largest remainders), each non-zero count
    at least one cell when there is room. In O(n log n): a stacked bar of thousands of
    job-name families is laid out at once."""
    total = float(sum(counts))
    if size <= 0 or total <= 0:
        return [0] * len(counts)
    exact = [c / total * size for c in counts]
    cells = [int(e) for e in exact]
    have = sum(cells)
    for index in sorted(range(len(counts)), key=lambda i: -(exact[i] - cells[i])):
        if have >= size:
            break
        cells[index] += 1
        have += 1
    # a part too small to round up to a cell still shows, taken from the largest (the
    # first of the largest, as max() picks it)
    if size >= sum(1 for c in counts if c > 0):
        largest = [(-cell, i) for i, cell in enumerate(cells)]
        heapq.heapify(largest)
        for index, count in enumerate(counts):
            if count > 0 and cells[index] == 0:
                while -largest[0][0] != cells[largest[0][1]]:  # an old entry
                    heapq.heappop(largest)
                top = largest[0][1]
                if cells[top] > 1:
                    cells[top] -= 1
                    cells[index] = 1
                    heapq.heappush(largest, (-cells[top], top))
                    heapq.heappush(largest, (-1, index))
    return cells
