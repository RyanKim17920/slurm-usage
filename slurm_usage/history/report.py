"""The answer to one query."""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from .errors import Note
from .select import Filter
from .stats import METRICS, Stats
from .timeline import Bucket, Window, local_date


class Lane(NamedTuple):
    key: str
    intervals: List[Tuple[int, int]]  # gaps shorter than the resolution are closed


class Utilization(NamedTuple):
    fraction: Optional[float]
    capacity: Optional[int]
    label: str


@dataclass
class Report:
    """Everything the screens need for one window and filter, from a single pass.

    ``groups`` maps a tuple of roles to {tuple of values: Stats}, and always has
    ("outcome",). Every grouping sums exactly to ``total`` (see Stats).
    """

    window: Window  # as requested
    start: int  # as resolved
    end: int  # never after as_of
    as_of: int  # "now" for running jobs: when Slurm was last asked
    where: Filter
    total: Stats
    groups: Dict[Tuple[str, ...], Dict[Tuple[str, ...], Stats]]
    outcomes: Dict[str, Stats]
    edges: List[int]  # local-hour bin edges
    values: Dict[str, List[int]]  # metric to usage per bin
    series_by: Optional[str] = None
    series_metric: str = "gpu"
    keyed_values: Dict[str, List[int]] = field(default_factory=dict)
    lanes: List[Lane] = field(default_factory=list)
    waits: Sequence[int] = ()  # sorted queue waits of attempts that started here
    complete: bool = True  # False if the window is not fully loaded
    efficiency_unchecked: Optional[int] = None  # finished attempts whose steps are not loaded
    notes: List[Note] = field(default_factory=list)
    floor: Optional[int] = None  # utilization counts time from here at the earliest

    def group(self, *roles: str) -> Dict[Tuple[str, ...], Stats]:
        return self.groups[roles]

    def ranked(self, *roles: str, metric: str = "gpu") -> List[Tuple[Tuple[str, ...], Stats]]:
        return sorted(self.groups[roles].items(),
                      key=lambda item: (-(item[1].amount(metric) or 0), -item[1].seconds, item[0]))

    def wait_quantile(self, fraction: float) -> Optional[int]:
        """Nearest rank."""
        if not self.waits:
            return None
        count = len(self.waits)
        return self.waits[min(count - 1, max(0, math.ceil(fraction * count) - 1))]

    def _bins(self, metric: str, key: Optional[str]) -> Optional[List[int]]:
        if metric not in METRICS:
            raise ValueError("unknown metric: %s" % metric)
        if self.total.amount(metric) is None:
            return None
        if key is None:
            return self.values[metric]
        if metric != self.series_metric:
            raise ValueError("per-key series were collected for %s" % self.series_metric)
        return self.keyed_values.get(key) or [0] * (len(self.edges) - 1)

    def buckets(self, metric: str = "gpu", unit: str = "hour",
                key: Optional[str] = None) -> Optional[List[Bucket]]:
        """Local hour or day buckets; None if the metric is unknown. Sums to the total."""
        if unit not in ("hour", "day"):
            raise ValueError("unit is hour or day")
        values = self._bins(metric, key)
        if values is None:
            return None
        edges, out = self.edges, []  # type: List[int], List[Bucket]
        day = None
        for at, value in enumerate(values):
            if unit == "day":
                today = local_date(edges[at])
                if today == day:
                    out[-1] = Bucket(out[-1].start, edges[at + 1], out[-1].value + value)
                    continue
                day = today
            out.append(Bucket(edges[at], edges[at + 1], value))
        return out

    def cumulative(self, metric: str = "gpu", unit: str = "hour",
                   key: Optional[str] = None) -> Optional[List[Tuple[int, int]]]:
        """(time, usage since the window start), from 0 at the start to the total at the end."""
        buckets = self.buckets(metric, unit, key)
        if buckets is None:
            return None
        points, running = [(self.start, 0)], 0
        for bucket in buckets:
            running += bucket.value
            points.append((bucket.end, running))
        return points

    def heatmap(self, metric: str = "gpu", key: Optional[str] = None) -> Optional[List[List[int]]]:
        """[weekday][hour] in local time, Monday first."""
        values = self._bins(metric, key)
        if values is None:
            return None
        grid = [[0] * 24 for _ in range(7)]
        for at, value in enumerate(values):
            if value:
                parts = time.localtime(self.edges[at])
                grid[parts.tm_wday][parts.tm_hour] += value
        return grid

    def utilization(self, capacity: Optional[int], metric: str = "gpu") -> Utilization:
        """Allocated share of ``capacity``: pass the live one, not slurmdbd's belief."""
        start = self.start if self.floor is None else max(self.start, self.floor)
        if self.window.start is None and self.floor is not None:
            start = min(self.start, self.floor)  # all of history: from the floor
        used, span = self.total.amount(metric), self.end - start
        fraction = used / (capacity * span) if used is not None and capacity and span > 0 else None
        return Utilization(fraction, capacity,
                           "assumes today's capacity held for the whole window")
