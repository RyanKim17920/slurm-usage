"""Run types: usage split by what kind of run it was, for one person or any filter."""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .aggregate import _Pass, usage_ratios
from .errors import Cancelled, canceller
from .select import Filter, Part, Selection, resolve
from .stats import Stats
from .timeline import Window

TYPE_ROLES = ("gpu_count", "name_family", "account", "qos", "partition", "outcome",
              "node_count")
FAILURES = ("failed", "node_fail", "timeout")

_UUID = re.compile(r"[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?"
                   r"[0-9a-fA-F]{12}")
_DATE = re.compile(r"(?<![0-9])(?:19|20)[0-9]{2}[-_.]?[01][0-9][-_.]?[0-3][0-9]"
                   r"(?:[T_-]?[0-2][0-9][:_-]?[0-5][0-9](?:[:_-]?[0-5][0-9])?)?(?![0-9])")
_HEX = re.compile(r"(?<![0-9A-Za-z])(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{6,}"
                  r"(?![0-9A-Za-z])")
_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_REPEAT = re.compile(r"(<n>)(?:[-_.x,]?<n>)+")


@lru_cache(maxsize=65536)
def name_family(name: str) -> str:
    """"train_lr3e-4_seed2_20260101" -> "train_lr<n>e-<n>_seed<n>_<date>".

    Dates, UUIDs, hex hashes and numbers (array indices, seeds, steps) become
    placeholders, so reruns of one script group together.
    """
    text = name.strip()
    if not text:
        return "(no name)"
    text = _UUID.sub("<id>", text)
    text = _DATE.sub("<date>", text)
    text = _HEX.sub("<hex>", text)
    text = _NUMBER.sub("<n>", text)
    return _REPEAT.sub("<n>", text)


def _node_count(nodes: str) -> int:
    from ..parse import expand_hostlist
    return len(expand_hostlist(nodes)) if nodes else 0


def type_of(by: str, part: Part, name: str, nodes: str) -> str:
    if by == "gpu_count":
        gpus = part.row[10]
        return "unknown" if gpus is None else str(gpus)
    if by == "name_family":
        return name_family(name)
    if by == "node_count":
        return str(_node_count(nodes))
    if by == "outcome":
        return part.outcome
    return getattr(part.roles, by)


@dataclass
class RunType:
    key: str
    by: str
    stats: Stats  # attempts, jobs, GPU/CPU seconds, waits, time-limit use, efficiency
    share: Optional[float]  # of the selection's total by the chosen metric
    outcomes: Dict[str, int] = field(default_factory=dict)  # attempts per outcome
    median_runtime: Optional[int] = None  # whole attempts, less suspension
    median_wait: Optional[int] = None
    example: str = ""  # one real name of this family

    @property
    def failure_rate(self) -> Optional[float]:
        """Failed, node-failed or timed-out attempts over attempts that ended."""
        ended = sum(n for outcome, n in self.outcomes.items() if outcome != "running")
        if not ended:
            return None
        return sum(self.outcomes.get(outcome, 0) for outcome in FAILURES) / ended


def _median(values: List[int]) -> Optional[int]:
    if not values:
        return None
    values.sort()
    return values[(len(values) - 1) // 2]


def run_types(store: Any, as_of: int, window: Optional[Window], where: Optional[Filter],
              by: str, metric: str = "gpu", gpu_tracked: Optional[bool] = None,
              cancel: Any = None) -> List[RunType]:
    if by not in TYPE_ROLES:
        raise ValueError("by is one of %s" % ", ".join(TYPE_ROLES))
    window, where, cancelled = window or Window(), where or Filter(), canceller(cancel)
    with store.lock:
        nodesets = store.nodesets()
        chosen = Selection(store.dims(), nodesets, where)
        first = store.first_start(chosen.dim_ids) if window.start is None else None
        span = resolve(window, as_of, first)
        repeated = store.repeated_jobs()
        groups = {}  # type: Dict[str, Tuple[Stats, Dict[str, int], List[int], List[int], str]]
        rows = store.scan(span.low, span.high, chosen.dim_ids, usage=True, extra=True)
        try:
            for number, part in enumerate(chosen.parts(rows, span)):
                if not number % 1024 and cancelled():
                    raise Cancelled()
                if part.problem:
                    continue
                row = part.row
                name = row[18] or ""
                key = type_of(by, part, name, nodesets.get(row[3], ""))
                entry = groups.get(key)
                if entry is None:
                    entry = groups[key] = (Stats(gpu_tracked), {}, [], [], name)
                stats, outcomes, runtimes, waits, _ = entry
                repeat = (row[1], part.roles.uid) if row[1] in repeated else None
                wait = limit = None
                if part.starts_here and row[5] is not None and row[5] <= row[6]:
                    wait = row[6] - row[5]
                    waits.append(wait)
                if (part.ends_here and part.outcome in ("completed", "timeout") and row[12]
                        and row[12] > 0):
                    limit = row[12]
                ratios = usage_ratios(row) if row[7] is not None else None
                _Pass.take(part, [stats], part.cpus, part.gpus, part.mem, repeat, wait, limit,
                           ratios)
                outcomes[part.outcome] = outcomes.get(part.outcome, 0) + 1
                if row[7] is not None:
                    runtimes.append(max(0, row[7] - row[6] - row[8]))
        finally:
            rows.close()
    total = sum(entry[0].amount(metric) or 0 for entry in groups.values())
    out = []
    for key, (stats, outcomes, runtimes, waits, example) in groups.items():
        stats.finish()
        amount = stats.amount(metric)
        out.append(RunType(key, by, stats, amount / total if total and amount is not None
                           else None, outcomes, _median(runtimes), _median(waits), example))
    out.sort(key=lambda t: (-(t.stats.amount(metric) or 0), -t.stats.attempts, t.key))
    return out
