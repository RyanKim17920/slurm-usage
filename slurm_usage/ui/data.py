"""The only place the UI touches ``slurm_usage.history``.

``HistoryData`` wraps a ``history.History`` (or anything with the same methods,
which is how tests run without Slurm) and adds what screens need on top: named
ranges with their previous period, trend rows, run-type breakdowns and an
events feed. Everything here is synchronous; the UI calls it from a worker
thread (see ``loader``).

``NoHistory`` stands in when history is switched off or unavailable: every
query raises ``HistoryError`` with a status the UI turns into a placeholder.
"""

import json
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from ..model import Status
from ..parse import expand_hostlist
from . import fmt

try:  # the history layer may be absent from a trimmed install
    from ..history import Filter, HistoryError, Window
except ImportError:  # pragma: no cover
    Filter = Window = None  # type: ignore

    class HistoryError(Exception):  # type: ignore
        def __init__(self, status: str, message: str = "") -> None:
            super().__init__(message or status)
            self.status, self.message = status, message or status

DAY = 86400
# (name, seconds); None = all of history
RANGES = (("day", DAY), ("week", 7 * DAY), ("month", 30 * DAY), ("year", 365 * DAY),
          ("all", None))
RANGE_NAMES = tuple(name for name, _ in RANGES)
MEMORY_RANGES = ("day", "week", "month")
_SECONDS = dict(RANGES)
OUTCOME_ORDER = ("running", "completed", "preempted", "timeout", "failed", "node_fail",
                 "cancelled", "requeued", "other")
FAILED = ("failed", "node_fail")  # failure bursts in the events feed
# The "fail" column of run types, everywhere (here and in history.runtypes): attempts
# that failed, lost their node or hit their time limit, among attempts that ended.
FAIL_OUTCOMES = ("failed", "node_fail", "timeout")
ATTEMPT_LIMIT = 20000  # per person page
JOB_PAGE = 500  # rows of a jobs pane loaded at once; more load as the cursor reaches the end


def in_range(event: Any, win: Any) -> bool:
    """Whether a history event has activity inside the window: it ends (or, without an
    end, happens) at or after the window's start."""
    start = getattr(win, "start", None)
    if start is None:
        return True
    last = getattr(event, "end", None)
    if last is None:
        last = getattr(event, "time", None)
    return last is None or last >= start


def window(name: str, now: Optional[float] = None) -> Any:
    seconds = _SECONDS[name]
    if seconds is None:
        return Window(None, None)
    end = int(time.time() if now is None else now)
    return Window(end - seconds, None)


def previous_window(name: str, now: Optional[float] = None) -> Any:
    seconds = _SECONDS[name]
    if seconds is None:
        return None
    end = int(time.time() if now is None else now) - seconds
    return Window(end - seconds, end)


def where(user: Optional[str] = None, account: Optional[str] = None, qos: Optional[str] = None,
          partition: Optional[str] = None) -> Any:
    return Filter(user=user, account=account, qos=qos, partition=partition)


def status_of(error: BaseException) -> str:
    return getattr(error, "status", Status.ERROR)


# -- results -----------------------------------------------------------------

class TrendRow(NamedTuple):
    range: str
    gpu_hours: Optional[float]
    cpu_hours: Optional[float]
    gpu_util: Optional[float]  # of live capacity
    cpu_util: Optional[float]
    gpu_eff: Any  # history.Efficiency or None
    cpu_eff: Any
    gpu_delta: Optional[float]  # previous period's GPU-hours, for fmt.delta
    cpu_delta: Optional[float]
    spark: Optional[List[float]]
    complete: bool
    attempts: int
    jobs: int
    previous_loaded: bool = True  # False: the previous period is not (fully) in the cache


class RunType(NamedTuple):
    dimension: str
    label: str
    attempts: int
    jobs: int
    gpu_hours: Optional[float]
    share: Optional[float]
    runtime: Optional[float]  # median, seconds
    wait: Optional[float]  # median, seconds
    limit_use: Optional[float]  # median elapsed/limit of completed and timed-out
    gpu_eff: Optional[float]
    gpu_cover: Optional[float]
    cpu_eff: Optional[float]
    cpu_cover: Optional[float]
    failure: Optional[float]  # failed, node_fail or timeout among finished (FAIL_OUTCOMES)
    rows: Optional[List[Any]]  # attempts, when computed here; None = drill by ``key``
    key: str = ""  # the history layer's key, for jobs(run_type=(dimension, key))
    complete: bool = True  # False: some sizes unknown, hours are a lower bound


class Event(NamedTuple):
    at: Optional[int]
    kind: str  # big_run, preempted, failures, timeouts, longest, gap, new_role, record_day
    text: str
    style: str  # canvas style name
    rows: List[Any]  # attempts to drill into
    job_id: Optional[str] = None
    where: Any = None  # history Filter to drill into
    window: Any = None  # history Window to drill into


# -- run families --------------------------------------------------------------

_HEXISH = re.compile(r"^(?=.*\d)[0-9a-f]{6,}$", re.I)
_TRAILING = re.compile(r"^(.*?[A-Za-z])(\d+)$")
_SPLIT = re.compile(r"([_\-.:/+= ]+)")

try:
    from ..history import name_family as _history_family
except ImportError:  # pragma: no cover
    _history_family = None


def _local_family(name: str) -> str:
    out = []
    for token in _SPLIT.split(name):
        if not token or _SPLIT.fullmatch(token):
            out.append(token)
        elif token.isdigit():
            out.append("<n>")
        elif _HEXISH.match(token):
            out.append("<hex>")
        else:
            match = _TRAILING.match(token)
            out.append(match.group(1) + "<n>" if match else token)
    return "".join(out)


def name_family(name: Optional[str]) -> str:
    """'train_lr3_0042' -> 'train_lr<n>_<n>': numbers, dates and hashes become placeholders."""
    if not name or not name.strip():
        return "(no name)"
    return _history_family(name) if _history_family is not None else _local_family(name)


def node_count(row: Any) -> int:
    text = getattr(row, "nodes", "") or ""
    if not text or text in ("None assigned", "(none)"):
        return 0
    names = expand_hostlist(text, 4096)
    return len(names) if names else 1


def gpu_class(row: Any) -> str:
    gpus = row.gpus
    if gpus is None:
        return "GPUs unknown"
    if node_count(row) > 1:
        return "multi-node"
    if gpus == 0:
        return "CPU only"
    return "x%d" % gpus


DIMENSIONS = (("gpu_count", "GPUs"), ("node_count", "nodes"), ("name_family", "job name"),
              ("account", "account"), ("qos", "QOS"), ("partition", "partition"),
              ("outcome", "outcome"))


def _key(row: Any, dimension: str) -> str:
    if dimension == "gpu_count":
        return "unknown" if row.gpus is None else str(row.gpus)
    if dimension == "node_count":
        return str(node_count(row))
    if dimension == "name_family":
        return name_family(row.name)
    return getattr(row, dimension)


def type_label(dimension: str, key: str) -> str:
    if dimension == "gpu_count":
        return "CPU only" if key == "0" else "x" + key if key.isdigit() else "GPUs " + key
    if dimension == "node_count":
        return "%s node%s" % (key, "" if key == "1" else "s") if key.isdigit() else key
    return key


def _weighted(rows: Sequence[Any], value: str, weight: str) -> Tuple[Optional[float],
                                                                      Optional[float]]:
    total = covered = used = 0.0
    for row in rows:
        amount = getattr(row, weight) or 0
        total += amount
        ratio = getattr(row, value)
        if ratio is not None and amount:
            covered += amount
            used += min(ratio, 1.0) * amount if value == "gpu_peak_utilization" else ratio * amount
    if not covered:
        return None, (0.0 if total else None)
    return used / covered, covered / total if total else None


def _numeric_order(key: str) -> Tuple[int, int]:
    return (0, int(key)) if key.isdigit() else (1, 0)


def sort_types(types: List[RunType], dimension: str) -> List[RunType]:
    if dimension in ("gpu_count", "node_count"):
        return sorted(types, key=lambda r: _numeric_order(r.key))
    if dimension == "outcome":
        return sorted(types, key=lambda r: OUTCOME_ORDER.index(r.key)
                      if r.key in OUTCOME_ORDER else 99)
    return sorted(types, key=lambda r: (-(r.gpu_hours or 0), -r.attempts, r.label))


def run_types(rows: Sequence[Any], dimension: str) -> List[RunType]:
    """Attempts grouped into kinds of runs, computed here from a list of attempts.
    GPU-hours are allocated, inside the window."""
    groups = OrderedDict()  # type: Dict[str, List[Any]]
    for row in rows:
        groups.setdefault(_key(row, dimension), []).append(row)
    known = [row.gpu_seconds for row in rows if row.gpu_seconds is not None]
    total = sum(known) if known else None
    out = []
    for key, members in groups.items():
        gpu = [m.gpu_seconds for m in members if m.gpu_seconds is not None]
        gpu_seconds = sum(gpu) if gpu else None
        finished = [m for m in members if m.outcome not in ("running", "other")]
        limits = [m.time_limit_used for m in members
                  if m.outcome in ("completed", "timeout") and m.time_limit_used is not None]
        gpu_eff, gpu_cover = _weighted(members, "gpu_peak_utilization", "gpu_seconds")
        cpu_eff, cpu_cover = _weighted(members, "cpu_efficiency", "cpu_seconds")
        out.append(RunType(
            dimension, type_label(dimension, key), len(members), len({m.job_id for m in members}),
            None if gpu_seconds is None else gpu_seconds / 3600.0,
            gpu_seconds / total if gpu_seconds is not None and total else None,
            fmt.median([m.elapsed for m in members if m.elapsed is not None]),
            fmt.median([m.wait for m in members if m.wait is not None]),
            fmt.median(limits), gpu_eff, gpu_cover, cpu_eff, cpu_cover,
            sum(m.outcome in FAIL_OUTCOMES for m in finished) / len(finished) if finished
            else None,
            members, key, len(gpu) == len(members)))
    return sort_types(out, dimension)


def from_history_type(item: Any, dimension: str) -> RunType:
    """history.RunType -> RunType."""
    stats = item.stats
    gpu_eff = _efficiency(stats, "gpu") or (None, None)
    cpu_eff = _efficiency(stats, "cpu") or (None, None)
    try:
        complete = stats.complete("gpu")
    except (AttributeError, ValueError):
        complete = True
    return RunType(dimension, type_label(dimension, item.key), stats.attempts, stats.jobs,
                   stats.gpu_hours, item.share, item.median_runtime, item.median_wait,
                   stats.time_limit_used, gpu_eff[0], gpu_eff[1], cpu_eff[0], cpu_eff[1],
                   item.failure_rate, None, item.key, complete)


_EVENT_STYLE = {"largest_allocation": "good", "record_day": "good", "longest_run": "plain",
                "preemption_run": "warn", "failure_burst": "bad", "idle_gap": "dim",
                "first_account": "plain", "first_qos": "plain", "change": "plain",
                "no_baseline": "dim"}


def from_history_event(item: Any) -> Event:
    return Event(item.time, item.kind, item.sentence, _EVENT_STYLE.get(item.kind, "plain"), [],
                 item.job_id, item.where, item.window)


# -- events ------------------------------------------------------------------

def _end(row: Any, now: int) -> int:
    if row.end is not None:
        return row.end
    return now if row.start is not None else 0


def _span_text(first: Optional[int], last: Optional[int]) -> str:
    if first is None or last is None:
        return ""
    if fmt.day(first) == fmt.day(last):
        return "on " + fmt.day(first)
    return "between %s and %s" % (fmt.day(first), fmt.day(last))


def _clusters(rows: List[Any], gap: int, minimum: int, now: int) -> List[List[Any]]:
    ordered = sorted(rows, key=lambda r: _end(r, now))
    out, current = [], []  # type: List[List[Any]], List[Any]
    for row in ordered:
        if current and _end(row, now) - _end(current[-1], now) > gap:
            if len(current) >= minimum:
                out.append(current)
            current = []
        current.append(row)
    if len(current) >= minimum:
        out.append(current)
    return out


def events(rows: Sequence[Any], range_name: str, now: Optional[float] = None,
           report: Any = None, previous: Any = None, limit: int = 40) -> List[Event]:
    """Notable things in a person's attempts, newest first. Plain sentences with numbers."""
    now = int(time.time() if now is None else now)
    found = []  # type: List[Event]
    started = [r for r in rows if r.start is not None]
    span = _SECONDS[range_name] or (now - min((r.start for r in started), default=now))

    # big runs: the largest allocations, once per job-name family and size
    sizes = sorted((r.gpus for r in started if r.gpus), reverse=True)
    if sizes and sizes[0] >= 4:
        threshold = max(4, sizes[0] // 2)
        seen = set()
        for row in sorted(started, key=lambda r: r.start):
            if (row.gpus or 0) < threshold:
                continue
            family = name_family(row.name)
            if (family, row.gpus) in seen:
                continue
            seen.add((family, row.gpus))
            members = [r for r in started if name_family(r.name) == family and r.gpus == row.gpus]
            nodes = node_count(row)
            where_ = " on %d nodes" % nodes if nodes > 1 else ""
            more = " (%d attempts at this size)" % len(members) if len(members) > 1 else ""
            found.append(Event(row.start, "big_run", "Started a %d-GPU run of %s (job %s)%s%s"
                               % (row.gpus, family, row.display_id, where_, more), "good",
                               members, row.job_id))

    # preemptions and timeouts: grouped by family, not called waste
    for outcome, kind, verb, style in (("preempted", "preempted", "were preempted", "warn"),
                                       ("timeout", "timeouts", "hit their time limit", "warn")):
        by_family = OrderedDict()  # type: Dict[str, List[Any]]
        for row in rows:
            if row.outcome == outcome:
                by_family.setdefault(name_family(row.name), []).append(row)
        for family, members in by_family.items():
            if len(members) < (3 if outcome == "preempted" else 2):
                continue
            ends = [r.end for r in members if r.end is not None]
            gpu = [r.gpu_seconds for r in members if r.gpu_seconds is not None]
            hours_ = (", %s GPU-h allocated" % fmt.hours(sum(gpu) / 3600.0)) if gpu else ""
            found.append(Event(max(ends) if ends else None, kind, "%d attempts of %s %s %s "
                               "(%d jobs%s)" % (len(members), family, verb,
                                                _span_text(min(ends, default=None),
                                                           max(ends, default=None)),
                                                len({r.job_id for r in members}), hours_),
                               style, members))

    # bursts of failures: three or more within an hour of each other
    failed = [r for r in rows if r.outcome in FAILED and r.end is not None]
    for burst in _clusters(failed, 3600, 3, now):
        first, last = burst[0].end, burst[-1].end
        families = sorted({name_family(r.name) for r in burst})
        codes = sorted({str(c) for c in (getattr(r, "exit_code", None) for r in burst)
                        if c is not None})
        exit_text = " with exit %s" % "/".join(codes[:3]) if codes else ""
        found.append(Event(last, "failures", "Burst of %d failed attempts%s in %s (%s)"
                           % (len(burst), exit_text, fmt.duration(last - first) if last > first
                              else "under a minute", ", ".join(families[:3])
                              + (", …" if len(families) > 3 else "")), "bad", burst))

    # the longest run
    timed = [r for r in started if r.elapsed]
    if timed:
        longest = max(timed, key=lambda r: r.elapsed)
        on = (" on %d GPUs" % longest.gpus) if longest.gpus else ""
        found.append(Event(longest.start, "longest", "Longest run: %s (job %s), %s%s, %s"
                           % (name_family(longest.name), longest.display_id,
                              fmt.duration(longest.elapsed), on, longest.outcome), "plain",
                           [longest], longest.job_id))

    # idle gaps: nothing allocated for a long stretch
    if started and span >= 7 * DAY:
        minimum = max(DAY, span // 10)
        intervals = sorted((r.start, _end(r, now)) for r in started)
        cover = intervals[0][1]
        for begin, end in intervals[1:]:
            if begin - cover >= minimum:
                found.append(Event(cover, "gap", "No jobs ran for %s (%s → %s)"
                                   % (fmt.duration(begin - cover), fmt.day(cover), fmt.day(begin)),
                                   "dim", []))
            cover = max(cover, end)

    # first use of an account, QOS or partition compared with the previous period
    if previous is not None:
        for role in ("account", "qos", "partition"):
            try:
                before = {key[0] for key in previous.group(role)}
            except (KeyError, AttributeError):
                continue
            firsts = OrderedDict()  # type: Dict[str, Any]
            for row in sorted(started, key=lambda r: r.start):
                value = getattr(row, role)
                if value not in before and value not in firsts:
                    firsts[value] = row
            for value, row in firsts.items():
                members = [r for r in rows if getattr(r, role) == value]
                found.append(Event(row.start, "new_role", "First use of %s %s this %s (job %s, "
                                   "%d attempts since)" % (role.replace("qos", "QOS"), value,
                                                           range_name, row.display_id,
                                                           len(members)), "plain", members,
                                   row.job_id))

    # record day
    if report is not None and span >= 7 * DAY:
        try:
            days = report.buckets("gpu", "day")
        except (ValueError, AttributeError, KeyError):
            days = None
        if days:
            best = max(days, key=lambda b: b.value)
            if best.value > 0:
                members = [r for r in started if r.start < best.end and _end(r, now) > best.start]
                found.append(Event(best.start, "record_day", "Record day: %s GPU-h allocated on "
                                   "%s (%d attempts)" % (fmt.hours(best.value / 3600.0),
                                                         fmt.day(best.start), len(members)),
                                   "good", members))

    found.sort(key=lambda e: e.at or 0, reverse=True)
    return found[:limit]


# -- one range in detail ---------------------------------------------------------

HOUR = 3600
# how a range is binned for its detail chart: day hourly, week by three hours, month
# daily, year weekly; all of history weekly, or monthly once it spans years
BIN_UNITS = {"day": "hour", "week": "3hour", "month": "day", "year": "week"}
UNIT_SECONDS = {"hour": HOUR, "3hour": 3 * HOUR, "day": DAY, "week": 7 * DAY,
                "month": 30 * DAY}
UNIT_WORDS = {"hour": "hour", "3hour": "3 hours", "day": "day", "week": "week",
              "month": "month"}


def bin_unit(range_name: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
    unit = BIN_UNITS.get(range_name)
    if unit is not None:
        return unit
    span = (end - start) if start is not None and end is not None else 0
    if span <= 45 * DAY:
        return "day"
    return "week" if span <= 3 * 365 * DAY else "month"


_UNIT_ORDER = ("hour", "3hour", "day", "week", "month")


def chart_unit(range_name: str, start: int, end: int, covered: int) -> str:
    """The range's bins, finer when its history begins well inside it (a year with
    six weeks of data is drawn by the day, not as six bars)."""
    unit = bin_unit(range_name, start, end)
    if covered > start + 2 * DAY:
        short = bin_unit("all", covered, end)
        if _UNIT_ORDER.index(short) < _UNIT_ORDER.index(unit):
            return short
    return unit


def bin_edges(start: int, end: int, unit: str) -> List[int]:
    """[start, boundaries on local clock/calendar edges..., end] for ``unit``."""
    import datetime
    edges = [start]
    if end <= start:
        return [start, start]
    if unit in ("hour", "3hour"):
        every = 3 if unit == "3hour" else 1
        at = start - start % HOUR
        while True:
            at += HOUR
            if at >= end:
                break
            if time.localtime(at).tm_hour % every == 0 and at > edges[-1]:
                edges.append(at)
    else:
        first = time.localtime(start)
        day = datetime.date(first.tm_year, first.tm_mon, first.tm_mday)
        while True:
            day += datetime.timedelta(days=1)
            if unit == "week" and day.weekday() != 0:
                continue
            if unit == "month" and day.day != 1:
                continue
            at = int(time.mktime((day.year, day.month, day.day, 0, 0, 0, 0, 1, -1)))
            if at >= end:
                break
            if at > edges[-1]:
                edges.append(at)
    edges.append(end)
    # a sliver of a bin at either end would read as a spike or a dip: it joins its
    # neighbour
    nominal = UNIT_SECONDS[unit]
    if len(edges) > 3 and edges[1] - edges[0] < nominal / 2:
        del edges[1]
    if len(edges) > 3 and edges[-1] - edges[-2] < nominal / 2:
        del edges[-2]
    return edges


def rebin(hour_edges: Sequence[int], values: Sequence[int], edges: Sequence[int],
          shift: int = 0) -> List[int]:
    """Sum per-hour usage (``values`` over ``hour_edges``) into the bins ``edges`` (moved
    back by ``shift`` seconds, for the previous period); nothing is lost or counted twice
    as long as every bin edge is an hour edge."""
    from bisect import bisect_right
    out = [0] * max(0, len(edges) - 1)
    if not out:
        return out
    moved = [e - shift for e in edges]
    last = len(out) - 1
    for at, value in enumerate(values):
        if not value:
            continue
        index = bisect_right(moved, hour_edges[at]) - 1
        out[min(last, max(0, index))] += value
    return out


def covered_start(total: Any, gpu_util: Optional[float], cpu_util: Optional[float],
                  gpu_capacity: Optional[int], cpu_capacity: Optional[int], end: float,
                  start: float) -> int:
    """Where a utilization was measured from: its span is usage / (capacity x share).
    (The rollup applies the capacity floor without returning it.)"""
    for util, capacity, metric in ((gpu_util, gpu_capacity, "gpu"),
                                   (cpu_util, cpu_capacity, "cpu")):
        used = total.amount(metric) if total is not None else None
        if util and capacity and used:
            return int(round(end - used / (capacity * util)))
    return int(start)


def _sum_amounts() -> Any:
    from ..history import Amounts
    return Amounts()


def _less(total: Any, part: Any) -> Any:
    """``total`` minus ``part`` (Amounts), never below zero."""
    out = _sum_amounts()
    for name in out.__slots__:
        setattr(out, name, max(0, getattr(total, name) - (getattr(part, name) if part else 0)))
    return out


def _trim(edges: List[int], series: Sequence[Optional[List[int]]], covered: int
          ) -> Tuple[List[int], List[Optional[List[int]]]]:
    """Bins from ``covered`` on: the ones wholly before it go (nothing was recorded
    there), the first one starts at it. Sums are unchanged."""
    if covered <= edges[0] or len(edges) < 3:
        return edges, list(series)
    drop = 0
    while drop < len(edges) - 2 and edges[drop + 1] <= covered and not any(
            values[drop] for values in series if values is not None):
        drop += 1
    edges = edges[drop:]
    edges[0] = max(edges[0], min(covered, edges[1] - 1))
    return edges, [None if values is None else values[drop:] for values in series]


class RangeSeries(NamedTuple):
    """One range for one scope (the cluster, a person, an account), binned for a chart.

    ``gpu``/``cpu`` are allocated seconds per bin and sum exactly to ``total``'s; the
    previous equal period is binned at the same offsets (None for all of history)."""

    range: str
    start: int
    end: int
    unit: str
    edges: List[int]
    gpu: Optional[List[int]]
    cpu: Optional[List[int]]
    prev_gpu: Optional[List[int]]
    prev_cpu: Optional[List[int]]
    total: Any  # history Stats
    previous: Any  # Stats of the previous period, or None
    complete: bool
    previous_complete: bool
    groups: Dict[str, List[Tuple[str, Any]]]  # role -> [(name, Stats)], largest first
    previous_groups: Dict[str, Dict[str, Any]]
    gpu_util: Optional[float]
    cpu_util: Optional[float]
    # where averages and utilization are measured from: the window's start, or later
    # when the cluster's history begins inside it (earlier for all of history when
    # accounting began before the first record)
    covered: Optional[int] = None

    @property
    def since(self) -> int:
        return self.start if self.covered is None else self.covered

    def seconds(self, metric: str) -> Optional[List[int]]:
        return self.gpu if metric == "gpu" else self.cpu

    def average(self, metric: str, previous: bool = False) -> Optional[List[float]]:
        """Mean GPUs (or CPUs) held over each bin."""
        values = (self.prev_gpu if metric == "gpu" else self.prev_cpu) if previous \
            else self.seconds(metric)
        if values is None:
            return None
        edges = self.edges
        return [v / float(edges[i + 1] - edges[i]) if edges[i + 1] > edges[i] else 0.0
                for i, v in enumerate(values)]

    def mean(self, metric: str) -> Optional[float]:
        """Mean GPUs (or CPUs) held over the covered span: the same span utilization
        is measured over, so mean / capacity is the utilization."""
        total = self.total.amount(metric) if self.total is not None else None
        span = self.end - self.since
        return total / float(span) if total is not None and span > 0 else None

    def peak(self, metric: str) -> Tuple[Optional[int], Optional[float]]:
        values = self.average(metric)
        if not values:
            return None, None
        index = max(range(len(values)), key=lambda i: (values[i], i))
        return index, values[index]


# -- the adapter ---------------------------------------------------------------

def _efficiency(stats: Any, metric: str) -> Any:
    try:
        return stats.efficiency(metric)
    except (AttributeError, ValueError, KeyError):
        return None


def _spark(report: Any, range_name: str) -> Optional[List[float]]:
    try:
        unit = "hour" if range_name in ("day", "week") else "day"
        buckets = report.buckets("gpu", unit)
        if buckets is None:
            buckets = report.buckets("cpu", unit)
    except (ValueError, AttributeError, KeyError):
        return None
    return None if buckets is None else [b.value for b in buckets]


class HistoryData:
    """History for the UI: named ranges, background loading, derived views."""

    def __init__(self, history: Any, clock: Callable[[], float] = time.time,
                 full_history: bool = True) -> None:
        self.history = history
        self.clock = clock
        self.full_history = full_history  # load all of history in the background
        self.paused = False  # history switched off in the app: no load runs
        self._persistent = None  # type: Optional[bool]
        self.generation = 0  # bumps when more history has been loaded
        self.progress = ""  # what the loader is doing, for the header
        self.loaded = set()  # type: set
        self.failure = None  # type: Optional[Tuple[str, str]]
        self.fraction = None  # type: Optional[float]  # of the current load, for a bar
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._preempt = threading.Event()  # stops only the load in progress
        self._loading = None  # type: Optional[str]
        self._bumped = time.time()
        self.results = None  # type: Optional[ResultCache]  # answers kept across runs

    def has(self, method: str) -> bool:
        return callable(getattr(self.history, method, None))

    # -- loading -------------------------------------------------------------

    def info(self) -> Any:
        found = self.history.info()
        note = self.limit_note()
        if note and not any(getattr(n, "code", "") == "background_limit"
                            for n in getattr(found, "notes", None) or ()):
            try:
                from ..history import Note
                found.notes = list(found.notes or ()) + [Note("background_limit", note)]
            except (AttributeError, ImportError, TypeError):
                pass
        return found

    def persistent(self) -> bool:
        """Whether loaded history survives this run (a cache file, not memory)."""
        if self._persistent is None:
            try:
                info = self.history.info()
                known = getattr(info, "persistent", None)  # set even when the path is hidden
                self._persistent = bool(getattr(info, "cache_path", None)
                                        if known is None else known)
            except Exception:
                return True  # unknown yet; decided once history answers
        return self._persistent

    def auto_ranges(self) -> Tuple[str, ...]:
        """Ranges loaded in the background without being asked for. Without a cache file
        everything would be held in memory and fetched again next run: stop at a month."""
        names = RANGE_NAMES if self.full_history else RANGE_NAMES[:-1]
        if not self.persistent():
            names = tuple(name for name in names if name in MEMORY_RANGES)
        return names

    def limit_note(self) -> str:
        if self._persistent is False:
            return ("no history cache file: only up to a month is loaded in the background; "
                    "pick a longer range to load it")
        if not self.full_history:
            return "recent history only: all of history is loaded only when you pick it (H)"
        return ""

    # One bar per load: reading records fills most of it, summarizing the rest.
    _PHASES = {"rollup": (0.85, 1.0), "aggregate": (0.85, 1.0)}

    def _progress(self, event: Any) -> None:
        total = getattr(event, "total", None)
        done = getattr(event, "done", 0)
        phase = getattr(event, "phase", "loading")
        if phase == "efficiency":
            self.fraction = min(1.0, done / float(total)) if total else None
            self.progress = "efficiency %s" % (fmt.pct(self.fraction) if total
                                               else fmt.num(done))
            if time.time() - self._bumped > 1.0:  # show the pieces read so far
                self._bumped = time.time()
                with self._lock:
                    self.generation += 1
            return
        else:
            low, high = self._PHASES.get(phase, (0.0, 0.85))
            if total:
                at = low + (high - low) * min(1.0, done / float(total))
                # never backwards within one load, whichever phase reports
                self.fraction = max(self.fraction or 0.0, at)
            label = self._loading or "history"
            self.progress = "%s %s" % (label, fmt.pct(self.fraction)
                                       if self.fraction is not None else fmt.num(done))
        if time.time() - self._bumped > 20:  # show what is loaded so far, marked partial
            self._bumped = time.time()
            with self._lock:
                self.generation += 1

    def load(self, range_name: str, update: bool = True) -> bool:
        """Make sure ``range_name`` is in the cache. Blocking; call off the UI thread.

        Returns False if prioritize() stopped it for a shorter range; what it had
        already committed stays in the cache and the next load continues from there.
        """
        if self.paused:
            return False
        self.progress = "%s …" % range_name
        self.fraction = None
        self._preempt.clear()
        self._loading = range_name
        before = self._data_version()

        def cancelled() -> bool:
            return self._cancel.is_set() or self._preempt.is_set() or self.paused
        try:
            self.history.refresh(None if _SECONDS[range_name] is None
                                 else window(range_name, self.clock()),
                                 progress=self._progress, cancel=cancelled, update=update)
            if self.has("build_rollup"):
                self.history.build_rollup(progress=self._progress, cancel=cancelled)
        except Exception as error:  # HistoryError, Cancelled, or a bug: shown, not raised
            if (self._preempt.is_set() or self.paused) and not self._cancel.is_set():
                return False
            self.failure = (status_of(error), str(getattr(error, "message", error)))
            raise
        finally:
            self.progress, self.fraction = "", None
            self._loading = None
        self.failure = None
        with self._lock:
            self.loaded.add(range_name)
            # what is on screen is recomputed only when the cache really changed
            if before is None or self._data_version() != before:
                self.generation += 1
        return True

    def _data_version(self) -> Any:
        """The history cache's own version, or None when it cannot be told."""
        history = getattr(self.history, "_history", self.history)
        store = getattr(history, "_store", None)
        version = getattr(store, "version", None)
        return None if version is None else (id(store), version)

    def prioritize(self, range_name: str) -> bool:
        """The viewer switched to ``range_name``: if a longer range is loading, stop it so
        the requested one is loaded next. Ranges nest, so a longer one never needs this."""
        current = self._loading
        if (current is None or range_name in self.loaded or range_name not in _SECONDS
                or RANGE_NAMES.index(range_name) >= RANGE_NAMES.index(current)):
            return False
        self._preempt.set()
        return True

    def load_in_order(self, first: str) -> None:
        """The chosen range first, then the others smallest first (older data only)."""
        order = [first] + [name for name in self.auto_ranges() if name != first]
        for number, name in enumerate(order):
            if self._cancel.is_set():
                return
            if name in self.loaded:
                continue
            try:
                if not self.load(name, update=number == 0):
                    continue
            except Exception:
                if self.failure and self.failure[0] in (Status.UNAVAILABLE, Status.DENIED):
                    return
                continue

    def update(self) -> None:
        """New records since the last load."""
        self.load("day", update=True)

    def stop(self) -> None:
        self._cancel.set()

    def pause(self, paused: bool) -> None:
        """Stop (and keep stopped) or allow background loading; loaded data stays."""
        self.paused = paused
        if paused:
            self._preempt.set()

    def stop_current(self) -> None:
        """Stop the load in progress; the next one may start."""
        if self._loading is not None:
            self._preempt.set()

    def forget(self) -> None:
        """The cache was emptied: nothing counts as loaded any more."""
        with self._lock:
            self.loaded = set()
            self.failure = None
            self._persistent = None
            self.generation += 1

    # -- queries ---------------------------------------------------------------

    MEMO_SECONDS = 60.0  # a range's answer is reused this long while the data is unchanged

    def _memo(self, key: Any, compute: Callable[[], Any]) -> Any:
        """One answer per (question, data generation) for a short while: the same query
        asked by several panes, or again when only the live capacity changed, is not
        computed twice. Windows move with the clock, so answers expire."""
        memo = self.__dict__.setdefault("_memos", OrderedDict())
        now = time.time()
        with self._lock:
            found = memo.get(key)
            if found is not None and found[0] == self.generation and \
                    now - found[1] < self.MEMO_SECONDS:
                return found[2]
            generation = self.generation
        value = compute()
        with self._lock:
            memo[key] = (generation, now, value)
            memo.move_to_end(key)
            while len(memo) > 64:
                memo.popitem(last=False)
        return value

    def report(self, range_name: str, filt: Any = None, group_by: Sequence[Any] = (),
               series_by: Optional[str] = None, lanes_by: Optional[str] = None,
               lane_resolution: Optional[int] = None, efficiency: bool = True,
               win: Any = None, lanes_metric: Optional[str] = None) -> Any:
        extra = {"lanes_metric": lanes_metric} if lanes_metric else {}

        def compute() -> Any:
            return self.history.query(win or window(range_name, self.clock()), filt,
                                      group_by=group_by, series_by=series_by,
                                      lanes_by=lanes_by, lane_resolution=lane_resolution,
                                      efficiency=efficiency, cancel=self._cancel, **extra)
        if win is not None:
            return compute()
        return self._memo(("report", range_name, filt, tuple(group_by), series_by, lanes_by,
                           lane_resolution, efficiency, lanes_metric), compute)

    def previous(self, range_name: str, filt: Any = None,
                 group_by: Sequence[Any] = ()) -> Any:
        if previous_window(range_name, self.clock()) is None:
            return None

        def compute() -> Any:
            return self.history.query(previous_window(range_name, self.clock()), filt,
                                      group_by=group_by, efficiency=False,
                                      cancel=self._cancel)
        return self._memo(("previous", range_name, filt, tuple(group_by)), compute)

    def trend(self, range_name: str, filt: Any = None, gpu_capacity: Optional[int] = None,
              cpu_capacity: Optional[int] = None) -> TrendRow:
        report = self.report(range_name, filt)
        total = report.total
        before, previous_loaded = None, range_name == "all"
        try:
            prev = self.previous(range_name, filt)
            if prev is not None and prev.complete:
                before, previous_loaded = prev.total, True
        except HistoryError:
            before = None
        gpu_util = report.utilization(gpu_capacity, "gpu").fraction if gpu_capacity else None
        cpu_util = report.utilization(cpu_capacity, "cpu").fraction if cpu_capacity else None
        return TrendRow(range_name, total.gpu_hours, total.cpu_hours, gpu_util, cpu_util,
                        _efficiency(total, "gpu"), _efficiency(total, "cpu"),
                        None if before is None else before.gpu_hours,
                        None if before is None else before.cpu_hours,
                        _spark(report, range_name), bool(report.complete),
                        total.attempts, total.jobs, previous_loaded)

    def overview(self, gpu_capacity: Optional[int], cpu_capacity: Optional[int]) -> Any:
        """history.Overview from the rollup (cache only), or None if this history has none."""
        if not self.has("overview"):
            return None
        try:
            from ..history.rollup import with_capacity
        except ImportError:  # pragma: no cover
            return self.history.overview(gpu_capacity=gpu_capacity, cpu_capacity=cpu_capacity)
        # computed once per data generation; the live capacity only changes percentages.
        # The overview depends on the cache alone (not the clock): one kept by an earlier
        # run for the same data generation is the same answer.
        def compute() -> Any:
            results = self.results
            generation = None
            if results is not None and self.has("generation"):
                try:
                    generation = self.history.generation()
                except Exception:
                    generation = None
            key = ("h", "overview@", repr(generation)) if generation is not None else None
            if key is not None:
                kept = results.load(key)
                if kept is not None:
                    return kept
            found = self.history.overview()
            if key is not None:
                results.save(key, found)
            return found
        found = self._memo(("overview",), compute)
        return with_capacity(found, gpu_capacity, cpu_capacity)

    def overview_trend(self, summary: Any) -> TrendRow:
        """history.RangeSummary -> TrendRow, with the rollup's sparkline when it has one
        (None otherwise: see spark())."""
        total, previous = summary.total, summary.previous
        loaded = previous is not None and summary.complete
        spark = None  # type: Optional[List[float]]
        if callable(getattr(summary, "spark", None)):
            values = summary.spark()
            spark = [float(v) for v in values] if values else None
        return TrendRow(summary.name, total.gpu_hours, total.cpu_hours, summary.gpu_utilization,
                        summary.cpu_utilization, total.efficiency("gpu"), total.efficiency("cpu"),
                        previous.gpu_hours if loaded else None,
                        previous.cpu_hours if loaded else None, spark, bool(summary.complete),
                        total.attempts, total.attempts, loaded or summary.name == "all")

    def range_series(self, range_name: str, filt: Any = None, roles: Sequence[str] = (),
                     win: Any = None, gpu_capacity: Optional[int] = None,
                     cpu_capacity: Optional[int] = None) -> RangeSeries:
        """``range_name`` in detail: per-bin usage now and in the previous equal period,
        totals, utilization and the ``roles`` ranked (each against the previous period)."""
        groups = tuple((role,) for role in roles)
        report = self.report(range_name, filt, group_by=groups, win=win, efficiency=False)
        start, end = report.start, report.end
        # the span utilization counts (Report.utilization): from the capacity floor
        floor = getattr(report, "floor", None)
        covered = start if floor is None else max(start, floor)
        if getattr(report.window, "start", 0) is None and floor is not None:
            covered = min(start, floor)
        unit = chart_unit(range_name, start, end, covered)
        edges = bin_edges(start, end, unit)
        values = getattr(report, "values", None) or {}
        hours = getattr(report, "edges", None) or []

        def binned(source: Any, metric: str, shift: int = 0) -> Optional[List[int]]:
            found = (getattr(source, "values", None) or {}).get(metric)
            if found is None or source.total.amount(metric) is None:
                return None
            return rebin(source.edges, found, edges, shift)

        gpu, cpu = binned(report, "gpu"), binned(report, "cpu")
        previous = prev_gpu = prev_cpu = None
        previous_complete, previous_groups = False, {}  # type: bool, Dict[str, Dict[str, Any]]
        length = _SECONDS.get(range_name)
        if length is not None and hours:
            try:
                before = self.report(range_name, filt, group_by=groups, efficiency=False,
                                     win=Window(start - (end - start), start))
            except HistoryError:
                before = None
            if before is not None:
                previous, previous_complete = before.total, bool(before.complete)
                shift = end - start
                prev_gpu, prev_cpu = binned(before, "gpu", shift), binned(before, "cpu", shift)
                for role in roles:
                    previous_groups[role] = {key[0]: stats for key, stats
                                             in before.group(role).items()}
        ranked = {}  # type: Dict[str, List[Tuple[str, Any]]]
        for role in roles:
            ranked[role] = [(key[0], stats) for key, stats in report.ranked(role)]
        gpu_util = report.utilization(gpu_capacity, "gpu").fraction if gpu_capacity else None
        cpu_util = report.utilization(cpu_capacity, "cpu").fraction if cpu_capacity else None
        edges, (gpu, cpu, prev_gpu, prev_cpu) = _trim(edges, (gpu, cpu, prev_gpu, prev_cpu),
                                                      covered)
        return RangeSeries(range_name, start, end, unit, edges, gpu, cpu, prev_gpu, prev_cpu,
                           report.total, previous, bool(report.complete), previous_complete,
                           ranked, previous_groups, gpu_util, cpu_util, covered)

    def rollup_series(self, range_name: str, gpu_capacity: Optional[int] = None,
                      cpu_capacity: Optional[int] = None) -> Optional[RangeSeries]:
        """The whole cluster's range from the hourly rollup (the overview's own figures,
        in a fraction of a second where a full query of a busy year takes many); None
        when this history keeps no rollup."""
        if not self.has("overview"):
            return None
        length = _SECONDS[range_name]
        ranges = [(range_name, length)]
        if length is not None:
            ranges.append(("twice", 2 * length))  # its first half is the previous period
        found = self.history.overview(gpu_capacity=gpu_capacity, cpu_capacity=cpu_capacity,
                                      ranges=ranges)
        if found is None or not found.ranges:
            return None
        summary = found.ranges[0]
        twice = found.ranges[1] if len(found.ranges) > 1 else None
        start, end, step = summary.start, summary.end, summary.series_step
        count = len(summary.gpu_series)
        source = [end - (count - i) * step for i in range(count + 1)]
        if source:
            source[0] = max(source[0], min(start, end))
        covered = covered_start(summary.total, summary.gpu_utilization,
                                summary.cpu_utilization, gpu_capacity, cpu_capacity, end, start)
        unit = chart_unit(range_name, start, end, covered)
        if unit == "day" and step == DAY and len(source) > 1:
            edges = list(source)  # whole days as the rollup keeps them
        else:
            edges = bin_edges(start, end, unit)
        gpu = rebin(source, summary.gpu_series, edges)
        cpu = rebin(source, summary.cpu_series, edges)
        prev_gpu = prev_cpu = None
        previous_complete = False
        users = {}  # type: Dict[str, Any]
        for item in summary.accounts.values():
            for user, amounts in item.users.items():
                users.setdefault(user, _sum_amounts()).add(amounts)
        previous_groups = {"account": {name: item.previous for name, item
                                       in summary.accounts.items() if item.previous is not None},
                           "user": {}}  # type: Dict[str, Dict[str, Any]]
        if twice is not None and length is not None:
            previous_complete = bool(twice.complete)
            older = len(twice.gpu_series) - count
            if older > 0 and twice.series_step == step:
                back = [e - length for e in source]
                prev_gpu = rebin(back, twice.gpu_series[:older][-count:], edges, length)
                prev_cpu = rebin(back, twice.cpu_series[:older][-count:], edges, length)
            both = {}  # type: Dict[str, Any]
            for item in twice.accounts.values():
                for user, amounts in item.users.items():
                    both.setdefault(user, _sum_amounts()).add(amounts)
            for user, amounts in both.items():
                previous_groups["user"][user] = _less(amounts, users.get(user))
        ranked = {
            "account": sorted(((name, item.amounts) for name, item in summary.accounts.items()),
                              key=lambda p: (-p[1].gpu_seconds, -p[1].cpu_seconds, p[0])),
            "user": sorted(users.items(), key=lambda p: (-p[1].gpu_seconds,
                                                          -p[1].cpu_seconds, p[0])),
        }
        edges, (gpu, cpu, prev_gpu, prev_cpu) = _trim(edges, (gpu, cpu, prev_gpu, prev_cpu),
                                                      covered)
        return RangeSeries(range_name, start, end, unit, edges, gpu, cpu, prev_gpu, prev_cpu,
                           summary.total, summary.previous, bool(summary.complete),
                           previous_complete, ranked, previous_groups,
                           summary.gpu_utilization, summary.cpu_utilization, covered)

    def spark(self, range_name: str, filt: Any = None) -> Optional[List[float]]:
        report = self.report(range_name, filt, efficiency=False)
        return _spark(report, range_name)

    def kind_trends(self, range_name: str, filt: Any, kinds: Sequence[str]
                    ) -> Optional[Tuple[List[int], Dict[str, Dict[str, List[int]]]]]:
        """GPU-seconds per chart bin of the range for each kind of run ({kind: {key:
        bins}}) with the bins' edges; None when the history layer cannot tell."""
        if not self.has("kind_series"):
            return None
        return self.history.kind_series(
            window(range_name, self.clock()), filt, kinds=kinds, cancel=self._cancel,
            make_bins=lambda start, end: bin_edges(start, end, bin_unit(range_name, start, end)))

    def run_types(self, range_name: str, filt: Any, dimension: str) -> List[RunType]:
        """By the history layer when it can; else from the attempts list."""
        win = window(range_name, self.clock())
        if self.has("run_types"):
            found = self.history.run_types(win, filt, by=dimension, cancel=self._cancel)
            return sort_types([from_history_type(item, dimension) for item in found], dimension)
        page = self.attempts(range_name, filt)
        return run_types(page.rows, dimension)

    def events(self, range_name: str, filt: Any) -> List[Event]:
        if self.has("events"):
            win = window(range_name, self.clock())
            found = [e for e in self.history.events(win, filt, cancel=self._cancel)
                     if in_range(e, win)]
            return sorted((from_history_event(e) for e in found), key=lambda e: e.at or 0,
                          reverse=True)
        page = self.attempts(range_name, filt)
        report = self.report(range_name, filt, efficiency=False)
        try:
            previous = self.previous(range_name, filt, (("account",), ("qos",), ("partition",)))
        except HistoryError:
            previous = None
        return events(page.rows, range_name, self.clock(), report, previous)

    def capacity(self) -> List[Any]:
        return self.history.capacity() if self.has("capacity") else []

    def attempts(self, range_name: str, filt: Any = None, limit: int = ATTEMPT_LIMIT,
                 win: Any = None, sort: str = "start") -> Any:
        return self.history.jobs(win or window(range_name, self.clock()), filt, by="attempt",
                                 sort=sort, descending=True, limit=limit, cancel=self._cancel)

    def jobs(self, range_name: str, filt: Any = None, sort: str = "gpu", limit: int = 0,
             win: Any = None, run_type: Optional[Tuple[str, str]] = None,
             match: str = "") -> Any:
        """``match``: keep jobs whose id, name or user contains it (searched by history,
        not only in the rows already shown)."""
        limit = limit or JOB_PAGE
        extra = {"run_type": run_type} if run_type is not None else {}  # type: Dict[str, Any]
        if match:
            extra["match"] = match
        return self.history.jobs(win or window(range_name, self.clock()), filt, by="job",
                                 sort=sort, descending=True, limit=limit, cancel=self._cancel,
                                 **extra)

    def job_attempts(self, job_id: str) -> List[Any]:
        return self.history.attempts(job_id)

    def job_detail(self, job_id: str) -> Any:
        return self.history.detail(job_id, cancel=self._cancel)

    def submit_line(self, job_id: str) -> Any:
        return self.history.submit_line(job_id)

    def batch_script(self, job_id: str) -> Any:
        return self.history.batch_script(job_id)

    def load_efficiency(self, range_name: str, filt: Any = None) -> int:
        """Read step usage for the range, then bring the rollup up to date: the usage
        written marks its hours for rebuilding, and until they are rebuilt the overview
        and every rollup-backed figure would read as partly loaded."""
        self.progress = "efficiency %s" % range_name
        try:
            count = self.history.load_efficiency(window(range_name, self.clock()), filt,
                                                 progress=self._progress, cancel=self._cancel)
            if self.has("build_rollup"):  # a no-op when nothing changed
                self.fraction = None
                self.history.build_rollup(progress=self._progress, cancel=self._cancel)
        finally:
            self.progress, self.fraction = "", None
        with self._lock:
            self.generation += 1
        return count


class ResultCache:
    """What history views showed last, kept beside the history cache so that a new run
    (or a view opened again) paints at once and then refreshes in place.

    Only history answers ("h" loader keys) are kept, only while ``active()`` (never
    anonymized), only for a history with a cache file, and only for the cache the
    answers were computed from. Values are pickled into an owner-only file in the
    owner-only cache directory; anything unreadable is ignored."""

    VERSION = 1
    LIMIT = 600  # answers kept; the oldest go first

    def __init__(self, history: Any, active: Callable[[], bool] = lambda: True) -> None:
        self._history, self._active = history, active
        self._db = None  # type: Any
        self._failed = False
        self._lock = threading.Lock()
        self._last = None  # type: Optional[str]  # the cache the stored answers belong to
        self._saves = 0
        self._pending = {}  # type: Dict[str, Any]

    def _ident(self) -> Optional[str]:
        """The history cache file in use, once it is open (never probes Slurm)."""
        store = getattr(self._history, "_store", None)
        path = getattr(store, "path", None)
        if not path:
            return None
        from .. import __version__  # answers of another version may be shaped differently
        return "%s %s" % (os.path.basename(path), __version__)

    def _open(self) -> Any:
        if self._db is not None or self._failed:
            return self._db
        try:
            import sqlite3
            from ..history.store import default_cache_dir, private_dir
            if not getattr(self._history, "_persist", False):
                raise OSError("no cache file")
            directory = private_dir(getattr(self._history, "_cache_dir", None)
                                    or default_cache_dir())
            path = os.path.join(directory, "views-v%d.sqlite" % self.VERSION)
            handle = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                found = os.fstat(handle)
                if found.st_uid != os.geteuid() or found.st_mode & 0o077:
                    raise OSError("not private")
            finally:
                os.close(handle)
            db = sqlite3.connect(path, timeout=2, check_same_thread=False)
            for pragma in ("journal_mode=TRUNCATE", "synchronous=OFF"):
                db.execute("PRAGMA " + pragma).fetchall()
            db.execute("CREATE TABLE IF NOT EXISTS views(k TEXT PRIMARY KEY, ident TEXT, "
                       "at REAL, v BLOB)")
            db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
            db.commit()
            row = db.execute("SELECT v FROM meta WHERE k='ident'").fetchone()
            self._last = row[0] if row else None
            self._db = db
        except Exception:
            self._failed = True
        return self._db

    def get_meta(self, key: str) -> Any:
        with self._lock:
            db = self._open()
            try:
                row = db.execute("SELECT v FROM meta WHERE k=?", ("x." + key,)).fetchone() \
                    if db is not None else None
                return json.loads(row[0]) if row else None
            except Exception:
                return None

    def set_meta(self, key: str, value: Any) -> None:
        with self._lock:
            db = self._open()
            if db is None:
                return
            try:
                db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", ("x." + key,
                                                                       json.dumps(value)))
                db.commit()
            except Exception:
                pass

    @staticmethod
    def _text(key: Any) -> Optional[str]:
        if not isinstance(key, tuple) or not key or key[0] != "h":
            return None
        text = repr(key)
        return None if " at 0x" in text else text

    def load(self, key: Any) -> Any:
        text = self._text(key)
        if text is None or not self._active():
            return None
        with self._lock:
            db = self._open()
            if db is None:
                return None
            try:
                row = db.execute("SELECT ident,v FROM views WHERE k=?", (text,)).fetchone()
                if row is None or row[0] != (self._ident() or self._last):
                    return None
                import pickle
                import zlib
                return pickle.loads(zlib.decompress(row[1]))
            except Exception:
                return None

    def save(self, key: Any, value: Any) -> None:
        """Remembered now, written by flush() (when nothing on screen is waiting)."""
        text = self._text(key)
        if text is None or value is None or not self._active():
            return
        with self._lock:
            self._pending[text] = value

    @property
    def pending(self) -> int:
        return len(self._pending)

    def flush(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, {}
        for text, value in pending.items():
            self._write(text, value)

    def _write(self, text: str, value: Any) -> None:
        ident = self._ident()
        if ident is None or not self._active():
            return
        try:
            import pickle
            import zlib
            blob = zlib.compress(pickle.dumps(value, protocol=4), 1)
        except Exception:
            return
        with self._lock:
            db = self._open()
            if db is None:
                return
            try:
                db.execute("INSERT OR REPLACE INTO views VALUES(?,?,?,?)",
                           (text, ident, time.time(), blob))
                if ident != self._last:
                    db.execute("INSERT OR REPLACE INTO meta VALUES('ident',?)", (ident,))
                    self._last = ident
                self._saves += 1
                if self._saves % 100 == 1:
                    db.execute("DELETE FROM views WHERE k NOT IN (SELECT k FROM views ORDER BY "
                               "at DESC LIMIT ?)", (self.LIMIT,))
                db.commit()
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass


class NoHistory:
    """History switched off or unavailable: every call says so."""

    def __init__(self, status: str = Status.UNAVAILABLE, message: str = "history is off") -> None:
        self.status, self.message = status, message

    def _fail(self, *args: Any, **kwargs: Any) -> Any:
        raise HistoryError(self.status, self.message)

    info = refresh = query = jobs = attempts = detail = submit_line = batch_script = _fail
    load_efficiency = run_types = events = overview = build_rollup = _fail

    def capacity(self) -> List[Any]:
        return []


def visibility(info: Any, invoker: Any, account: str, my_accounts: Sequence[str]) -> str:
    """How much of an account's history the viewer can see."""
    if info is None:
        return fmt.NOT_YET
    accounting = getattr(info, "accounting", None)
    if accounting == Status.UNAVAILABLE:
        return "no accounting"
    if accounting == Status.DENIED:
        return "denied"
    seen = getattr(info, "visibility", "unknown")
    if seen == "all" or invoker.privileged:
        return "full"
    if account in (invoker.coordinator_accounts or ()):
        return "full"
    if seen == "restricted":
        return "own only" if account in my_accounts else "denied"
    return "own only" if account in my_accounts else "unknown"
