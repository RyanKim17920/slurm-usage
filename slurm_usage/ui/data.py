"""The only place the UI touches ``slurm_usage.history``.

``HistoryData`` wraps a ``history.History`` (or anything with the same methods,
which is how tests run without Slurm) and adds what screens need on top: named
ranges with their previous period, trend rows, run-type breakdowns and an
events feed. Everything here is synchronous; the UI calls it from a worker
thread (see ``loader``).

``NoHistory`` stands in when history is switched off or unavailable: every
query raises ``HistoryError`` with a status the UI turns into a placeholder.
"""

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
            self.generation += 1
        return True

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

    def report(self, range_name: str, filt: Any = None, group_by: Sequence[Any] = (),
               series_by: Optional[str] = None, lanes_by: Optional[str] = None,
               lane_resolution: Optional[int] = None, efficiency: bool = True,
               win: Any = None, lanes_metric: Optional[str] = None) -> Any:
        extra = {"lanes_metric": lanes_metric} if lanes_metric else {}
        return self.history.query(win or window(range_name, self.clock()), filt,
                                  group_by=group_by, series_by=series_by, lanes_by=lanes_by,
                                  lane_resolution=lane_resolution, efficiency=efficiency,
                                  cancel=self._cancel, **extra)

    def previous(self, range_name: str, filt: Any = None,
                 group_by: Sequence[Any] = ()) -> Any:
        win = previous_window(range_name, self.clock())
        if win is None:
            return None
        return self.history.query(win, filt, group_by=group_by, efficiency=False,
                                  cancel=self._cancel)

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
        return self.history.overview(gpu_capacity=gpu_capacity, cpu_capacity=cpu_capacity)

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

    def spark(self, range_name: str, filt: Any = None) -> Optional[List[float]]:
        report = self.report(range_name, filt, efficiency=False)
        return _spark(report, range_name)

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
