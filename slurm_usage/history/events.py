"""Notable, data-derived events for a window: what stands out against the same
selection's own earlier history. No site constants: every threshold comes from
that history.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .errors import Cancelled, canceller
from .records import OUTCOMES, outcome_of
from .select import Filter, Selection, resolve, unresolved
from .store import OPEN_STOP
from .timeline import DAY, HOUR, Window, local_date, local_midnight

KINDS = ("largest_allocation", "longest_run", "preemption_run", "failure_burst", "idle_gap",
         "first_account", "first_qos", "record_day", "change", "no_baseline")
_FAILED = frozenset(("failed", "node_fail"))
_MIN_GAP = 6 * HOUR  # below this a pause is not news anywhere
PER_KIND = 10  # the biggest of each kind; the rest are counted in values["others"]


@dataclass
class Event:
    kind: str
    time: int  # when it happened (start of the span)
    end: Optional[int]
    sentence: str  # plain, already safe to display
    values: Dict[str, Any] = field(default_factory=dict)
    where: Filter = field(default_factory=Filter)  # to drill into
    window: Optional[Window] = None
    job_id: Optional[str] = None


def _hours(seconds: float) -> str:
    hours = seconds / 3600.0
    if hours >= 48:
        return "%.1f days" % (hours / 24)
    return "%.1f h" % hours if hours < 10 else "%.0f h" % hours


def _when(epoch: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch))


def _quantile(values: List[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def prior_loaded(covered_from: Optional[int], start: int, length: int) -> bool:
    """Whether loaded history before ``start`` can serve as a baseline: all of it, or at
    least ``length`` seconds of it. Ingest always reads a little before a window; that
    overlap is not history to compare against."""
    if covered_from is None:
        return False
    return covered_from == 0 or covered_from <= start - max(length, 1)


NO_BASELINE = ("No earlier history loaded: records, firsts and the change against the "
               "previous period need it")


def find_events(store: Any, as_of: int, window: Optional[Window], where: Optional[Filter],
                covered_from: Optional[int], cancel: Any = None,
                limit: int = 200) -> List[Event]:
    """Chronological. Baselines use everything loaded before the window."""
    if getattr(store, "has_agg", False):
        return _find_events_sql(store, as_of, window, where, covered_from, cancel, limit)
    return find_events_scan(store, as_of, window, where, covered_from, cancel, limit)


def find_events_scan(store: Any, as_of: int, window: Optional[Window], where: Optional[Filter],
                     covered_from: Optional[int], cancel: Any = None,
                     limit: int = 200) -> List[Event]:
    """find_events() by reading every row up to the window's end, in Python."""
    window, where, cancelled = window or Window(), where or Filter(), canceller(cancel)
    events = []  # type: List[Event]
    with store.reading():
        dims = store.dims()
        chosen = Selection(dims, {}, where)
        first = store.first_start(chosen.dim_ids)
        span = resolve(window, as_of, first)
        low, high = span.start, span.end
        history = Window(None, high)
        whole = resolve(history, as_of, first)
        # Baselines from before the window
        prior_max_gpus = prior_max_cpus = prior_max_run = 0
        prior_gaps = []  # type: List[int]
        seen_accounts, seen_qos = set(), set()  # type: set, set
        prior_fail_days = {}  # type: Dict[Any, int]
        cover = None  # type: Optional[int]
        # In the window
        top = None  # type: Optional[tuple]
        longest = None  # type: Optional[tuple]
        gaps = []  # type: List[Tuple[int, int]]
        failures = {}  # type: Dict[Tuple[Any, str], List[tuple]]
        chains = {}  # type: Dict[Tuple[int, Any], List[tuple]]
        firsts = []  # type: List[Event]
        has_prior = prior_loaded(covered_from, low, high - low)

        rows = store.scan(None, high, chosen.dim_ids, extra=True)
        try:
            for number, part in enumerate(chosen.parts(rows, whole)):
                if not number % 1024 and cancelled():
                    raise Cancelled()
                if part.problem:
                    continue
                row, roles = part.row, part.roles
                begun, ended = row[6], row[7]
                runtime = (part.real_stop - begun - row[8])
                inside = begun >= low
                gpus, cpus = row[10] or 0, row[9] or 0
                if cover is not None and begun > cover:
                    gap = (cover, begun)
                    if gap[1] <= low:
                        prior_gaps.append(gap[1] - gap[0])
                    elif gap[1] > low:
                        gaps.append((max(gap[0], low), gap[1]))
                cover = max(cover or 0, part.real_stop)
                if part.outcome in ("preempted", "requeued") or (row[1], roles.uid) in chains:
                    chains.setdefault((row[1], roles.uid), []).append(
                        (begun, part.real_stop, part.outcome, gpus * max(0, runtime)))
                if not inside:
                    prior_max_gpus = max(prior_max_gpus, gpus)
                    prior_max_cpus = max(prior_max_cpus, cpus)
                    if ended is not None:
                        prior_max_run = max(prior_max_run, runtime)
                    seen_accounts.add(roles.account)
                    seen_qos.add(roles.qos)
                    if part.outcome in _FAILED:
                        day = local_date(ended or begun)
                        prior_fail_days[day] = prior_fail_days.get(day, 0) + 1
                    continue
                label = str(row[1])
                size = (gpus, cpus)
                if top is None or size > top[0]:
                    top = (size, begun, label, roles)
                if ended is not None and (longest is None or runtime > longest[0]):
                    longest = (runtime, begun, ended, label, gpus)
                if has_prior and roles.account not in seen_accounts:
                    seen_accounts.add(roles.account)
                    firsts.append(Event("first_account", begun, None,
                                        "First use of account %s" % roles.account,
                                        {"account": roles.account},
                                        where.replace(account=roles.account),
                                        Window(begun, high), label))
                if has_prior and roles.qos not in seen_qos:
                    seen_qos.add(roles.qos)
                    firsts.append(Event("first_qos", begun, None,
                                        "First use of QOS %s" % roles.qos, {"qos": roles.qos},
                                        where.replace(qos=roles.qos), Window(begun, high), label))
                if part.outcome in _FAILED:
                    key = (local_date(ended or begun), row[19] or "?")
                    failures.setdefault(key, []).append((begun, ended or begun, label))
        finally:
            rows.close()
        if cover is not None and cover < high and high - cover > 0:
            gaps.append((max(cover, low), high))

    return _conclude(where, low, high, has_prior, top, longest, prior_max_gpus,
                     prior_max_cpus, prior_max_run, chains, prior_fail_days, failures,
                     prior_gaps, gaps, firsts, limit)


def _conclude(where: Filter, low: int, high: int, has_prior: bool, top: Any, longest: Any,
              prior_max_gpus: int, prior_max_cpus: int, prior_max_run: int,
              chains: Dict[Tuple[int, Any], List[tuple]], prior_fail_days: Dict[Any, int],
              failures: Dict[Tuple[Any, str], List[tuple]], prior_gaps: List[int],
              gaps: List[Tuple[int, int]], firsts: List[Event], limit: int) -> List[Event]:
    events = []  # type: List[Event]
    if top is not None and (top[0][0] or top[0][1]):
        (gpus, cpus), begun, label, roles = top
        record = has_prior and (gpus > prior_max_gpus if gpus else cpus > prior_max_cpus)
        what = ("%d GPU%s" % (gpus, "" if gpus == 1 else "s") if gpus
                else "%d CPU%s" % (cpus, "" if cpus == 1 else "s"))
        events.append(Event("largest_allocation", begun, None,
                            "Largest allocation: %s (job %s)%s" % (
                                what, label, ", the most ever" if record else ""),
                            {"gpus": gpus, "cpus": cpus, "record": record},
                            where.replace(user=roles.user), Window(begun, high), label))
    if longest is not None and longest[0] > 0:
        runtime, begun, ended, label, gpus = longest
        record = has_prior and runtime > prior_max_run
        events.append(Event("longest_run", begun, ended, "Longest run: %s (job %s)%s" % (
            _hours(runtime), label, ", the longest ever" if record else ""),
            {"seconds": runtime, "gpus": gpus, "record": record}, where, Window(begun, ended),
            label))

    # A preemption run is news only when it is longer than this selection's usual ones.
    prior_runs = [sum(1 for c in chain if c[2] in ("preempted", "requeued"))
                  for chain in chains.values() if chain[0][0] < low]
    usual_run = _quantile(prior_runs, 0.9) or 1
    # Counted and dated by what happened inside the window: a long chain that began
    # weeks ago shows only its preemptions here, from where the window starts.
    runs = []
    for (job, uid), chain in chains.items():
        if chain[-1][1] < low or chain[0][0] >= high:
            continue
        preempted = [c for c in chain if c[2] in ("preempted", "requeued")
                     and c[1] >= low and c[0] < high]
        if len(preempted) < 2 or len(preempted) <= usual_run:
            continue
        runs.append((job, chain, preempted))
    runs.sort(key=lambda item: -sum(c[3] for c in item[2]))
    for job, chain, preempted in runs[:PER_KIND]:
        finished = chain[-1][2] if chain[-1][2] not in ("preempted", "requeued") else None
        every = sum(1 for c in chain if c[2] in ("preempted", "requeued"))
        begun = max(low, preempted[0][0])
        events.append(Event("preemption_run", begun, chain[-1][1],
                            "Job %d was preempted or requeued %d times%s%s" % (
                                job, len(preempted),
                                " (%d since it was submitted)" % every
                                if every > len(preempted) else "",
                                ", then %s" % finished if finished else ", still unfinished"),
                            {"attempts": len(chain), "preempted": len(preempted),
                             "preempted_all": every,
                             "gpu_seconds": sum(c[3] for c in preempted),
                             "usual": usual_run, "others": max(0, len(runs) - PER_KIND)},
                            where, Window(begun, chain[-1][1]), str(job)))

    daily = list(prior_fail_days.values())
    usual = max(2.0, 2 * (_quantile(daily, 0.9) or 0))
    bursts = sorted(failures.items(), key=lambda item: -len(item[1]))[:PER_KIND]
    for (day, code), items in bursts:
        if len(items) < usual:
            continue
        start = min(i[0] for i in items)
        end = max(i[1] for i in items)
        events.append(Event("failure_burst", start, end,
                            "%d jobs failed with exit code %s on %s (usually at most %d a day)"
                            % (len(items), code, day.isoformat(), int(usual / 2)),
                            {"count": len(items), "exit_code": code, "usual": usual / 2},
                            where.replace(outcome=["failed", "node_fail"]),
                            Window(local_midnight(day), local_midnight(day) + DAY)))

    long_gap = max(_MIN_GAP, _quantile(prior_gaps, 0.9) or 0)
    for start, end in sorted(gaps, key=lambda g: g[0] - g[1])[:PER_KIND]:
        if end - start >= long_gap:
            events.append(Event("idle_gap", start, end, "No allocations for %s" % _hours(
                end - start), {"seconds": end - start, "usual_max": long_gap}, where,
                Window(start, end)))
    events.extend(firsts)
    events.sort(key=lambda e: (e.time, KINDS.index(e.kind)))
    return events[:limit]




def _find_events_sql(store: Any, as_of: int, window: Optional[Window], where: Optional[Filter],
                     covered_from: Optional[int], cancel: Any = None,
                     limit: int = 200) -> List[Event]:
    """find_events_scan()'s results with the baselines from before the window computed by
    SQLite: only the rows that started inside the window are read here."""
    window, where, cancelled = window or Window(), where or Filter(), canceller(cancel)
    with store.reading():
        dims = store.dims()
        chosen = Selection(dims, {}, where)
        first = store.first_start(chosen.dim_ids)
        span = resolve(window, as_of, first)
        low, high = span.start, span.end
        whole = resolve(Window(None, high), as_of, first)
        has_prior = prior_loaded(covered_from, low, high - low)
        picked = set(dims) if chosen.dim_ids is None else set(chosen.dim_ids) & set(dims)
        outcomes = where.outcome
        finished = {d for d in picked
                    if outcomes is None or outcome_of(dims[d].state, True) in outcomes}
        running = {d for d in picked
                   if outcomes is None or outcome_of(dims[d].state, False) in outcomes}
        skip = [r[0] for r in store.open_rows()
                if r[2] in dims and unresolved(dims[r[2]].state, None, r[8])]
        everything = chosen.dim_ids is None and outcomes is None
        rows = store.event_rows(high, whole.end, as_of, None if everything else finished,
                                None if everything else running, skip)
        failed = {d for d in finished if outcome_of(dims[d].state, True) in _FAILED}
        chained = {d for d in finished
                   if outcome_of(dims[d].state, True) in ("preempted", "requeued")}
        prior, ends, found_gaps, cover, chain_rows = store.event_prior(rows, low, failed,
                                                                       chained)

        prior_max_gpus = max([r[1] or 0 for r in prior] or [0])
        prior_max_cpus = max([r[2] or 0 for r in prior] or [0])
        prior_max_run = max([r[3] for r in prior if r[3] is not None] or [0])
        seen_accounts = {dims[r[0]].account for r in prior}
        seen_qos = {dims[r[0]].qos for r in prior}
        prior_fail_days = {}  # type: Dict[Any, int]
        for ended in ends:
            day = local_date(ended)
            prior_fail_days[day] = prior_fail_days.get(day, 0) + 1
        prior_gaps = []  # type: List[int]
        gaps = []  # type: List[Tuple[int, int]]
        for begun, before in found_gaps:
            if begun <= low:
                prior_gaps.append(begun - before)
            else:
                gaps.append((max(before, low), begun))
        if cover is not None and cover < high and high - cover > 0:
            gaps.append((max(cover, low), high))
        chains = {}  # type: Dict[Tuple[int, Any], List[tuple]]
        for begun, real_stop, job, dim, gpus, suspended, stop in chain_rows:
            roles = dims[dim]
            outcome = outcome_of(roles.state, stop < OPEN_STOP)
            if outcome in ("preempted", "requeued") or (job, roles.uid) in chains:
                chains.setdefault((job, roles.uid), []).append(
                    (begun, real_stop, outcome, gpus * max(0, real_stop - begun - suspended)))

        if cancelled():
            raise Cancelled()
        # inside the window: the largest, the longest, firsts and failures, from SQLite
        biggest, long_row, first_rows, failed_rows = store.event_inside(rows, low, failed)
        top = longest = None  # type: Optional[tuple]
        if biggest is not None:
            gpus, cpus, begun, job, dim = biggest
            top = ((gpus, cpus), begun, str(job), dims[dim])
        if long_row is not None:
            runtime, begun, ended, job, gpus = long_row
            longest = (runtime, begun, ended, str(job), gpus)
        firsts = []  # type: List[Event]
        if has_prior:
            for dim, begun, job in sorted(first_rows, key=lambda r: r[1]):
                roles = dims[dim]
                label = str(job)
                if roles.account not in seen_accounts:
                    seen_accounts.add(roles.account)
                    firsts.append(Event("first_account", begun, None,
                                        "First use of account %s" % roles.account,
                                        {"account": roles.account},
                                        where.replace(account=roles.account),
                                        Window(begun, high), label))
                if roles.qos not in seen_qos:
                    seen_qos.add(roles.qos)
                    firsts.append(Event("first_qos", begun, None,
                                        "First use of QOS %s" % roles.qos, {"qos": roles.qos},
                                        where.replace(qos=roles.qos), Window(begun, high), label))
        failures = {}  # type: Dict[Tuple[Any, str], List[tuple]]
        for begun, ended, job, code in failed_rows:
            key = (local_date(ended or begun), code or "?")
            failures.setdefault(key, []).append((begun, ended or begun, str(job)))
    return _conclude(where, low, high, has_prior, top, longest, prior_max_gpus,
                     prior_max_cpus, prior_max_run, chains, prior_fail_days, failures,
                     prior_gaps, gaps, firsts, limit)


def add_period_events(events: List[Event], days: List[Tuple[int, int, int]],
                      prior_best: Optional[int], total: Optional[int], previous: Optional[int],
                      window: Window, where: Filter, low: int, high: int) -> List[Event]:
    """Record days and the change against the previous equal period. ``prior_best`` and
    ``previous`` are None when that history is not fully loaded."""
    if prior_best is None and previous is None and low > 0:
        events.append(Event("no_baseline", high, None, NO_BASELINE, {}, where, window))
    best = prior_best or 0
    for start, end, value in days:
        if prior_best is not None and value > best and value > 0:
            events.append(Event("record_day", start, end, "Record day: %s GPU-h" % (
                "%.0f" % (value / 3600.0)), {"gpu_seconds": value, "previous_best": best},
                where, Window(start, end)))
            best = value
    if total is not None and previous:
        change = total / previous - 1.0
        events.append(Event("change", high, None, "%s%.0f%% GPU-hours compared with the "
                                                   "previous %s" % (
                                                       "+" if change >= 0 else "", 100 * change,
                                                       _hours(high - low)),
                            {"gpu_seconds": total, "previous": previous, "change": change},
                            where, window))
    events.sort(key=lambda e: (e.time, KINDS.index(e.kind)))
    return events
