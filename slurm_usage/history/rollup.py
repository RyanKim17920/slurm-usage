"""An hourly usage rollup by role, kept in the cache, for the landing screen.

Each range also carries its usage over time (hourly bins for day and week, daily
bins otherwise) read from the same rollup, so every trend sparkline is instant.

Hours are epoch hours (multiples of 3600 s), so the ranges here are rolling
(the last 24 hours, 7 days, ...) and immune to DST. Only hours before the last
refresh are stored, so a stored hour never changes unless the data under it
does; the store records which span changed, and only that span is rebuilt,
newest first, in chunks that become visible as they finish.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .aggregate import start_counts, usage_ratios
from .errors import Cancelled, Progress, ProgressFn, canceller
from .families import name_family
from .select import Filter, Selection, Span, effective
from .store import AGG_VERSION, FAMILY_VERSION, OPEN_STOP
from .timeline import DAY, HOUR

CHUNK = 7 * DAY
RANGES = (("day", DAY), ("week", 7 * DAY), ("month", 30 * DAY), ("year", 365 * DAY),
          ("all", None))
_GIB = 1024 ** 3


def floor_hour(when: int) -> int:
    return when - when % HOUR


def capacity_floor(store: Any, covered_from: Optional[int]) -> Optional[int]:
    """When the cluster's usage can begin, for every utilization denominator: the
    accounting origin if known, else the first record once all history is loaded, else
    where loaded history begins. One rule for every range, so a range and a longer one
    are always comparable."""
    origin = store.get_meta("origin")
    if origin:
        return int(origin)
    if covered_from == 0:
        return store.first_start()
    return covered_from


def util_span(start: int, end: int, floor: Optional[int]) -> int:
    return end - max(start, floor if floor is not None else start)


class Amounts:
    """Additive totals. Seconds of resource; efficiency sums weighted by allocation."""

    __slots__ = ("gpu_seconds", "cpu_seconds", "mem_byte_seconds", "attempts", "cpu_covered",
                 "cpu_used", "mem_covered", "mem_used", "gpu_covered", "gpu_used")

    def __init__(self, values: Optional[tuple] = None) -> None:
        values = values or (0,) * 10
        for name, value in zip(self.__slots__, values):
            setattr(self, name, value or 0)

    def add(self, other: "Amounts") -> None:
        for name in self.__slots__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @property
    def gpu_hours(self) -> float:
        return self.gpu_seconds / 3600.0

    @property
    def cpu_hours(self) -> float:
        return self.cpu_seconds / 3600.0

    @property
    def mem_gib_hours(self) -> float:
        return self.mem_byte_seconds / (3600.0 * _GIB)

    def amount(self, metric: str) -> int:
        return {"gpu": self.gpu_seconds, "cpu": self.cpu_seconds,
                "mem": self.mem_byte_seconds}[metric]

    def efficiency(self, metric: str) -> Tuple[Optional[float], Optional[float]]:
        """(value, coverage) as in Stats.efficiency."""
        covered, used = getattr(self, metric + "_covered"), getattr(self, metric + "_used")
        total = self.amount(metric)
        return (used / covered if covered else None, covered / total if total else None)

    def change(self, previous: Optional["Amounts"], metric: str = "gpu") -> Optional[float]:
        """Relative change from the previous period; None if it had nothing."""
        if previous is None or not previous.amount(metric):
            return None
        return self.amount(metric) / previous.amount(metric) - 1.0


@dataclass
class AccountSummary:
    account: str
    amounts: Amounts
    previous: Optional[Amounts]
    users: Dict[str, Amounts]
    share: Optional[float]  # of the range's total, by the overview's metric
    first: Optional[int]  # first and last hour with usage in all loaded history
    last: Optional[int]
    running_gpus: Optional[int]  # held at the last refresh
    running_cpus: Optional[int]
    attempts_on_record: int  # rows the viewer can see, all time


@dataclass
class RangeSummary:
    name: str
    start: int
    end: int  # the last full hour before the last refresh
    total: Amounts
    previous: Optional[Amounts]  # the equal period before; None for "all"
    gpu_utilization: Optional[float]  # of capacity passed in, which is today's
    cpu_utilization: Optional[float]
    previous_gpu_utilization: Optional[float]
    accounts: Dict[str, AccountSummary]
    complete: bool  # False while part of it is unloaded or still being rolled up
    label: str = "percentages assume today's capacity for the whole range"
    # usage over the range, oldest bin first, bins of series_step seconds ending at ``end``
    # (hours for day and week, days otherwise); gpu- and cpu-seconds per bin
    series_step: int = HOUR
    gpu_series: List[int] = field(default_factory=list)
    cpu_series: List[int] = field(default_factory=list)
    # the seconds utilization is measured over, this range's and the previous period's
    span: int = 0
    previous_span: int = 0

    def spark(self) -> List[int]:
        """GPU usage per bin, or CPU usage on a cluster without GPU usage."""
        return self.gpu_series if self.total.gpu_seconds else self.cpu_series


@dataclass
class Overview:
    as_of: int
    covered_from: Optional[int]  # 0 = all history
    earliest: Optional[int]  # the oldest hour with any usage visible to you
    visibility: str
    ranges: List[RangeSummary] = field(default_factory=list)
    pending_rollup: Optional[Tuple[int, int]] = None  # span not yet rolled up
    notes: List[Any] = field(default_factory=list)


def build(store: Any, as_of: int, progress: ProgressFn = None, cancel: Any = None) -> int:
    """Bring the rollup up to date, newest first. Returns hours rebuilt."""
    cancelled = canceller(cancel)
    settled = floor_hour(as_of)
    agg = getattr(store, "has_agg", False)
    with store.lock:
        last = store.get_meta("rollup_until")
        dirty = store.get_meta("dirty")
        families = agg and getattr(store, "has_families", False)
        if agg and (store.get_meta("agg") != AGG_VERSION or
                    families and store.get_meta("agg_fam") != FAMILY_VERSION):
            last = None  # a cache from before these aggregates: build everything once
            store.agg_reset()
            store.set_meta("agg", AGG_VERSION)
            if families:
                store.set_meta("agg_fam", FAMILY_VERSION)
        if last is None:
            first = store.first_start()
            dirty = None if first is None else [first, settled]
        elif settled > last:
            dirty = [min(last, dirty[0]) if dirty else last, settled]
        if not dirty:
            store.set_meta("rollup_until", settled)
            return 0
        low = floor_hour(max(0, dirty[0]))
        high = min(settled, floor_hour(min(dirty[1], OPEN_STOP)) + HOUR)
        store.set_meta("rollup_until", settled)
        if high <= low:
            store.rollup_replace(0, 0, (), None)
            return 0
        store.rollup_replace(0, 0, (), [low, high])

    dims, done, top = store.dims(), 0, high
    chosen = Selection(dims, {}, Filter())
    while high > low:
        if cancelled():
            raise Cancelled()
        begin = max(low, high - CHUNK)
        rows, extra = _chunk(store, chosen, begin, high, as_of, agg)
        store.rollup_replace(begin, high, rows, None if begin <= low else [low, begin], extra)
        done += (high - begin) // HOUR
        high = begin
        if progress:
            progress(Progress("rollup", top - high, top - low))
    return done


def _chunk(store: Any, chosen: Selection, low: int, high: int, as_of: int,
           agg: bool = False) -> Tuple[List[tuple], Optional[tuple]]:
    span = Span(low, high, low, high, as_of)
    sums = {}  # type: Dict[Tuple[int, int], list]
    exact = {}  # type: Dict[Tuple[int, int], list]  # finished allocations only
    starts = {}  # type: Dict[Tuple[int, int], list]
    families = getattr(store, "has_families", False)  # the aggregates by job-name family
    rows = store.scan(low, high, usage=True, ordered=False,
                      by_start=store.cheaper_by_start(low, high), extra=families)
    try:
        for part in chosen.parts(rows, span):
            if part.problem or part.stop <= part.start and not part.starts_here:
                continue
            row = part.row
            dim, begun, suspended = row[2], row[6], row[8]
            gpus, cpus, mem = part.gpus or 0, part.cpus or 0, part.mem or 0
            ratios = usage_ratios(row) if row[7] is not None else (None, None, None)
            # what the exact aggregates hold: allocations that ended, and not before they began
            finished = agg and row[7] is not None and row[7] >= begun
            duration = part.real_stop - begun
            held = min(suspended, duration)
            hour, p = floor_hour(part.start), part.start
            size = -1 if row[10] is None else row[10]  # GPUs per allocation, -1 unknown
            family = name_family(row[18] or "") if families else None
            if part.starts_here:
                key = (floor_hour(begun), dim)
                sums.setdefault(key, [0] * 10)[3] += 1
                if finished:
                    counts = start_counts(row, part.outcome, ratios)
                    key = key + (size, family)
                    found = starts.get(key)
                    if found is None:
                        starts[key] = list(counts)
                    else:
                        for at, value in enumerate(counts):
                            if at == 7:
                                if value is not None and (found[7] is None or value > found[7]):
                                    found[7] = value
                            else:
                                found[at] += value
            while p < part.stop:
                q = min(part.stop, hour + HOUR)
                seconds = (effective(q - begun, duration, held) - effective(p - begun, duration,
                                                                             held)
                           if held else q - p)
                if finished:
                    e = exact.get((hour, dim, size, family))
                    if e is None:
                        e = exact[(hour, dim, size, family)] = (
                            [0, 0, 0, 0, 0, 0.0, 0, 0.0, 0, 0.0] + [None] * 8)
                    for at, held_now in ((10, True), (12, gpus), (14, cpus), (16, mem)):
                        if held_now:
                            if e[at] is None or p < e[at]:
                                e[at] = p
                            if e[at + 1] is None or q > e[at + 1]:
                                e[at + 1] = q
                if seconds:
                    entry = sums.setdefault((hour, dim), [0] * 10)
                    entry[0] += gpus * seconds
                    entry[1] += cpus * seconds
                    entry[2] += mem * seconds
                    cpu, rss, gpu = ratios
                    if cpu is not None and cpus:
                        entry[4] += cpus * seconds
                        entry[5] += cpu * cpus * seconds
                    if rss is not None and mem:
                        entry[6] += mem * seconds
                        entry[7] += rss * mem * seconds
                    if gpu is not None and gpus:
                        entry[8] += gpus * seconds
                        entry[9] += min(gpu, 1.0) * gpus * seconds
                    if finished:
                        e[0] += gpus * seconds
                        e[1] += cpus * seconds
                        e[2] += mem * seconds
                        e[3] += seconds
                        if cpu is not None and cpus:
                            e[4] += cpus * seconds
                            e[5] += cpu * cpus * seconds
                        if rss is not None and mem:
                            e[6] += mem * seconds
                            e[7] += rss * mem * seconds
                        if gpu is not None and gpus:
                            e[8] += gpus * seconds
                            e[9] += min(gpu, 1.0) * gpus * seconds
                p, hour = q, hour + HOUR
    finally:
        rows.close()
    out = [(hour, dim) + tuple(values) for (hour, dim), values in sums.items()]
    if not agg:
        return out, None
    # kept by (hour, dim, GPUs per allocation) and, apart, by (hour, dim, family)
    hours, by_family = {}, {}  # type: Dict[tuple, list], Dict[tuple, list]
    for (hour, dim, size, family), values in exact.items():
        found = hours.get((hour, dim, size))
        if found is None:
            hours[(hour, dim, size)] = list(values)
        else:
            for at in range(10):
                found[at] += values[at]
            for at in range(10, 18, 2):  # first and last use
                if values[at] is not None and (found[at] is None or values[at] < found[at]):
                    found[at] = values[at]
                if values[at + 1] is not None and (found[at + 1] is None
                                                   or values[at + 1] > found[at + 1]):
                    found[at + 1] = values[at + 1]
        if families:
            found = by_family.get((hour, dim, family))
            if found is None:
                by_family[(hour, dim, family)] = values[:10]
            else:
                for at in range(10):
                    found[at] += values[at]
    begun, begun_family = {}, {}  # type: Dict[tuple, list], Dict[tuple, list]
    for (hour, dim, size, family), counts in starts.items():
        for into, key in ((begun, (hour, dim, size)), (begun_family, (hour, dim, family))):
            if into is begun_family and not families:
                continue
            found = into.get(key)
            if found is None:
                into[key] = list(counts)
                continue
            for at, value in enumerate(counts):
                if at == 7:
                    if value is not None and (found[7] is None or value > found[7]):
                        found[7] = value
                else:
                    found[at] += value
    ids = store.family_ids({key[2] for key in by_family} | {key[2] for key in begun_family}) \
        if families else {}
    return out, ([key + tuple(values) for key, values in hours.items()],
                 [key + tuple(values) for key, values in begun.items()],
                 [key[:2] + (ids[key[2]],) + tuple(values) for key, values in by_family.items()],
                 [key[:2] + (ids[key[2]],) + tuple(values)
                  for key, values in begun_family.items()])


def _sum(store: Any, low: int, high: int, dims: Dict[int, Any]
         ) -> Tuple[Amounts, Dict[str, Amounts], Dict[Tuple[str, str], Amounts]]:
    total, accounts, users = Amounts(), {}, {}  # type: Amounts, Dict[str, Amounts], Dict
    for row in store.rollup_sum(low, high):
        roles = dims.get(row[0])
        if roles is None:
            continue
        amounts = Amounts(row[1:])
        total.add(amounts)
        accounts.setdefault(roles.account, Amounts()).add(amounts)
        users.setdefault((roles.account, roles.user), Amounts()).add(amounts)
    return total, accounts, users


def _segments(store: Any, end: int, ranges: Any, dims: Dict[int, Any]) -> Tuple[Any, list]:
    """(_sum() for the spans the ranges need, (dim, first hour, last hour) with usage),
    from one grouped read of the rollup."""
    if not hasattr(store, "rollup_segments"):
        return (lambda low, high: _sum(store, low, high, dims)), store.rollup_activity()
    bounds = {end - length for _, length in ranges if length is not None}
    bounds |= {end - 2 * length for _, length in ranges if length is not None}
    bounds.add(end)
    order = sorted(bounds, reverse=True)  # order[0] == end: segment 0 is empty
    width = 11
    per_dim = {}  # type: Dict[int, List[list]]
    spans = {}  # type: Dict[int, List[int]]
    for row in store.rollup_segments(order, end):
        dim, segment = row[0], row[1]
        found = per_dim.get(dim)
        if found is None:
            found = per_dim[dim] = [[0] * width for _ in range(len(order) + 1)]
        found[segment] = [value or 0 for value in row[2:13]]
        pair = spans.setdefault(dim, [row[13], row[14]])
        pair[0], pair[1] = min(pair[0], row[13]), max(pair[1], row[14])
    oldest = min((pair[0] for pair in spans.values()), default=None)
    position = {bound: at for at, bound in enumerate(order)}
    # usage by (account, user): what overview() reports; cumulative[k] holds the hours
    # from order[k] up to end, i.e. segments 1..k
    by_role = {}  # type: Dict[Tuple[str, str], List[list]]
    for dim, parts in per_dim.items():
        roles = dims.get(dim)
        if roles is None:
            continue
        key = (roles.account, roles.user)
        have = by_role.get(key)
        if have is None:
            by_role[key] = [list(part) for part in parts]
        else:
            for into, part in zip(have, parts):
                for at, value in enumerate(part):
                    into[at] += value
    cumulative = {}  # type: Dict[Tuple[str, str], List[tuple]]
    for key, parts in by_role.items():
        running, out = [0] * width, []
        for at, part in enumerate(parts):
            if at:
                running = [a + b for a, b in zip(running, part)]
            out.append(tuple(running))
        cumulative[key] = out

    def summed(low: int, high: int) -> Tuple[Amounts, Dict[str, Amounts],
                                              Dict[Tuple[str, str], Amounts]]:
        a, b = position.get(low), position.get(high)
        if a is None and oldest is not None and low <= oldest:
            a = len(order)  # everything there is
        if a is None and oldest is None:
            a = len(order)
        if a is None or b is None or a < b:
            return _sum(store, low, high, dims)
        total, accounts, users = Amounts(), {}, {}  # type: Amounts, Dict[str, Amounts], Dict
        for (account, user), values in cumulative.items():
            upper, lower = values[a], values[b]
            if upper[10] == lower[10]:
                continue  # no rollup row in the span
            amounts = Amounts(tuple(x - y for x, y in zip(upper[:10], lower[:10])))
            total.add(amounts)
            accounts.setdefault(account, Amounts()).add(amounts)
            users.setdefault((account, user), Amounts()).add(amounts)
        return total, accounts, users

    return summed, [(dim, pair[0], pair[1]) for dim, pair in spans.items()]


def overview(store: Any, as_of: int, covered_from: Optional[int], visibility: str,
             gpu_capacity: Optional[int] = None, cpu_capacity: Optional[int] = None,
             ranges: Any = RANGES, metric: str = "gpu") -> Overview:
    """Totals, change against the previous equal period and per-account detail, per range."""
    end = floor_hour(as_of)
    dims = store.dims()
    dirty = store.get_meta("dirty")
    built = store.get_meta("rollup_until")
    activity = {}  # type: Dict[str, List[int]]
    summed, spans = _segments(store, end, ranges, dims)
    for dim, first, last in spans:
        roles = dims.get(dim)
        if roles is not None:
            pair = activity.setdefault(roles.account, [first, last])
            pair[0], pair[1] = min(pair[0], first), max(pair[1], last)
    counts = {}  # type: Dict[str, int]
    for dim, count in store.count_by_dim().items():
        roles = dims.get(dim)
        if roles is not None:
            counts[roles.account] = counts.get(roles.account, 0) + count
    running_gpus, running_cpus = {}, {}  # type: Dict[str, int], Dict[str, int]
    for row in store.scan(as_of, as_of + 1, ordered=False):
        roles = dims.get(row[2])
        if roles is not None and row[7] is None and row[13] in (0, 3) and roles.state in (
                "RUNNING", "COMPLETING", "CONFIGURING", "STOPPED", "SIGNALING", "RESIZING"):
            running_gpus[roles.account] = running_gpus.get(roles.account, 0) + (row[10] or 0)
            running_cpus[roles.account] = running_cpus.get(roles.account, 0) + (row[9] or 0)
    earliest = min((pair[0] for pair in activity.values()), default=None)
    floor = capacity_floor(store, covered_from)
    result = Overview(as_of, covered_from, earliest, visibility,
                      pending_rollup=tuple(dirty) if dirty else None)

    series = {}  # type: Dict[int, Tuple[int, Dict[int, Tuple[int, int]]]]

    def binned(start: int, step: int) -> Tuple[List[int], List[int]]:
        """Per-bin usage over [start, end): one grouped query per step, sliced per range."""
        count = -(-(end - start) // step) if end > start else 0
        low, found = series.get(step, (None, None))
        if low is None or low > start:
            low = start
            found = {at: (gpu or 0, cpu or 0)
                     for at, gpu, cpu in store.rollup_series(low, end, step)}
            series[step] = (low, found)
        gpu = [found.get(at, (0, 0))[0] for at in range(count - 1, -1, -1)]
        cpu = [found.get(at, (0, 0))[1] for at in range(count - 1, -1, -1)]
        return gpu, cpu

    lengths = [length for _, length in ranges]
    if any(length is None or length > 7 * DAY for length in lengths):
        starts = [end - length for length in lengths if length and length > 7 * DAY]
        if None in lengths:
            starts.append(earliest if earliest is not None else end)
        binned(min(min(starts), end), DAY)  # the widest daily query first, so others slice it
    if any(length is not None and length <= 7 * DAY for length in lengths):
        binned(end - max(length for length in lengths if length and length <= 7 * DAY), HOUR)

    for name, length in ranges:
        start = (earliest if earliest is not None else end) if length is None else end - length
        total, accounts, users = summed(start, end)
        previous = previous_accounts = None
        if length is not None:
            previous, previous_accounts, _ = summed(start - length, start)
        span = util_span(start if length is not None else min(start, floor or start), end,
                         floor)
        loaded = covered_from is not None and (covered_from == 0 or covered_from <= (
            start - (length or 0)))
        complete = (loaded and built is not None and built >= end
                    and not (dirty and dirty[0] < end and dirty[1] > start - (length or 0)))

        def share(seconds: int, capacity: Optional[int], width: int) -> Optional[float]:
            return seconds / (capacity * width) if capacity and width > 0 else None

        before = util_span(start - (length or 0), start, floor)
        summary = RangeSummary(
            name, start, end, total, previous,
            share(total.gpu_seconds, gpu_capacity, span),
            share(total.cpu_seconds, cpu_capacity, span),
            share(previous.gpu_seconds, gpu_capacity, before) if previous else None, {},
            complete)
        summary.span, summary.previous_span = span, before
        summary.series_step = HOUR if length is not None and length <= 7 * DAY else DAY
        summary.gpu_series, summary.cpu_series = binned(start, summary.series_step)
        grand = total.amount(metric)
        for account, amounts in accounts.items():
            pair = activity.get(account, [None, None])
            summary.accounts[account] = AccountSummary(
                account, amounts, (previous_accounts or {}).get(account),
                {user: a for (acct, user), a in users.items() if acct == account},
                amounts.amount(metric) / grand if grand else None, pair[0], pair[1],
                running_gpus.get(account, 0), running_cpus.get(account, 0),
                counts.get(account, 0))
        result.ranges.append(summary)
    return result


def with_capacity(found: Overview, gpu_capacity: Optional[int],
                  cpu_capacity: Optional[int]) -> Overview:
    """``found`` (computed for any capacity) with its utilizations for these capacities,
    as overview() would give them; ``found`` itself is not changed."""
    import copy

    def share(seconds: int, capacity: Optional[int], width: int) -> Optional[float]:
        return seconds / (capacity * width) if capacity and width > 0 else None

    out = copy.copy(found)
    out.ranges = []
    for summary in found.ranges:
        summary = copy.copy(summary)
        summary.gpu_utilization = share(summary.total.gpu_seconds, gpu_capacity, summary.span)
        summary.cpu_utilization = share(summary.total.cpu_seconds, cpu_capacity, summary.span)
        summary.previous_gpu_utilization = share(
            summary.previous.gpu_seconds, gpu_capacity, summary.previous_span) \
            if summary.previous else None
        out.ranges.append(summary)
    return out


def describe_span(start: int, end: int) -> str:
    fmt = "%Y-%m-%d %H:%M"
    return "%s to %s" % (time.strftime(fmt, time.localtime(start)),
                         time.strftime(fmt, time.localtime(end)))
