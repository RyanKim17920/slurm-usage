"""One pass over the stored allocations that yields a whole Report."""

from array import array
from bisect import bisect_left, bisect_right
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from collections.abc import Sequence as _SequenceABC

from .errors import Cancelled, Progress, ProgressFn, canceller
from .records import NONE, OUTCOMES, Roles, outcome_of
from .report import Lane, Report
from .select import ROLES, Filter, Part, Selection, effective, resolve
from .series import Series
from .stats import METRICS, Stats
from .store import AGG_VERSION, OPEN_STOP
from .timeline import HOUR, Window, hour_edges

DEFAULT_GROUPS = (("user",), ("account",), ("qos",), ("partition",))
_PROGRESS_EVERY = 25000


def _group_specs(group_by: Sequence[Any]) -> List[Tuple[str, ...]]:
    specs = [("outcome",)]
    for item in group_by:
        spec = (item,) if isinstance(item, str) else tuple(item)
        for role in spec:
            if role not in ROLES:
                raise ValueError("unknown role: %s" % role)
        if spec and spec not in specs:
            specs.append(spec)
    return specs


def _value(role: str, roles: Roles, outcome: str, node: str) -> str:
    if role == "outcome":
        return outcome
    return node if role == "node" else getattr(roles, role)


def usage_ratios(row: tuple) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(cpu, mem, gpu) efficiency of one whole finished attempt; None where unmeasured."""
    begun, end, suspended, cpus, gpus, mem = row[6:12]
    used_cpu, used_rss, util = row[14:17]
    if end is None or begun is None:
        return None, None, None
    elapsed = end - begun - suspended
    cpu = (used_cpu / 1000.0 / (cpus * elapsed)
           if used_cpu is not None and cpus and elapsed > 0 else None)
    rss = used_rss / mem if used_rss is not None and mem else None
    gpu = util / (100.0 * gpus) if util is not None and gpus else None
    return cpu, rss, gpu


def start_counts(row: tuple, outcome: str, ratios: tuple) -> tuple:
    """agg_start's per-attempt columns for one finished allocation, counted as a window
    that holds all of it counts them (aggregate._Pass.take)."""
    begun, end, suspended = row[6], row[7], row[8]
    cpus, gpus, mem, limit = row[9], row[10], row[11], row[12]
    duration = end - begun
    seconds = duration - min(suspended, duration) if duration > 0 else 0
    wait = begun - row[5] if row[5] is not None and row[5] <= begun else None
    limited = outcome in ("completed", "timeout") and bool(limit) and limit > 0
    gpu = ratios[2]
    return (1, int(cpus is not None), int(mem is not None), int(gpus is not None),
            int(bool(suspended)), int(wait is not None), wait or 0, wait,
            int(limited), (end - begun - suspended) if limited else 0, limit if limited else 0,
            int(bool(seconds) and gpu is not None and bool(gpus) and gpu > 1.0),
            int(duration > 0 and not row[17]))


class LazyStats(Stats):
    """Stats from the exact aggregates. The fields that do not add up over hours and
    roles (first, last, peak_gpus, active_seconds, gpu_active_seconds) are computed by
    one full pass over the window, the first time any of them is read."""

    __slots__ = ("_resolver",)

    def __init__(self, gpu_tracked: Optional[bool] = None) -> None:
        self._resolver = None  # type: Any
        Stats.__init__(self, gpu_tracked)


def _lazy(name: str) -> property:
    slot = Stats.__dict__[name]

    def get(self: LazyStats) -> Any:
        if self._resolver is not None:
            self._resolver()
        return slot.__get__(self, Stats)

    def put(self: LazyStats, value: Any) -> None:
        slot.__set__(self, value)

    return property(get, put)


LAZY_FIELDS = ("first", "last", "peak_gpus", "active_seconds", "gpu_active_seconds")
for _name in LAZY_FIELDS:
    setattr(LazyStats, _name, _lazy(_name))


class _Waits(_SequenceABC):
    """Sorted queue waits, read from the cache the first time they are needed."""

    def __init__(self, load: Any) -> None:
        self._load, self._values = load, None  # type: Any, Optional[List[int]]

    def _get(self) -> List[int]:
        if self._values is None:
            self._values = self._load()
            self._load = None
        return self._values

    def __getitem__(self, index: Any) -> Any:
        return self._get()[index]

    def __len__(self) -> int:
        return len(self._get())

    def __eq__(self, other: object) -> bool:
        return list(self._get()) == list(other)  # type: ignore

    def __repr__(self) -> str:
        return repr(self._get())


class _Pass:
    """The state of one aggregation."""

    def __init__(self, specs: List[Tuple[str, ...]], gpu_tracked: Optional[bool],
                 edges: List[int], series_by: Optional[str], series_metric: str,
                 lanes_by: Optional[str], resolution: int,
                 lanes_metric: Optional[str] = None, stats_class: Any = Stats) -> None:
        self.gpu_tracked = gpu_tracked
        self.stats_class = stats_class
        # lanes from attempts holding this resource only (None: any allocation)
        self.lanes_at = None if lanes_metric is None else METRICS.index(lanes_metric)
        self.total = stats_class(gpu_tracked)
        self.groups = {spec: {} for spec in specs}  # type: Dict[tuple, Dict[tuple, Stats]]
        self.node_specs = [s for s in specs if "node" in s]
        self.plain_specs = [s for s in specs if "node" not in s]
        self.edges = edges
        self.series = [Series(edges) for _ in METRICS]
        self.series_by, self.metric_at = series_by, METRICS.index(series_metric)
        self.keyed = {}  # type: Dict[str, Series]
        self.lanes_by, self.resolution = lanes_by, resolution
        self.segments = {}  # type: Dict[str, List[List[int]]]
        self.raw_lanes = False
        self.waits = array("q")
        self._targets = {}  # type: Dict[tuple, List[Stats]]

    def targets(self, roles: Roles, outcome: str, node: Optional[str] = None) -> List[Stats]:
        key = (roles, outcome, node)
        found = self._targets.get(key)
        if found is None:
            found = [self.total] if node is None else []
            for spec in self.plain_specs if node is None else self.node_specs:
                values = tuple(_value(role, roles, outcome, node or NONE) for role in spec)
                group = self.groups[spec]
                if values not in group:
                    group[values] = self.stats_class(self.gpu_tracked)
                found.append(group[values])
            self._targets[key] = found
        return found

    @staticmethod
    def take(part: Part, targets: List[Stats], cpus: Optional[int], gpus: Optional[int],
             mem: Optional[int], repeat: Optional[tuple], wait: Optional[int],
             limit: Optional[int], ratios: Optional[tuple],
             amount: Optional[int] = None) -> None:
        """``amount``: the seconds to count usage for, when not all of part.seconds
        (the rest is in the aggregates)."""
        row, _, _, p, q, seconds, _, problem = part[:8]
        if problem:
            for stats in targets:
                stats.unresolved += 1
            return
        counted = seconds if amount is None else amount
        for stats in targets:
            stats.add(p, q, counted, cpus, gpus, mem)
        if repeat is not None:
            for stats in targets:
                stats.note_repeat(repeat)
        if row[7] is None:
            for stats in targets:
                stats.running += 1
        if row[8]:
            for stats in targets:
                stats.suspension_estimated += 1
        if wait is not None:
            for stats in targets:
                stats.wait_count += 1
                stats.wait_total += wait
                if stats.wait_max is None or wait > stats.wait_max:
                    stats.wait_max = wait
        if limit is not None:
            elapsed = part.real_stop - row[6] - row[8]
            for stats in targets:
                stats.limit_count += 1
                stats.limit_used += elapsed
                stats.limit_requested += limit
        if ratios is None or not seconds:
            return
        cpu, rss, gpu = ratios
        seconds = counted
        if cpu is not None and cpus:
            weight = cpus * seconds
            for stats in targets:
                stats.cpu_covered += weight
                stats.cpu_used += cpu * weight
        if rss is not None and mem:
            weight = mem * seconds
            for stats in targets:
                stats.mem_covered += weight
                stats.mem_used += rss * weight
        if gpu is not None and gpus:
            weight = gpus * seconds
            capped = min(gpu, 1.0)
            for stats in targets:
                stats.gpu_covered += weight
                stats.gpu_used += capped * weight
                if gpu > 1.0:
                    stats.gpu_over_100 += 1

    def draw(self, part: Part) -> None:
        p, q = part.start, part.stop
        edges = self.edges
        i, j = bisect_right(edges, p) - 1, bisect_left(edges, q) - 1
        row = part.row
        begun, suspended = row[6], row[8]
        duration = part.real_stop - begun
        held = min(suspended, duration)

        def draw_one(series: Series, rate: int) -> None:
            if held:
                series.add_suspended(i, j, p, q, rate, begun, duration, held)
            else:
                series.add(i, j, p, q, rate)

        amounts = (part.gpus, part.cpus, part.mem)
        for series, rate in zip(self.series, amounts):
            if rate:
                draw_one(series, rate)
        by = self.series_by
        if by is not None:
            if by == "node":
                keyed = [(s[0], (s[2], s[1], s[3])[self.metric_at]) for s in part.shares or ()]
            else:
                keyed = [(_value(by, part.roles, part.outcome, NONE), amounts[self.metric_at])]
            for key, rate in keyed:
                if rate:
                    if key not in self.keyed:
                        self.keyed[key] = Series(edges)
                    draw_one(self.keyed[key], rate)
        by = self.lanes_by
        if by is not None and (self.lanes_at is None or amounts[self.lanes_at]):
            keys = ([s[0] for s in part.shares or ()] if by == "node"
                    else [_value(by, part.roles, part.outcome, NONE)])
            for key in keys:
                lane = self.segments.setdefault(key, [])
                if self.raw_lanes:  # merged once all are in (they come in any order)
                    lane.append([p, q])
                elif lane and p <= lane[-1][1] + self.resolution:
                    if q > lane[-1][1]:
                        lane[-1][1] = q
                else:
                    lane.append([p, q])

    def finish(self) -> None:
        self.total.finish()
        for group in self.groups.values():
            for stats in group.values():
                stats.finish()


def _fast_span(store: Any, span: Any, edges: List[int]) -> Optional[Tuple[int, int]]:
    """Whole hours [low, high) of the window the exact aggregates can answer, or None."""
    if not getattr(store, "has_agg", False) or store.get_meta("agg") != AGG_VERSION:
        return None
    built = store.get_meta("rollup_until")
    if built is None:
        return None
    low = -(-span.start // HOUR) * HOUR
    high = min(span.end - span.end % HOUR, built)
    dirty = store.get_meta("dirty")
    if dirty:
        bad_low = max(0, dirty[0]) // HOUR * HOUR
        bad_high = min(dirty[1], OPEN_STOP) // HOUR * HOUR + HOUR
        if bad_low < high and bad_high > low:
            left, right = (low, min(high, bad_low)), (max(low, bad_high), high)
            low, high = max(left, right, key=lambda pair: pair[1] - pair[0])
    if high - low < 2 * HOUR:
        return None
    # every hour must fall in one bin (zones a whole number of hours from UTC)
    if any(at % HOUR for at in edges[bisect_right(edges, low):bisect_left(edges, high)]):
        return None
    return low, high


def _edge_seconds(part: Part, low: int, high: int) -> int:
    """part.seconds outside [low, high)."""
    p, q = part.start, part.stop
    if p >= low and q <= high:
        return 0
    row = part.row
    begun, suspended = row[6], row[8]
    duration = part.real_stop - begun
    held = min(suspended, duration) if duration > 0 else 0
    total = 0
    for a, b in ((p, min(q, low)), (max(p, high), q)):
        if b > a:
            total += (effective(b - begun, duration, held) - effective(a - begun, duration, held)
                      if held else b - a)
    return total


def _add_counts(stats: Stats, counts: Sequence[Any], efficiency: bool) -> None:
    attempts = counts[0]
    stats.attempts += attempts
    stats.cpu_known += counts[1]
    stats.cpu_unknown += attempts - counts[1]
    stats.mem_known += counts[2]
    stats.mem_unknown += attempts - counts[2]
    stats.gpu_known += counts[3]
    stats.gpu_unknown += attempts - counts[3]
    stats.suspension_estimated += counts[4]
    stats.wait_count += counts[5]
    stats.wait_total += counts[6]
    if counts[7] is not None and (stats.wait_max is None or counts[7] > stats.wait_max):
        stats.wait_max = counts[7]
    stats.limit_count += counts[8]
    stats.limit_used += counts[9]
    stats.limit_requested += counts[10]
    if efficiency:
        stats.gpu_over_100 += counts[11]


def _add_hours(stats: Stats, sums: Sequence[Any], efficiency: bool) -> None:
    stats.gpu_seconds += sums[0]
    stats.cpu_seconds += sums[1]
    stats.mem_byte_seconds += sums[2]
    stats.seconds += sums[3]
    if efficiency:
        stats.cpu_covered += sums[4]
        stats.cpu_used += sums[5]
        stats.mem_covered += sums[6]
        stats.mem_used += sums[7]
        stats.gpu_covered += sums[8]
        stats.gpu_used += sums[9]


def _fast_pass(store: Any, chosen: Selection, span: Any, state: _Pass, hours: Tuple[int, int],
               repeated: Set[int], efficiency: bool, cancelled: Any, progress: ProgressFn,
               targets_of: Any = None, skip_problem: bool = False) -> Tuple[int, Any]:
    """The window from the aggregates of its whole hours plus the rows at its edges.
    Returns (finished attempts whose steps are unread, the sorted waits).

    targets_of(roles, outcome, gpus) -> [Stats] splits by GPUs per allocation as well
    (run types); skip_problem leaves out attempts not counted up to now."""
    low, high = hours
    by_gpus = targets_of is not None
    if targets_of is None:
        def targets_of(roles: Roles, outcome: str, gpus: Any) -> List[Stats]:
            return state.targets(roles, outcome)
    dims, outcomes = chosen.dims, chosen.where.outcome
    finished_dims = None  # type: Optional[Set[int]]
    if chosen.dim_ids is not None or outcomes is not None:
        finished_dims = {d for d in (chosen.dim_ids if chosen.dim_ids is not None else dims)
                         if d in dims and (outcomes is None
                                           or outcome_of(dims[d].state, True) in outcomes)}
    unchecked = 0
    less = {}  # type: Dict[int, List[Any]]  # counts of rows the aggregates hold, done exactly
    rows = store.scan_edges(span.start, low, high, span.high, chosen.dim_ids, usage=efficiency)
    for number, part in enumerate(chosen.parts(rows, span)):
        if not number % 1024 and cancelled():
            raise Cancelled()
        row, roles, outcome, p, q = part[:5]
        if row[7] is not None and span.low is not None and row[7] < span.low:
            continue  # as scan() would: it ended before the window
        if skip_problem and part.problem:
            continue
        repeat = wait = limit = ratios = None
        if not part.problem:
            if row[1] in repeated:
                repeat = (row[1], roles.uid)
            if part.starts_here and row[5] is not None and row[5] <= row[6]:
                wait = row[6] - row[5]
                state.waits.append(wait)
            if (part.ends_here and outcome in ("completed", "timeout") and row[12]
                    and row[12] > 0):
                limit = row[12]
            if efficiency and row[7] is not None:
                ratios = usage_ratios(row)
                if q > p and not row[17]:
                    unchecked += 1
        held = row[7] is not None and row[7] >= row[6]  # in the aggregates
        amount = None
        if held:
            amount = _edge_seconds(part, low, high)
            if low <= row[6] < high:
                counts = start_counts(row, outcome, ratios or (None, None, None))
                found = less.setdefault((row[2], row[10]) if by_gpus else row[2], [0] * 13)
                for at, value in enumerate(counts):
                    if at != 7:
                        found[at] += value
        state.take(part, targets_of(roles, outcome, row[10]), part.cpus, part.gpus, part.mem,
                   repeat, wait, limit, ratios, amount)
        if q > p and not part.problem:
            if not held:
                state.draw(part)
            else:
                for a, b in ((p, min(q, low)), (max(p, high), q)):
                    if b > a:
                        state.draw(part._replace(start=a, stop=b))

    by_hour, by_start = store.agg_sums(low, high, finished_dims, by_gpus)
    for key in set(by_hour) | set(by_start):
        dim, gpus = key if by_gpus else (key, None)
        roles = dims.get(dim)
        if roles is None:
            continue
        outcome = outcome_of(roles.state, True)
        if outcomes is not None and outcome not in outcomes:
            continue
        targets = targets_of(roles, outcome, gpus)
        sums, counts = by_hour.get(key), by_start.get(key)
        if counts is not None:
            counts = list(counts)
            for at, value in enumerate(less.get(key, ())):
                if at != 7:
                    counts[at] = (counts[at] or 0) - value
            if efficiency:
                unchecked += counts[12] or 0
        for stats in targets:
            if sums is not None:
                _add_hours(stats, sums, efficiency)
            if counts is not None:
                _add_counts(stats, counts, efficiency)

    # distinct jobs: the attempts of jobs with several, inside the whole hours
    for begun, stop, job, dim, gpus in store.repeated_rows(low, high, finished_dims):
        roles = dims.get(dim)
        if roles is None:
            continue
        for stats in targets_of(roles, outcome_of(roles.state, True), gpus):
            stats.note_repeat((job, roles.uid))

    edges, by = state.edges, state.series_by
    metric_at = state.metric_at
    for hour, dim, gpu, cpu, mem in store.agg_series(low, high, finished_dims, by is not None):
        i = bisect_right(edges, hour) - 1
        for series, amount in zip(state.series, (gpu, cpu, mem)):
            if amount:
                series.partial[i] += amount
        if by is not None:
            roles = dims.get(dim)
            amount = (gpu, cpu, mem)[metric_at]
            if roles is None or not amount:
                continue
            key = _value(by, roles, outcome_of(roles.state, True), NONE)
            if key not in state.keyed:
                state.keyed[key] = Series(edges)
            state.keyed[key].partial[i] += amount

    if state.lanes_by is not None:
        metric = None if state.lanes_at is None else METRICS[state.lanes_at]
        for hour, dim, first, last in store.agg_busy(low, high, finished_dims, metric):
            roles = dims.get(dim)
            if roles is None:
                continue
            key = _value(state.lanes_by, roles, outcome_of(roles.state, True), NONE)
            state.segments.setdefault(key, []).append([first, last])
        resolution = state.resolution
        for key, pieces in state.segments.items():
            pieces.sort()
            merged = []  # type: List[List[int]]
            for p, q in pieces:
                if merged and p <= merged[-1][1] + resolution:
                    if q > merged[-1][1]:
                        merged[-1][1] = q
                else:
                    merged.append([p, q])
            state.segments[key] = merged

    edge_waits = state.waits

    def waits() -> List[int]:
        return sorted(list(edge_waits) + store.agg_waits(low, high, finished_dims))

    return unchecked, _Waits(waits)


def build_report(store: Any, as_of: int, window: Optional[Window] = None,
                 where: Optional[Filter] = None, group_by: Sequence[Any] = DEFAULT_GROUPS,
                 series_by: Optional[str] = None, series_metric: str = "gpu",
                 lanes_by: Optional[str] = None, lane_resolution: Optional[int] = None,
                 efficiency: bool = False, gpu_tracked: Optional[bool] = None,
                 progress: ProgressFn = None, cancel: Any = None,
                 lanes_metric: Optional[str] = None, exact: bool = False) -> Report:
    """exact=True reads every row of the window instead of the aggregates of its whole
    hours; the results are the same (floating-point sums up to rounding)."""
    specs = _group_specs(group_by)
    if lanes_metric is not None and lanes_metric not in METRICS:
        raise ValueError("unknown metric: %s" % lanes_metric)
    for role in (series_by, lanes_by):
        if role is not None and role not in ROLES:
            raise ValueError("unknown role: %s" % role)
    if series_metric not in METRICS:
        raise ValueError("unknown metric: %s" % series_metric)
    window, where, cancelled = window or Window(), where or Filter(), canceller(cancel)

    with store.lock:
        need_nodes = any("node" in spec for spec in specs) or "node" in (series_by, lanes_by)
        chosen = Selection(store.dims(), store.nodesets(), where, need_nodes)
        first = store.first_start(chosen.dim_ids) if window.start is None else None
        span = resolve(window, as_of, first)
        resolution = (max(1, (span.end - span.start) // 1000) if lane_resolution is None
                      else lane_resolution)
        edges = hour_edges(span.start, span.end)
        hours = None
        # lanes close gaps up to the resolution, so an hour's first and last use stand for
        # all of it once the resolution is an hour or more
        if not exact and not chosen.need_nodes and (lanes_by is None or resolution >= HOUR):
            hours = _fast_span(store, span, edges)
        state = _Pass(specs, gpu_tracked, edges, series_by, series_metric, lanes_by,
                      resolution, lanes_metric, Stats if hours is None else LazyStats)
        repeated = store.repeated_jobs()  # type: Set[int]
        unchecked = 0
        waits = None  # type: Any

        if hours is not None:
            state.raw_lanes = True
            unchecked, waits = _fast_pass(store, chosen, span, state, hours, repeated,
                                          efficiency, cancelled, progress)
        else:
            rows = store.scan(span.low, span.high, chosen.dim_ids, usage=efficiency)
            try:
                for number, part in enumerate(chosen.parts(rows, span)):
                    if not number % 1024 and cancelled():
                        raise Cancelled()
                    if progress and number and not number % _PROGRESS_EVERY:
                        progress(Progress("aggregate", number))
                    row, roles, outcome, p, q = part[:5]
                    repeat = wait = limit = ratios = None
                    if not part.problem:
                        if row[1] in repeated:
                            repeat = (row[1], roles.uid)
                        if part.starts_here and row[5] is not None and row[5] <= row[6]:
                            wait = row[6] - row[5]
                            state.waits.append(wait)
                        if (part.ends_here and outcome in ("completed", "timeout") and row[12]
                                and row[12] > 0):
                            limit = row[12]
                        if efficiency and row[7] is not None:
                            ratios = usage_ratios(row)
                            if q > p and not row[17]:
                                unchecked += 1
                    state.take(part, state.targets(roles, outcome), part.cpus, part.gpus,
                               part.mem, repeat, wait, limit, ratios)
                    if state.node_specs:
                        for node, cpus, gpus, mem in part.shares or ():
                            state.take(part, state.targets(roles, outcome, node), cpus, gpus,
                                       mem, repeat, wait, limit, ratios)
                    if q > p and not part.problem:
                        state.draw(part)
            finally:
                rows.close()

        for roles, outcome, ended, count in chosen.idle(store.idle_counts(span.low, span.high)):
            targets = list(state.targets(roles, outcome))
            if state.node_specs:
                targets += state.targets(roles, outcome, NONE)
            for stats in targets:
                if ended:
                    stats.never_started += count
                else:
                    stats.pending += count
    state.finish()
    if waits is None:
        waits = sorted(state.waits)

    by_outcome = state.groups[("outcome",)]
    report = Report(
        window=window, start=span.start, end=span.end, as_of=as_of, where=where,
        total=state.total, groups=state.groups,
        outcomes={name: by_outcome.get((name,)) or Stats(gpu_tracked) for name in OUTCOMES},
        edges=state.edges,
        values={name: s.values() for name, s in zip(METRICS, state.series)},
        series_by=series_by, series_metric=series_metric,
        keyed_values={key: s.values() for key, s in state.keyed.items()},
        lanes=[Lane(key, [(a, b) for a, b in segments])
               for key, segments in sorted(state.segments.items())],
        waits=waits,
        efficiency_unchecked=unchecked if efficiency else None)
    if hours is not None:
        _defer(state, lambda: build_report(
            store, as_of, window, where, group_by, series_by, series_metric, lanes_by,
            lane_resolution, False, gpu_tracked, None, cancel, lanes_metric, exact=True))
    return report


def _defer(state: _Pass, exact: Any) -> None:
    """Attach the one full pass that fills in the non-additive fields when first read."""
    members = [((), (), state.total)] + [(spec, key, stats) for spec, group in state.groups.items()
                                         for key, stats in group.items()]

    def resolve_all() -> None:
        full = exact()
        for _, _, stats in members:
            stats._resolver = None
        for spec, key, stats in members:
            source = full.total if spec == () else full.groups.get(spec, {}).get(key)
            for name in LAZY_FIELDS:
                setattr(stats, name, getattr(source, name) if source is not None
                        else getattr(Stats(), name))

    for _, _, stats in members:
        stats._resolver = resolve_all
