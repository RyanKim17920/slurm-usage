"""One pass over the stored allocations that yields a whole Report."""

from array import array
from bisect import bisect_left, bisect_right
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .errors import Cancelled, Progress, ProgressFn, canceller
from .records import NONE, OUTCOMES, Roles
from .report import Lane, Report
from .select import ROLES, Filter, Part, Selection, resolve
from .series import Series
from .stats import METRICS, Stats
from .timeline import Window, hour_edges

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


class _Pass:
    """The state of one aggregation."""

    def __init__(self, specs: List[Tuple[str, ...]], gpu_tracked: Optional[bool],
                 edges: List[int], series_by: Optional[str], series_metric: str,
                 lanes_by: Optional[str], resolution: int,
                 lanes_metric: Optional[str] = None) -> None:
        self.gpu_tracked = gpu_tracked
        # lanes from attempts holding this resource only (None: any allocation)
        self.lanes_at = None if lanes_metric is None else METRICS.index(lanes_metric)
        self.total = Stats(gpu_tracked)
        self.groups = {spec: {} for spec in specs}  # type: Dict[tuple, Dict[tuple, Stats]]
        self.node_specs = [s for s in specs if "node" in s]
        self.plain_specs = [s for s in specs if "node" not in s]
        self.edges = edges
        self.series = [Series(edges) for _ in METRICS]
        self.series_by, self.metric_at = series_by, METRICS.index(series_metric)
        self.keyed = {}  # type: Dict[str, Series]
        self.lanes_by, self.resolution = lanes_by, resolution
        self.segments = {}  # type: Dict[str, List[List[int]]]
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
                    group[values] = Stats(self.gpu_tracked)
                found.append(group[values])
            self._targets[key] = found
        return found

    @staticmethod
    def take(part: Part, targets: List[Stats], cpus: Optional[int], gpus: Optional[int],
             mem: Optional[int], repeat: Optional[tuple], wait: Optional[int],
             limit: Optional[int], ratios: Optional[tuple]) -> None:
        row, _, _, p, q, seconds, _, problem = part[:8]
        if problem:
            for stats in targets:
                stats.unresolved += 1
            return
        for stats in targets:
            stats.add(p, q, seconds, cpus, gpus, mem)
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
                if lane and p <= lane[-1][1] + self.resolution:
                    if q > lane[-1][1]:
                        lane[-1][1] = q
                else:
                    lane.append([p, q])

    def finish(self) -> None:
        self.total.finish()
        for group in self.groups.values():
            for stats in group.values():
                stats.finish()


def build_report(store: Any, as_of: int, window: Optional[Window] = None,
                 where: Optional[Filter] = None, group_by: Sequence[Any] = DEFAULT_GROUPS,
                 series_by: Optional[str] = None, series_metric: str = "gpu",
                 lanes_by: Optional[str] = None, lane_resolution: Optional[int] = None,
                 efficiency: bool = False, gpu_tracked: Optional[bool] = None,
                 progress: ProgressFn = None, cancel: Any = None,
                 lanes_metric: Optional[str] = None) -> Report:
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
        state = _Pass(specs, gpu_tracked, hour_edges(span.start, span.end), series_by,
                      series_metric, lanes_by, resolution, lanes_metric)
        repeated = store.repeated_jobs()  # type: Set[int]
        unchecked = 0

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
                state.take(part, state.targets(roles, outcome), part.cpus, part.gpus, part.mem,
                           repeat, wait, limit, ratios)
                if state.node_specs:
                    for node, cpus, gpus, mem in part.shares or ():
                        state.take(part, state.targets(roles, outcome, node), cpus, gpus, mem,
                                   repeat, wait, limit, ratios)
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

    by_outcome = state.groups[("outcome",)]
    return Report(
        window=window, start=span.start, end=span.end, as_of=as_of, where=where,
        total=state.total, groups=state.groups,
        outcomes={name: by_outcome.get((name,)) or Stats(gpu_tracked) for name in OUTCOMES},
        edges=state.edges,
        values={name: s.values() for name, s in zip(METRICS, state.series)},
        series_by=series_by, series_metric=series_metric,
        keyed_values={key: s.values() for key, s in state.keyed.items()},
        lanes=[Lane(key, [(a, b) for a, b in segments])
               for key, segments in sorted(state.segments.items())],
        waits=sorted(state.waits),
        efficiency_unchecked=unchecked if efficiency else None)
