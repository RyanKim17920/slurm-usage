"""Run types: usage split by what kind of run it was, for one person or any filter."""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .aggregate import LAZY_FIELDS, LazyStats, _fast_pass, _fast_span, _Pass, usage_ratios
from .errors import Cancelled, canceller
from .families import name_family
from .records import outcome_of
from .select import Filter, Part, Selection, resolve, unresolved
from .stats import Stats
from .store import FAMILY_VERSION
from .timeline import Window, hour_edges

# kinds the exact aggregates can tell apart: by GPUs per allocation, role or job-name family
_FAST = ("gpu_count", "account", "qos", "partition", "outcome", "name_family")

TYPE_ROLES = ("gpu_count", "name_family", "account", "qos", "partition", "outcome",
              "node_count")
FAILURES = ("failed", "node_fail", "timeout")


@lru_cache(maxsize=16384)
def _node_count(nodes: str) -> int:
    """Nodes in a host list (the same few lists come up again and again)."""
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


class RunType:
    """One kind of run. The medians and the example name may be worked out on first
    use (one query for every kind of the same answer): screens never read them."""

    __slots__ = ("key", "by", "stats", "share", "outcomes", "_median_runtime",
                 "_median_wait", "_example", "_later", "_set")

    def __init__(self, key: str, by: str, stats: Stats, share: Optional[float],
                 outcomes: Optional[Dict[str, int]] = None,
                 median_runtime: Optional[int] = None, median_wait: Optional[int] = None,
                 example: str = "", later: Any = None) -> None:
        self.key, self.by, self.stats, self.share = key, by, stats, share
        self.outcomes = {} if outcomes is None else outcomes  # attempts per outcome
        self._median_runtime = median_runtime  # whole attempts, less suspension
        self._median_wait = median_wait
        self._example = example  # one real name of this family
        self._later = later  # () -> None: fills the three in for every kind at once
        self._set = ()  # type: Tuple[str, ...]  # assigned, so kept when they are filled in

    def _settle(self) -> None:
        later = self._later
        if later is not None:
            later()

    @property
    def median_runtime(self) -> Optional[int]:
        self._settle()
        return self._median_runtime

    @median_runtime.setter
    def median_runtime(self, value: Optional[int]) -> None:
        self._set += ("median_runtime",)
        self._median_runtime = value

    @property
    def median_wait(self) -> Optional[int]:
        self._settle()
        return self._median_wait

    @median_wait.setter
    def median_wait(self, value: Optional[int]) -> None:
        self._set += ("median_wait",)
        self._median_wait = value

    @property
    def example(self) -> str:
        self._settle()
        return self._example

    @example.setter
    def example(self, value: str) -> None:
        self._set += ("example",)
        self._example = value

    def __getstate__(self) -> Any:
        self._settle()
        return {name: getattr(self, name) for name in self.__slots__
                if name not in ("_later", "_set")}

    def __setstate__(self, state: Any) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        self._later, self._set = None, ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RunType):
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name) for name in (
            "key", "by", "stats", "share", "outcomes", "median_runtime", "median_wait",
            "example"))

    def __repr__(self) -> str:
        return "RunType(key=%r, by=%r, share=%r, outcomes=%r)" % (
            self.key, self.by, self.share, self.outcomes)

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


def _key(by: str, roles: Any, outcome: str, gpus: Any) -> str:
    """``gpus``: GPUs per allocation, or by name_family the family."""
    if by == "name_family":
        return gpus
    if by == "gpu_count":
        return "unknown" if gpus is None else str(gpus)
    if by == "outcome":
        return outcome
    return getattr(roles, by)


def _fast_types(store: Any, as_of: int, window: Window, where: Filter, by: str, metric: str,
                gpu_tracked: Optional[bool], cancelled: Any, cancel: Any
                ) -> Optional[List[RunType]]:
    """run_types() from the exact aggregates, or None where they cannot answer."""
    if by not in _FAST or where.node is not None or not getattr(store, "has_agg", False):
        return None
    family = by == "name_family"
    if family and (not getattr(store, "has_families", False)
                   or store.get_meta("agg_fam") != FAMILY_VERSION):
        return None
    with store.lock:
        chosen = Selection(store.dims(), {}, where)
        first = store.first_start(chosen.dim_ids) if window.start is None else None
        span = resolve(window, as_of, first)
        edges = hour_edges(span.start, span.end)
        hours = _fast_span(store, span, edges)
        if hours is None:
            return None
        groups = {}  # type: Dict[str, Tuple[LazyStats, Dict[str, Stats]]]

        def targets_of(roles: Any, outcome: str, gpus: Optional[int]) -> List[Stats]:
            key = _key(by, roles, outcome, gpus)
            found = groups.get(key)
            if found is None:
                found = groups[key] = (LazyStats(gpu_tracked), {})
            stats, outcomes = found
            if outcome not in outcomes:
                outcomes[outcome] = Stats(gpu_tracked)
            return [stats, outcomes[outcome]]

        state = _Pass([("outcome",)], gpu_tracked, edges, None, "gpu", None, 1)
        _fast_pass(store, chosen, span, state, hours, store.repeated_jobs(), True, cancelled,
                   None, targets_of, skip_problem=True,
                   **({"by_family": True} if family else {}))
        dims, outcomes_wanted = chosen.dims, where.outcome
        finished = None  # type: Any
        if chosen.dim_ids is not None or outcomes_wanted is not None:
            finished = {d for d in (chosen.dim_ids if chosen.dim_ids is not None else dims)
                        if d in dims and (outcomes_wanted is None or
                                          outcome_of(dims[d].state, True) in outcomes_wanted)}
        # medians and one real name per kind, from SQLite
        live = None  # type: Any
        if chosen.dim_ids is not None or outcomes_wanted is not None:
            live = {d for d in (chosen.dim_ids if chosen.dim_ids is not None else dims)
                    if d in dims and (outcomes_wanted is None or
                                      outcome_of(dims[d].state, False) in outcomes_wanted)}
        skip = [r[0] for r in store.open_rows()
                if r[2] in dims and unresolved(dims[r[2]].state, None, r[8])]
        # an unfinished row's outcome is not its dim's finished one
        kinds = None if by in ("gpu_count", "name_family") else {
            d: (_key(by, roles, outcome_of(roles.state, True), None),
                _key(by, roles, outcome_of(roles.state, False), None))
            for d, roles in dims.items()}
        medians_of = (span.low, span.high, span.start, span.end, as_of, finished, live, skip,
                      kinds) + ((name_family,) if family else ())
        lookup = (lambda key: -1 if key == "unknown" else int(key)) if by == "gpu_count" \
            else (lambda key: key)
    total = 0
    out = []
    for key, (stats, outcomes) in groups.items():
        stats.finish()
        if not stats.attempts:
            continue
        total += stats.amount(metric) or 0
    for key, (stats, outcomes) in groups.items():
        if not stats.attempts:
            continue
        amount = stats.amount(metric)
        out.append(RunType(key, by, stats, amount / total if total and amount is not None
                           else None, {o: s.attempts for o, s in outcomes.items() if s.attempts}))

    def medians() -> None:
        """Run times, waits and the first name of each kind: a read of every row, so
        only when one of them is asked for."""
        for item, _ in keyed:
            item._later = None
        runtimes, waits, first_names = store.type_medians(*medians_of)
        for item, original in keyed:  # the key as made here (a wrapper may rename it)
            key = lookup(original)
            for name, value in (("median_runtime", runtimes.get(key)),
                                ("median_wait", waits.get(key)),
                                ("example", first_names.get(key, ""))):
                if name not in item._set:
                    setattr(item, "_" + name, value)

    keyed = [(item, item.key) for item in out]
    for item in out:
        item._later = medians
    members = [item.stats for item in out]

    def resolve_lazy() -> None:
        full = {item.key: item.stats for item in run_types(
            store, as_of, window, where, by, metric, gpu_tracked, cancel, exact=True)}
        for stats in members:
            stats._resolver = None
        for item in out:
            source = full.get(item.key) or Stats()
            for name in LAZY_FIELDS:
                setattr(item.stats, name, getattr(source, name))

    for stats in members:
        stats._resolver = resolve_lazy
    out.sort(key=lambda t: (-(t.stats.amount(metric) or 0), -t.stats.attempts, t.key))
    return out


def run_types(store: Any, as_of: int, window: Optional[Window], where: Optional[Filter],
              by: str, metric: str = "gpu", gpu_tracked: Optional[bool] = None,
              cancel: Any = None, exact: bool = False) -> List[RunType]:
    if by not in TYPE_ROLES:
        raise ValueError("by is one of %s" % ", ".join(TYPE_ROLES))
    window, where, cancelled = window or Window(), where or Filter(), canceller(cancel)
    if not exact:
        found = _fast_types(store, as_of, window, where, by, metric, gpu_tracked, cancelled,
                            cancel)
        if found is not None:
            return found
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
