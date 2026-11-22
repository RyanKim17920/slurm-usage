"""Filters, and the one place that decides how much of an allocation counts."""

from typing import Any, Dict, FrozenSet, Iterable, Iterator, List, NamedTuple, Optional, Set, Tuple

from ..model import PENDING_STATES
from ..parse import expand_hostlist
from .records import ACTIVE_STATES, NONE, OUTCOMES, Roles, outcome_of
from .store import FLAG_RUNAWAY, FLAG_STALE, Row
from .timeline import Window

ROLES = ("user", "account", "qos", "partition", "node", "state", "outcome")
_DIM_ROLES = ("user", "account", "qos", "partition", "state")
_NODE_LIMIT = 1 << 17

RUNAWAY, STALE, MISSING_END = "runaway", "stale", "missing_end"


class Filter:
    """Each role takes a value or a collection of values; None matches everything.

    Use records.NONE for "has no value". ``state`` is Slurm's state, ``outcome``
    one of records.OUTCOMES. A node filter counts only that node's share.
    """

    __slots__ = ROLES

    def __init__(self, user: Any = None, account: Any = None, qos: Any = None,
                 partition: Any = None, node: Any = None, state: Any = None,
                 outcome: Any = None) -> None:
        for role, value in zip(ROLES, (user, account, qos, partition, node, state, outcome)):
            if value is not None:
                value = frozenset((value,) if isinstance(value, str) else value)
            setattr(self, role, value)
        if self.outcome and not self.outcome <= frozenset(OUTCOMES):
            raise ValueError("unknown outcome: %s" % ", ".join(sorted(self.outcome)))

    def items(self) -> List[Tuple[str, FrozenSet[str]]]:
        return [(r, getattr(self, r)) for r in ROLES if getattr(self, r) is not None]

    def replace(self, **changes: Any) -> "Filter":
        values = {role: getattr(self, role) for role in ROLES}
        values.update(changes)
        return Filter(**values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Filter) and self.items() == other.items()

    def __hash__(self) -> int:
        return hash(tuple(self.items()))

    def __repr__(self) -> str:
        return "Filter(%s)" % ", ".join("%s=%r" % (r, sorted(v)) for r, v in self.items())


def matching_dims(dims: Dict[int, Roles], where: Filter) -> Optional[Set[int]]:
    """Ids passing the filter on stored roles; None means all of them."""
    tests = [(Roles._fields.index(role), getattr(where, role))
             for role in _DIM_ROLES if getattr(where, role) is not None]
    if not tests:
        return None
    return {i for i, roles in dims.items() if all(roles[at] in wanted for at, wanted in tests)}


def unresolved(state: str, end: Optional[int], flag: int) -> Optional[str]:
    """Why an allocation without an end must not be counted up to now, if it must not."""
    if end is not None:
        return None
    if state not in ACTIVE_STATES:
        return MISSING_END
    if flag == FLAG_RUNAWAY:
        return RUNAWAY
    if flag == FLAG_STALE:
        return STALE
    return None


def effective(elapsed: int, duration: int, suspended: int) -> int:
    """Running seconds in the first ``elapsed`` of ``duration``.

    sacct gives total suspension but not when it happened, so it is spread
    evenly. This is integral and telescopes: any split of a window still sums
    to the same total.
    """
    return elapsed - suspended * elapsed // duration


class Part(NamedTuple):
    """One allocation's contribution to a window."""

    row: Row
    roles: Roles
    outcome: str
    start: int  # clipped to the window
    stop: int
    seconds: int  # stop - start, less suspension
    real_stop: int  # unclipped; "now" while running
    problem: Optional[str]
    starts_here: bool
    ends_here: bool
    cpus: Optional[int]  # what counts: the share of the filtered nodes, else all
    gpus: Optional[int]
    mem: Optional[int]
    shares: Optional[List[tuple]]  # (node, cpus, gpus, mem), only when nodes matter


class Span(NamedTuple):
    """A window resolved against what is known."""

    low: Optional[int]  # what was asked for
    high: int
    start: int  # what is drawn
    end: int
    as_of: int


def resolve(window: Optional[Window], as_of: int, first: Optional[int]) -> Span:
    window = window or Window()
    high = as_of if window.end is None else int(window.end)
    end = min(high, as_of)
    if window.start is None:
        start = end if first is None else min(first, end)
        return Span(None, high, start, end, as_of)
    low = int(window.start)
    if low >= high:
        raise ValueError("the window is empty")
    return Span(low, high, min(low, end), end, as_of)


class Selection:
    """What a filter selects, with the lookups needed to evaluate it row by row."""

    def __init__(self, dims: Dict[int, Roles], nodesets: Dict[int, str], where: Filter,
                 need_nodes: bool = False) -> None:
        self.dims = dims
        self.where = where
        self.dim_ids = matching_dims(dims, where)
        self.need_nodes = need_nodes or where.node is not None
        self._nodesets = nodesets
        self._hosts = {}  # type: Dict[int, List[Tuple[int, str]]]
        self._host_counts = {}  # type: Dict[int, int]

    def _hosts_of(self, set_id: int) -> List[Tuple[int, str]]:
        hosts = self._hosts.get(set_id)
        if hosts is None:
            text = self._nodesets.get(set_id, "")
            names = expand_hostlist(text, _NODE_LIMIT) or [NONE]
            self._host_counts[set_id] = len(names)
            wanted = self.where.node
            hosts = [(i, n) for i, n in enumerate(names) if wanted is None or n in wanted]
            self._hosts[set_id] = hosts
        return hosts

    def _shares(self, set_id: int, hosts: List[Tuple[int, str]], cpus: Optional[int],
                gpus: Optional[int], mem: Optional[int]) -> List[tuple]:
        count = self._host_counts[set_id]
        if count == 1:
            return [(hosts[0][1], cpus, gpus, mem)]
        shares = []
        for index, name in hosts:
            share = [name]  # type: List[Any]
            for amount in (cpus, gpus, mem):
                share.append(None if amount is None
                             else amount // count + (1 if index < amount % count else 0))
            shares.append(tuple(share))
        return shares

    def parts(self, rows: Iterable[Row], span: Span) -> Iterator[Part]:
        low, high = span.low, span.high
        window_start, window_end, as_of = span.start, span.end, span.as_of
        outcomes, node_filter = self.where.outcome, self.where.node
        cache = {}  # type: Dict[Tuple[int, bool, int], Tuple[Roles, str, Optional[str]]]
        for row in rows:
            dim, set_id, begun, end, suspended = row[2], row[3], row[6], row[7], row[8]
            cpus, gpus, mem, flag = row[9], row[10], row[11], row[13]
            kind = (dim, end is not None, flag)
            found = cache.get(kind)
            if found is None:
                roles = self.dims.get(dim)
                if roles is None:
                    continue  # written by another process after we looked
                problem = unresolved(roles.state, end, flag)
                found = roles, outcome_of(roles.state, end is not None), problem
                if problem:
                    found = roles, "other", problem
                cache[kind] = found
            roles, outcome, problem = found
            if outcomes is not None and outcome not in outcomes:
                continue

            stop = begun if problem else (end if end is not None else as_of)
            if stop < begun:
                stop = begun
            starts_here = begun < high and (low is None or begun >= low)
            p = begun if begun > window_start else window_start
            q = stop if stop < window_end else window_end
            if q > p:
                if suspended:
                    duration = stop - begun
                    held = min(suspended, duration)
                    seconds = (effective(q - begun, duration, held)
                               - effective(p - begun, duration, held))
                else:
                    seconds = q - p
            else:
                # No time in the window. It still counts if it began here and took no time,
                # or if it is a broken record that claims to be running through the window.
                if not problem and not (starts_here and stop == begun):
                    continue
                seconds = 0
                q = p

            shares = None
            if self.need_nodes:
                hosts = self._hosts_of(set_id)
                if not hosts:
                    continue
                shares = self._shares(set_id, hosts, cpus, gpus, mem)
                if node_filter is not None:
                    cpus = None if cpus is None else sum(s[1] for s in shares)
                    gpus = None if gpus is None else sum(s[2] for s in shares)
                    mem = None if mem is None else sum(s[3] for s in shares)

            ends_here = (end is not None and end <= high and (low is None or end >= low))
            yield Part(row, roles, outcome, p, q, seconds, stop, problem, starts_here,
                       ends_here, cpus, gpus, mem, shares)

    def idle(self, counts: Iterable[Tuple[int, bool, int]]) -> Iterator[Tuple[Roles, str, bool, int]]:
        """(roles, outcome, ended, rows) for records that never got an allocation."""
        if self.where.node is not None:
            return
        for dim, ended, count in counts:
            roles = self.dims.get(dim)
            if roles is None or (self.dim_ids is not None and dim not in self.dim_ids):
                continue
            if not ended and roles.state not in PENDING_STATES:
                continue
            outcome = outcome_of(roles.state, True) if ended else "other"
            if self.where.outcome is not None and outcome not in self.where.outcome:
                continue
            yield roles, outcome, ended, count
