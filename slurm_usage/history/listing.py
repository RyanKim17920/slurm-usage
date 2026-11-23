"""The drill-down lists: jobs in a window, and the attempts of one job."""

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

from .aggregate import usage_ratios
from .errors import Cancelled, canceller
from .records import NONE, UNLIMITED, outcome_of
from .select import Filter, Part, Selection, resolve, unresolved
from .store import ANY, NEWEST, Info, Row
from .timeline import Window

SORTS = ("gpu", "cpu", "mem", "start", "end", "elapsed", "wait", "attempts")


@dataclass
class JobRow:
    """One attempt, or when grouped by job its latest attempt in the window plus
    the sums over those in the window. History.attempts() has every attempt.
    """

    key: str
    job_id: str  # numeric
    display_id: str
    name: str
    user: str
    uid: Optional[int]
    account: str
    qos: str
    partition: str
    state: str
    outcome: str
    nodes: str  # hostlist
    submit: Optional[int]
    eligible: Optional[int]
    start: Optional[int]
    end: Optional[int]  # None while running
    elapsed: Optional[int]  # seconds, less suspension
    cpus: Optional[int]
    gpus: Optional[int]
    mem: Optional[int]  # bytes
    time_limit: Optional[int]  # seconds
    time_unlimited: bool
    restarts: Optional[int]  # Slurm's counter for this attempt; never add them up
    problem: Optional[str]  # runaway, stale, missing_end: not counted up to now
    cpu_efficiency: Optional[float] = None
    mem_efficiency: Optional[float] = None
    gpu_peak_utilization: Optional[float] = None
    # inside the window
    gpu_seconds: Optional[int] = None
    cpu_seconds: Optional[int] = None
    mem_byte_seconds: Optional[int] = None
    attempts: int = 1
    first_start: Optional[int] = None

    @property
    def wait(self) -> Optional[int]:
        if self.start is None or self.eligible is None or self.eligible > self.start:
            return None
        return self.start - self.eligible

    @property
    def time_limit_used(self) -> Optional[float]:
        if self.elapsed is None or not self.time_limit:
            return None
        return self.elapsed / self.time_limit


@dataclass
class JobPage:
    rows: List[JobRow] = field(default_factory=list)
    total: Optional[int] = 0  # matches before limit and offset; None: not counted
    complete: bool = True


class _Entry(NamedTuple):
    part: Part
    gpu: Optional[int]
    cpu: Optional[int]
    mem: Optional[int]
    attempts: int
    first_start: int


def _plus(a: Optional[int], b: Optional[int]) -> Optional[int]:
    return None if a is None or b is None else a + b


def _times(amount: Optional[int], seconds: int) -> Optional[int]:
    return None if amount is None else amount * seconds


def _sort_value(entry: _Entry, sort: str) -> Optional[int]:
    if sort in ("gpu", "cpu", "mem", "attempts"):
        return getattr(entry, sort)
    row = entry.part.row
    if sort == "start":
        return row[6]
    if sort == "end":
        return row[7] if row[7] is not None else entry.part.real_stop
    if sort == "elapsed":
        return entry.part.real_stop - row[6] - row[8]
    return row[6] - row[5] if row[5] is not None and row[5] <= row[6] else None


def make_row(row: Row, roles: Any, info: Optional[Info], nodes: str, outcome: str,
             problem: Optional[str], real_stop: Optional[int]) -> JobRow:
    begun = row[6]
    elapsed = None
    if begun is not None and real_stop is not None and not problem:
        elapsed = max(0, real_stop - begun - row[8])
    limit = row[12]
    cpu, rss, gpu = usage_ratios(row)
    return JobRow(
        key=row[0], job_id=str(row[1]), display_id=info.label if info else str(row[1]),
        name=info.name if info else "", user=roles.user if roles else NONE,
        uid=roles.uid if roles else None, account=roles.account if roles else NONE,
        qos=roles.qos if roles else NONE, partition=roles.partition if roles else NONE,
        state=roles.state if roles else "UNKNOWN", outcome=outcome, nodes=nodes,
        submit=row[4], eligible=row[5], start=begun, end=row[7], elapsed=elapsed,
        cpus=row[9], gpus=row[10], mem=row[11],
        time_limit=None if limit == UNLIMITED else limit, time_unlimited=limit == UNLIMITED,
        restarts=info.restarts if info else None, problem=problem,
        cpu_efficiency=cpu, mem_efficiency=rss, gpu_peak_utilization=gpu, first_start=begun)


def _matches(part: Any, needle: str) -> bool:
    """A job id, name or user containing ``needle`` (already lower case)."""
    row = part.row
    return (needle in str(row[1]) or needle in (row[18] or "").lower()
            or needle in (part.roles.user or "").lower())


def list_jobs(store: Any, as_of: int, window: Optional[Window] = None,
              where: Optional[Filter] = None, by: str = "job", sort: str = "gpu",
              descending: bool = True, limit: int = 200, offset: int = 0,
              cancel: Any = None, run_type: Optional[Tuple[str, str]] = None,
              match: Optional[str] = None, count: bool = True) -> JobPage:
    """Jobs (or attempts) with usage in the window. ``nodes`` filters count only their share.

    run_type=(by, key) keeps only attempts of one types.run_types() row. count=False
    may leave page.total None (not counted): the newest attempts are then read newest
    first and no further than the page.
    """
    from .runtypes import TYPE_ROLES, type_of
    if run_type is not None and run_type[0] not in TYPE_ROLES:
        raise ValueError("unknown run type: %s" % run_type[0])
    if sort not in SORTS:
        raise ValueError("unknown sort: %s" % sort)
    if by not in ("job", "attempt"):
        raise ValueError("by is job or attempt")
    window, cancelled = window or Window(), canceller(cancel)
    page = JobPage()
    needle = (match or "").strip().lower()

    with store.lock:
        nodesets = store.nodesets()
        chosen = Selection(store.dims(), nodesets, where or Filter())
        first = store.first_start(chosen.dim_ids) if window.start is None else None
        span = resolve(window, as_of, first)
        repeated = store.repeated_jobs() if by == "job" else set()

        # what SQLite can rule out for a run type (the check below stays: this only
        # narrows); for finished rows, which are never unresolved, it is exact
        dim_ids, gpus, finished_ids = chosen.dim_ids, ANY, None  # type: Any, Any, Any
        sql_type = run_type is None
        if run_type is not None:
            kind, key = run_type
            dims = chosen.dims
            pool = dims if dim_ids is None else dim_ids
            if kind in ("account", "qos", "partition"):
                dim_ids = {d for d in pool if d in dims and getattr(dims[d], kind) == key}
                sql_type = True
            elif kind == "outcome":
                dim_ids = {d for d in pool if d in dims and key in (
                    outcome_of(dims[d].state, True), outcome_of(dims[d].state, False))}
                finished_ids = {d for d in dim_ids if outcome_of(dims[d].state, True) == key}
                sql_type = True
            elif kind == "gpu_count" and (key == "unknown" or key.isdigit()):
                gpus = None if key == "unknown" else int(key)
                sql_type = True
            elif kind == "name_family" and getattr(store, "has_families", False):
                sql_type = True
        narrow = {} if gpus is ANY else {"gpus": gpus}  # type: Dict[str, Any]
        if run_type is not None and run_type[0] == "name_family" and sql_type:
            narrow["family"] = run_type[1]
        extra = run_type is not None or bool(needle)
        newest = (not count and by == "attempt" and sort == "start" and descending
                  and not chosen.need_nodes)
        # Finished jobs with one attempt, sorted by a column SQLite can compute, come
        # from SQLite already ranked; the rest (unfinished rows, requeued jobs) from here.
        quick = (sort in ("gpu", "cpu", "start") and sql_type and not newest
                 and not needle and not chosen.need_nodes
                 and callable(getattr(store, "top_finished", None)))
        ranked = []  # type: List[_Entry]
        if quick:
            dims, outcomes = chosen.dims, chosen.where.outcome
            picked = dim_ids if finished_ids is None else finished_ids
            if outcomes is not None:
                picked = {d for d in (picked if picked is not None else dims)
                          if d in dims and outcome_of(dims[d].state, True) in outcomes}
            counted, keys = store.top_finished(span.low, span.high, span.start, span.end,
                                               picked, sort, descending, offset + limit,
                                               by == "job", **narrow)
            page.total += counted
            for part in chosen.parts(store.rows_of(keys, extra), span):
                s = part.seconds
                ranked.append(_Entry(part, _times(part.gpus, s), _times(part.cpus, s),
                                     _times(part.mem, s), 1, part.row[6]))

        def entries() -> Iterator[_Entry]:
            merged = {}  # only jobs with several attempts are held
            # measured usage is read for the rows shown only (usage_of below)
            if quick:
                rows = store.scan_special(span.low, span.high, dim_ids, extra, by == "job",
                                          **narrow)
            else:
                more = dict(narrow)
                if needle and callable(getattr(store, "match_sql", None)):
                    more["match"] = needle  # SQLite reads only the rows that may match
                rows = store.scan(span.low, span.high, dim_ids, usage=False,
                                  ordered=NEWEST if newest else False, extra=extra, **more)
            try:
                for number, part in enumerate(chosen.parts(rows, span)):
                    if not number % 1024 and cancelled():
                        raise Cancelled()
                    if run_type is not None and (part.problem or type_of(
                            run_type[0], part, part.row[18] or "",
                            nodesets.get(part.row[3], "")) != run_type[1]):
                        continue
                    if needle and not _matches(part, needle):
                        continue
                    s = part.seconds
                    entry = _Entry(part, _times(part.gpus, s), _times(part.cpus, s),
                                   _times(part.mem, s), 1, part.row[6])
                    job = part.row[1]
                    if job in repeated:
                        identity = (job, part.roles.uid)
                        old = merged.get(identity)
                        if old is not None:
                            latest = max(entry, old, key=lambda e: (e.part.row[6],
                                                                    e.part.row[4] or 0))
                            entry = _Entry(latest.part, _plus(old.gpu, entry.gpu),
                                           _plus(old.cpu, entry.cpu), _plus(old.mem, entry.mem),
                                           old.attempts + 1,
                                           min(old.first_start, entry.first_start))
                        merged[identity] = entry
                        continue
                    page.total += 1
                    yield entry
            finally:
                rows.close()
            for entry in merged.values():
                page.total += 1
                yield entry

        def order(entry: _Entry) -> Tuple[float, int, str]:
            value = _sort_value(entry, sort)
            if value is None:  # unknown goes last either way
                value = float("-inf") if descending else float("inf")
            return value, entry.part.row[6], entry.part.row[0]

        pick = heapq.nlargest if descending else heapq.nsmallest
        found = entries()
        if newest:  # already in this order: the first ones are the page
            best = list(itertools.islice(found, offset + limit))[offset:]
            found.close()
            page.total = None  # type: ignore  # not counted
        else:
            if ranked:
                found = itertools.chain(ranked, found)
            best = pick(offset + limit, found, key=order)[offset:]
        info = store.info([entry.part.row[0] for entry in best])
        measured = store.usage_of([entry.part.row[0] for entry in best])
        for entry in best:
            part = entry.part
            row = part.row
            row = row[:14] + measured.get(row[0], (None, None, None, False)) + tuple(row[18:])
            job = make_row(row, part.roles, info.get(part.row[0]),
                           nodesets.get(part.row[3], ""), part.outcome, part.problem,
                           part.real_stop)
            job.gpu_seconds, job.cpu_seconds, job.mem_byte_seconds = entry.gpu, entry.cpu, entry.mem
            job.attempts, job.first_start = entry.attempts, entry.first_start
            page.rows.append(job)
    return page


def list_attempts(store: Any, as_of: int, job_id: str) -> List[JobRow]:
    """Every cached attempt of one job, oldest first, with whole-attempt usage."""
    with store.lock:
        found = store.by_job(job_id)
        if not found:
            return []
        dims, nodesets = store.dims(), store.nodesets()
        info = store.info([row[0] for row in found])
    out = []
    for row in found:
        roles = dims.get(row[2])
        state = roles.state if roles else "UNKNOWN"
        begun, end, problem = row[6], row[7], None
        real_stop = None
        if begun is not None:
            problem = unresolved(state, end, row[13])
            real_stop = begun if problem else max(begun, end if end is not None else as_of)
        if problem or (begun is None and end is None):
            outcome = "other"
        else:
            outcome = outcome_of(state, end is not None)
        job = make_row(row, roles, info.get(row[0]), nodesets.get(row[3], ""), outcome, problem,
                       real_stop)
        if job.elapsed is not None:
            job.gpu_seconds = _times(job.gpus, job.elapsed)
            job.cpu_seconds = _times(job.cpus, job.elapsed)
            job.mem_byte_seconds = _times(job.mem, job.elapsed)
        out.append(job)
    return out
