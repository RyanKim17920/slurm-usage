"""Lazy, numeric-only step ingest for CPU, memory and GPU efficiency.

Step queries cost an order of magnitude more than allocation queries, so they
run only when asked, only where a finished attempt has not been read yet.
"""

from typing import Any, Callable, Dict, Optional, Tuple

from ..model import Status
from ..parse import parse_duration, parse_time, parse_usage_tres
from .commands import bulk_sacct, stamp, stream_lines
from .environment import Environment
from .errors import HistoryError, Progress, ProgressFn
from .ingest import OVERLAP
from .records import NONE, new_delimiter, read_rows, record_key
from .select import Filter, Selection, resolve
from .timeline import DAY, HOUR, Window

USAGE_FIELDS = ("DBIndex", "JobIDRaw", "Cluster", "Submit", "End", "State", "TotalCPU",
                "TRESUsageInTot")
FEW_LINES, MANY_LINES = 20000, 60000
FIRST_PIECE, SMALLEST_PIECE, LARGEST_PIECE = DAY, HOUR, 7 * DAY
TIMEOUT = 300.0
_RSS_SANE, _UTIL_SANE = 2 ** 53, 100 * 4096


class AttemptUsage:
    """Reduces the rows of one attempt to three numbers; absent stays None."""

    __slots__ = ("job_cpu", "step_cpu", "rss", "util")

    def __init__(self, job_cpu: Optional[str] = None) -> None:
        self.job_cpu = parse_duration(job_cpu)
        self.step_cpu = 0.0
        self.rss = None  # type: Optional[int]
        self.util = None  # type: Optional[float]

    def step(self, total_cpu: Optional[str], usage_total: Optional[str]) -> None:
        self.step_cpu += parse_duration(total_cpu) or 0
        usage = parse_usage_tres(usage_total)
        rss = usage.get("mem")
        sampled = bool(rss) and rss < _RSS_SANE  # zero means the sampler never ran
        if sampled and (self.rss is None or rss > self.rss):
            self.rss = int(rss)
        util = usage.get("gres/gpuutil")
        if util is None or util > _UTIL_SANE or (not util and not sampled):
            return
        # Summed over GPUs and tasks, and the peak sample. Steps may overlap or follow
        # each other, so the largest is the only figure that is safe to claim.
        if self.util is None or util > self.util:
            self.util = float(util)

    def result(self) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        cpu = self.job_cpu or self.step_cpu
        return int(cpu * 1000) if cpu else None, self.rss, self.util


def load_efficiency(runner: Any, store: Any, env: Environment, as_of: int,
                    window: Optional[Window], where: Filter, progress: ProgressFn,
                    cancelled: Callable[[], bool]) -> int:
    """Read the steps of finished attempts in the window. Returns how many needed it."""
    fields = env.supported(USAGE_FIELDS)
    if "TRESUsageInTot" not in fields and "TotalCPU" not in fields:
        raise HistoryError(Status.UNAVAILABLE, "this sacct records no step usage")
    chosen = Selection(store.dims(), {}, where)
    span = resolve(window, as_of, None)
    todo = [row for row in store.unchecked(span.low, span.high)
            if chosen.dim_ids is None or row[1] in chosen.dim_ids]
    if not todo:
        return 0

    scope = []
    uids = {chosen.dims[row[1]].uid for row in todo}
    if where.user and len(where.user) == 1 and len(uids) == 1 and None not in uids:
        scope = ["-u", str(next(iter(uids)))]
    if where.account and len(where.account) == 1:
        account = next(iter(where.account))
        if account != NONE and not account.startswith("-") and account.isascii():
            scope += ["-A", account]

    piece, at, finished = FIRST_PIECE, 0, set()
    while at < len(todo):
        begin, upto = todo[at][3], at
        while upto < len(todo) and todo[upto][3] < begin + piece:
            upto += 1
        wanted = [row[0] for row in todo[at:upto] if row[0] not in finished]
        if wanted:
            try:
                lines = _fetch(runner, store, env, fields, begin, todo[upto - 1][3], as_of,
                               scope, cancelled, finished)
            except HistoryError as error:
                if error.status != Status.TIMEOUT or piece <= SMALLEST_PIECE:
                    raise
                piece = max(SMALLEST_PIECE, piece // 4)
                continue
            store.mark_checked(wanted)  # nothing to read is an answer too
            if lines > MANY_LINES:
                piece = max(SMALLEST_PIECE, piece // 2)
            elif lines < FEW_LINES:
                piece = min(LARGEST_PIECE, piece * 2)
        at = upto
        if progress:
            progress(Progress("efficiency", at, len(todo)))
    return len(todo)


def _fetch(runner: Any, store: Any, env: Environment, fields: Tuple[str, ...], begin: int,
           end: int, as_of: int, scope: list, cancelled: Callable[[], bool],
           finished: set) -> int:
    separator = new_delimiter()
    until = "now" if end + OVERLAP >= as_of else stamp(end + OVERLAP)
    argv = bulk_sacct(fields, separator, "-S", stamp(begin - OVERLAP), "-E", until, *scope)
    lines = stream_lines(runner, argv, cancelled, TIMEOUT)
    found = []  # (key, cpu ms, rss, util)
    key, job, usage, count = None, "", None, 0

    def close_attempt() -> None:
        if key is not None and usage is not None:
            found.append((key,) + usage.result())
            finished.add(key)

    try:
        for row in read_rows(lines, separator, fields):
            count += 1
            if row is None:
                key = usage = None  # never attach later steps to the wrong attempt
                continue
            raw = row.get("JobIDRaw", "")
            if "." in raw:
                if usage is not None and raw.partition(".")[0] == job:
                    usage.step(row.get("TotalCPU"), row.get("TRESUsageInTot"))
                continue
            close_attempt()
            key, job, usage = None, raw, None
            cluster = row.get("Cluster")
            if cluster and env.cluster and cluster.lower() != env.cluster.lower():
                continue
            if parse_time(row.get("End")) is not None and raw.isdigit():
                key = record_key(row.get("DBIndex"), int(raw), parse_time(row.get("Submit")))
                usage = AttemptUsage(row.get("TotalCPU"))
        close_attempt()
    finally:
        lines.close()
    if found:
        store.put_usage(found)
    return count
