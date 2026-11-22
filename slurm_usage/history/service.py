"""History: the object a front end talks to."""

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, List, Optional, Sequence

from ..model import CapacityCheck, Invoker, Status
from . import detail as detail_module
from .aggregate import DEFAULT_GROUPS, build_report
from .efficiency import load_efficiency
from .environment import capacity, probe
from .errors import HistoryError, Note, ProgressFn, canceller
from .ingest import Ingestor
from .listing import JobPage, JobRow, list_attempts, list_jobs
from .report import Report
from . import events as events_module
from . import runtypes
from . import rollup as rollup_module
from .select import Filter
from .store import cache_name, open_store
from .timeline import Window

_FRESH = 600  # a window reaching "now" is complete if refreshed this recently


@dataclass
class HistoryInfo:
    accounting: str
    visibility: str
    cluster: Optional[str]
    invoker: Invoker
    gpu_tracked: Optional[bool]
    covered_from: Optional[int] = None  # 0 = all of history
    as_of: Optional[int] = None
    cache_path: Optional[str] = None  # None = this process only
    notes: List[Note] = field(default_factory=list)


class History:
    """Windowed, filtered usage history as far as Slurm shows it to the invoker.

    refresh() and load_efficiency() talk to Slurm and can take a while: run them
    off the UI thread. query(), jobs() and attempts() read only the cache and
    may be called meanwhile. detail(), submit_line() and batch_script() ask
    Slurm about one job and cache nothing.
    """

    def __init__(self, runner: Any = None, cache_dir: Optional[str] = None,
                 persist: bool = True, clock: Callable[[], float] = time.time) -> None:
        if runner is None:
            from ..runner import Runner
            runner = Runner()
        self.runner = runner
        self._cache_dir, self._persist, self._clock = cache_dir, persist, clock
        self._opening = threading.Lock()
        self._working = threading.Lock()  # one ingest at a time
        self._env = None  # type: Any
        self._store = None  # type: Any
        self._ingest = None  # type: Optional[Ingestor]
        self._failure = None  # type: Optional[HistoryError]

    def _open(self) -> None:
        with self._opening:
            if self._env is None:
                env = probe(self.runner)
                if env.accounting != Status.UNAVAILABLE:
                    self._store = open_store(self._cache_dir, cache_name(*env.identity),
                                             self._persist)
                    self._ingest = Ingestor(self.runner, self._store, env)
                self._env = env

    def _require(self) -> None:
        self._open()
        if self._store is None:
            raise HistoryError(Status.UNAVAILABLE, self._env.accounting_detail)

    def _as_of(self) -> int:
        self._require()
        known = self._ingest.coverage()
        if known is None:
            if self._failure is not None:
                raise HistoryError(self._failure.status, self._failure.message)
            raise HistoryError(Status.UNKNOWN, "history has not been loaded yet")
        return known[1]

    def _gpu_tracked(self) -> bool:
        # Recorded allocations outrank a configuration that may have changed since.
        return bool(self._env.gpu_tracked) or self._store.has_gpus()

    def _notes(self) -> List[Note]:
        notes = list(self._env.notes)
        if self._store is None:
            return notes
        notes.extend(self._store.notes)
        if not self._gpu_tracked():
            notes.append(Note("no_gpu_tres", "accounting does not track GPUs (gres/gpu): "
                                             "GPU usage is unknown, not zero")
                         if self._env.gpu_tracked is False else
                         Note("gpu_unknown", "no GPU allocation on record and cannot tell "
                                             "whether GPUs are tracked"))
        if self._ingest.malformed:
            notes.append(Note("malformed", "%d unreadable sacct records were skipped"
                              % self._ingest.malformed))
        if self._failure is not None:
            notes.append(Note("refresh_failed", "last refresh failed: %s"
                              % self._failure.message))
        return notes

    def info(self) -> HistoryInfo:
        self._open()
        env = self._env
        info = HistoryInfo(env.accounting, env.visibility, env.cluster, replace(env.invoker),
                           env.gpu_tracked)
        if self._store is not None:
            known = self._ingest.coverage()
            if known:
                info.covered_from, info.as_of = known
                info.accounting = Status.OK
            if self._failure is not None:
                info.accounting = self._failure.status
            info.gpu_tracked = self._gpu_tracked()
            info.cache_path = self._store.path
        info.notes = self._notes()
        return info

    def refresh(self, window: Optional[Window] = None, progress: ProgressFn = None,
                cancel: Any = None, update: bool = True) -> HistoryInfo:
        """Load the window (None = all history). update=False only fills in older history.

        Raises HistoryError or Cancelled; pieces already loaded remain usable.
        """
        self._require()
        low = None if window is None else window.start
        with self._working:
            try:
                self._ingest.update(low, int(self._clock()), update, progress, canceller(cancel))
            except HistoryError as error:
                self._failure = error
                raise
            self._failure = None
        return self.info()

    def query(self, window: Optional[Window] = None, where: Optional[Filter] = None,
              group_by: Sequence[Any] = DEFAULT_GROUPS, series_by: Optional[str] = None,
              series_metric: str = "gpu", lanes_by: Optional[str] = None,
              lane_resolution: Optional[int] = None, efficiency: bool = False,
              progress: ProgressFn = None, cancel: Any = None) -> Report:
        """Everything for one window and filter. Raises rather than showing zeros when unloaded."""
        as_of = self._as_of()
        window = window or Window()
        report = build_report(self._store, as_of, window, where, group_by, series_by,
                              series_metric, lanes_by, lane_resolution, efficiency,
                              self._gpu_tracked(), progress, cancel)
        report.complete = self._is_complete(window)
        report.floor = rollup_module.capacity_floor(self._store, self._ingest.coverage()[0]
                                                    if self._ingest.coverage() else None)
        report.notes = self._notes()
        total = report.total
        if total.unresolved:
            report.notes.append(Note("unresolved", "%d records without an end are not really "
                                     "running and are not counted up to now" % total.unresolved))
        if total.suspension_estimated:
            report.notes.append(Note("suspension", "suspended time is subtracted evenly over "
                                                   "each job: its timing is not recorded"))
        if not report.complete:
            report.notes.append(Note("incomplete", "this window is not fully loaded"))
        return report

    def _is_complete(self, window: Window) -> bool:
        start, to = self._ingest.coverage()
        covered = start == 0 or (window.start is not None and start <= window.start)
        now = self._clock()
        until = now if window.end is None else min(now, window.end)
        return covered and until - to <= _FRESH

    def jobs(self, window: Optional[Window] = None, where: Optional[Filter] = None,
             by: str = "job", sort: str = "gpu", descending: bool = True, limit: int = 200,
             offset: int = 0, cancel: Any = None,
             run_type: Optional[Any] = None) -> JobPage:
        """run_type=(by, key) drills into one run_types() row."""
        window = window or Window()
        as_of = self._as_of()
        page = list_jobs(self._store, as_of, window, where, by, sort, descending, limit, offset,
                         cancel, run_type)
        page.complete = self._is_complete(window)
        return page

    def attempts(self, job_id: object) -> List[JobRow]:
        as_of = self._as_of()
        return list_attempts(self._store, as_of, detail_module.check_job_id(job_id))

    def load_efficiency(self, window: Optional[Window] = None, where: Optional[Filter] = None,
                        progress: ProgressFn = None, cancel: Any = None) -> int:
        """Read step usage for any user or the whole cluster; then query(efficiency=True)."""
        as_of = self._as_of()
        with self._working:
            return load_efficiency(self.runner, self._store, self._env, as_of, window,
                                   where or Filter(), progress, canceller(cancel))

    def detail(self, job_id: object, cancel: Any = None) -> detail_module.JobDetail:
        job = detail_module.check_job_id(job_id)
        self._open()
        if self._env.accounting == Status.UNAVAILABLE:
            return detail_module.JobDetail(job, Status.UNAVAILABLE, self._env.accounting_detail)
        flags = {row[0]: row[13] for row in self._store.by_job(job)}
        return detail_module.fetch_detail(self.runner, self._env, job, int(self._clock()),
                                          flags, canceller(cancel))

    def submit_line(self, job_id: object, key: Optional[str] = None) -> detail_module.Text:
        """On an explicit request only. ``key`` picks an attempt (JobRow.key)."""
        self._open()
        if not self._env.supported(("SubmitLine",)):
            return detail_module.Text(Status.UNAVAILABLE, message="this sacct has no submit lines")
        return detail_module.fetch_submit_line(self.runner, job_id, key)

    def batch_script(self, job_id: object) -> detail_module.Text:
        """On an explicit request only. Slurm decides who may read it."""
        return detail_module.fetch_batch_script(self.runner, job_id)

    def run_types(self, window: Optional[Window] = None, where: Optional[Filter] = None,
                  by: str = "gpu_count", metric: str = "gpu",
                  cancel: Any = None) -> List[runtypes.RunType]:
        """Usage per kind of run: by gpu_count, name_family, account, qos, partition,
        outcome or node_count. Drill in with jobs(run_type=(by, row.key))."""
        return runtypes.run_types(self._store, self._as_of(), window, where, by, metric,
                                  self._gpu_tracked(), cancel)

    def events(self, window: Optional[Window] = None, where: Optional[Filter] = None,
               cancel: Any = None) -> List[events_module.Event]:
        """Notable events in the window, judged against this selection's own earlier history."""
        as_of = self._as_of()
        window = window or Window()
        covered_from, _ = self._ingest.coverage()
        found = events_module.find_events(self._store, as_of, window, where, covered_from,
                                          cancel)
        report = self.query(window, where, group_by=(), cancel=cancel)
        days = [(b.start, b.end, b.value) for b in report.buckets("gpu", "day") or ()]
        length = report.end - report.start
        # baselines only from fully loaded history, never from ingest's overlap
        prior = (window.start is not None and report.start > 0 and length > 0
                 and events_module.prior_loaded(covered_from, report.start, length))
        prior_best = None
        if prior:
            before = self.query(Window(None, report.start), where, group_by=(), cancel=cancel)
            if before.end > before.start:
                prior_best = max([b.value for b in before.buckets("gpu", "day") or ()] or [0])
        previous = None
        if prior:
            previous = self.query(Window(report.start - length, report.start), where,
                                  group_by=(), cancel=cancel).total.amount("gpu")
        return events_module.add_period_events(found, days, prior_best,
                                               report.total.amount("gpu"), previous, window,
                                               where or Filter(), report.start, report.end)

    def build_rollup(self, progress: ProgressFn = None, cancel: Any = None) -> int:
        """Update the hourly rollup behind overview(). Call after refresh(); off the UI thread."""
        as_of = self._as_of()
        with self._working:
            return rollup_module.build(self._store, as_of, progress, canceller(cancel))

    def overview(self, gpu_capacity: Optional[int] = None, cpu_capacity: Optional[int] = None,
                 ranges: Any = rollup_module.RANGES,
                 metric: str = "gpu") -> rollup_module.Overview:
        """Day/week/month/year/all from the rollup, each against the previous equal period.

        Pass live capacity (not slurmdbd's). Reads only the cache: fast.
        """
        as_of = self._as_of()
        start, _ = self._ingest.coverage()
        result = rollup_module.overview(self._store, as_of, start, self._env.visibility,
                                        gpu_capacity, cpu_capacity, ranges, metric)
        result.notes = self._notes()
        return result

    def capacity(self) -> List[CapacityCheck]:
        self._open()
        return capacity(self.runner, self._env.cluster)

    def rebuild(self) -> None:
        self._require()
        with self._working:
            self._store.clear()
            self._failure = None

    def close(self) -> None:
        if self._store is not None:
            self._store.close()

    def __enter__(self) -> "History":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
