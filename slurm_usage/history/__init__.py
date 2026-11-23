"""Usage history from Slurm accounting, as far as the invoker may see it.

    history = History()
    history.refresh(Window.last(30 * 86400), progress=show, cancel=event)
    report = history.query(Window.last(7 * 86400), Filter(user="alice"),
                           group_by=[("qos",), ("user", "qos")])
    report.total.gpu_hours, report.ranked("qos"), report.heatmap("gpu")

Modules: records (parsing), store (private cache), ingest (bounded sync),
select/stats/series/aggregate (one-pass aggregation), report, listing
(drill-down), efficiency (lazy step ingest), detail (one job, uncached),
environment (what is recorded and visible), service (History).

Unknown is None, never 0. Usage is integer unit-seconds (memory in byte-seconds),
so that groupings and buckets sum exactly.
"""

from ..model import CapacityCheck, Status, Step
from .errors import Cancelled, HistoryError, Note, Progress
from .records import NONE, OUTCOMES
from .report import Lane, Report, Utilization
from .runtypes import TYPE_ROLES, RunType, name_family
from .rollup import RANGES, AccountSummary, Amounts, Overview, RangeSummary
from .select import MISSING_END, ROLES, RUNAWAY, STALE, Filter
from .service import History, HistoryInfo
from .stats import METRICS, Efficiency, Stats
from .timeline import Bucket, Window

__all__ = [
    "Event", "KINDS", "RunType", "TYPE_ROLES", "name_family",
    "AccountSummary", "Amounts", "Overview", "RANGES", "RangeSummary", "AttemptDetail", "Bucket", "Cancelled", "CapacityCheck", "Efficiency", "Filter", "History",
    "HistoryError", "HistoryInfo", "JobDetail", "JobPage", "JobRow", "Lane", "LogPath",
    "METRICS", "MISSING_END", "NONE", "Note", "OUTCOMES", "Progress", "ROLES", "RUNAWAY",
    "Report", "SORTS", "STALE", "Stats", "Status", "Step", "Text", "Utilization", "Window",
]


# imported when first asked for: a summary run never needs them
_LATER = {"AttemptDetail": "detail", "JobDetail": "detail", "Text": "detail",
          "SORTS": "listing", "JobPage": "listing", "JobRow": "listing", "LogPath": "paths",
          "KINDS": "events", "Event": "events"}


def __getattr__(name: str):  # PEP 562
    module = _LATER.get(name)
    if module is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    import importlib
    value = getattr(importlib.import_module("." + module, __name__), name)
    globals()[name] = value
    return value
