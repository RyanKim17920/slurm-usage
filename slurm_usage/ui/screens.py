"""Every screen: a function from (Context, View) to a list of panes.

Screens never block: data comes from ``Context.live``/``Context.hist``, which
return loader entries that may still be pending (drawn as ``?``).
"""

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..live import applicable_limits, group_by
from ..model import Status
from . import data as hd
from . import fmt
from .canvas import BAD, DIM, GOOD, HEAD, PLAIN, RESOURCE, USER, WARN
from .loader import FAILED, Entry
from .widgets import Column, Line, Pane, Table, Target, text_line, wrap

TABS = (("overview", "Overview", "Ovw", "o"), ("me", "Me", "Me", "m"),
        ("nodes", "Nodes", "Nodes", "n"), ("queue", "Queue", "Queue", "j"),
        ("people", "People", "Ppl", "p"), ("roles", "Roles", "Roles", "a"),
        ("timeline", "Timeline", "Time", "t"))
ROLE_GROUPS = ("account", "qos", "partition")
LANE_GROUPS = ("user", "account", "qos", "partition")


class View:
    """One entry of the navigation stack."""

    def __init__(self, kind: str, params: Optional[Dict[str, Any]] = None,
                 title: str = "") -> None:
        self.kind = kind
        self.params = dict(params or {})
        self.title = title or kind.capitalize()
        self.panes = {}  # type: Dict[str, Any]  # pane id -> PaneState
        self.focus = 0
        self.extra = {}  # type: Dict[str, Any]


class Context:
    """What a screen may read while being built."""

    def __init__(self, app: Any, width: int) -> None:
        self.app = app
        self.snap = app.snapshot
        self.data = app.data
        self.range = app.range
        self.width = width
        self.ascii = app.ascii
        self.now = time.time()
        self.collector = app.collector

    def hist(self, key: Tuple[Any, ...], fn: Callable[[], Any]) -> Entry:
        return self.app.loader.get(("h",) + key, fn, generation=self.data.generation)

    def live(self, key: Tuple[Any, ...], fn: Callable[[], Any]) -> Entry:
        return self.app.loader.get(("l",) + key, fn, generation=self.app.snap_generation)

    def once(self, key: Tuple[Any, ...], fn: Callable[[], Any]) -> Entry:
        """Fetched on request and kept until refresh."""
        return self.app.loader.get(("o",) + key, fn, generation=self.app.refreshes)


# -- helpers -------------------------------------------------------------------

def value(entry: Optional[Entry]) -> Any:
    return None if entry is None else entry.value


def missing(entry: Optional[Entry]) -> str:
    if entry is None:
        return fmt.NOT_YET
    if entry.state == FAILED and entry.value is None:
        return fmt.placeholder(entry.status)
    return fmt.NOT_YET


def eff_cell(eff: Any) -> Tuple[str, str]:
    """'54%·80%': efficiency over coverage of the allocation."""
    if eff is None:
        return fmt.NA, DIM
    val, cover = eff
    if val is None:  # nothing measured yet over an allocation: not sampled; no allocation: n/a
        return (fmt.NOT_YET if cover == 0 else fmt.NOT_APPLICABLE), DIM
    style = GOOD if val >= 0.7 else WARN if val >= 0.3 else BAD
    text = fmt.pct(val)
    if cover is not None and cover < 0.995:
        text += "·" + fmt.pct(cover)
    return text, style


def delta_cell(now: Optional[float], before: Optional[float],
               loaded: bool = True) -> Tuple[str, str]:
    if not loaded:
        return fmt.NOT_YET, DIM
    text = fmt.delta(now, before)
    return text, DIM if text in fmt.PLACEHOLDERS else PLAIN


def filt_key(params: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((k, v) for k, v in params.items() if v is not None))


def make_filter(params: Dict[str, Any]) -> Any:
    keys = ("user", "account", "qos", "partition")
    return hd.where(**{k: params.get(k) for k in keys})


def snap_jobs(ctx: Context) -> List[Any]:
    return [job for job in (ctx.snap.jobs or ()) if job.active] if ctx.snap else []


def status_line(label: str, entry: Optional[Entry]) -> Line:
    return text_line("%s: %s %s" % (label, missing(entry),
                                    entry.message if entry is not None else ""), DIM)


def capacity(ctx: Context) -> Tuple[Optional[int], Optional[int]]:
    if ctx.snap is None:
        return None, None
    return ctx.snap.totals.gpus.total, ctx.snap.totals.cpus.total


def trend_entry(ctx: Context, range_name: str, params: Dict[str, Any]) -> Entry:
    gpus, cpus = capacity(ctx)
    filt = make_filter(params)
    return ctx.hist(("trend", range_name, filt_key(params), gpus, cpus),
                    lambda: ctx.data.trend(range_name, filt, gpus, cpus))


def range_report(ctx: Context, params: Dict[str, Any], group: Sequence[Any] = (),
                 series_by: Optional[str] = None, range_name: Optional[str] = None) -> Entry:
    name = range_name or ctx.range
    filt = make_filter(params)
    groups = tuple(tuple(g) if not isinstance(g, str) else (g,) for g in group)
    return ctx.hist(("report", name, filt_key(params), groups, series_by),
                    lambda: ctx.data.report(name, filt, group_by=groups, series_by=series_by))


def resource_line(label: str, res: Any, ascii_only: bool, width: int) -> Line:
    if res is None or res.total is None:
        return Line([(label.ljust(5), HEAD), (fmt.NA, DIM)])
    parts = [(label.ljust(5), HEAD), (fmt.num(res.total), PLAIN), (" total  ", DIM),
             (fmt.num(res.alloc), GOOD), (" alloc  ", DIM)]
    if res.completing:
        parts += [(fmt.num(res.completing), WARN), (" completing  ", DIM)]
    if res.shared:
        parts += [(fmt.num(res.shared), WARN), (" shared  ", DIM)]
    parts += [(fmt.num(res.free), PLAIN), (" free  ", DIM)]
    if res.unavailable:
        parts += [(fmt.num(res.unavailable), BAD), (" unavail  ", DIM)]
    if res.reserved:
        parts += [(fmt.num(res.reserved), DIM), (" resv  ", DIM)]
    used = sum(fmt.width(t) for t, _ in parts)
    room = width - used - 8
    fraction = (res.alloc + (res.completing or 0)) / res.total if res.total and res.alloc is not None else None
    if room >= 6:
        parts += [(fmt.bar(fraction, min(room, 30), ascii_only), GOOD), (" " + fmt.pct(fraction), PLAIN)]
    return Line(parts)


# -- shared panes ----------------------------------------------------------------

TREND_COLUMNS = (
    Column("Range", 5, "<", 9),
    Column("GPU-h alloc", 11, ">", 9),
    Column("util", 5, ">", 6),
    Column("CPU-h alloc", 11, ">", 8),
    Column("util", 5, ">", 4),
    Column("GPU util*", 9, ">", 5),
    Column("CPU eff", 9, ">", 3),
    Column("ΔGPU", 6, ">", 7),
    Column("ΔCPU", 6, ">", 2),
    Column("trend", 8, ">", 6, flex=True),
)

PERSON_TREND_COLUMNS = (
    Column("Range", 5, "<", 9),
    Column("GPU-h alloc", 11, ">", 9),
    Column("share", 6, ">", 8),
    Column("CPU-h alloc", 11, ">", 7),
    Column("GPU util*", 9, ">", 5),
    Column("CPU eff", 9, ">", 3),
    Column("ΔGPU", 6, ">", 7),
    Column("jobs", 6, ">", 4),
    Column("trend", 8, ">", 6, flex=True),
)


def overview_entry(ctx: Context) -> Entry:
    gpus, cpus = capacity(ctx)
    return ctx.hist(("overview", gpus, cpus), lambda: ctx.data.overview(gpus, cpus))


def cluster_trend(ctx: Context, name: str) -> Tuple[Any, Entry]:
    """TrendRow for the whole cluster: from the rollup when the history layer has one."""
    if ctx.data.has("overview"):
        entry = overview_entry(ctx)
        found = value(entry)
        if found is not None:
            for summary in found.ranges:
                if summary.name == name:
                    row = ctx.data.overview_trend(summary)
                    if row.spark is None and not hasattr(summary, "series_step"):
                        # an older history layer without a series in its rollup
                        spark = ctx.hist(("spark", name), lambda: ctx.data.spark(name))
                        row = row._replace(spark=value(spark))
                    return row, entry
            return None, entry
        if entry.state != FAILED or entry.status not in (Status.ERROR,):
            return None, entry
    entry = trend_entry(ctx, name, {})
    return value(entry), entry


def trend_pane(ctx: Context, params: Dict[str, Any], person: bool = False) -> Pane:
    table = Table(PERSON_TREND_COLUMNS if person else TREND_COLUMNS, ctx.width - 2)
    spark_width = next((size for i, col, size in table.layout if col.name == "trend"), 8)
    lines = []
    incomplete = False
    for name in hd.RANGE_NAMES:
        if person:
            entry = trend_entry(ctx, name, params)
            row = value(entry)
        else:
            row, entry = cluster_trend(ctx, name)
        label = (name, HEAD if name == ctx.range else PLAIN)
        if row is None:
            ph = missing(entry)
            cells = [label] + [ph] * (len(table.columns) - 1)
            lines.append(table.row(cells, key=name))
            continue
        mark = "" if row.complete else "≥"
        incomplete = incomplete or not row.complete
        spark = (fmt.sparkline(row.spark, spark_width, ctx.ascii), GOOD)
        if person:
            cluster = cluster_trend(ctx, name)[0]
            share = None
            if cluster is not None and cluster.gpu_hours and row.gpu_hours is not None:
                share = row.gpu_hours / cluster.gpu_hours
            cells = [label, mark + fmt.hours(row.gpu_hours), fmt.pct(share, 1),
                     mark + fmt.hours(row.cpu_hours), eff_cell(row.gpu_eff),
                     eff_cell(row.cpu_eff), delta_cell(row.gpu_hours, row.gpu_delta, row.previous_loaded),
                     mark + fmt.num(row.jobs), spark]
        else:
            util = [fmt.pct(value) for value in (row.gpu_util, row.cpu_util)]
            util = [u if u in fmt.PLACEHOLDERS else mark + u for u in util]
            cells = [label, mark + fmt.hours(row.gpu_hours), util[0],
                     mark + fmt.hours(row.cpu_hours), util[1],
                     eff_cell(row.gpu_eff), eff_cell(row.cpu_eff),
                     delta_cell(row.gpu_hours, row.gpu_delta, row.previous_loaded),
                     delta_cell(row.cpu_hours, row.cpu_delta, row.previous_loaded), spark]
        lines.append(table.row(cells, key=name))
    note = "GPU util* = peak sample·coverage"
    if incomplete:
        note = "≥ partly loaded · " + note
    elif not person and ctx.width >= 100:
        note = "util = alloc ÷ live capacity · " + note
    return Pane("trend", "Trend" + (" · " + params["user"] if params.get("user") else ""),
                lines, table.header(), grow=False, status=note)


def warnings_pane(ctx: Context, extra: Sequence[str] = ()) -> Optional[Pane]:
    texts = [(w.message, WARN) for w in (ctx.snap.warnings if ctx.snap else ())]
    texts += [(t, WARN) for t in extra]
    if not ctx.app.no_history:
        checks = value(ctx.hist(("capacity",), ctx.data.capacity)) or ()
        for check in checks:
            if check.mismatch and not any(check.metric in t for t, _ in texts):
                texts.append(("%s capacity differs: nodes have %d, the accounting database "
                              "records %d; percentages use the nodes" % (
                                  check.metric, check.live, check.accounting), WARN))
    info = value(ctx.app.history_info)
    for note in getattr(info, "notes", ()) or ():
        texts.append(("history: " + note.message, DIM))
    if ctx.data.failure:
        texts.append(("history %s: %s" % ctx.data.failure, BAD))
    if not texts:
        return None
    lines = []
    for text, style in texts:
        for part in wrap(text, ctx.width - 3):
            lines.append(text_line(part, style))
    return Pane("warnings", "Warnings (%d)" % len(texts), lines, grow=False, max_lines=2)


# -- Overview ------------------------------------------------------------------

def overview(ctx: Context, view: View) -> List[Pane]:
    panes = [cluster_now(ctx), trend_pane(ctx, {}), accounts_pane(ctx)]
    warn = warnings_pane(ctx)
    if warn is not None:
        panes.append(warn)
    return panes


def cluster_now(ctx: Context) -> Pane:
    snap = ctx.snap
    if snap is None:
        entry = ctx.app.snapshot_entry
        return Pane("now", "Cluster now", [status_line("live", entry)], grow=False)
    t, h = snap.totals, snap.health
    nodes_target = Target("nodes", {}, "Nodes")
    queue_target = Target("queue", {"state": "pending"}, "Queue")
    lines = [resource_line("GPU", t.gpus, ctx.ascii, ctx.width),
             resource_line("CPU", t.cpus, ctx.ascii, ctx.width)]
    lines[0].action = lines[1].action = nodes_target
    down = sum(c for s, c in h.node_states.items() if "DOWN" in s.upper())
    drain = sum(c for s, c in h.node_states.items() if "DRAIN" in s.upper())
    completing = sum(1 for job in snap.jobs or () if job.state == "COMPLETING")
    parts = [("Nodes", HEAD), (" %s" % fmt.num(t.nodes), PLAIN), (" (%s schedulable)  " %
                                                                  fmt.num(t.nodes_schedulable), DIM),
             ("%d down" % down, BAD if down else DIM), ("  ", DIM),
             ("%d drain" % drain, BAD if drain else DIM), ("  ", DIM),
             ("%d completing" % completing, WARN if completing else DIM)]
    lines.append(Line(parts, nodes_target))
    stranded = None
    if h.gpus_stranded_cpu is not None or h.gpus_stranded_mem is not None:
        stranded = (h.gpus_stranded_cpu or 0) + (h.gpus_stranded_mem or 0)
    block = h.largest_gpu_block
    parts = [("Free ", HEAD)]
    if block is not None:
        parts += [("largest block %d GPUs on " % block.gpus, PLAIN), (block.node, RESOURCE)]
    elif t.gpus.total:
        parts += [("no whole free GPU block", WARN)]
    else:
        parts += [("no GPUs", DIM)]
    parts += [("  stranded GPUs ", DIM), (fmt.num(stranded) if stranded is not None else fmt.NA,
                                          BAD if stranded else DIM)]
    if h.scheduler:
        down_hosts = [s.host for s in h.scheduler if s.up is False]
        parts += [("  controller ", DIM), ("down: " + ",".join(down_hosts) if down_hosts else "up",
                                           BAD if down_hosts else GOOD)]
    lines.append(Line(parts, nodes_target))
    pending = h.pending_by_reason
    parts = [("Queue", HEAD)]
    running = sum(1 for job in snap.jobs or () if job.holds_resources)
    waiting = sum(job.task_count for job in snap.jobs or () if job.pending)
    parts += [(" %d running" % running, GOOD), ("  %d pending" % waiting, WARN if waiting else DIM)]
    if pending:
        parts.append((": ", DIM))
        for number, usage in enumerate(pending[:6]):
            if number:
                parts.append((" · ", DIM))
            parts.append(("%s %d" % (usage.key, usage.pending), PLAIN))
    elif snap.jobs is None:
        parts.append(("  " + fmt.placeholder(snap.capabilities.get("jobs").status
                                             if snap.capabilities.get("jobs") else None), DIM))
    lines.append(Line(parts, queue_target))
    return Pane("now", "Cluster now · " + (snap.cluster.name or "?"), lines, grow=False)


ACCOUNT_COLUMNS = (
    Column("Account", 12, "<", 10, flex=True),
    Column("users", 5, ">", 5),
    Column("run", 4, ">", 8),
    Column("pend", 5, ">", 7),
    Column("GPUs", 5, ">", 8),
    Column("cap", 9, ">", 4),
    Column("GPU-h", 7, ">", 9),
    Column("share", 6, ">", 8),
    Column("GPU util*", 9, ">", 3),
    Column("trend", 8, ">", 5),
    Column("visible", 8, "<", 6),
)


def _gpu_cap(ctx: Context, account: str) -> Optional[Tuple[Optional[float], float]]:
    for limit in (ctx.snap.limits or ()) if ctx.snap else ():
        if (limit.scope == "assoc" and limit.user is None and limit.account == account
                and limit.metric == "gres/gpu" and limit.name.startswith("GrpTRES")
                and limit.limit is not None and not limit.name.endswith("Mins")):
            return limit.used, limit.limit
    return None


def accounts_pane(ctx: Context) -> Pane:
    snap = ctx.snap
    table = Table(ACCOUNT_COLUMNS, ctx.width - 2)
    spark_width = next((size for i, col, size in table.layout if col.name == "trend"), 8)
    entry = range_report(ctx, {}, (("account",), ("user", "account")), series_by="account")
    report = value(entry)
    names = set()
    users = {}  # type: Dict[str, set]
    for role in (snap.roles or ()) if snap else ():
        names.add(role.account)
        if role.user:
            users.setdefault(role.account, set()).add(role.user)
    live = {u.key: u for u in group_by(snap_jobs(ctx), "account")} if snap else {}
    names.update(k for k in live if k)
    hist = {}  # type: Dict[str, Tuple[Optional[float], Optional[float], Any, bool]]
    summary = None
    if ctx.data.has("overview"):
        ov_entry = overview_entry(ctx)
        found = value(ov_entry)
        if found is not None:
            summary = next((r for r in found.ranges if r.name == ctx.range), None)
        entry = ov_entry if report is None else entry
    if summary is not None:
        for name, item in summary.accounts.items():
            hist[name] = (item.amounts.gpu_hours, item.share, item.amounts.efficiency("gpu"),
                          summary.complete)
            users.setdefault(name, set()).update(item.users)
        names.update(hist)
    elif report is not None:
        total_gpu = report.total.gpu_seconds
        for key, stats in report.group("account").items():
            hist[key[0]] = (stats.gpu_hours, stats.gpu_seconds / total_gpu if total_gpu else None,
                            stats.efficiency("gpu"), report.complete and stats.complete("gpu"))
        names.update(hist)
        for (user, account), _ in report.group("user", "account").items():
            users.setdefault(account, set()).add(user)
    loaded = summary is not None or report is not None
    invoker = snap.invoker if snap else None
    mine = [r.account for r in (snap.my_roles or ())] if snap else []
    info = value(ctx.app.history_info)
    lines = []
    for name in sorted(n for n in names if n):
        use = live.get(name)
        stats = hist.get(name)
        cap = _gpu_cap(ctx, name)
        if stats is not None:
            gpu_h, share, eff, complete = stats
            spark = None
            if report is not None:
                try:
                    buckets = report.buckets("gpu", "day" if ctx.range not in ("day", "week")
                                             else "hour", key=name)
                    spark = [b.value for b in buckets] if buckets else None
                except (ValueError, KeyError):
                    spark = None
            hist_cells = [("" if complete else "≥") + fmt.hours(gpu_h), fmt.pct(share, 1),
                          eff_cell(eff), (fmt.sparkline(spark, spark_width, ctx.ascii), GOOD)]
        else:
            ph = "0" if loaded else missing(entry)
            hist_cells = [ph, fmt.NOT_APPLICABLE if loaded else ph, fmt.NOT_APPLICABLE, ""]
        if ctx.data.failure and not loaded:
            hist_cells = [fmt.placeholder(ctx.data.failure[0])] * 3 + [""]
        vis = hd.visibility(info, invoker, name, mine) if invoker else fmt.NOT_YET
        if ctx.app.no_history:
            vis = "history off"
        vis_style = GOOD if vis == "full" else WARN if vis == "own only" else DIM
        cap_text = fmt.NOT_APPLICABLE
        cap_style = DIM
        if cap is not None:
            used, limit = cap
            cap_text = "%s/%s" % (fmt.num(used) if used is not None else "?", fmt.num(limit))
            cap_style = BAD if used is not None and used >= limit else PLAIN
        cells = [(name, RESOURCE), fmt.num(len(users.get(name, ()))) if name in users else fmt.NA,
                 str(use.running) if use else "0", str(use.pending) if use else "0",
                 fmt.num(use.gpus) if use else "0", (cap_text, cap_style)] + hist_cells + [
                     (vis, vis_style)]
        lines.append(table.row(
            cells, Target("people", {"account": name}, "People in " + name),
            {"name": name, "GPU-h": (stats[0] or 0) if stats else 0,
             "running": use.running if use else 0, "pending": use.pending if use else 0,
             "GPUs": (use.gpus or 0) if use else 0}, key=name))
    lines.sort(key=lambda line: -(line.sort.get("GPU-h") or 0) - 1e12 * (line.sort.get("GPUs") or 0))
    return Pane("accounts", "Accounts · %s" % ctx.range, lines, table.header(),
                sorts=("GPU-h", "GPUs", "running", "pending", "name"),
                status="hours allocated" if loaded else "history " + missing(entry))


# -- People ------------------------------------------------------------------

PEOPLE_COLUMNS = (
    Column("User", 12, "<", 10, flex=True),
    Column("accounts", 14, "<", 3),
    Column("run", 4, ">", 8),
    Column("pend", 5, ">", 7),
    Column("GPUs", 5, ">", 8),
    Column("GPU-h", 7, ">", 9),
    Column("share", 6, ">", 8),
    Column("CPU-h", 7, ">", 5),
    Column("GPU util*", 9, ">", 4),
    Column("jobs", 6, ">", 3),
)


def people(ctx: Context, view: View) -> List[Pane]:
    params = {k: view.params.get(k) for k in ("account", "qos", "partition")}
    entry = range_report(ctx, params, (("user",), ("user", "account")))
    report = value(entry)
    jobs = [j for j in snap_jobs(ctx) if all(
        params[k] is None or (getattr(j, k) or "") == params[k] or
        (k == "partition" and params[k] in (j.partition or "").split(","))
        for k in params)]
    live = {u.key: u for u in group_by(jobs, "user")}
    accounts = {}  # type: Dict[str, set]
    for job in jobs:
        accounts.setdefault(job.user, set()).add(job.account)
    hist = {}
    total = None
    if report is not None:
        hist = {key[0]: stats for key, stats in report.group("user").items()}
        total = report.total.gpu_seconds
        for (user, account), _ in report.group("user", "account").items():
            accounts.setdefault(user, set()).add(account)
    if ctx.snap and params["account"]:
        for role in ctx.snap.roles or ():
            if role.account == params["account"] and role.user:
                accounts.setdefault(role.user, set()).add(role.account)
    table = Table(PEOPLE_COLUMNS, ctx.width - 2)
    lines = []
    me = ctx.snap.invoker.user if ctx.snap else None
    for user in sorted(set(live) | set(hist) | set(accounts)):
        if not user:
            continue
        use, stats = live.get(user), hist.get(user)
        hist_cells = ([fmt.hours(stats.gpu_hours),
                       fmt.pct(stats.gpu_seconds / total if total else None, 1),
                       fmt.hours(stats.cpu_hours), eff_cell(stats.efficiency("gpu")),
                       fmt.num(stats.jobs)] if stats is not None
                      else [missing(entry) if report is None else "0"] * 5)
        name_style = (USER, ) if user != me else (USER,)
        cells = [(user + (" (you)" if user == me else ""), name_style[0]),
                 ",".join(sorted(a for a in accounts.get(user, ()) if a)),
                 str(use.running) if use else "0", str(use.pending) if use else "0",
                 fmt.num(use.gpus) if use else "0"] + hist_cells
        lines.append(table.row(cells, Target("person", {"user": user}, user),
                               {"GPU-h": stats.gpu_seconds if stats else 0,
                                "GPUs": (use.gpus or 0) if use else 0,
                                "pending": use.pending if use else 0, "user": user}, key=user))
    lines.sort(key=lambda l: (-(l.sort["GPUs"]), -(l.sort["GPU-h"] or 0), l.sort["user"]))
    scope = " · ".join("%s %s" % (k, v) for k, v in params.items() if v)
    return [Pane("people", "People" + (" · " + scope if scope else "") + " · " + ctx.range,
                 lines, table.header(), sorts=("GPU-h", "GPUs", "pending", "user"),
                 status="hours allocated")]


# -- Roles -------------------------------------------------------------------

ROLE_COLUMNS = (
    Column("Name", 14, "<", 10, flex=True),
    Column("users", 5, ">", 6),
    Column("run", 4, ">", 8),
    Column("pend", 5, ">", 7),
    Column("GPUs", 5, ">", 8),
    Column("CPUs", 6, ">", 4),
    Column("GPU-h", 7, ">", 9),
    Column("share", 6, ">", 8),
    Column("CPU-h", 7, ">", 3),
    Column("GPU util*", 9, ">", 3),
    Column("limits", 7, ">", 2),
)


def roles(ctx: Context, view: View) -> List[Pane]:
    by = view.extra.get("group", "account")
    entry = range_report(ctx, {}, ((by,), ("user", by)))
    report = value(entry)
    jobs = snap_jobs(ctx)
    if by == "partition":
        live = {}  # type: Dict[str, Any]
        for usage in group_by(jobs, lambda j: (j.partition or "").split(",")[0]):
            live[usage.key] = usage
    else:
        live = {u.key: u for u in group_by(jobs, by)}
    hist, users = {}, {}  # type: Dict[str, Any], Dict[str, set]
    total = None
    if report is not None:
        hist = {key[0]: stats for key, stats in report.group(by).items()}
        total = report.total.gpu_seconds
        for (user, key), _ in report.group("user", by).items():
            users.setdefault(key, set()).add(user)
    for job in jobs:
        key = getattr(job, by) if by != "partition" else (job.partition or "").split(",")[0]
        users.setdefault(key, set()).add(job.user)
    names = set(live) | set(hist)
    limits = {}  # type: Dict[str, int]
    snap = ctx.snap
    if snap is not None:
        if by == "account":
            names.update(r.account for r in snap.roles or ())
            for limit in snap.limits or ():
                if limit.scope == "assoc" and limit.limit is not None:
                    limits[limit.account] = limits.get(limit.account, 0) + 1
        elif by == "qos":
            for qos in snap.qos or ():
                names.add(qos.name)
                limits[qos.name] = sum(1 for l in qos.limits if l.limit is not None)
        else:
            names.update(p.name for p in snap.partitions or ())
    table = Table(ROLE_COLUMNS, ctx.width - 2)
    lines = []
    for name in sorted(n for n in names if n):
        use, stats = live.get(name), hist.get(name)
        hist_cells = ([fmt.hours(stats.gpu_hours),
                       fmt.pct(stats.gpu_seconds / total if total else None, 1),
                       fmt.hours(stats.cpu_hours), eff_cell(stats.efficiency("gpu"))]
                      if stats is not None else [missing(entry) if report is None else "0"] * 4)
        cells = [(name, RESOURCE), fmt.num(len(users.get(name, ()))),
                 str(use.running) if use else "0", str(use.pending) if use else "0",
                 fmt.num(use.gpus) if use else "0", fmt.num(use.cpus) if use else "0"
                 ] + hist_cells + [str(limits[name]) if name in limits else fmt.NOT_APPLICABLE]
        lines.append(table.row(cells, Target("people", {by: name}, "%s %s" % (by, name)),
                               {"GPU-h": stats.gpu_seconds if stats else 0,
                                "GPUs": (use.gpus or 0) if use else 0, "name": name}, key=name))
    lines.sort(key=lambda l: (-l.sort["GPUs"], -(l.sort["GPU-h"] or 0), l.sort["name"]))
    label = {"account": "Accounts", "qos": "QOS", "partition": "Partitions"}[by]
    return [Pane("roles", "Roles: %s · %s" % (label, ctx.range), lines, table.header(),
                 sorts=("GPU-h", "GPUs", "name"), status="g: account/QOS/partition")]


# -- Nodes -------------------------------------------------------------------

NODE_COLUMNS = (
    Column("Node", 10, "<", 10, flex=True),
    Column("state", 10, "<", 9),
    Column("partitions", 12, "<", 2),
    Column("CPU", 9, ">", 8),
    Column("mem", 11, ">", 4),
    Column("GPU", 7, ">", 8),
    Column("type", 6, "<", 3),
    Column("load", 6, ">", 5),
    Column("reason", 16, "<", 6, flex=False),
)


def node_style(node: Any) -> str:
    state = (node.state or "").upper() + " " + " ".join(node.state_flags).upper()
    if any(word in state for word in ("DOWN", "DRAIN", "FAIL", "NOT_RESPONDING")):
        return BAD
    if node.availability == "reserved" or "MAINT" in state or "RESERVED" in state:
        return DIM
    if "MIXED" in state or "ALLOC" in state:
        return GOOD
    if "COMPLETING" in state:
        return WARN
    return PLAIN


def nodes(ctx: Context, view: View) -> List[Pane]:
    snap = ctx.snap
    if snap is None or snap.nodes is None:
        cap = snap.capabilities.get("nodes") if snap else None
        return [Pane("nodes", "Nodes", [text_line("nodes: %s %s" % (
            fmt.placeholder(cap.status if cap else None), cap.detail if cap else ""), DIM)])]
    table = Table(NODE_COLUMNS, ctx.width - 2)
    lines = []
    for node in snap.nodes:
        state = (node.state or "?") + ("+" + "+".join(node.state_flags) if node.state_flags else "")
        gpu = fmt.ratio(node.gpus_alloc, node.gpus_total) if node.gpus_total else fmt.NOT_APPLICABLE
        cells = [(node.name, RESOURCE), (state, node_style(node)), ",".join(node.partitions),
                 fmt.ratio(node.cpus_alloc, node.cpus_total),
                 "%s/%s" % (fmt.mem(node.mem_alloc), fmt.mem(node.mem_total))
                 if node.mem_alloc is not None else fmt.mem(node.mem_total),
                 gpu, ",".join(sorted(node.gpu_types)) or fmt.NOT_APPLICABLE,
                 "%.1f" % node.cpu_load if node.cpu_load is not None else fmt.NA,
                 (node.reason or "", WARN if node.reason else PLAIN)]
        free = node.gpus_free
        lines.append(table.row(cells, Target("queue", {"node": node.name}, "jobs on " + node.name),
                               {"name": node.name, "GPUs free": free, "CPUs free": node.cpus_free,
                                "load": node.cpu_load, "state": state}, key=node.name))
    problems = sum(1 for node in snap.nodes if node_style(node) == BAD)
    return [Pane("nodes", "Nodes (%d)" % len(snap.nodes), lines, table.header(),
                 sorts=("name", "GPUs free", "CPUs free", "load", "state"),
                 status="%d need attention" % problems if problems else "")]


# -- Queue -------------------------------------------------------------------

QUEUE_COLUMNS = (
    Column("Job", 11, "<", 10),
    Column("user", 9, "<", 9),
    Column("account", 9, "<", 5),
    Column("qos", 7, "<", 3),
    Column("partition", 9, "<", 4),
    Column("state", 8, "<", 8),
    Column("GPUs", 4, ">", 8),
    Column("CPUs", 5, ">", 6),
    Column("time", 7, ">", 7),
    Column("reason/nodes", 14, "<", 6, flex=True),
    Column("name", 14, "<", 2),
)


def job_style(job: Any) -> str:
    if job.holds_resources:
        return WARN if job.state == "COMPLETING" else GOOD
    if job.pending:
        return WARN
    return DIM


def queue(ctx: Context, view: View) -> List[Pane]:
    snap = ctx.snap
    if snap is None or snap.jobs is None:
        cap = snap.capabilities.get("jobs") if snap else None
        return [Pane("queue", "Queue", [text_line("jobs: %s %s" % (
            fmt.placeholder(cap.status if cap else None), cap.detail if cap else ""), DIM)])]
    params = view.params
    jobs = []
    for job in snap_jobs(ctx):
        if params.get("user") and job.user != params["user"]:
            continue
        if params.get("account") and job.account != params["account"]:
            continue
        if params.get("node") and params["node"] not in job.nodes:
            continue
        if params.get("state") == "pending" and not job.pending:
            continue
        jobs.append(job)
    drilled = any(params.get(k) for k in ("user", "account", "qos", "partition"))
    columns = QUEUE_COLUMNS if drilled else QUEUE_COLUMNS[:-1]
    table = Table(columns, ctx.width - 2)
    lines = []
    for job in jobs:
        if job.pending:
            spent = ctx.now - job.submit if job.submit else None
            where = job.reason or ""
            gpus = job.gpus_req
            cpus = job.cpus_req
        else:
            spent = job.elapsed if job.elapsed is not None else (
                ctx.now - job.start if job.start else None)
            where = ",".join(job.nodes[:3]) + ("…" if len(job.nodes) > 3 else "")
            gpus, cpus = job.gpus_alloc, job.cpus_alloc
        count = ""
        cells = [job.display_id or job.job_id, (job.user or "?", USER), (job.account or "", RESOURCE),
                 job.qos or "", job.partition or "", ((job.state or "?") + count, job_style(job)),
                 fmt.num(gpus) if gpus is not None else fmt.NA,
                 fmt.num(cpus) if cpus is not None else fmt.NA, fmt.duration(spent),
                 (where, WARN if job.pending else RESOURCE), job.name or ""]
        lines.append(table.row(cells, Target("job", {"job_id": job.job_id}, "job " + (job.display_id or job.job_id)),
                               {"time": spent, "GPUs": gpus, "user": job.user or "",
                                "priority": job.priority, "state": job.state or ""},
                               key=job.job_id))
    lines.sort(key=lambda l: (l.sort["state"] != "RUNNING", -(l.sort["GPUs"] or 0)))
    scope = " · ".join("%s %s" % (k, v) for k, v in sorted(params.items()) if v)
    running = sum(1 for j in jobs if j.holds_resources)
    return [Pane("queue", "Queue" + (" · " + scope if scope else ""), lines, table.header(),
                 sorts=("time", "GPUs", "priority", "user", "state"),
                 status="%d running · %d pending" % (running, sum(j.task_count for j in jobs if j.pending)))]


# -- Person ------------------------------------------------------------------

RUN_COLUMNS = (
    Column("Run type", 14, "<", 10, flex=True),
    Column("att", 5, ">", 8),
    Column("jobs", 5, ">", 5),
    Column("GPU-h", 7, ">", 9),
    Column("share", 5, ">", 8),
    Column("runtime", 7, ">", 7),
    Column("wait", 6, ">", 6),
    Column("limit", 5, ">", 4),
    Column("GPU util*", 9, ">", 5),
    Column("CPU eff", 8, ">", 2),
    Column("fail", 4, ">", 6),
)


def person_attempts(ctx: Context, user: str) -> Entry:
    filt = make_filter({"user": user})
    name = ctx.range
    return ctx.hist(("attempts", name, user), lambda: ctx.data.attempts(name, filt))


def event_target(event: Any, user: Optional[str]) -> Optional[Target]:
    title = event.text if len(event.text) <= 40 else event.text[:39] + "…"
    if event.job_id:
        return Target("job", {"job_id": event.job_id}, "job " + event.job_id)
    if event.rows:
        return jobs_target(title, event.rows)
    if event.window is not None or event.where is not None:
        params = {"user": user}  # type: Dict[str, Any]
        for role, values in (event.where.items() if event.where is not None else ()):
            if role in ("user", "account", "qos", "partition") and len(values) == 1:
                params[role] = next(iter(values))
        win = event.window
        if win is not None:
            low, high = win.start, win.end
            if event.kind == "idle_gap" and low is not None and high is not None:
                low, high = low - 86400, high + 86400  # what ran either side of the pause
            params["window"] = (low, high)
        return Target("jobs", params, title)
    return None


def jobs_target(title: str, rows: Sequence[Any]) -> Target:
    return Target("jobs", {"rows": list(rows)}, title)


def person(ctx: Context, view: View) -> List[Pane]:
    user = view.params.get("user")
    params = {"user": user}
    panes = [trend_pane(ctx, params, person=True)]

    # live jobs
    live = [j for j in snap_jobs(ctx) if j.user == user]
    lines = []
    for job in sorted(live, key=lambda j: (not j.holds_resources, -(j.gpus_alloc or j.gpus_req or 0))):
        size = job.gpus_alloc if job.holds_resources else job.gpus_req
        spent = (job.elapsed if job.holds_resources else (ctx.now - job.submit if job.submit else None))
        where = ",".join(job.nodes[:2]) if job.holds_resources else (job.reason or "")
        lines.append(Line([(fmt.fit(job.display_id or job.job_id, 12), PLAIN),
                           (fmt.fit(job.state or "?", 9), job_style(job)),
                           (fmt.fit("%s GPU" % fmt.num(size) if size else "CPU", 7), PLAIN),
                           (fmt.fit(fmt.duration(spent), 8, ">") + " ", DIM),
                           (fmt.fit(job.account or "", 10), RESOURCE),
                           (fmt.fit(where, 14) + " ", WARN if job.pending else RESOURCE),
                           (job.name or "", PLAIN)],
                          Target("job", {"job_id": job.job_id}, "job " + (job.display_id or job.job_id))))
    run = sum(1 for j in live if j.holds_resources)
    status = "%d running · %d pending" % (run, sum(j.task_count for j in live if j.pending))
    if ctx.snap is None:
        lines, status = [status_line("live", ctx.app.snapshot_entry)], ""
    panes.append(Pane("live", "Now", lines, status=status, grow=False, max_lines=4))

    # kinds of runs
    dim = view.extra.get("dim", "gpu_count")
    dim_label = dict(hd.DIMENSIONS)[dim]
    filt = make_filter(params)
    name = ctx.range
    types_entry = ctx.hist(("runtypes", name, user, dim),
                          lambda: ctx.data.run_types(name, filt, dim))
    kinds = value(types_entry)
    table = Table(RUN_COLUMNS, ctx.width - 2)
    lines = []
    if kinds is None:
        lines.append(text_line("history: %s %s" % (missing(types_entry), types_entry.message), DIM))
    else:
        for kind in kinds:
            style = RESOURCE if dim in ("account", "qos", "partition") else PLAIN
            if dim == "outcome":
                style = OUTCOME_STYLE.get(kind.key, PLAIN)
            fail_style = BAD if kind.failure and kind.failure >= 0.25 else PLAIN
            cells = [(kind.label, style), fmt.num(kind.attempts), fmt.num(kind.jobs),
                     ("" if kind.complete else "≥") + fmt.hours(kind.gpu_hours),
                     fmt.pct(kind.share), fmt.duration(kind.runtime),
                     fmt.duration(kind.wait), fmt.pct(kind.limit_use),
                     eff_cell((kind.gpu_eff, kind.gpu_cover)),
                     eff_cell((kind.cpu_eff, kind.cpu_cover)),
                     (fmt.pct(kind.failure), fail_style) if kind.failure is not None
                     else (fmt.NOT_APPLICABLE, DIM)]
            title = "%s · %s %s" % (user, dim_label, kind.label)
            target = (jobs_target(title, kind.rows) if kind.rows is not None else
                      Target("jobs", {"user": user, "run_type": (dim, kind.key)}, title))
            lines.append(table.row(cells, target,
                                   {"GPU-h": kind.gpu_hours, "attempts": kind.attempts,
                                    "fail": kind.failure, "runtime": kind.runtime}, key=kind.key))
    panes.append(Pane("runs", "What kinds of runs · by %s · %s" % (dim_label, ctx.range), lines,
                      table.header(), sorts=("GPU-h", "attempts", "fail", "runtime"),
                      status="g: GPUs/nodes/name/account/QOS/partition/outcome"))

    # events
    report_entry = range_report(ctx, params, (("account",), ("qos",), ("partition",)))
    report = value(report_entry)
    events_entry = ctx.hist(("events", name, user), lambda: ctx.data.events(name, filt))
    found = value(events_entry)
    lines = []
    if found is not None:
        for event in found:
            style = {"good": GOOD, "bad": BAD, "warn": WARN, "dim": DIM}.get(event.style, PLAIN)
            lines.append(Line([(fmt.fit(fmt.when(event.at, ctx.now), 12) + " ", DIM),
                               (event.text, style)], event_target(event, user),
                              {"time": event.at}))
    else:
        lines.append(text_line("history: %s %s" % (missing(events_entry), events_entry.message),
                               DIM))
    panes.append(Pane("events", "Events · " + ctx.range, lines, sorts=("time",)))

    # roles and outcomes
    lines = []
    if report is not None:
        total = report.total.gpu_seconds or 0
        for role in ("account", "qos", "partition"):
            ranked = report.ranked(role)
            parts = [(fmt.fit(role.replace("qos", "QOS"), 10), HEAD)]
            for number, (key, stats) in enumerate(ranked[:4]):
                if number:
                    parts.append(("  ", DIM))
                share = stats.gpu_seconds / total if total else None
                parts += [(key[0], RESOURCE), (" %s GPU-h %s" % (fmt.hours(stats.gpu_hours),
                                                                 fmt.pct(share)), PLAIN)]
            lines.append(Line(parts, Target("people", {role: ranked[0][0][0]}, role)
                              if ranked else None))
        parts = [(fmt.fit("outcome", 10), HEAD)]
        first = True
        for name in hd.OUTCOME_ORDER:
            stats = report.outcomes.get(name)
            if stats is None or not stats.attempts:
                continue
            if not first:
                parts.append(("  ", DIM))
            first = False
            style = {"completed": GOOD, "running": GOOD, "failed": BAD, "node_fail": BAD,
                     "timeout": WARN, "preempted": WARN}.get(name, PLAIN)
            parts += [(name, style), (" %d" % stats.attempts, PLAIN)]
        lines.append(Line(parts))
    else:
        lines.append(text_line("history: %s" % missing(report_entry), DIM))
    panes.append(Pane("split", "Roles and outcomes · " + ctx.range, lines, grow=False))

    # activity, efficiency, waits
    panes.append(activity_pane(ctx, params, report, report_entry))

    # jobs list
    jobs_entry = ctx.hist(("jobs", ctx.range, user, "gpu"),
                          lambda name=ctx.range: ctx.data.jobs(name, make_filter(params)))
    jpage = value(jobs_entry)
    panes.append(jobs_pane(ctx, "jobs", "Jobs · " + ctx.range,
                           jpage.rows if jpage is not None else None, jobs_entry))
    return panes


def activity_pane(ctx: Context, params: Dict[str, Any], report: Any, entry: Entry) -> Pane:
    lines = []
    if report is None:
        return Pane("activity", "Activity", [text_line("history: %s" % missing(entry), DIM)],
                    grow=False)
    total = report.total
    unit = "hour" if ctx.range == "day" else "day"
    buckets = report.buckets("gpu", unit) or report.buckets("cpu", unit)
    metric = "GPU" if report.buckets("gpu", unit) else "CPU"
    if buckets:
        peak = max(b.value for b in buckets) / 3600.0
        lines.append(Line([("%s-h per %s " % (metric, unit), HEAD),
                           (fmt.sparkline([b.value for b in buckets], max(10, ctx.width - 40),
                                          ctx.ascii), GOOD),
                           (" peak %s" % fmt.hours(peak), DIM)]))
    grid = report.heatmap("gpu") or report.heatmap("cpu")
    if grid:
        top = max(max(row) for row in grid) or 1
        step = 2 if ctx.width >= 60 else 1
        lines.append(Line([("     ", DIM), ("".join(fmt.fit(str(h), step) if h % (3 if step == 2 else 6) == 0
                                                  else " " * step for h in range(24)), DIM),
                           ("  local hour", DIM)]))
        for number, label in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
            cells = "".join(fmt.shade(v / top, ctx.ascii) * step for v in grid[number])
            lines.append(Line([(label + "  ", DIM), (cells, GOOD)]))
    gpu, cpu = total.efficiency("gpu"), total.efficiency("cpu")
    parts = [("Efficiency ", HEAD), ("GPU util (peak sample) ", DIM), eff_cell(gpu),
             ("  CPU ", DIM), eff_cell(cpu), ("  (value·coverage)", DIM)]
    if report.efficiency_unchecked:
        parts.append(("  %d attempts not measured (E loads)" % report.efficiency_unchecked, DIM))
    lines.append(Line(parts))
    lines.append(Line([("Waits      ", HEAD),
                       ("median %s  p90 %s  max %s  over %d starts" % (
                           fmt.duration(report.wait_quantile(0.5)),
                           fmt.duration(report.wait_quantile(0.9)),
                           fmt.duration(total.wait_max), total.wait_count), PLAIN)]))
    used = total.time_limit_used
    lines.append(Line([("Time limit ", HEAD),
                       ("%s of requested time used (completed + timed out, %d attempts)"
                        % (fmt.pct(used), total.limit_count) if used is not None
                        else "no finished attempt with a limit", PLAIN if used is not None else DIM)]))
    return Pane("activity", "Activity · " + ctx.range, lines, grow=False)


JOB_COLUMNS = (
    Column("Job", 12, "<", 10),
    Column("name", 14, "<", 6, flex=True),
    Column("user", 9, "<", 4),
    Column("outcome", 9, "<", 9),
    Column("start", 12, "<", 5),
    Column("elapsed", 7, ">", 8),
    Column("GPUs", 4, ">", 7),
    Column("GPU-h", 6, ">", 8),
    Column("wait", 6, ">", 3),
    Column("att", 3, ">", 2),
)

OUTCOME_STYLE = {"completed": GOOD, "running": GOOD, "failed": BAD, "node_fail": BAD,
                 "timeout": WARN, "preempted": WARN, "cancelled": DIM}


def jobs_pane(ctx: Context, pane_id: str, title: str, rows: Optional[Sequence[Any]],
              entry: Optional[Entry] = None) -> Pane:
    table = Table(JOB_COLUMNS, ctx.width - 2)
    lines = []
    if rows is None:
        return Pane(pane_id, title, [text_line("history: %s %s" % (
            missing(entry), entry.message if entry else ""), DIM)])
    for row in rows:
        cells = [row.display_id, row.name, (row.user, USER),
                 (row.outcome, OUTCOME_STYLE.get(row.outcome, PLAIN)),
                 fmt.when(row.start, ctx.now), fmt.duration(row.elapsed),
                 fmt.num(row.gpus) if row.gpus is not None else fmt.NA,
                 fmt.hours(row.gpu_seconds / 3600.0) if row.gpu_seconds is not None else fmt.NA,
                 fmt.duration(row.wait), str(row.attempts)]
        lines.append(table.row(cells, Target("job", {"job_id": row.job_id, "row": row},
                                             "job " + row.display_id),
                               {"GPU-h": row.gpu_seconds, "start": row.start,
                                "elapsed": row.elapsed, "wait": row.wait, "GPUs": row.gpus},
                               key=row.key))
    return Pane(pane_id, title + " (%d)" % len(rows), lines, table.header(),
                sorts=("GPU-h", "start", "elapsed", "wait", "GPUs"))


def jobs_list(ctx: Context, view: View) -> List[Pane]:
    rows = view.params.get("rows")
    if rows is not None:
        return [jobs_pane(ctx, "jobs", view.title, rows)]
    spec = view.params
    win = spec.get("window")
    filt_params = {k: spec.get(k) for k in ("user", "account", "qos", "partition")}
    run_type = spec.get("run_type")
    entry = ctx.hist(("joblist", ctx.range, filt_key(filt_params), win, run_type),
                     lambda name=ctx.range: ctx.data.jobs(name, make_filter(filt_params),
                                                          win=hd.Window(*win) if win else None,
                                                          run_type=run_type))
    page = value(entry)
    return [jobs_pane(ctx, "jobs", view.title, page.rows if page is not None else None, entry)]


# -- Timeline ----------------------------------------------------------------

def timeline(ctx: Context, view: View) -> List[Pane]:
    by = view.extra.get("lanes", "user")
    zoom = view.extra.get("zoom", 1)
    entry = ctx.hist(("lanes", ctx.range, by),
                     lambda name=ctx.range: ctx.data.report(name, None, lanes_by=by,
                                                            efficiency=False))
    report = value(entry)
    if report is None:
        return [Pane("timeline", "Timeline", [text_line("history: %s %s" % (
            missing(entry), entry.message), DIM)])]
    label_width = min(16, max(8, ctx.width // 6))
    bins = max(10, ctx.width - label_width - 4)
    span = max(1, report.end - report.start)
    shown = max(bins, span // zoom)
    pan = view.extra.get("pan", 0.0)  # 0 = ends at the window's end
    end = report.end - int(pan * shown)
    end = max(report.start + shown, min(report.end, end))
    start = end - shown
    size = shown / float(bins)
    cursor = max(0, min(bins - 1, view.extra.get("bin", bins - 1)))
    view.extra["bin"] = cursor
    view.extra["bins"] = (start, size, bins)
    axis = [" "] * bins
    pattern = "%b %d" if shown > 2 * 86400 else "%H:%M"
    sample = time.strftime(pattern, time.localtime(start))
    for mark in range(0, bins - len(sample), len(sample) + 3):
        label = time.strftime(pattern, time.localtime(start + mark * size))
        for i, ch in enumerate(label):
            if mark + i < bins:
                axis[mark + i] = ch
    header = Line([(fmt.fit(by, label_width) + " ", DIM), ("".join(axis), DIM)])
    lines = []
    for lane in report.lanes:
        cover = [0.0] * bins
        total = 0
        for a, b in lane.intervals:
            total += b - a
            if b <= start or a >= end:
                continue
            a, b = max(a, start), min(b, end)
            first, last = int((a - start) / size), min(bins - 1, int((b - start - 1) / size))
            for i in range(first, last + 1):
                lo, hi = start + i * size, start + (i + 1) * size
                cover[i] += max(0.0, min(b, hi) - max(a, lo)) / size
        cells = "".join(fmt.shade(min(1.0, c), ctx.ascii) if c > 0 else "·" if ctx.ascii is False and False else (" " if c <= 0 else fmt.shade(c, ctx.ascii)) for c in cover)
        style = USER if by == "user" else RESOURCE
        bin_start = int(start + cursor * size)
        bin_end = int(start + (cursor + 1) * size) + 1
        parts = [(fmt.fit(lane.key, label_width) + " ", style), (cells[:cursor], GOOD),
                 (cells[cursor] if cells[cursor] != " " else "┊", WARN), (cells[cursor + 1:], GOOD)]
        params = {by: lane.key, "window": (bin_start, bin_end)}
        lines.append(Line(parts, Target("jobs", params, "%s %s at %s" % (by, lane.key,
                                                                         fmt.when(bin_start, ctx.now))),
                          {"busy": total, "name": lane.key}, key=lane.key))
    lines.sort(key=lambda l: -l.sort["busy"])
    status = "bin %s · %s → %s" % (fmt.duration(size), fmt.when(start + cursor * size, ctx.now),
                                   fmt.when(start + (cursor + 1) * size, ctx.now))
    return [Pane("timeline", "Timeline · lanes by %s · %s · zoom %dx" % (by, ctx.range, zoom),
                 lines, header, sorts=("busy", "name"), status=status)]


# -- Job ---------------------------------------------------------------------

STEP_COLUMNS = (
    Column("Step", 10, "<", 10),
    Column("name", 12, "<", 5, flex=True),
    Column("state", 10, "<", 8),
    Column("elapsed", 8, ">", 8),
    Column("tasks", 5, ">", 3),
    Column("CPU eff", 7, ">", 9),
    Column("max RSS", 8, ">", 7),
    Column("GPU util", 8, ">", 7),
    Column("GPU mem", 8, ">", 4),
)


def step_cpu_eff(step: Any) -> Optional[float]:
    cpus = step.tres_alloc.get("cpu") if step.tres_alloc else None
    if step.total_cpu is None or not cpus or not step.elapsed:
        return None
    return step.total_cpu / (cpus * step.elapsed)


def job(ctx: Context, view: View) -> List[Pane]:
    job_id = str(view.params.get("job_id"))
    live = ctx.snap.job(job_id) if ctx.snap else None
    row = view.params.get("row")
    attempts_entry = ctx.hist(("job_attempts", job_id), lambda: ctx.data.job_attempts(job_id))
    attempts = value(attempts_entry) or []
    if row is None and attempts:
        row = attempts[-1]
    detail_entry = ctx.once(("detail", job_id), lambda: ctx.data.job_detail(job_id))
    detail = value(detail_entry)

    # summary
    lines = []

    def kv(label: str, *parts: Tuple[str, str]) -> None:
        lines.append(Line([(fmt.fit(label, 10), HEAD)] + list(parts)))

    if live is not None:
        src = live
        kv("who", (src.user or "?", USER), ("  account ", DIM), (src.account or "?", RESOURCE),
           ("  qos ", DIM), (src.qos or "?", RESOURCE), ("  partition ", DIM), (src.partition or "?", RESOURCE))
        kv("state", (src.state or "?", job_style(src)), ("  " + (src.reason or "") if src.pending else "", WARN))
        kv("request", ("%s CPUs  %s GPUs  %s mem  limit %s" % (
            fmt.num(src.cpus_req if src.pending else src.cpus_alloc),
            fmt.num(src.gpus_req if src.pending else src.gpus_alloc),
            fmt.mem(src.mem_req if src.pending else src.mem_alloc),
            "unlimited" if src.time_unlimited else fmt.duration(src.time_limit)), PLAIN))
        if src.nodes:
            kv("nodes", (",".join(src.nodes[:8]) + (" …" if len(src.nodes) > 8 else ""), RESOURCE))
        wait = src.start - (src.eligible or src.submit) if src.start and (src.eligible or src.submit) and not src.pending else None
        kv("timing", ("submit %s  start %s%s  end %s" % (
            fmt.when(src.submit, ctx.now), fmt.when(src.start, ctx.now),
            " (est.)" if src.pending else "", fmt.when(src.end, ctx.now) + (" (est.)" if src.end and src.end > ctx.now else "")), PLAIN))
        used = src.elapsed / src.time_limit if src.elapsed and src.time_limit else None
        kv("", ("waited %s  elapsed %s  limit used %s" % (fmt.duration(wait), fmt.duration(src.elapsed),
                                                          fmt.pct(used)), DIM))
        if src.name:
            kv("name", (src.name, PLAIN))
        if src.workdir:
            kv("workdir", (src.workdir, DIM))
    elif row is not None:
        kv("who", (row.user, USER), ("  account ", DIM), (row.account, RESOURCE), ("  qos ", DIM),
           (row.qos, RESOURCE), ("  partition ", DIM), (row.partition, RESOURCE))
        kv("outcome", (row.outcome, OUTCOME_STYLE.get(row.outcome, PLAIN)), ("  " + row.state, DIM))
        kv("request", ("%s CPUs  %s GPUs  %s mem  limit %s" % (
            fmt.num(row.cpus), fmt.num(row.gpus), fmt.mem(row.mem),
            "unlimited" if row.time_unlimited else fmt.duration(row.time_limit)), PLAIN))
        if row.nodes:
            kv("nodes", (row.nodes, RESOURCE))
        kv("timing", ("submit %s  start %s  end %s" % (fmt.when(row.submit, ctx.now),
                                                       fmt.when(row.start, ctx.now),
                                                       fmt.when(row.end, ctx.now)), PLAIN))
        kv("", ("waited %s  elapsed %s  limit used %s" % (
            fmt.duration(row.wait), fmt.duration(row.elapsed), fmt.pct(row.time_limit_used)), DIM))
        if row.name:
            kv("name", (row.name, PLAIN))
        effs = [("CPU ", DIM), (fmt.pct(row.cpu_efficiency), PLAIN), ("  mem peak ", DIM),
                (fmt.pct(row.mem_efficiency), PLAIN), ("  GPU peak ", DIM),
                (fmt.pct(row.gpu_peak_utilization), PLAIN)]
        kv("efficiency", *effs)
    else:
        lines.append(text_line("not in the live queue; history: %s %s" % (
            missing(attempts_entry), attempts_entry.message), DIM))
    name = (live.name if live is not None else row.name if row is not None else "") or ""
    panes = [Pane("summary", "Job %s%s" % (job_id, " · " + name if name else ""), lines, grow=False)]

    # attempts
    lines = []
    for number, att in enumerate(attempts):
        lines.append(Line([(fmt.fit("#%d" % (number + 1), 4), DIM),
                           (fmt.fit(att.outcome, 10), OUTCOME_STYLE.get(att.outcome, PLAIN)),
                           (fmt.fit(fmt.when(att.start, ctx.now), 13), PLAIN),
                           (fmt.fit(fmt.when(att.end, ctx.now), 13), PLAIN),
                           (fmt.fit(fmt.duration(att.elapsed), 8, ">") + "  ", PLAIN),
                           (fmt.fit("%s GPU" % fmt.num(att.gpus), 7), PLAIN),
                           (att.nodes, RESOURCE)]))
    if not attempts:
        lines.append(text_line("attempts: %s %s" % (
            missing(attempts_entry) if attempts_entry.state != "ok" else fmt.NOT_APPLICABLE,
            attempts_entry.message or ("none in history" if attempts_entry.state == "ok" else "")), DIM))
    panes.append(Pane("attempts", "Attempts (%d)" % len(attempts), lines, grow=False, max_lines=4))

    # steps
    steps, note = [], ""
    if live is not None and (live.holds_resources or live.state == "SUSPENDED") and ctx.collector:
        step_entry = ctx.live(("steps", job_id), lambda: ctx.collector.steps(live))
        report = value(step_entry)
        if report is not None:
            steps = report.steps
            note = "" if report.status in Status.USABLE else "%s: %s" % (fmt.placeholder(report.status), report.detail)
        else:
            note = missing(step_entry)
    if not steps and detail is not None:
        for att in getattr(detail, "attempts", ()):
            steps = att.steps or steps
        if getattr(detail, "status", Status.OK) not in Status.USABLE:
            note = "%s: %s" % (fmt.placeholder(detail.status), detail.message)
    elif not steps and detail is None and not note:
        note = missing(detail_entry)
    table = Table(STEP_COLUMNS, ctx.width - 2)
    lines = []
    for step in steps:
        cells = [step.step_id, step.name or "", step.state or "", fmt.duration(step.elapsed),
                 fmt.num(step.ntasks) if step.ntasks is not None else fmt.NA,
                 fmt.pct(step_cpu_eff(step)), fmt.mem(step.max_rss),
                 ("%.0f" % step.gpu_util_max) if step.gpu_util_max is not None else fmt.NA,
                 fmt.mem(step.gpu_mem_max)]
        lines.append(table.row(cells))
    if note:
        lines.append(text_line(note, DIM if fmt.DENIED not in note else BAD))
    panes.append(Pane("steps", "Steps", lines, table.header(), grow=False, max_lines=5,
                      status="GPU util is Slurm's gpuutil, may sum GPUs"))

    # logs and on-demand text
    lines = []
    paths = None
    if live is not None and ctx.collector:
        path_entry = ctx.live(("paths", job_id), lambda: ctx.collector.log_paths(live))
        lp = value(path_entry)
        if lp is not None:
            if lp.status in Status.USABLE:
                paths = (lp.stdout, lp.raw_stdout, lp.stderr, lp.raw_stderr)
            else:
                lines.append(text_line("log paths: %s %s" % (fmt.placeholder(lp.status), lp.detail),
                                       BAD if lp.status == Status.DENIED else DIM))
    if paths is None and detail is not None and getattr(detail, "attempts", None):
        last = detail.attempts[-1]
        out, err = last.stdout, last.stderr
        paths = (out.display if out else None, out.raw if out else None,
                 err.display if err else None, err.raw if err else None)
    started = None
    if live is not None and live.start and not live.pending:
        started = live.start
    elif attempts and attempts[0].start:
        started = attempts[0].start
    elif row is not None:
        started = row.start
    workdir = None
    if live is not None and live.raw_workdir:
        workdir = (live.workdir, live.raw_workdir)
    elif detail is not None and getattr(detail, "attempts", None) and detail.attempts[-1].workdir:
        last = detail.attempts[-1]
        workdir = (last.workdir, getattr(last, "raw_workdir", None))
    if workdir is not None and workdir[1]:
        view.extra["raw_workdir"] = workdir[1]

    def path_line(label: str, shown_path: Optional[str], raw: Optional[str],
                  directory: bool = False) -> None:
        parts = [(fmt.fit(label, 8), HEAD), (shown_path or fmt.NOT_APPLICABLE, PLAIN)]
        if raw:
            state = ctx.app.path_state(raw, started, directory)
            if state is None:
                parts.append(("  [checking ?]", DIM))
            else:
                style = {"good": GOOD, "warn": WARN, "bad": BAD}.get(state.style, DIM)
                parts.append(("  [%s]" % state.note, style))
        lines.append(Line(parts))

    if ctx.app.anonymized:
        lines.append(text_line("paths, scripts, submit lines and logs are hidden by --anonymize",
                               DIM))
    else:
        if workdir is not None:
            path_line("workdir", workdir[0], workdir[1], directory=True)
        if paths is not None:
            path_line("stdout", paths[0], paths[1])
            if paths[2] and paths[2] != paths[0]:
                path_line("stderr", paths[2], paths[3])
            view.extra["paths"] = paths
        script = view.extra.get("script_path")
        if script:
            path_line("script", script, script)
        lines.append(text_line("paths as Slurm recorded them; checked with os.stat, never read",
                               DIM))
    lines.append(Line([("keys    ", HEAD), ("l/e stdout/stderr tail · b batch script · "
                                            "u submit line (on request)", DIM)]))
    panes.append(Pane("logs", "Logs", lines, grow=False))

    shown = view.extra.get("show")
    if shown:
        entry = ctx.app.loader.peek(("o", "fetch", job_id, shown))
        text_lines = []
        if entry is None or entry.at is None:
            text_lines.append(text_line("asking Slurm… ?", DIM))
        elif entry.state == FAILED:
            text_lines.append(text_line("%s: %s" % (fmt.placeholder(entry.status), entry.message), BAD))
        else:
            status, body, detail_text = entry.value
            if status not in Status.USABLE:
                style = BAD if status == Status.DENIED else DIM
                text_lines.append(text_line("%s: %s" % (fmt.placeholder(status), detail_text), style))
            for text in body:
                text_lines.append(text_line(text))
            if not body and status in Status.USABLE:
                text_lines.append(text_line("(empty) " + detail_text, DIM))
        label = {"stdout": "stdout tail", "stderr": "stderr tail", "script": "Batch script",
                 "submit": "Submit line"}[shown]
        panes.append(Pane("output", label, text_lines))
    return panes


# -- Help --------------------------------------------------------------------

HELP = [
    ("Navigation", HEAD),
    ("  o m n j p a t   Overview, Me, Nodes, Queue (jobs), People, Roles (accounts), Timeline", PLAIN),
    ("  ← →             previous / next tab (in Timeline: move the bin cursor)", PLAIN),
    ("  ↑ ↓ PgUp PgDn   move in the focused pane;  Home/End  first/last", PLAIN),
    ("  Tab Shift-Tab   next / previous pane", PLAIN),
    ("  Enter           drill into the row (›);  Esc / Backspace  back up the breadcrumb", PLAIN),
    ("  /  s  S         filter the pane;  cycle sort column;  reverse sort", PLAIN),
    ("  [ ]  1-5        range: day, week, month, year, all — shared by every view", PLAIN),
    ("  g               Roles: account/QOS/partition;  Person: run types by GPUs, nodes, job name,", PLAIN),
    ("                  account, QOS, partition, outcome;", PLAIN),
    ("                  Timeline: lanes by user/account/QOS/partition", PLAIN),
    ("  + - , .         Timeline zoom and pan", PLAIN),
    ("  l e b u         Job: stdout tail, stderr tail, batch script, submit line (on request only)", PLAIN),
    ("  E               load efficiency (step usage) for the current range and person", PLAIN),
    ("  r  ?  q         refresh now, this help, quit;  mouse: wheel scrolls, click selects, double-click opens", PLAIN),
    ("", PLAIN),
    ("Colors", HEAD),
    ("  cyan     a resource: node, account, QOS, partition", RESOURCE),
    ("  magenta  a person", USER),
    ("  green    productive: allocated, running, completed, efficient", GOOD),
    ("  amber    partial or waiting: pending, preempted, timed out, partly loaded", WARN),
    ("  red      needs attention: down, drained, failed, refused", BAD),
    ("  gray     context, or not available", DIM),
    ("", PLAIN),
    ("Placeholders", HEAD),
    ("  n/a      unavailable here      —  not applicable      ?  not loaded yet", PLAIN),
    ("  denied   Slurm refused you     ERR  the query failed   ≥  window only partly loaded", PLAIN),
    ("", PLAIN),
    ("Glossary", HEAD),
    ("  GPU-h / CPU-h alloc   allocated hours (GPUs or CPUs held × wall time), not measured use", PLAIN),
    ("  util                  allocated hours ÷ (today's live capacity × window length)", PLAIN),
    ("  GPU eff               peak sampled GPU utilization of measured attempts; CPU eff = CPU time ÷ allocated", PLAIN),
    ("  ·cov                  share of allocated hours the efficiency figure describes", PLAIN),
    ("  attempt / job         a requeue or preemption starts another attempt of the same job", PLAIN),
    ("  preempted             stopped by the scheduler for other work; allocated time still counts and", PLAIN),
    ("                        is not a failure or waste", PLAIN),
    ("  fail                  failed or node_fail among finished attempts", PLAIN),
    ("  limit                 elapsed ÷ requested time limit of completed and timed-out attempts", PLAIN),
    ("  stranded GPUs         free GPUs that cannot start: no free core or memory left on the node", PLAIN),
    ("  share                 part of the cluster's allocated GPU-hours in the same range", PLAIN),
    ("  visible               how much of an account's history you can see: full, own only,", PLAIN),
    ("                        denied, or no accounting", PLAIN),
    ("  Job environments are never shown.  Nothing here changes anything in Slurm.", DIM),
]


def help_screen(ctx: Context, view: View) -> List[Pane]:
    lines = []
    for text, style in HELP:
        indent = 18 if text.startswith("  ") and len(text) > 18 and text[17] == " " else 11
        if text.startswith("  ") and "   " in text[2:24]:
            indent = text.index(next(w for w in text[2:].split("  ") if w.strip()), 0)
            indent = len(text) - len(text.lstrip()) + text.lstrip().find("  ")
            while indent < len(text) and text[indent] == " ":
                indent += 1
        for part in wrap(text, ctx.width - 3, indent):
            lines.append(text_line(part, style))
    return [Pane("help", "Help", lines)]


def me(ctx: Context, view: View) -> List[Pane]:
    user = ctx.snap.invoker.user if ctx.snap else view.params.get("user")
    if user is None:
        user = ctx.app.user
    view.params["user"] = user
    return person(ctx, view)


SCREENS = {
    "overview": overview, "me": me, "person": person, "nodes": nodes, "queue": queue,
    "people": people, "roles": roles, "timeline": timeline, "job": job, "jobs": jobs_list,
    "help": help_screen,
}

FOOTERS = {
    "overview": "↵ open",
    "me": "g run types · E efficiency",
    "person": "g run types · E efficiency",
    "roles": "g group",
    "timeline": "g lanes · ←→ bin · +- zoom · ,. pan",
    "job": "l e b u fetch",
}
