"""Every screen: a function from (Context, View) to a list of panes.

Screens never block: data comes from ``Context.live``/``Context.hist``, which
return loader entries that may still be pending (drawn as ``?``).
"""

import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..live import applicable_limits, group_by
from ..model import Status
from . import charts
from . import data as hd
from . import fmt
from . import theme
from .canvas import (ACCENT, BAD, BRIGHT, DIM, FAINT, GOOD, HEAD, KEY, PLAIN, RESOURCE, SEL,
                     USER, WARN, bold, on)
from .loader import FAILED, Entry
from .widgets import Column, Line, Pane, Table, Target, fit_parts, text_line, wrap

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
        # what a pane's content may use: the screen less its box and padding (2 more
        # cells are left for the scroll hint, as tables are built to ctx.width - 2)
        self.width = width - 2
        self.ascii = app.ascii
        self.color = getattr(app, "color", True)
        self.now = time.time()
        self.collector = app.collector

    def hist(self, key: Tuple[Any, ...], fn: Callable[[], Any], version: Any = None) -> Entry:
        """``version``: inputs that change the answer but should keep the old one on
        screen while the new one is computed (a longer page, a search)."""
        return self.app.loader.get(("h",) + key, fn, generation=(self.data.generation, version))

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


def stale(entry: Optional[Entry]) -> bool:
    """The last recomputation failed; what is shown is the previous answer."""
    return entry is not None and entry.state == FAILED and entry.value is not None


def busy(entry: Optional[Entry]) -> bool:
    return entry is not None and (entry.busy or entry.at is None)


SPARK_MOST = 28  # a trend wider than this is drawn this wide, against the right edge
EFFICIENCY = ("GPU util*", "CPU eff")  # columns left out while nothing in them is measured
MIN_COVER = 0.05  # an efficiency measured over less of the allocation than this is not shown


def eff_cell(eff: Any) -> Tuple[str, str]:
    """'54%·80%': efficiency over coverage of the allocation. Measured over almost none of
    it (below MIN_COVER), it is '?': a value from a sliver would be read as the whole."""
    if eff is None:
        return fmt.NA, FAINT
    val, cover = eff
    if val is None:  # nothing measured yet over an allocation: not sampled; no allocation: n/a
        return (fmt.NOT_YET if cover == 0 else fmt.NOT_APPLICABLE), FAINT
    if cover is not None and cover < MIN_COVER:
        return fmt.NOT_YET, FAINT
    style = GOOD if val >= 0.7 else WARN if val >= 0.3 else BAD
    text = fmt.pct(val)
    if cover is not None and cover < 0.995:
        text += "·" + fmt.pct(cover)
    return text, style


def delta_cell(now: Optional[float], before: Optional[float],
               loaded: bool = True) -> List[Tuple[str, str]]:
    return charts.delta(now, before, loaded)


def share_cell(share: Optional[float], ctx: "Context", bar: int = 5) -> Any:
    """A share as a small bar and a percentage: '██▍   35.2%'."""
    if share is None:
        return (fmt.NOT_APPLICABLE, FAINT)
    text = fmt.pct(share, 1)
    return charts.gauge(share, bar, ctx.ascii, ACCENT) + [
        (" " + text.rjust(5)[:-1], PLAIN), ("%", DIM)]


def filt_key(params: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((k, v) for k, v in params.items() if v is not None))


def make_filter(params: Dict[str, Any]) -> Any:
    keys = ("user", "account", "qos", "partition")
    return hd.where(**{k: params.get(k) for k in keys})


def snap_jobs(ctx: Context) -> List[Any]:
    return [job for job in (ctx.snap.jobs or ()) if job.active] if ctx.snap else []


def status_line(label: str, entry: Optional[Entry]) -> Line:
    return text_line("%s: %s %s" % (label, missing(entry),
                                    entry.message if entry is not None else ""), FAINT)


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

# priority: what answers "how much, how well, what's wrong" survives a narrow screen;
# the CPU twins, measured efficiency and raw counts come back as the width allows
TREND_COLUMNS = (
    Column("Range", 5, "<", 10),
    Column("GPU-h", 6, ">", 9),
    Column("util", 4, ">", 8),
    Column("ΔGPU", 5, ">", 7),
    Column("CPU-h", 6, ">", 5),
    Column("util", 4, ">", 4),
    Column("ΔCPU", 5, ">", 1),
    Column("GPU util*", 9, ">", 2),
    Column("CPU eff", 8, ">", 1),
    Column("trend", 20, ">", 6, flex=True, most=SPARK_MOST),
)

PERSON_TREND_COLUMNS = (
    Column("Range", 5, "<", 10),
    Column("GPU-h", 6, ">", 9),
    Column("share", 6, ">", 8),
    Column("ΔGPU", 5, ">", 7),
    Column("jobs", 5, ">", 5),
    Column("CPU-h", 6, ">", 3),
    Column("GPU util*", 9, ">", 4),
    Column("CPU eff", 8, ">", 2),
    Column("trend", 16, ">", 6, flex=True, most=SPARK_MOST),
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


def person_share(row: Any, cluster: Any) -> Any:
    """A person's part of the cluster's GPU-hours, only from equally loaded figures: both
    complete gives the share; the person partly loaded but the cluster complete gives a
    lower bound (≥); a partly loaded cluster total could be further behind than the
    person's, so the share is not known (?)."""
    if cluster is None or row.gpu_hours is None or not cluster.gpu_hours:
        return fmt.NOT_YET if cluster is None else fmt.NOT_APPLICABLE, DIM
    if not cluster.complete:
        return fmt.NOT_YET, DIM
    share = min(1.0, row.gpu_hours / cluster.gpu_hours)
    return ("" if row.complete else "≥") + fmt.pct(share, 1), PLAIN


def trend_pane(ctx: Context, params: Dict[str, Any], person: bool = False) -> Pane:
    table = Table(PERSON_TREND_COLUMNS if person else TREND_COLUMNS, ctx.width - 2,
                  prune=EFFICIENCY)
    lines = []
    incomplete = False
    for name in hd.RANGE_NAMES:
        if person:
            entry = trend_entry(ctx, name, params)
            row = value(entry)
        else:
            row, entry = cluster_trend(ctx, name)
        label = (name, KEY if name == ctx.range else PLAIN)
        if row is None:
            ph = missing(entry)
            cells = [label] + [(ph, FAINT)] * (len(table.columns) - 1)
            lines.append(table.row(cells, key=name))
            continue
        mark = "" if row.complete else "≥"
        incomplete = incomplete or not row.complete
        spark = (lambda size, data=row.spark: charts.spark(data, min(size, SPARK_MOST),
                                                           ctx.ascii, ctx.color))
        if person:
            cells = [label, mark + fmt.hours(row.gpu_hours),
                     person_share(row, cluster_trend(ctx, name)[0]),
                     delta_cell(row.gpu_hours, row.gpu_delta, row.previous_loaded),
                     mark + fmt.num(row.jobs), mark + fmt.hours(row.cpu_hours),
                     eff_cell(row.gpu_eff), eff_cell(row.cpu_eff), spark]
        else:
            util = [fmt.pct(value) for value in (row.gpu_util, row.cpu_util)]
            util = [u if u in fmt.PLACEHOLDERS else mark + u for u in util]
            cells = [label, mark + fmt.hours(row.gpu_hours), util[0],
                     delta_cell(row.gpu_hours, row.gpu_delta, row.previous_loaded),
                     mark + fmt.hours(row.cpu_hours), util[1],
                     delta_cell(row.cpu_hours, row.cpu_delta, row.previous_loaded),
                     eff_cell(row.gpu_eff), eff_cell(row.cpu_eff), spark]
        lines.append(table.row(cells, key=name))
    note = "≥ partly loaded" if incomplete else ""
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


class Card:
    """One tile of the overview's top row: a title, a figure line, a chart line and
    a detail line, each a list of styled parts (or a callable of the inner width)."""

    def __init__(self, key: str, title: str, weight: int, rows: Sequence[Any],
                 note: Sequence[Tuple[str, str]] = ()) -> None:
        self.key, self.title, self.weight, self.rows, self.note = key, title, weight, rows, note


def _figure(left: List[Tuple[str, str]], right: List[Tuple[str, str]]) -> Callable[[int], Any]:
    """Parts at the left, parts flush right, as room allows (the right goes first)."""
    def make(size: int) -> List[Tuple[str, str]]:
        used = charts.width(left) + charts.width(right)
        if used + 1 > size:
            return fit_parts(left, size)
        return list(left) + [(" " * (size - used), PLAIN)] + list(right)
    return make


def _chips(items: Sequence[List[Tuple[str, str]]]) -> Callable[[int], Any]:
    """Short items joined by two spaces; the last ones dropped when they do not fit."""
    def make(size: int) -> List[Tuple[str, str]]:
        out = []  # type: List[Tuple[str, str]]
        for item in items:
            gap = [("  ", PLAIN)] if out else []
            if charts.width(out) + charts.width(gap + item) > size:
                if not out:
                    return fit_parts(item, size)
                break
            out += gap + item
        return out
    return make


def _count(number: Any, word: str, style: str = PLAIN) -> List[Tuple[str, str]]:
    return [(fmt.num(number) if isinstance(number, (int, float)) else str(number), style),
            (" " + word, DIM)]


def resource_card(key: str, title: str, res: Any, ctx: Context,
                  spark: Optional[Sequence[float]] = None) -> Card:
    if res is None or res.total is None:
        return Card(key, title, 5, [[(fmt.NA, FAINT)]])
    fraction = ((res.alloc + (res.completing or 0)) / res.total
                if res.total and res.alloc is not None else None)
    tone = charts.ramp_style(fraction)
    figure = _figure([(fmt.num(res.alloc), BRIGHT), (" / " + fmt.num(res.total), DIM),
                      (" alloc", DIM)],
                     [(fmt.pct(fraction)[:-1], bold(tone)), ("%", tone)])
    details = [_count(res.free, "free")]
    if res.unavailable:
        details.append(_count(res.unavailable, "unavail", BAD))
    if res.completing:
        details.append(_count(res.completing, "completing", WARN))
    if res.shared:
        details.append(_count(res.shared, "shared", WARN))
    if res.reserved:
        details.append(_count(res.reserved, "resv", DIM))
    chips = _chips(details)

    def detail(size: int) -> List[Tuple[str, str]]:
        parts = chips(size)
        room = size - charts.width(parts) - 2
        if spark is not None and room >= 8:
            line = charts.spark(spark, min(room, 24), ctx.ascii, ctx.color)
            return parts + [(" " * (size - charts.width(parts) - charts.width(line)), PLAIN)] + line
        return parts

    return Card(key, title, 5, [figure, lambda size: charts.gauge(fraction, size, ctx.ascii),
                                detail])


def cluster_now(ctx: Context) -> Pane:
    """The top row of cards: GPUs, CPUs, nodes and the queue."""
    snap = ctx.snap
    if snap is None:
        entry = ctx.app.snapshot_entry
        return Pane("now", "Cluster now", [status_line("live", entry)], grow=False)
    t, h = snap.totals, snap.health
    nodes_target = Target("nodes", {}, "Nodes")
    queue_target = Target("queue", {"state": "pending"}, "Queue")
    series = {}  # type: Dict[str, Any]
    row, _ = cluster_trend(ctx, ctx.range) if not ctx.app.no_history else (None, None)
    if row is not None and row.spark is not None:
        series["gpu"] = row.spark
    cards = [resource_card("gpu", "GPUs", t.gpus, ctx, series.get("gpu")),
             resource_card("cpu", "CPUs", t.cpus, ctx)]
    if not t.gpus.total:
        cards = cards[1:]

    # nodes: how many, what is wrong, where a job would start
    down, drain = fmt.down_drain(h.node_states)
    completing = sum(1 for job in snap.jobs or () if job.state == "COMPLETING")
    problems = []  # type: List[List[Tuple[str, str]]]
    if down:
        problems.append([("● ", BAD)] + _count(down, "down", BAD))
    if drain:
        problems.append([("● ", BAD)] + _count(drain, "drain", BAD))
    if completing:
        problems.append([("● ", WARN)] + _count(completing, "completing", WARN))
    down_hosts = [s.host for s in h.scheduler or () if s.up is False]
    if down_hosts:
        problems.insert(0, [("● ", BAD), ("controller down", BAD)])
    if not problems:
        problems = [[("● ", GOOD), ("all healthy", DIM)]]
    stranded = None
    if h.gpus_stranded_cpu is not None or h.gpus_stranded_mem is not None:
        stranded = (h.gpus_stranded_cpu or 0) + (h.gpus_stranded_mem or 0)
    block = h.largest_gpu_block
    if block is not None:
        free = [[(str(block.gpus), PLAIN), (" free on ", DIM), (block.node, RESOURCE)]]
    elif t.gpus.total:
        free = [[("no whole free GPU block", WARN)]]
    else:
        free = [[("no GPUs", DIM)]]
    if stranded:
        free.append(_count(stranded, "stranded", BAD))
    figure = _figure([(fmt.num(t.nodes), BRIGHT), (" nodes", DIM)],
                     [(fmt.num(t.nodes_schedulable), PLAIN), (" sched", DIM)])
    cards.append(Card("nodes", "Nodes", 4, [figure, _chips(problems), _chips(free)]))

    # the queue: running against waiting, and why
    pending = h.pending_by_reason
    running = sum(1 for job in snap.jobs or () if job.holds_resources)
    waiting = sum(job.task_count for job in snap.jobs or () if job.pending)
    total = running + waiting
    figure = _figure([(str(running), bold(GOOD)), (" run  ", DIM),
                      (str(waiting), bold(WARN) if waiting else DIM), (" pend", DIM)], [])
    mix = (lambda size: charts.stacked([(running / total, GOOD), (waiting / total, WARN)],
                                       size, ctx.ascii, ctx.color)) if total else \
        (lambda size: [(fmt.fit("empty", size), FAINT)])
    if pending:
        reasons = [[(usage.key, DIM), (" %d" % usage.pending, PLAIN)] for usage in pending[:4]]
    elif snap.jobs is None:
        cap = snap.capabilities.get("jobs")
        reasons = [[(fmt.placeholder(cap.status if cap else None), FAINT)]]
    else:
        reasons = [[("nothing waiting", DIM)]]
    cards.append(Card("queue", "Queue", 4, [figure, mix, _chips(reasons)]))

    targets = {"gpu": nodes_target, "cpu": nodes_target, "nodes": nodes_target,
               "queue": queue_target}
    lines = []
    for card in cards:
        text = card.title + " " + " ".join(
            "".join(t for t, _ in (r(40) if callable(r) else r)) for r in card.rows)
        line = Line([(text, PLAIN)], targets[card.key], key=card.key)
        lines.append(line)
    by_key = {card.key: card for card in cards}

    def draw(canvas: Any, top: int, rows: int, width: int, state: Any, focused: bool,
             shown: List[Line]) -> List[Tuple[int, ...]]:
        return draw_cards(canvas, top, rows, width, [by_key[l.key] for l in shown
                                                      if l.key in by_key],
                          state.cursor if focused else -1, theme.box(ctx.ascii))

    return Pane("now", "Cluster now", lines, grow=False, draw=draw, fixed=5, frame=False,
                min_lines=5, max_lines=5)


def draw_cards(canvas: Any, top: int, rows: int, width: int, cards: Sequence[Card],
               cursor: int, glyphs: theme.Box) -> List[Tuple[int, ...]]:
    """Cards side by side, one cell apart, widths by weight; the selected one in the
    accent color."""
    hits = []  # type: List[Tuple[int, ...]]
    if not cards or rows <= 0:
        return hits
    gap = 1
    room = width - gap * (len(cards) - 1)
    weights = sum(card.weight for card in cards)
    sizes = [room * card.weight // weights for card in cards]
    sizes[-1] += room - sum(sizes)
    col = 0
    for index, (card, size) in enumerate(zip(cards, sizes)):
        chosen = index == cursor
        border = ACCENT if chosen else FAINT
        inner = size - 4
        body = rows - 2 if rows >= 3 else rows
        if rows >= 3:
            canvas.put(top, col, glyphs.tl + glyphs.h + " ", border)
            end = canvas.put(top, col + 3, card.title, KEY if chosen else HEAD,
                             limit=max(0, size - 6))
            canvas.put(top, end, " " + glyphs.h * max(0, col + size - 1 - end - 1), border)
            canvas.put(top, col + size - 1, glyphs.tr, border)
            canvas.put(top + rows - 1, col, glyphs.bl + glyphs.h * (size - 2) + glyphs.br,
                       border)
        start = top + 1 if rows >= 3 else top
        for number in range(body):
            at = start + number
            if rows >= 3:
                canvas.put(at, col, glyphs.v, border)
                canvas.put(at, col + size - 1, glyphs.v, border)
            if number < len(card.rows):
                made = card.rows[number]
                parts = made(inner) if callable(made) else made
                canvas.segments(at, col + 2, fit_parts(parts, inner) if charts.width(parts)
                                > inner else parts, inner)
        for at in range(top, top + rows):
            hits.append((at, index, col, col + size))
        col += size + gap
    return hits


ACCOUNT_COLUMNS = (
    Column("Account", 12, "<", 10, flex=True, most=22),
    Column("run", 4, ">", 8),
    Column("pend", 4, ">", 7),
    Column("GPUs", 4, ">", 8),
    Column("cap", 7, ">", 5),
    Column("GPU-h", 6, ">", 9),
    Column("share", 6, ">", 8),
    Column("GPU util*", 9, ">", 2),
    Column("members", 7, ">", 3),
    Column("trend", 10, ">", 6, flex=True, most=SPARK_MOST),
    Column("visible", 8, "<", 1),
)


def _gpu_cap(ctx: Context, account: str) -> Optional[Tuple[Optional[float], float]]:
    for limit in (ctx.snap.limits or ()) if ctx.snap else ():
        if (limit.scope == "assoc" and limit.user is None and limit.account == account
                and limit.metric == "gres/gpu" and limit.name.startswith("GrpTRES")
                and limit.limit is not None and not limit.name.endswith("Mins")):
            return limit.used, limit.limit
    return None


def rollup_summary(ctx: Context, range_name: Optional[str] = None) -> Any:
    """The history layer's hourly summary of a range, if it has one loaded (else None)."""
    if not ctx.data.has("overview"):
        return None
    found = value(overview_entry(ctx))
    if found is None:
        return None
    name = range_name or ctx.range
    return next((r for r in found.ranges if r.name == name), None)


def summary_note(summary: Any, ctx: Context) -> str:
    end = getattr(summary, "end", None)
    return "hourly summary to %s, exact figures computing…" % fmt.when(end, ctx.now) \
        if end else "hourly summary, exact figures computing…"


def accounts_pane(ctx: Context) -> Pane:
    """Accounts in the range. The figures come from the same query as Roles (so the two
    always agree); while it computes, the hourly summary stands in and says so."""
    snap = ctx.snap
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
    summary = None if report is not None else rollup_summary(ctx)
    if report is not None:
        total_gpu = report.total.gpu_seconds
        for key, stats in report.group("account").items():
            hist[key[0]] = (stats.gpu_hours, stats.gpu_seconds / total_gpu if total_gpu else None,
                            stats.efficiency("gpu"), report.complete and stats.complete("gpu"))
        names.update(hist)
        for (user, account), _ in report.group("user", "account").items():
            users.setdefault(account, set()).add(user)
    elif summary is not None:
        for name, item in summary.accounts.items():
            hist[name] = (item.amounts.gpu_hours, item.share, item.amounts.efficiency("gpu"),
                          False)
            users.setdefault(name, set()).update(item.users)
        names.update(hist)
    loaded = summary is not None or report is not None
    invoker = snap.invoker if snap else None
    mine = [r.account for r in (snap.my_roles or ())] if snap else []
    info = value(ctx.app.history_info)
    # "visible" earns its column only when some account is not fully visible
    seen = {name: (hd.visibility(info, invoker, name, mine) if invoker else fmt.NOT_YET)
            for name in names if name}
    columns = list(ACCOUNT_COLUMNS)
    if ctx.app.no_history or all(v == "full" for v in seen.values()):
        columns = [c for c in columns if c.name != "visible"]
    else:
        columns = [c._replace(priority=8) if c.name == "visible" else c for c in columns]
    table = Table(columns, ctx.width - 2, prune=EFFICIENCY)
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
                          eff_cell(eff), (lambda size, data=spark: charts.spark(
                              data, min(size, SPARK_MOST), ctx.ascii, ctx.color))
                          if spark is not None else ""]
        else:
            ph = "0" if loaded else missing(entry)
            hist_cells = [ph, fmt.NOT_APPLICABLE if loaded else ph,
                          (fmt.NOT_APPLICABLE if loaded else ph, FAINT), ""]
        if ctx.data.failure and not loaded:
            hist_cells = [fmt.placeholder(ctx.data.failure[0])] * 3 + [""]
        vis = seen.get(name, fmt.NOT_YET)
        if ctx.app.no_history:
            vis = "history off"
        vis_style = GOOD if vis == "full" else WARN if vis == "own only" else DIM
        cap_text = fmt.NOT_APPLICABLE
        cap_style = FAINT
        if cap is not None:
            used, limit = cap
            cap_text = "%s/%s" % (fmt.num(used) if used is not None else "?", fmt.num(limit))
            cap_style = BAD if used is not None and used >= limit else PLAIN
        quiet = lambda n: (str(n), PLAIN) if n else ("0", FAINT)
        cells = [(name, RESOURCE), quiet(use.running if use else 0),
                 (str(use.pending), WARN) if use and use.pending else ("0", FAINT),
                 quiet(use.gpus if use else 0), (cap_text, cap_style)] + hist_cells[:3] + [
                     fmt.num(len(users.get(name, ()))) if name in users else fmt.NA,
                     hist_cells[3], (vis, vis_style)]
        lines.append(table.row(
            cells, Target("people", {"account": name}, "People in " + name),
            {"name": name, "GPU-h": (stats[0] or 0) if stats else 0,
             "running": use.running if use else 0, "pending": use.pending if use else 0,
             "GPUs": (use.gpus or 0) if use else 0}, key=name))
    lines.sort(key=lambda line: (-(line.sort.get("GPU-h") or 0), -(line.sort.get("GPUs") or 0),
                                 line.sort.get("name") or ""))
    if report is not None:
        status = ""
    elif summary is not None:
        status = summary_note(summary, ctx)
    else:
        status = "history " + missing(entry)
    return Pane("accounts", "Accounts · %s" % ctx.range, lines, table.header(),
                sorts=("GPU-h", "GPUs", "running", "pending", "name"), status=status,
                natural="GPU-h")


# -- People ------------------------------------------------------------------

PEOPLE_COLUMNS = (
    Column("User", 12, "<", 10, flex=True, most=22),
    Column("accounts", 12, "<", 3, flex=True, most=30),
    Column("run", 4, ">", 8),
    Column("pend", 4, ">", 7),
    Column("GPUs", 4, ">", 8),
    Column("GPU-h", 6, ">", 9),
    Column("share", 12, ">", 8),
    Column("CPU-h", 6, ">", 4),
    Column("GPU util*", 9, ">", 2),
    Column("jobs", 5, ">", 5),
)

SHARE_OF = {"account": "of acct", "qos": "of QOS", "partition": "of part", "user": "of user"}


def share_columns(columns: Sequence[Column], params: Dict[str, Any]) -> List[Column]:
    """A drilled-into list's "share" is of its parent's total: say which."""
    scope = [k for k in ("partition", "qos", "account", "user") if params.get(k)]
    if not scope:
        return list(columns)
    label = SHARE_OF[scope[0]] if len(scope) == 1 else "of sel."
    return [c._replace(name=label, width=max(c.width, 7)) if c.name == "share" else c
            for c in columns]


def live_cells(use: Any, cpus: bool = False) -> List[Any]:
    """run, pend (and CPUs) now: zeros recede, waiting is amber."""
    running = use.running if use else 0
    pending = use.pending if use else 0
    gpus = (use.gpus or 0) if use else 0
    cells = [(str(running), PLAIN) if running else ("0", FAINT),
             (str(pending), WARN) if pending else ("0", FAINT),
             (fmt.num(gpus), PLAIN) if gpus else ("0", FAINT)]
    if cpus:
        count = (use.cpus or 0) if use else 0
        cells.append((fmt.num(count), PLAIN) if count else ("0", FAINT))
    return cells


class _Amounts:
    """Rollup amounts shaped like Stats for the People and Roles tables."""

    def __init__(self, amounts: Any) -> None:
        self.gpu_seconds, self.gpu_hours = amounts.gpu_seconds, amounts.gpu_hours
        self.cpu_hours = amounts.cpu_hours
        self.jobs = None
        self._amounts = amounts

    def efficiency(self, metric: str) -> Any:
        return self._amounts.efficiency(metric)


def rollup_users(summary: Any, account: Optional[str] = None) -> Dict[str, Any]:
    """Per-user totals from the hourly summary (all accounts, or one)."""
    out = {}  # type: Dict[str, Any]
    for name, item in summary.accounts.items():
        if account is not None and name != account:
            continue
        for user, amounts in item.users.items():
            if user not in out:
                out[user] = type(amounts)()
            out[user].add(amounts)
    return out


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
    hist = {}  # type: Dict[str, Any]
    total = None
    summary = None
    if report is not None:
        hist = {key[0]: stats for key, stats in report.group("user").items()}
        total = report.total.gpu_seconds
        for (user, account), _ in report.group("user", "account").items():
            accounts.setdefault(user, set()).add(account)
    elif not params["qos"] and not params["partition"]:
        summary = rollup_summary(ctx)  # stands in while the exact figures compute
        if summary is not None:
            found = rollup_users(summary, params["account"])
            hist = {user: _Amounts(amounts) for user, amounts in found.items()}
            total = sum(a.gpu_seconds for a in found.values())
            for name, item in summary.accounts.items():
                if params["account"] is None or name == params["account"]:
                    for user in item.users:
                        accounts.setdefault(user, set()).add(name)
    if ctx.snap and params["account"]:
        for role in ctx.snap.roles or ():
            if role.account == params["account"] and role.user:
                accounts.setdefault(role.user, set()).add(role.account)
    table = Table(share_columns(PEOPLE_COLUMNS, params), ctx.width - 2, prune=EFFICIENCY)
    lines = []
    me = ctx.snap.invoker.user if ctx.snap else None
    loaded = report is not None or summary is not None
    mark = "" if report is not None else "≥" if summary is not None else ""
    for user in sorted(set(live) | set(hist) | set(accounts)):
        if not user:
            continue
        use, stats = live.get(user), hist.get(user)
        if stats is not None:
            hist_cells = [mark + fmt.hours(stats.gpu_hours),
                          share_cell(stats.gpu_seconds / total if total else None, ctx),
                          mark + fmt.hours(stats.cpu_hours), eff_cell(stats.efficiency("gpu")),
                          fmt.num(stats.jobs) if stats.jobs is not None else (fmt.NOT_YET, FAINT)]
        elif loaded:
            hist_cells = [("0", FAINT), fmt.NOT_APPLICABLE, ("0", FAINT),
                          (fmt.NOT_APPLICABLE, FAINT),
                          ("0", FAINT) if report is not None else (fmt.NOT_YET, FAINT)]
        else:
            hist_cells = [missing(entry)] * 5
        cells = [(user + (" (you)" if user == me else ""), bold(USER) if user == me else USER),
                 (",".join(sorted(a for a in accounts.get(user, ()) if a)), RESOURCE)
                 ] + live_cells(use) + hist_cells
        line = table.row(cells, Target("person", {"user": user}, user),
                         {"GPU-h": stats.gpu_seconds if stats else 0,
                          "GPUs": (use.gpus or 0) if use else 0,
                          "pending": use.pending if use else 0, "user": user}, key=user)
        line.pinned = user == me
        lines.append(line)
    lines.sort(key=lambda l: (-(l.sort["GPU-h"] or 0), -(l.sort["GPUs"]), l.sort["user"]))
    scope = " · ".join("%s %s" % (k, v) for k, v in params.items() if v)
    if report is not None:
        status = ""
    elif summary is not None:
        status = summary_note(summary, ctx)
    else:
        status = "computing…" if busy(entry) else "history " + missing(entry)
    return [Pane("people", "People" + (" · " + scope if scope else "") + " · " + ctx.range,
                 lines, table.header(), sorts=("GPU-h", "GPUs", "pending", "user"),
                 status=status, natural="GPU-h")]


# -- Roles -------------------------------------------------------------------

ROLE_COLUMNS = (
    Column("Name", 14, "<", 10, flex=True, most=24),
    Column("active", 6, ">", 5),
    Column("run", 4, ">", 8),
    Column("pend", 4, ">", 7),
    Column("GPUs", 4, ">", 8),
    Column("CPUs", 5, ">", 3),
    Column("GPU-h", 6, ">", 9),
    Column("share", 12, ">", 8),
    Column("CPU-h", 6, ">", 3),
    Column("GPU util*", 9, ">", 2),
    Column("limits", 6, ">", 4),
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
    summary = None
    if report is not None:
        hist = {key[0]: stats for key, stats in report.group(by).items()}
        total = report.total.gpu_seconds
        for (user, key), _ in report.group("user", by).items():
            users.setdefault(key, set()).add(user)
    elif by == "account":
        summary = rollup_summary(ctx)  # stands in while the exact figures compute
        if summary is not None:
            hist = {name: _Amounts(item.amounts) for name, item in summary.accounts.items()}
            total = summary.total.gpu_seconds
            for name, item in summary.accounts.items():
                users.setdefault(name, set()).update(item.users)
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
    table = Table(ROLE_COLUMNS, ctx.width - 2, prune=EFFICIENCY)
    lines = []
    loaded = report is not None or summary is not None
    mark = "≥" if summary is not None else ""
    for name in sorted(n for n in names if n):
        use, stats = live.get(name), hist.get(name)
        if stats is not None:
            hist_cells = [mark + fmt.hours(stats.gpu_hours),
                          share_cell(stats.gpu_seconds / total if total else None, ctx),
                          mark + fmt.hours(stats.cpu_hours), eff_cell(stats.efficiency("gpu"))]
        elif loaded:  # no usage in the range: nothing to measure, not a measured zero
            hist_cells = [("0", FAINT), fmt.NOT_APPLICABLE, ("0", FAINT),
                          (fmt.NOT_APPLICABLE, FAINT)]
        else:
            hist_cells = [missing(entry)] * 4
        cells = [(name, RESOURCE), fmt.num(len(users.get(name, ())))] + live_cells(use, True) \
            + hist_cells + [str(limits[name]) if name in limits else fmt.NOT_APPLICABLE]
        lines.append(table.row(cells, Target("people", {by: name}, "%s %s" % (by, name)),
                               {"GPU-h": stats.gpu_seconds if stats else 0,
                                "GPUs": (use.gpus or 0) if use else 0, "name": name}, key=name))
    lines.sort(key=lambda l: (-(l.sort["GPU-h"] or 0), -l.sort["GPUs"], l.sort["name"]))
    label = {"account": "Accounts", "qos": "QOS", "partition": "Partitions"}[by]
    status = ""
    if summary is not None:
        status = summary_note(summary, ctx)
    elif report is None and busy(entry):
        status = "computing…"
    return [Pane("roles", "Roles: %s · %s" % (label, ctx.range), lines, table.header(),
                 sorts=("GPU-h", "GPUs", "name"), status=status, natural="GPU-h")]


# -- Nodes -------------------------------------------------------------------

NODE_COLUMNS = (
    Column("Node", 8, "<", 10, flex=True, most=16),
    Column("state", 10, "<", 9, flex=True, most=22),
    Column("GPUs", 13, "<", 8),
    Column("CPUs", 7, ">", 7),
    Column("mem", 11, ">", 3),
    Column("type", 5, "<", 4, flex=True, most=12),
    Column("load", 5, ">", 2),
    Column("partitions", 10, "<", 1, flex=True, most=12),
    Column("reason", 12, "<", 6, flex=True),
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
    untracked = False
    for node in snap.nodes:
        state = (node.state or "?") + ("+" + "+".join(node.state_flags) if node.state_flags else "")
        if node.gpus_total:
            gpu = charts.pips(node.gpus_alloc, node.gpus_total, 8, ctx.ascii) + [
                (" " + fmt.ratio(node.gpus_alloc, node.gpus_total), PLAIN)]
        else:
            gpu = (fmt.NOT_APPLICABLE, FAINT)
        if node.mem_alloc is not None:
            mem = "%s/%s" % (fmt.mem(node.mem_alloc), fmt.mem(node.mem_total))
        else:  # Slurm does not track memory allocation here: show the size, not a usage
            untracked = untracked or node.mem_total is not None
            mem = (fmt.NA + "/" + fmt.mem(node.mem_total), FAINT)
        types = ",".join(sorted({fmt.gpu_model(t) for t in node.gpu_types}))
        cells = [(node.name, RESOURCE), (state, node_style(node)), gpu,
                 fmt.ratio(node.cpus_alloc, node.cpus_total), mem, (types or fmt.NOT_APPLICABLE,
                                                                     DIM if types else FAINT),
                 "%.1f" % node.cpu_load if node.cpu_load is not None else fmt.NA,
                 (",".join(node.partitions), DIM),
                 (node.reason or "", WARN if node.reason else PLAIN)]
        free = node.gpus_free
        lines.append(table.row(cells, Target("queue", {"node": node.name}, "jobs on " + node.name),
                               {"name": node.name, "GPUs free": free, "CPUs free": node.cpus_free,
                                "load": node.cpu_load, "state": state}, key=node.name))
    problems = sum(1 for node in snap.nodes if node_style(node) == BAD)
    notes = [[("● ", BAD), ("%d need attention" % problems, BAD)]] if problems else []
    return [Pane("nodes", "Nodes (%d)" % len(snap.nodes), lines, table.header(),
                 sorts=("name", "GPUs free", "CPUs free", "load", "state"),
                 status=notes)]


# -- Queue -------------------------------------------------------------------

QUEUE_COLUMNS = (
    Column("Job", 10, "<", 10, flex=True, most=14),
    Column("user", 8, "<", 9, flex=True, most=14),
    Column("account", 8, "<", 5, flex=True, most=12),
    Column("qos", 6, "<", 2),
    Column("partition", 9, "<", 1),
    Column("state", 8, "<", 8, flex=True, most=11),
    Column("GPUs", 4, ">", 8),
    Column("CPUs", 4, ">", 3),
    Column("time", 7, ">", 7),
    Column("reason/nodes", 14, "<", 6, flex=True),
    Column("name", 14, "<", 4, flex=True),
)


def job_style(job: Any) -> str:
    if job.holds_resources:
        return WARN if job.state == "COMPLETING" else GOOD
    if job.pending:
        return WARN
    return DIM


def live_elapsed(job: Any, now: float) -> Optional[float]:
    """A live job's run time. squeue reports 0 for a COMPLETING job (and none for some
    states): then it is start to end, or to now while the end is not known."""
    if job.elapsed and job.state != "COMPLETING":
        return job.elapsed
    if job.start:
        stop = job.end if job.end and job.end <= now and job.state == "COMPLETING" else now
        return max(job.elapsed or 0, stop - job.start)
    return job.elapsed


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
            spent = live_elapsed(job, ctx.now)
            where = ",".join(job.nodes[:3]) + ("…" if len(job.nodes) > 3 else "")
            gpus, cpus = job.gpus_alloc, job.cpus_alloc
        count = ""
        cells = [job.display_id or job.job_id, (job.user or "?", USER), (job.account or "", RESOURCE),
                 (job.qos or "", DIM), (job.partition or "", DIM),
                 ((job.state or "?") + count, job_style(job)),
                 fmt.num(gpus) if gpus is not None else fmt.NA,
                 fmt.num(cpus) if cpus is not None else fmt.NA, fmt.duration(spent),
                 (where, WARN if job.pending else RESOURCE), (job.name or "", DIM)]
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
    Column("Run type", 14, "<", 10, flex=True, most=40),
    Column("att", 4, ">", 3),
    Column("jobs", 5, ">", 8),
    Column("GPU-h", 6, ">", 9),
    Column("share", 5, ">", 8),
    Column("runtime", 7, ">", 6),
    Column("wait", 6, ">", 3),
    Column("limit", 5, ">", 2),
    Column("GPU util*", 9, ">", 4),
    Column("CPU eff", 8, ">", 1),
    Column("fail", 4, ">", 7),
)

LIVE_COLUMNS = (
    Column("Job", 10, "<", 10),
    Column("state", 8, "<", 9),
    Column("GPUs", 6, ">", 8),
    Column("time", 7, ">", 7),
    Column("account", 8, "<", 4, flex=True, most=12),
    Column("reason/nodes", 12, "<", 6, flex=True, most=24),
    Column("name", 8, "<", 5, flex=True),
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
    widest = {"account": max([fmt.width(j.account or "") for j in live] + [7]),
              "reason/nodes": max([fmt.width(",".join(j.nodes[:2]) if j.holds_resources
                                             else j.reason or "") for j in live] + [8])}
    table = Table([c._replace(width=min(24, widest[c.name]), flex=False)
                   if c.name in widest else c for c in LIVE_COLUMNS], ctx.width - 2)
    for job in sorted(live, key=lambda j: (not j.holds_resources, -(j.gpus_alloc or j.gpus_req or 0))):
        size = job.gpus_alloc if job.holds_resources else job.gpus_req
        spent = (live_elapsed(job, ctx.now) if job.holds_resources
                 else (ctx.now - job.submit if job.submit else None))
        where = ",".join(job.nodes[:2]) if job.holds_resources else (job.reason or "")
        lines.append(table.row(
            [job.display_id or job.job_id, (job.state or "?", job_style(job)),
             [(fmt.num(size), PLAIN), (" GPU", DIM)] if size else ("CPU", DIM),
             (fmt.duration(spent), DIM),
             (job.account or "", RESOURCE), (where, WARN if job.pending else RESOURCE),
             job.name or ""],
            Target("job", {"job_id": job.job_id}, "job " + (job.display_id or job.job_id))))
    run = sum(1 for j in live if j.holds_resources)
    status = "%d running · %d pending" % (run, sum(j.task_count for j in live if j.pending))
    if ctx.snap is None:
        lines, status = [status_line("live", ctx.app.snapshot_entry)], ""
    panes.append(Pane("live", "Now", lines, status=status, grow=False, max_lines=4))
    more = "more"  # one box at the bottom: events, jobs, activity, roles as tabs

    # kinds of runs. Every lambda binds its inputs now: it runs later on a worker thread,
    # after this function has moved on (a late-bound loop variable once turned the range
    # into an outcome name there, and every query failed with ERR 'other').
    dim = view.extra.get("dim", "gpu_count")
    dim_label = dict(hd.DIMENSIONS)[dim]
    filt = make_filter(params)
    range_name = ctx.range
    types_entry = ctx.hist(("runtypes", range_name, user, dim),
                           lambda r=range_name, f=filt, d=dim: ctx.data.run_types(r, f, d))
    kinds = value(types_entry)
    table = Table(share_columns(RUN_COLUMNS, {"user": user}), ctx.width - 2, prune=EFFICIENCY)
    lines = []
    if kinds is None:
        lines.append(text_line("history: %s %s" % (missing(types_entry), types_entry.message), DIM))
    else:
        for kind in kinds:
            style = RESOURCE if dim in ("account", "qos", "partition") else PLAIN
            if dim == "outcome":
                style = OUTCOME_STYLE.get(kind.key, PLAIN)
            fail_style = (BAD if kind.failure and kind.failure >= 0.25 else
                          FAINT if not kind.failure else PLAIN)
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
    runs_status = [[("g", KEY), (" group", DIM)]]  # type: List[Any]
    if stale(types_entry):
        runs_status.insert(0, "stale: %s %s" % (fmt.placeholder(types_entry.status),
                                                types_entry.message))
    panes.append(Pane("runs", "What kinds of runs · by %s · %s" % (dim_label, ctx.range), lines,
                      table.header(), sorts=("GPU-h", "attempts", "fail", "runtime"),
                      status=runs_status))

    # events
    report_entry = range_report(ctx, params, (("account",), ("qos",), ("partition",)))
    report = value(report_entry)
    events_entry = ctx.hist(("events", range_name, user),
                            lambda r=range_name, f=filt: ctx.data.events(r, f))
    found = value(events_entry)
    lines = []
    if found is not None:
        for event in found:
            style = {"good": GOOD, "bad": BAD, "warn": WARN, "dim": DIM}.get(event.style, PLAIN)
            lines.append(Line([(fmt.fit(fmt.when(event.at, ctx.now), 12, ">") + "  ", DIM),
                               ("● " if event.style in ("good", "bad", "warn") else "· ",
                                style if style != PLAIN else FAINT),
                               (event.text, style)], event_target(event, user),
                              {"time": event.at}))
    else:
        lines.append(text_line("history: %s %s" % (missing(events_entry), events_entry.message),
                               FAINT))
    panes.append(Pane("events", "Events · " + ctx.range, lines, sorts=("time",), group=more,
                      tab="Events"))

    # roles and outcomes
    lines = []
    if report is not None:
        total = report.total.gpu_seconds or 0
        for role in ("account", "qos", "partition"):
            ranked = report.ranked(role)
            parts = [(fmt.fit(role.replace("qos", "QOS"), 10), DIM)]
            for number, (key, stats) in enumerate(ranked[:4]):
                if number:
                    parts.append(("  ", DIM))
                share = stats.gpu_seconds / total if total else None
                parts += [(key[0], RESOURCE), (" " + fmt.hours(stats.gpu_hours), PLAIN),
                          (" GPU-h ", DIM), (fmt.pct(share), PLAIN)]
            lines.append(Line(parts, Target("people", {role: ranked[0][0][0]}, role)
                              if ranked else None))
        parts = [(fmt.fit("outcome", 10), DIM)]
        first = True
        for outcome in hd.OUTCOME_ORDER:
            stats = report.outcomes.get(outcome)
            if stats is None or not stats.attempts:
                continue
            if not first:
                parts.append(("  ", DIM))
            first = False
            style = {"completed": GOOD, "running": GOOD, "failed": BAD, "node_fail": BAD,
                     "timeout": WARN, "preempted": WARN}.get(outcome, PLAIN)
            parts += [(outcome, style), (" %d" % stats.attempts, PLAIN)]
        lines.append(Line(parts))
    else:
        lines.append(text_line("history: %s" % missing(report_entry), DIM))
    split = Pane("split", "Roles and outcomes · " + ctx.range, lines, grow=False,
                 group=more, tab="Roles")

    # jobs list: a page at a time; a search asks history, not only the rows loaded
    jobs = paged_jobs(ctx, view, "Jobs · " + ctx.range, ("jobs", ctx.range, user, "gpu"), filt)
    jobs.group, jobs.tab = more, jobs.title.replace(" · " + ctx.range, "")
    activity = activity_pane(ctx, params, report, report_entry)
    activity.group, activity.tab = more, "Activity"
    panes += [jobs, activity, split]
    return panes


def column_chart(values: Sequence[float], size: int, rows: int, ascii_only: bool,
                 color: bool) -> List[List[Tuple[str, str]]]:
    """A bar chart ``rows`` tall in eighth-cell steps, one bar per cell, bars shaded on
    the ramp by height; the bottom row keeps a faint baseline under empty bars."""
    data = fmt.resample(list(values), size)
    top = max(data) if data else 0
    marks = charts.ASCII_SPARKS if ascii_only else charts.EIGHTHS
    out = []
    for level in range(rows - 1, -1, -1):
        cells = []  # type: List[Tuple[str, str]]
        for value in data:
            height = value / top * rows * 8 if top > 0 else 0
            filled = height - level * 8
            tone = charts.heat_style(0.35 + 0.65 * (value / top)) if color and top else PLAIN
            if filled >= 8:
                cells.append((marks[-1], tone))
            elif filled >= 1:
                cells.append((marks[max(0, int(filled) - 1)], tone))
            elif level == 0:
                cells.append((marks[0] if color else " ", charts.FAINT))
            else:
                cells.append((" ", PLAIN))
        out.append(charts._join(cells))
    return out


def activity_pane(ctx: Context, params: Dict[str, Any], report: Any, entry: Entry) -> Pane:
    lines = []
    if report is None:
        return Pane("activity", "Activity", [text_line("history: %s" % missing(entry), FAINT)],
                    grow=False)
    total = report.total
    unit = "hour" if ctx.range == "day" else "day"
    buckets = report.buckets("gpu", unit) or report.buckets("cpu", unit)
    metric = "GPU" if report.buckets("gpu", unit) else "CPU"
    label = 12
    if buckets:
        values = [b.value for b in buckets]
        peak = max(values) / 3600.0
        size = max(10, min(len(values) * max(1, (ctx.width - label - 4) // max(1, len(values))),
                           ctx.width - label - 4))
        chart = column_chart(values, size, 3, ctx.ascii, ctx.color)
        heads = [[("%s-h" % metric, DIM)], [("per " + unit, DIM)], [("peak ", DIM),
                                                                    (fmt.hours(peak), PLAIN)]]
        for head, row in zip(heads, chart):
            lines.append(Line(fit_parts(head, label) + row))
    grid = report.heatmap("gpu") or report.heatmap("cpu")
    if grid:
        top = max(max(row) for row in grid) or 1
        step = 2 if ctx.width >= 60 else 1
        axis = "".join(fmt.fit(str(h), step * 3) for h in range(0, 24, 3))
        lines.append(Line([(fmt.fit("", label), PLAIN), (axis, FAINT), ("  local hour", FAINT)]))
        for number, day in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
            cells = []  # type: List[Tuple[str, str]]
            for v in grid[number]:
                cells += charts.heat_cells([v / top], ctx.ascii, ctx.color, "█") * 1 \
                    if v > 0 else [(("·" if not ctx.ascii else ".") if ctx.color else " ",
                                    charts.FAINT)]
            row = []  # type: List[Tuple[str, str]]
            for text, style in cells:
                row.append((text * step, style))
            tail = charts.legend(ctx.ascii, ctx.color, "█") if number == 6 else []
            lines.append(Line([(fmt.fit(day, label), DIM)] + charts._join(row) +
                              ([("  ", PLAIN)] + tail if tail else [])))
    gpu, cpu = total.efficiency("gpu"), total.efficiency("cpu")
    parts = [(fmt.fit("Efficiency", label), DIM), ("GPU util ", DIM), eff_cell(gpu),
             ("  CPU ", DIM), eff_cell(cpu)]
    if report.efficiency_unchecked:
        parts += [("  %d attempts not measured (" % report.efficiency_unchecked, DIM),
                  ("E", KEY), (" loads)", DIM)]
    lines.append(Line(parts))
    lines.append(Line([(fmt.fit("Waits", label), DIM), ("median ", DIM),
                       (fmt.duration(report.wait_quantile(0.5)), PLAIN), ("  p90 ", DIM),
                       (fmt.duration(report.wait_quantile(0.9)), PLAIN), ("  max ", DIM),
                       (fmt.duration(total.wait_max), PLAIN),
                       ("  over %d starts" % total.wait_count, DIM)]))
    used = total.time_limit_used
    if used is not None:
        lines.append(Line([(fmt.fit("Time limit", label), DIM)] +
                          charts.gauge(used, 10, ctx.ascii, ACCENT) +
                          [(" " + fmt.pct(used), PLAIN),
                           (" of requested time used (completed + timed out, %d attempts)"
                            % total.limit_count, DIM)]))
    else:
        lines.append(Line([(fmt.fit("Time limit", label), DIM),
                           ("no finished attempt with a limit", FAINT)]))
    return Pane("activity", "Activity · " + ctx.range, lines, grow=False)


JOB_COLUMNS = (
    Column("Job", 10, "<", 10),
    Column("name", 12, "<", 7, flex=True, most=40),
    Column("user", 8, "<", 4, flex=True, most=14),
    Column("outcome", 9, "<", 9),
    Column("start", 12, "<", 5),
    Column("elapsed", 7, ">", 8),
    Column("GPUs", 4, ">", 7),
    Column("GPU-h", 6, ">", 8),
    Column("wait", 6, ">", 2),
    Column("att", 3, ">", 1),
)

OUTCOME_STYLE = {"completed": GOOD, "running": GOOD, "failed": BAD, "node_fail": BAD,
                 "timeout": WARN, "preempted": WARN, "cancelled": DIM}


def jobs_pane(ctx: Context, pane_id: str, title: str, rows: Optional[Sequence[Any]],
              entry: Optional[Entry] = None, total: Optional[int] = None,
              match: str = "", hours: str = "GPU-h") -> Pane:
    """``total``: how many jobs there are when ``rows`` is only the first of them.
    ``hours``: the GPU-hours column's label ("GPU-h in bin" for a timeline bin)."""
    columns = [c._replace(name=hours, width=max(c.width, len(hours))) if c.name == "GPU-h"
               else c for c in JOB_COLUMNS]
    table = Table(columns, ctx.width - 2)
    lines = []
    if rows is None:
        return Pane(pane_id, title, [text_line("history: %s %s" % (
            missing(entry), entry.message if entry else ""), DIM)])
    for row in rows:
        cells = [row.display_id, row.name, (row.user, USER),
                 (row.outcome, OUTCOME_STYLE.get(row.outcome, PLAIN)),
                 (fmt.when(row.start, ctx.now), DIM), fmt.duration(row.elapsed),
                 fmt.num(row.gpus) if row.gpus is not None else fmt.NA,
                 fmt.hours(row.gpu_seconds / 3600.0) if row.gpu_seconds is not None else fmt.NA,
                 (fmt.duration(row.wait), DIM),
                 (str(row.attempts), PLAIN if row.attempts > 1 else FAINT)]
        lines.append(table.row(cells, Target("job", {"job_id": row.job_id, "row": row},
                                             "job " + row.display_id),
                               {"GPU-h": row.gpu_seconds, "start": row.start,
                                "elapsed": row.elapsed, "wait": row.wait, "GPUs": row.gpus},
                               key=row.key))
    more = total is not None and total > len(rows)
    count = ("%s of %s" % (fmt.num(len(rows)), fmt.num(total)) if more
             else fmt.num(len(rows)))
    status = ""
    if match:
        status = "matching “%s”" % match
    if more:
        status = (status + "  " if status else "") + "↓ at the end loads more"
    if entry is not None and busy(entry) and rows:
        status = (status + "  " if status else "") + "loading…"
    return Pane(pane_id, title + " (%s)" % count, lines, table.header(),
                sorts=("GPU-h", "start", "elapsed", "wait", "GPUs"), more=more, status=status)


def paged_jobs(ctx: Context, view: View, title: str, key: Tuple[Any, ...], filt: Any,
               win: Any = None, run_type: Any = None, pane_id: str = "jobs") -> Pane:
    """The first JOB_PAGE jobs, more as the cursor reaches the end (app.py raises
    view.extra["jobs_limit"]); the pane's / filter is also sent to history as a search."""
    state = view.panes.get(pane_id)
    match = (state.filter if state is not None else "").strip()
    limit = view.extra.get("jobs_limit", hd.JOB_PAGE)
    entry = ctx.hist(key, lambda r=ctx.range, f=filt, l=limit, m=match: ctx.data.jobs(
        r, f, limit=l, win=win, run_type=run_type, match=m), version=(limit, match))
    page = value(entry)
    total = getattr(page, "total", None) if page is not None else None
    return jobs_pane(ctx, pane_id, title, page.rows if page is not None else None, entry,
                     total, match, "GPU-h in bin" if win is not None else "GPU-h")


def jobs_list(ctx: Context, view: View) -> List[Pane]:
    rows = view.params.get("rows")
    if rows is not None:
        return [jobs_pane(ctx, "jobs", view.title, rows)]
    spec = view.params
    win = spec.get("window")
    filt_params = {k: spec.get(k) for k in ("user", "account", "qos", "partition")}
    run_type = spec.get("run_type")
    return [paged_jobs(ctx, view, view.title,
                       ("joblist", ctx.range, filt_key(filt_params), win, run_type),
                       make_filter(filt_params), hd.Window(*win) if win else None, run_type)]


# -- Timeline ----------------------------------------------------------------

def timeline(ctx: Context, view: View) -> List[Pane]:
    by = view.extra.get("lanes", "user")
    zoom = view.extra.get("zoom", 1)
    gpus_now = ctx.snap.totals.gpus.total if ctx.snap is not None else None
    metric = view.extra.get("metric") or ("gpu" if gpus_now != 0 else "cpu")
    entry = ctx.hist(("lanes", ctx.range, by, metric),
                     lambda name=ctx.range, b=by, m=metric: lanes_report(ctx, name, b, m))
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
    axis = timeline_axis(start, size, bins, shown)
    header = Line([(fmt.fit(by, label_width) + " ", DIM), ("".join(axis), DIM)])
    glyph = "▄"
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
        cells = charts.heat_cells([min(1.0, c) for c in cover], ctx.ascii, ctx.color, glyph)
        flat = [(ch, style) for text, style in cells for ch in text]
        here = flat[cursor]
        flat[cursor] = ((here[0], bold(ACCENT)) if here[0] != " " else ("┊", ACCENT))
        style = USER if by == "user" else RESOURCE
        bin_start = int(start + cursor * size)
        bin_end = int(start + (cursor + 1) * size) + 1
        parts = [(fmt.fit(lane.key, label_width) + " ", style)] + charts._join(flat)
        params = {by: lane.key, "window": (bin_start, bin_end)}
        lines.append(Line(parts, Target("jobs", params, "%s %s · jobs in the bin at %s" % (
            by, lane.key, fmt.when(bin_start, ctx.now))),
                          {"busy": total, "name": lane.key}, key=lane.key))
    lines.sort(key=lambda l: -l.sort["busy"])
    status = "bin %s · %s → %s" % (fmt.duration(size), fmt.when(start + cursor * size, ctx.now),
                                   fmt.when(start + (cursor + 1) * size, ctx.now))
    shade = [("%s held " % ("GPUs" if metric == "gpu" else "CPUs"), DIM)] + charts.legend(
        ctx.ascii, ctx.color, glyph)
    # the bin first: the legend is dropped whole on a narrow screen, never cut mid-word
    return [Pane("timeline", "Timeline · %s lanes by %s · %s · zoom %dx" % (
        "GPU" if metric == "gpu" else "CPU", by, ctx.range, zoom),
                 lines, header, sorts=("busy", "name"), status=(status, shade),
                 select_width=label_width + 1)]


def lanes_report(ctx: Context, range_name: str, by: str, metric: str) -> Any:
    """Lanes from attempts holding the metric's resource (GPU view: GPU jobs only), so a
    CPU-only partition is not drawn solid on the GPU view."""
    try:
        return ctx.data.report(range_name, None, lanes_by=by, efficiency=False,
                               lanes_metric=metric)
    except TypeError:  # a history layer without per-metric lanes
        return ctx.data.report(range_name, None, lanes_by=by, efficiency=False)


def timeline_axis(start: float, size: float, bins: int, shown: float) -> List[str]:
    """Time labels along the bins; a label equal to the previous one is left out, and
    when labels are less than a day apart they carry the hour too."""
    axis = [" "] * bins
    daily = shown > 2 * 86400
    sample = time.strftime("%b %d", time.localtime(start)) if daily else "00:00"
    gap = (len(sample) + 3) * size
    if daily and gap < 86400:
        pattern = "%d %Hh"
    else:
        pattern = "%b %d" if daily else "%H:%M"
    width = len(time.strftime(pattern, time.localtime(start)))
    previous = None
    for mark in range(0, max(0, bins - width), width + 3):
        label = time.strftime(pattern, time.localtime(start + mark * size))
        if label == previous:
            continue
        previous = label
        for i, ch in enumerate(label):
            if mark + i < bins:
                axis[mark + i] = ch
    return axis


# -- Job ---------------------------------------------------------------------

STEP_COLUMNS = (
    Column("Step", 10, "<", 10),
    Column("name", 12, "<", 5, flex=True, most=40),
    Column("state", 10, "<", 8),
    Column("elapsed", 8, ">", 8),
    Column("tasks", 5, ">", 3),
    Column("CPU eff", 7, ">", 9),
    Column("max RSS", 8, ">", 7),
    Column("GPU util*", 9, ">", 7),
    Column("GPU mem", 8, ">", 4),
)
STANDARD_STEPS = ("batch", "extern", "interactive")


def step_cpu_eff(step: Any, elapsed: Optional[float] = None) -> Optional[float]:
    cpus = step.tres_alloc.get("cpu") if step.tres_alloc else None
    elapsed = step.elapsed or elapsed
    if step.total_cpu is None or not cpus or not elapsed:
        return None
    return step.total_cpu / (cpus * elapsed)


def step_gpu_util(step: Any, job_gpus: Optional[int] = None) -> Optional[float]:
    """Peak gpuutil per allocated GPU, capped at 100% as everywhere else. Slurm's raw
    figure may be summed over a task's GPUs (645 on a 16-GPU step is about 40% each)."""
    raw = step.gpu_util_max
    gpus = (step.tres_alloc or {}).get("gres/gpu") or job_gpus
    if raw is None or not gpus:
        return None
    return min(1.0, raw / (100.0 * gpus))


def from_steps(steps: Sequence[Any], cpus: Optional[int], mem: Optional[int],
               elapsed: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """(CPU efficiency, memory peak) of a whole attempt from its steps, when history has
    not measured it yet: CPU time of every step over the attempt's allocation, and the
    largest step RSS over the requested memory."""
    used = [s.total_cpu for s in steps if s.total_cpu is not None]
    cpu = sum(used) / (cpus * elapsed) if used and cpus and elapsed else None
    rss = [s.max_rss for s in steps if s.max_rss]
    peak = max(rss) / mem if rss and mem else None
    return cpu, peak


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
    sep = (" · ", FAINT)

    def kv(label: str, *parts: Tuple[str, str]) -> None:
        lines.append(Line([(fmt.fit(label, 11), DIM)] + list(parts)))

    def who(user: Any, account: Any, qos: Any, partition: Any) -> None:
        kv("who", (user or "?", bold(USER)), sep, (account or "?", RESOURCE), sep,
           ("qos ", DIM), (qos or "?", RESOURCE), sep, ("partition ", DIM),
           (partition or "?", RESOURCE))

    def elapsed_line(spent: Optional[float], limit: Optional[float], unlimited: bool,
                     waited: Optional[float]) -> None:
        used = spent / limit if spent and limit else None
        parts = [(fmt.duration(spent), bold(PLAIN))]  # type: List[Tuple[str, str]]
        if unlimited:
            parts.append(("  no time limit", DIM))
        elif limit:
            parts += [(" of ", DIM), (fmt.duration(limit), PLAIN), ("  ", PLAIN)]
            parts += charts.gauge(used, 20, ctx.ascii) + [(" " + fmt.pct(used), PLAIN)]
        parts += [sep, ("waited ", DIM), (fmt.duration(waited), PLAIN)]
        kv("elapsed", *parts)

    if live is not None:
        src = live
        who(src.user, src.account, src.qos, src.partition)
        kv("state", (src.state or "?", bold(job_style(src))),
           ("  " + (src.reason or "") if src.pending else "", WARN))
        memory = fmt.mem(src.mem_req if src.pending else src.mem_alloc)
        mem_note = ""
        if not src.pending and src.mem_alloc is None and src.mem_req is not None:
            memory, mem_note = fmt.mem(src.mem_req), " (requested)"  # AllocTRES has no memory
        kv("request", (fmt.num(src.gpus_req if src.pending else src.gpus_alloc), PLAIN),
           (" GPUs", DIM), sep, (fmt.num(src.cpus_req if src.pending else src.cpus_alloc), PLAIN),
           (" CPUs", DIM), sep, (memory, PLAIN), (" mem" + mem_note, DIM))
        if src.nodes:
            kv("nodes", (",".join(src.nodes[:8]) + (" …" if len(src.nodes) > 8 else ""), RESOURCE))
        wait = src.start - (src.eligible or src.submit) if src.start and (src.eligible or src.submit) and not src.pending else None
        kv("timing", ("submit ", DIM), (fmt.when(src.submit, ctx.now), PLAIN), sep,
           ("start ", DIM), (fmt.when(src.start, ctx.now) + (" (est.)" if src.pending else ""),
                             PLAIN), sep,
           ("end ", DIM), (fmt.when(src.end, ctx.now) + (" (est.)" if src.end and src.end > ctx.now
                                                         else ""), PLAIN))
        spent = live_elapsed(src, ctx.now) if not src.pending else None
        elapsed_line(spent, src.time_limit, src.time_unlimited, wait)
    elif row is not None:
        who(row.user, row.account, row.qos, row.partition)
        kv("outcome", (row.outcome, bold(OUTCOME_STYLE.get(row.outcome, PLAIN))),
           ("  " + row.state, DIM))
        kv("request", (fmt.num(row.gpus), PLAIN), (" GPUs", DIM), sep, (fmt.num(row.cpus), PLAIN),
           (" CPUs", DIM), sep, (fmt.mem(row.mem), PLAIN), (" mem", DIM))
        if row.nodes:
            kv("nodes", (row.nodes, RESOURCE))
        kv("timing", ("submit ", DIM), (fmt.when(row.submit, ctx.now), PLAIN), sep,
           ("start ", DIM), (fmt.when(row.start, ctx.now), PLAIN), sep,
           ("end ", DIM), (fmt.when(row.end, ctx.now), PLAIN))
        elapsed_line(row.elapsed, row.time_limit, row.time_unlimited, row.wait)
        cpu_eff, mem_peak = row.cpu_efficiency, row.mem_efficiency
        note = ""
        if cpu_eff is None or mem_peak is None:  # not measured by history yet: the steps
            last = [a for a in getattr(detail, "attempts", ()) or ()]
            steps_of = last[-1].steps if last else []
            got = from_steps(steps_of, row.cpus, row.mem, row.elapsed)
            if (cpu_eff is None and got[0] is not None) or (mem_peak is None
                                                            and got[1] is not None):
                note = "  (from steps)"
            cpu_eff = cpu_eff if cpu_eff is not None else got[0]
            mem_peak = mem_peak if mem_peak is not None else got[1]
        gpu_peak = row.gpu_peak_utilization

        def figure(fraction: Optional[float]) -> Tuple[str, str]:
            if fraction is None:
                return fmt.NA, FAINT
            return fmt.pct(fraction), PLAIN

        effs = [("CPU ", DIM), figure(cpu_eff), sep, ("mem peak ", DIM), figure(mem_peak), sep,
                ("GPU peak ", DIM), figure(min(1.0, gpu_peak) if gpu_peak is not None else None),
                (note, DIM)]
        kv("efficiency", *effs)
    else:
        lines.append(text_line("not in the live queue; history: %s %s" % (
            missing(attempts_entry), attempts_entry.message), FAINT))
    name = (live.name if live is not None else row.name if row is not None else "") or ""
    panes = [Pane("summary", "Job %s%s" % (job_id, " · " + name if name else ""), lines, grow=False,
                  focusable=False)]

    # attempts
    lines = []
    for number, att in enumerate(attempts):
        elapsed = att.elapsed
        if att.end is None and att.start:  # still running: as now, not as of the last load
            elapsed = (live_elapsed(live, ctx.now) if live is not None and live.start == att.start
                       else ctx.now - att.start)
        lines.append(Line([(fmt.fit("#%d" % (number + 1), 4), FAINT),
                           (fmt.fit(att.outcome, 10), OUTCOME_STYLE.get(att.outcome, PLAIN)),
                           (fmt.fit(fmt.when(att.start, ctx.now), 12), DIM), ("→ ", FAINT),
                           (fmt.fit(fmt.when(att.end, ctx.now) if att.end else "running", 13),
                            DIM if att.end else GOOD),
                           (fmt.fit(fmt.duration(elapsed), 8, ">") + "  ", PLAIN),
                           (fmt.fit(fmt.plural(att.gpus, "GPU"), 8), PLAIN),
                           (att.nodes, RESOURCE)]))
    if not attempts:
        lines.append(text_line("attempts: %s %s" % (
            missing(attempts_entry) if attempts_entry.state != "ok" else fmt.NOT_APPLICABLE,
            attempts_entry.message or ("none in history" if attempts_entry.state == "ok" else "")), DIM))
    panes.append(Pane("attempts", "Attempts (%d)" % len(attempts), lines, grow=False, max_lines=4,
                      group="detail"))

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
    running = live is not None and live.holds_resources
    job_gpus = (live.gpus_alloc if live is not None else row.gpus if row is not None else None)
    whole = live_elapsed(live, ctx.now) if running else None
    for step in steps:
        # sstat on a running job leaves name, state and elapsed out: the standard steps are
        # named after themselves, and batch/extern run as long as the job
        name = step.name or (step.step_id if step.step_id in STANDARD_STEPS else "")
        state = step.state or ("RUNNING" if running else "")
        elapsed = step.elapsed
        if elapsed is None and running and step.step_id in ("batch", "extern"):
            elapsed = whole
        shown = (fmt.duration(elapsed) if elapsed is not None
                 else ("running", DIM) if running else fmt.NA)
        util = step_gpu_util(step, job_gpus)
        cells = [step.step_id, name, (state, GOOD if state == "RUNNING" else
                                      BAD if state in ("FAILED", "NODE_FAIL", "OUT_OF_MEMORY")
                                      else PLAIN), shown,
                 fmt.num(step.ntasks) if step.ntasks is not None else fmt.NA,
                 fmt.pct(step_cpu_eff(step, elapsed)), fmt.mem(step.max_rss),
                 fmt.pct(util) if util is not None else fmt.NA, fmt.mem(step.gpu_mem_max)]
        lines.append(table.row(cells))
    if note:
        for part in wrap(note, ctx.width - 3):
            lines.append(text_line(part, DIM if fmt.DENIED not in note else BAD))
    # the steps first: what the job is doing (or did); its attempts are the next tab
    panes.insert(len(panes) - 1, Pane("steps", "Steps (%d)" % len(steps), lines, table.header(),
                                      grow=False, max_lines=5, group="detail"))

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
    # logs are judged against the end of the last attempt; while it runs they are being
    # written. A script is judged against the first start (what ran was read then).
    started = None
    if live is not None and live.start and not live.pending:
        started = live.start
    elif attempts and attempts[0].start:
        started = attempts[0].start
    elif row is not None:
        started = row.start
    running = live is not None and (live.holds_resources or live.state == "SUSPENDED")
    ended = None
    if not running:
        last = attempts[-1] if attempts else row
        ended = getattr(last, "end", None) or (live.end if live is not None else None)
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
        parts = [(fmt.fit(label, 8), DIM)]
        tail = []  # type: List[Tuple[str, str]]
        if raw:
            if label == "script":
                state = ctx.app.path_state(raw, started, False, False, "the job started")
            else:
                state = ctx.app.path_state(raw, ended, directory, running and not directory)
            if state is None:
                tail = [("  ○ checking ?", FAINT)]
            else:
                style = {"good": GOOD, "warn": WARN, "bad": BAD}.get(state.style, DIM)
                tail = [("  ● " if state.style in ("good", "warn", "bad") else "  ○ ", style),
                        (state.note, style if state.style != "good" else DIM)]
        # a long path loses its start, never its file name
        room = max(12, ctx.width - 3 - 8 - charts.width(tail))
        text = shown_path or fmt.NOT_APPLICABLE
        if fmt.width(text) > room:
            text = "…" + text[-(room - 1):]
        parts.append((text, PLAIN if shown_path else FAINT))
        lines.append(Line(parts + tail))

    if ctx.app.anonymized:
        lines.append(text_line("paths, scripts, submit lines and logs are hidden while anonymized (A)",
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
    panes.append(Pane("logs", "Logs", lines, grow=False,
                      status=[[("l", KEY), (" out  ", DIM), ("e", KEY), (" err  ", DIM),
                               ("b", KEY), (" script  ", DIM), ("u", KEY), (" submit", DIM)]]))

    shown = view.extra.get("show")
    if shown:
        entry = ctx.app.loader.peek(("o", "fetch", job_id, shown))
        text_lines = []
        room = ctx.width - 3
        tail = shown in ("stdout", "stderr")

        def say(text: str, style: str = PLAIN, indent: int = 2) -> None:
            for part in wrap(text, room, indent):  # messages and commands wrap, never clip
                text_lines.append(text_line(part, style))

        if entry is None or entry.at is None:
            text_lines.append(text_line("asking Slurm… ?", DIM))
        elif entry.state == FAILED:
            say("%s: %s" % (fmt.placeholder(entry.status), entry.message), BAD)
        else:
            status, body, detail_text = entry.value
            if status not in Status.USABLE:
                say("%s: %s" % (fmt.placeholder(status), detail_text),
                    BAD if status == Status.DENIED else DIM)
            for text in body:
                if tail:  # a log keeps its lines; long ones are cut at the edge
                    text_lines.append(text_line(text))
                else:
                    say(text, PLAIN, 4)
            if not body and status in Status.USABLE:
                say("(empty) " + detail_text, DIM)
        label = {"stdout": "stdout tail", "stderr": "stderr tail", "script": "Batch script",
                 "submit": "Submit line"}[shown]
        panes.append(Pane("output", label, text_lines, follow=tail))
    return panes


# -- Help --------------------------------------------------------------------

HELP = [
    ("Navigation", HEAD),
    ("  o m n j p a t   Overview, Me, Nodes, Queue (jobs), People, Roles (accounts), Timeline", PLAIN),
    ("  ← →             previous / next tab (in Timeline: move the bin cursor)", PLAIN),
    ("  ↑ ↓ PgUp PgDn   move in the focused pane (PgUp/PgDn a page);  Home/End  first/last", PLAIN),
    ("  Tab Shift-Tab   next / previous pane", PLAIN),
    ("  Enter           drill into the row (›);  Esc / Backspace  back up the breadcrumb", PLAIN),
    ("  /  s  S         filter the pane as you type (Ctrl-U clears; a jobs list searches all of", PLAIN),
    ("                  history);  cycle sort column (▼▲ in its header);  reverse sort", PLAIN),
    ("  [ ]  1-5        range: day, week, month, year, all — shared by every view", PLAIN),
    ("  g               Roles: account/QOS/partition;  Person: run types by GPUs, nodes, job name,", PLAIN),
    ("                  account, QOS, partition, outcome;", PLAIN),
    ("                  Timeline: lanes by user/account/QOS/partition", PLAIN),
    ("  + - , .  c      Timeline zoom and pan;  c: GPU or CPU lanes", PLAIN),
    ("  l e b u         Job: stdout tail, stderr tail, batch script, submit line (on request only)", PLAIN),
    ("  E               load efficiency (step usage) for the current range and person", PLAIN),
    ("  r  ?  q         refresh now, this help, quit;  mouse: wheel scrolls, click selects, double-click opens", PLAIN),
    ("", PLAIN),
    ("Settings (any screen, no restart)", HEAD),
    ("  A               anonymize on/off: pseudonyms for people, accounts, nodes and job names;", PLAIN),
    ("                  closes open paths, scripts, submit lines and logs", PLAIN),
    ("  C  U            colors on/off;  Unicode or ASCII glyphs", PLAIN),
    ("  H               history: full (all of it in the background), recent (the chosen range and", PLAIN),
    ("                  shorter), off (live view only; stops loading)", PLAIN),
    ("  O               settings: every switch and its state, the cache location, clear the cache", PLAIN),
    ("", PLAIN),
    ("Colors", HEAD),
    ("  cyan     a resource: node, account, QOS, partition", RESOURCE),
    ("  magenta  a person", USER),
    ("  green    productive: allocated, running, completed, efficient", GOOD),
    ("  amber    partial or waiting: pending, preempted, timed out, partly loaded", WARN),
    ("  red      needs attention: down, drained, failed, refused", BAD),
    ("  gray     context, or not available", DIM),
    ("  blue     focus, keys and the cursor", ACCENT),
    ("  ramp     charts and lanes: darker = less, paler = more (shades without color)", PLAIN),
    ("", PLAIN),
    ("Placeholders", HEAD),
    ("  n/a      unavailable here      —  not applicable      ?  not loaded yet", PLAIN),
    ("  denied   Slurm refused you     ERR  the query failed   ≥  window only partly loaded", PLAIN),
    ("", PLAIN),
    ("Glossary", HEAD),
    ("  GPU-h / CPU-h alloc   allocated hours (GPUs or CPUs held × wall time), not measured use", PLAIN),
    ("  util                  allocated hours ÷ (today's live capacity × window length)", PLAIN),
    ("  GPU util*             peak sampled GPU utilization per allocated GPU of measured attempts", PLAIN),
    ("                        (capped at 100%); CPU eff = CPU time ÷ allocated", PLAIN),
    ("  ·cov                  share of allocated hours the efficiency figure describes; under 5%", PLAIN),
    ("                        the figure is shown as ? (too little measured)", PLAIN),
    ("  attempt / job         a requeue or preemption starts another attempt of the same job", PLAIN),
    ("  preempted             stopped by the scheduler for other work; allocated time still counts and", PLAIN),
    ("                        is not a failure or waste", PLAIN),
    ("  fail                  failed, node_fail or timeout among attempts that ended", PLAIN),
    ("  limit                 elapsed ÷ requested time limit of completed and timed-out attempts", PLAIN),
    ("  stranded GPUs         free GPUs that cannot start: no free core or memory left on the node", PLAIN),
    ("  largest block         the most GPUs free on one node: the biggest job that could start now", PLAIN),
    ("  mem n/a/size          Nodes: Slurm does not track memory allocation there; the size is shown", PLAIN),
    ("  lanes                 Timeline: each cell is one bin, shaded by the part of it with GPUs", PLAIN),
    ("                        (or CPUs) held", PLAIN),
    ("  paths                 Job: as Slurm recorded them; checked with os.stat, never read", PLAIN),
    ("  share                 part of the cluster's allocated GPU-hours in the same range;", PLAIN),
    ("                        of acct / of QOS / of part / of user: part of that one's GPU-hours", PLAIN),
    ("  members               Overview: users with an association or usage in the account", PLAIN),
    ("  active                Roles: users with jobs now or usage in the range", PLAIN),
    ("  visible               how much of an account's history you can see: full, own only,", PLAIN),
    ("                        denied, or no accounting", PLAIN),
    ("  Job environments are never shown.  Nothing here changes anything in Slurm.", DIM),
]


def help_screen(ctx: Context, view: View) -> List[Pane]:
    """Keys and terms in a column of their own (accent for keys, bold for terms), the
    explanation beside them, wrapped under itself."""
    lines = []
    section = ""
    room = ctx.width - 3
    for text, style in HELP:
        if style == HEAD:
            section = text
            lines.append(text_line(text, bold(ACCENT)))
            continue
        if not text.startswith("  ") or style not in (PLAIN,) and section != "Colors":
            for part in wrap(text, room):
                lines.append(text_line(part, style))
            continue
        body = text[2:]
        lead = len(body) - len(body.lstrip(" "))
        if lead:  # a continuation: under the explanation
            for part in wrap(text, room):
                lines.append(text_line(part, PLAIN))
            continue
        # the key column ends at the longest run of spaces near the start
        runs = [(len(m.group(0)), m.start(), m.end()) for m in re.finditer(r" {2,}", body[:26])]
        if not runs:
            lines.append(text_line(text, style))
            continue
        _, gap, after = max(runs, key=lambda r: (r[0], -r[1]))
        indent = 2 + after
        key, desc = body[:gap], body[after:]
        key_style = (KEY if section.startswith(("Navigation", "Settings")) else
                     bold(style) if section == "Colors" else HEAD)
        wrapped = wrap(" " * indent + desc, room, indent)
        first = wrapped[0][indent:]
        lines.append(Line([("  ", PLAIN), (key, key_style),
                           (" " * (indent - 2 - fmt.width(key)), PLAIN),
                           (first, PLAIN if section != "Colors" else DIM)]))
        for part in wrapped[1:]:
            lines.append(text_line(part, PLAIN))
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
    "overview": [("↵", "open")],
    "me": [("g", "run types"), ("E", "efficiency")],
    "person": [("g", "run types"), ("E", "efficiency")],
    "roles": [("g", "group")],
    "timeline": [("g", "lanes"), ("←→", "bin"), ("+-", "zoom"), (",.", "pan")],
    "job": [("l e b u", "fetch")],
}
