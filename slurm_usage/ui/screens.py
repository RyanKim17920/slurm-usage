"""Every screen: a function from (Context, View) to a list of panes.

Screens never block: data comes from ``Context.live``/``Context.hist``, which
return loader entries that may still be pending (drawn as ``?``).
"""

import re
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from ..live import applicable_limits, group_by
from ..model import Status
from . import breakdown, charts
from . import data as hd
from . import fmt
from . import plot
from . import theme
from .canvas import (ACCENT, BAD, BRIGHT, DIM, FAINT, GOOD, HEAD, KEY, PLAIN, RESOURCE, SEL,
                     TAB, USER, WARN, bold, on)
from .loader import FAILED, Entry
from .widgets import (Column, Line, Pane, PaneState, Table, Target, fit_parts, text_line,
                      wrap)

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

    def __init__(self, app: Any, width: int, height: int = 40) -> None:
        self.app = app
        self.height = height  # the whole screen's rows
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

    def hist(self, key: Tuple[Any, ...], fn: Callable[[], Any], version: Any = None,
             later: bool = False) -> Entry:
        """``version``: inputs that change the answer but should keep the old one on
        screen while the new one is computed (a longer page, a search). ``later``: for a
        pane not on show (a tab behind another): asked for once nothing else is being
        worked out, so what is on screen is answered first."""
        loader = self.app.loader
        if later and loader.busy and loader.peek(("h",) + key) is None:
            return Entry()  # pending; the next frame asks again
        return loader.get(("h",) + key, fn, generation=(self.data.generation, version))

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


def kept_lines(ctx: Context, name: Tuple[Any, ...], inputs: Sequence[Any],
               build: Callable[[], List[Line]]) -> List[Line]:
    """Lines that depend only on ``inputs`` (answers, compared by identity) and the
    layout: built once per answer, not on every frame (a list of thousands of job-name
    families took longer to lay out than a frame lasts)."""
    cache = ctx.app.__dict__.setdefault("_kept_lines", {})
    layout = (ctx.width, ctx.ascii, ctx.color)
    found = cache.get(name)
    if (found is not None and found[0] == layout and len(found[1]) == len(inputs)
            and all(a is b for a, b in zip(found[1], inputs))):
        return found[2]
    lines = build()
    if len(cache) > 32:
        cache.clear()
    cache[name] = (layout, tuple(inputs), lines)
    return lines


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
    """Live capacity; before the first snapshot, what it was last run (so the answers
    kept from then match and nothing is computed twice)."""
    if ctx.snap is None:
        return getattr(ctx.app, "capacity_hint", None) or (None, None)
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


def rollup_stand_in(ctx: Context, params: Dict[str, Any], name: str) -> Any:
    """(amounts, previous GPU-hours or None, CPU-hours) of one account or person in a
    range from the hourly summary, or None: shown (marked ≥) while the exact figures
    compute, so a drill does not open on a column of placeholders."""
    if set(k for k, v in params.items() if v) - {"user", "account"}:
        return None
    summary = rollup_summary(ctx, name)
    if summary is None:
        return None
    account, user = params.get("account"), params.get("user")
    if user:
        found = rollup_users(summary, account).get(user)
        return (found, None, found.cpu_hours) if found is not None else None
    item = summary.accounts.get(account) if account else None
    if item is None:
        return None
    before = item.previous.gpu_hours if item.previous is not None and summary.complete \
        and name != "all" else None
    return item.amounts, before, item.amounts.cpu_hours


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
        target = range_target(name, params)
        if row is None:
            stand_in = rollup_stand_in(ctx, params, name) if person else None
            if stand_in is not None:  # the hourly summary's figures while exact ones compute
                gpu, before, cpu = stand_in
                incomplete = True
                cells = [label, "≥" + fmt.hours(gpu.gpu_hours), (fmt.NOT_YET, FAINT),
                         delta_cell(gpu.gpu_hours, before, before is not None),
                         (fmt.NOT_YET, FAINT), "≥" + fmt.hours(cpu),
                         (fmt.NOT_YET, FAINT), (fmt.NOT_YET, FAINT), ""]
                lines.append(table.row(cells[:len(table.columns)], target, key=name))
                continue
            ph = missing(entry)
            cells = [label] + [(ph, FAINT)] * (len(table.columns) - 1)
            lines.append(table.row(cells, target, key=name))
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
        lines.append(table.row(cells, target, key=name))
    note = "≥ partly loaded" if incomplete else ""
    who = params.get("user") or params.get("account")
    return Pane("trend", "Trend" + (" · " + who if who else ""),
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
    panes = [hero_pane(ctx), range_tiles(ctx, view)]
    accounts = accounts_pane(ctx)
    warn = warnings_pane(ctx)
    # the chosen range's chart goes between the tiles and the accounts when it fits
    # without taking rows the accounts need
    used = 2 + 5 + 5 + 3 + len(accounts.lines) + (1 + min(2, len(warn.lines)) if warn else 0)
    free = ctx.height - used - 1
    if free >= 8 and not ctx.app.no_history:
        panes.append(overview_chart(ctx, min(free, 16)))
    panes.append(accounts)
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


# -- the hero: GPUs in use now, large -------------------------------------------------

RANGE_LENGTHS = {"day": "24h", "week": "7 days", "month": "30 days", "year": "365 days",
                 "all": "all history"}
RANGE_WORDS = {"day": "day", "week": "week", "month": "month", "year": "year",
               "all": "history"}


SCOPE_ROLES = ("user", "account", "qos", "partition")


def scope_name(params: Dict[str, Any]) -> Optional[str]:
    """Who a range detail is about: the person, else the account, QOS or partition."""
    for role in SCOPE_ROLES:
        if params.get(role):
            return params[role] if role in ("user", "account") else "%s %s" % (
                "QOS" if role == "qos" else role, params[role])
    return None


def range_target(name: str, params: Dict[str, Any]) -> Target:
    """Range detail for ``name``: the cluster, or one person, account, QOS or partition."""
    keep = {k: params[k] for k in SCOPE_ROLES if params.get(k)}
    who = scope_name(keep) or "cluster"
    return Target("range", dict(keep, range=name), "%s · %s" % (name.capitalize(), who))


def gpu_metric(ctx: Context) -> str:
    """GPU figures, unless the cluster has no GPUs."""
    if ctx.snap is not None and not ctx.snap.totals.gpus.total and ctx.snap.totals.cpus.total:
        return "cpu"
    return "gpu"


def _box(canvas: Any, top: int, col: int, size: int, rows: int, title: str, note: str,
         chosen: bool, glyphs: theme.Box) -> None:
    border = ACCENT if chosen else FAINT
    canvas.put(top, col, glyphs.tl + glyphs.h + " ", border)
    end = canvas.put(top, col + 3, title, KEY if chosen else HEAD, limit=max(0, size - 6))
    room = col + size - 1 - end - 1
    if note and room >= fmt.width(note) + 4:
        rule = room - fmt.width(note) - 2
        end = canvas.put(top, end, " " + glyphs.h * rule + " ", border)
        end = canvas.put(top, end, note, DIM)
        canvas.put(top, end, " " + glyphs.h, border)
    else:
        canvas.put(top, end, " " + glyphs.h * max(0, room), border)
    canvas.put(top, col + size - 1, glyphs.tr, border)
    for at in range(top + 1, top + rows - 1):
        canvas.put(at, col, glyphs.v, border)
        canvas.put(at, col + size - 1, glyphs.v, border)
    canvas.put(top + rows - 1, col, glyphs.bl + glyphs.h * (size - 2) + glyphs.br, border)


def hero_pane(ctx: Context) -> Pane:
    """GPUs in use now, in large figures, with the gauge and what else is going on."""
    snap = ctx.snap
    metric = gpu_metric(ctx)
    word = "GPUs" if metric == "gpu" else "CPUs"
    res = None if snap is None else (snap.totals.gpus if metric == "gpu" else snap.totals.cpus)
    used = total = fraction = None
    if res is not None and res.total is not None and res.alloc is not None:
        used, total = res.alloc + (res.completing or 0), res.total
        fraction = used / float(total) if total else None
    tone = charts.ramp_style(fraction)
    target = range_target("day", {})
    text = "%s in use %s / %s %s" % (word, fmt.num(used), fmt.num(total), fmt.pct(fraction))
    lines = [Line([(text, PLAIN)], target, key="hero")]

    def chips() -> List[List[Tuple[str, str]]]:
        """What sits beside the gauge: free, unavailable, the biggest free block."""
        if snap is None:
            entry = ctx.app.snapshot_entry
            return [[("live state " + ("loading…" if entry is None or busy(entry)
                                       else missing(entry)), FAINT)]]
        out = [_count(res.free, "free")] if res is not None and res.free is not None else []
        if res is not None and res.unavailable:
            out.append(_count(res.unavailable, "unavailable", BAD))
        if res is not None and res.completing:
            out.append(_count(res.completing, "completing", WARN))
        block = snap.health.largest_gpu_block
        if metric == "gpu" and block is not None:
            out.append([("largest free block ", DIM), (str(block.gpus), PLAIN), (" on ", DIM),
                        (block.node, RESOURCE)])
        return out

    def context() -> List[List[Tuple[str, str]]]:
        """The rest of the cluster in one line: CPUs, nodes, the queue."""
        if snap is None:
            return []
        t, h = snap.totals, snap.health
        out = []  # type: List[List[Tuple[str, str]]]
        other = t.cpus if metric == "gpu" else None
        if other is not None and other.total:
            frac = (other.alloc + (other.completing or 0)) / float(other.total) \
                if other.alloc is not None else None
            out.append([("CPUs ", DIM), (fmt.num(other.alloc), PLAIN),
                        ("/" + fmt.num(other.total) + " ", DIM),
                        (fmt.pct(frac), charts.ramp_style(frac))])
        down, drain = fmt.down_drain(h.node_states)
        nodes = [(fmt.num(t.nodes), PLAIN), (" nodes", DIM)]
        if down:
            nodes += [("  ● ", BAD), (str(down), BAD), (" down", BAD)]
        if drain:
            nodes += [("  ● ", BAD), (str(drain), BAD), (" drain", BAD)]
        out.append(nodes)
        running = sum(1 for job in snap.jobs or () if job.holds_resources)
        waiting = sum(job.task_count for job in snap.jobs or () if job.pending)
        queue = [(str(running), GOOD), (" running  ", DIM),
                 (str(waiting), WARN if waiting else DIM), (" pending", DIM)]
        out.append(queue)
        reasons = h.pending_by_reason
        if reasons:  # why the queue waits, while there is room
            top = reasons[0]
            out.append([("(", FAINT), (top.key, DIM), (" %d" % top.pending, PLAIN),
                        (")", FAINT)])
        return out

    def draw(canvas: Any, top: int, rows: int, width: int, state: Any, focused: bool,
             shown: List[Line]) -> List[Tuple[int, ...]]:
        glyphs = theme.box(ctx.ascii)
        if rows < 3:
            canvas.segments(top, 1, [(text, PLAIN)], width - 2)
            ctx.app.click_targets.append((top, 0, width, target))
            return []
        _box(canvas, top, 0, width, rows, word + " in use", "now", False, glyphs)
        inner = width - 4
        figure = "%s/%s" % (fmt.num(used) if used is not None else "?",
                            fmt.num(total) if total is not None else "?")
        percent = fmt.pct(fraction) if fraction is not None else ""
        big_left = plot.big_rows(fmt.num(used) if used is not None else "?", ctx.ascii)
        big_slash = plot.big_rows("/", ctx.ascii)
        big_total = plot.big_rows(fmt.num(total) if total is not None else "?", ctx.ascii)
        big_pct = plot.big_rows(percent, ctx.ascii) if percent else ["", "", ""]
        size = sum(fmt.width(r[0]) for r in (big_left, big_slash, big_total)) + 2
        with_pct = size + 3 + fmt.width(big_pct[0])
        body = min(3, rows - 2)
        if size + 22 > inner:  # too narrow for large figures: one line of text
            parts = [(figure.replace("/", " / "), bold(BRIGHT)), (" " + word.lower() + " in use  ",
                                                                   DIM), (percent, bold(tone))]
            canvas.segments(top + 1, 2, parts, inner)
            canvas.segments(top + 2, 2, charts.gauge(fraction, inner, ctx.ascii), inner)
            if body >= 3:
                canvas.segments(top + 3, 2, _chips(chips())(inner), inner)
        else:
            show_pct = with_pct + 24 <= inner
            left = with_pct if show_pct else size
            for number in range(body):
                parts = [(big_left[number], bold(BRIGHT) if used is not None else FAINT),
                         (" ", PLAIN), (big_slash[number], FAINT), (" ", PLAIN),
                         (big_total[number], DIM)]
                if show_pct:
                    parts += [("   ", PLAIN), (big_pct[number], bold(tone))]
                canvas.segments(top + 1 + number, 2, parts, inner)
            room = inner - left - 3
            at = 2 + left + 3
            gauge = charts.gauge(fraction, room - (0 if show_pct else 5), ctx.ascii)
            if not show_pct and percent:
                gauge += [(" " + percent.rjust(4), bold(tone))]
            canvas.segments(top + 1, at, gauge, room)
            if body >= 2:
                canvas.segments(top + 2, at, _chips(chips())(room), room)
            if body >= 3:
                canvas.segments(top + 3, at, _chips(context())(room), room)
        ctx.app.click_targets.append((top, 0, width, target))
        for at in range(top + 1, top + rows):
            ctx.app.click_targets.append((at, 0, width, target))
        return []

    return Pane("hero", word + " in use", lines, grow=False, draw=draw, fixed=5, frame=False,
                min_lines=5, max_lines=5, focusable=False)


# -- the range tiles: day, week, month, year, all of history -------------------------

def range_tiles(ctx: Context, view: View) -> Pane:
    """One tile per range: hours, utilization and change against the previous period, a
    small trend; the chosen range is the cursor, and a tile opens its range detail."""
    metric = gpu_metric(ctx)
    unit = "GPU-h" if metric == "gpu" else "CPU-h"
    rows = {}  # type: Dict[str, Any]
    if not ctx.app.no_history:
        for name in hd.RANGE_NAMES:
            rows[name] = cluster_trend(ctx, name)[0]
    lines = [Line([(name, PLAIN)], range_target(name, {}), key=name) for name in hd.RANGE_NAMES]
    state = view.panes.setdefault("ranges", PaneState())
    state.cursor = hd.RANGE_NAMES.index(ctx.range)  # the tiles' cursor is the range
    loading = getattr(ctx.data, "_loading", None)
    fraction = getattr(ctx.data, "fraction", None)
    auto = ctx.data.auto_ranges() if hasattr(ctx.data, "auto_ranges") else hd.RANGE_NAMES

    def titled(name: str, inner: int, note: str) -> bool:
        """Whether ``note`` fits in the tile's title (as _box decides)."""
        return inner >= len(name) + fmt.width(note) + 5

    def since_label(name: str) -> str:
        """'since Jul 25' when the cluster's history begins inside the range: its hours
        and utilization are over that span (the same one the range detail uses)."""
        row = rows.get(name)
        cap = capacity(ctx)[0 if metric == "gpu" else 1]
        if row is None:
            return ""
        hours = row.gpu_hours if metric == "gpu" else row.cpu_hours
        util = row.gpu_util if metric == "gpu" else row.cpu_util
        if not (hours and util and cap):
            return ""
        span = hours * 3600.0 / (cap * util)
        length = dict(hd.RANGES).get(name)
        if length is not None and span > length - hd.DAY:
            return ""
        end = ctx.now - ctx.now % 3600
        return "since " + time.strftime("%b %d", time.localtime(end - span))

    def content(name: str, inner: int) -> List[List[Tuple[str, str]]]:
        row = rows.get(name)
        if ctx.app.no_history:
            return [[("history off", FAINT)], [], []]
        spark = row.spark if row is not None else None
        # figures once the range itself is loaded (its previous period may not be: then
        # the change is "?"); until then what loading it is doing, never partial sums
        if row is None or not (row.complete or name in ctx.data.loaded):
            if loading == name:
                bar = charts.gauge(fraction, max(1, inner - 5), ctx.ascii, ACCENT)
                second = bar + [(" " + (fmt.pct(fraction) if fraction is not None
                                        else "…").rjust(4), DIM)]
                first = [("loading…", DIM)]
            elif name in auto:
                first, second = [("loading…", DIM)], [("next in line", FAINT)]
            else:  # not loaded in the background: choosing it loads it
                first, second = [("not loaded", DIM)], [("open to load", FAINT)]
            trace = charts.spark(spark, inner, ctx.ascii, False, FAINT) if spark else []
            return [first, second, trace]
        hours = row.gpu_hours if metric == "gpu" else row.cpu_hours
        util = row.gpu_util if metric == "gpu" else row.cpu_util
        before = row.gpu_delta if metric == "gpu" else row.cpu_delta
        change = delta_cell(hours, before, row.previous_loaded) if name != "all" else []
        if change and change[0][0] in fmt.PLACEHOLDERS:
            change = []
        figure = [(fmt.hours(hours), bold(BRIGHT)), (" " + unit, DIM)]
        pct = [(fmt.pct(util), bold(charts.ramp_style(util)) if util is not None else FAINT),
               (" util", DIM)]
        since = since_label(name)
        if since:  # history begins inside the range: no "new", say where it begins
            change = []
        under = None  # type: Optional[List[Tuple[str, str]]]
        if since and not titled(name, inner, since):  # under the figures, not the title
            label = [(since.replace("since", "from"), WARN)]
            room = inner - charts.width(label) - 1
            trend = charts.spark(spark, room, ctx.ascii, ctx.color) if room >= 3 else []
            under = label + ([(" ", PLAIN)] + trend if trend else [])
        if inner >= 18:  # one layout for every tile, so the strip reads across
            first = _figure(figure, change)(inner)
            room = inner - charts.width(pct) - 1
            second = pct + ([(" ", PLAIN)] + charts.gauge(util, room, ctx.ascii)
                            if room >= 4 else [])
        else:  # narrow: the change moves down beside the trend
            first, second = figure, pct
            room = inner - charts.width(pct) - 1
            if room >= 4:
                second = pct + [(" ", PLAIN)] + charts.gauge(util, room, ctx.ascii)
            if under is not None:
                return [first, second, under]
            if change:
                room = inner - charts.width(change) - 1
                trend = charts.spark(spark, room, ctx.ascii, ctx.color) if room >= 3 else []
                return [first, second, change + ([(" ", PLAIN)] + trend if trend else [])]
        return [first, second, under if under is not None
                else charts.spark(spark, inner, ctx.ascii, ctx.color)]

    def draw(canvas: Any, top: int, height: int, width: int, state: Any, focused: bool,
             shown: List[Line]) -> List[Tuple[int, ...]]:
        hits = []  # type: List[Tuple[int, ...]]
        glyphs = theme.box(ctx.ascii)
        count = len(hd.RANGE_NAMES)
        gap = 1
        room = width - gap * (count - 1)
        sizes = [room // count] * count
        for extra in range(room - sum(sizes)):
            sizes[extra] += 1
        col = 0
        current = hd.RANGE_NAMES.index(ctx.range)
        for index, (name, size) in enumerate(zip(hd.RANGE_NAMES, sizes)):
            chosen = index == current
            inner = size - 4
            if height >= 3:
                label = since_label(name) or RANGE_LENGTHS[name]
                note = label if titled(name, inner, label) else ""
                _box(canvas, top, col, size, height, name.capitalize(), note, chosen, glyphs)
            body = content(name, max(1, inner))
            start = top + 1 if height >= 3 else top
            for number in range(min(len(body), height - 2 if height >= 3 else height)):
                parts = body[number]
                if parts:
                    canvas.segments(start + number, col + 2, fit_parts(parts, inner)
                                    if charts.width(parts) > inner else parts, inner)
            for at in range(top, top + height):
                hits.append((at, index, col, col + size))
            col += size + gap
        return hits

    return Pane("ranges", "Ranges", lines, grow=False, draw=draw, fixed=5, frame=False,
                min_lines=5, max_lines=5, click_opens=True, range_cursor=True)


# -- Range detail ---------------------------------------------------------------------

def _live_used(ctx: Context, params: Dict[str, Any], metric: str) -> Optional[float]:
    """GPUs (or CPUs) held now by the scope: the cluster, a person or an account."""
    snap = ctx.snap
    if snap is None:
        return None
    if not any(params.get(role) for role in SCOPE_ROLES):
        res = snap.totals.gpus if metric == "gpu" else snap.totals.cpus
        if res.alloc is None:
            return None
        return float(res.alloc + (res.completing or 0))
    total = 0
    for job in snap_jobs(ctx):
        if not job.holds_resources:
            continue
        if params.get("user") and job.user != params["user"]:
            continue
        if params.get("account") and job.account != params["account"]:
            continue
        if params.get("qos") and job.qos != params["qos"]:
            continue
        if params.get("partition") and params["partition"] not in (job.partition or "").split(","):
            continue
        total += (job.gpus_alloc or 0) if metric == "gpu" else (job.cpus or 0)
    return float(total)


def range_entry(ctx: Context, params: Dict[str, Any], name: str, roles: Sequence[str]) -> Entry:
    """The range binned for its chart. For the whole cluster the figures are the tiles'
    exactly: a day or a week is queried over the tiles' window (hourly or 3-hour bins
    and the previous period at the same resolution), longer ranges come from the hourly
    rollup itself (a busy year in a fraction of a second, not many seconds)."""
    gpus, cpus = capacity(ctx)
    filt = make_filter(params)
    win = None
    if not params and ctx.data.has("overview"):
        if name in ("month", "year", "all"):  # long ranges of the cluster: the rollup
            return ctx.hist(("rollup-range", name, gpus, cpus),
                            lambda n=name: ctx.data.rollup_series(n, gpus, cpus))
        summary = rollup_summary(ctx, name)  # short ones: exact, over the tiles' window
        if summary is not None:
            win = hd.Window(summary.start, summary.end)
    return ctx.hist(("range", name, filt_key(params), tuple(roles), win, gpus, cpus),
                    lambda n=name, f=filt, r=tuple(roles), w=win: ctx.data.range_series(
                        n, f, r, w, gpus, cpus))


def _headline(series: Any, metric: str, name: str, ctx: Context, scoped: bool,
              cap: Optional[int]) -> List[List[Tuple[str, str]]]:
    """The figures that matter, most important first: total, average in use,
    utilization, peak, change, the other metric."""
    unit = "GPU" if metric == "gpu" else "CPU"
    total = series.total.amount(metric)
    out = []  # type: List[List[Tuple[str, str]]]
    mark = "" if series.complete else "≥"
    out.append([(mark + fmt.hours(total / 3600.0 if total is not None else None), bold(BRIGHT)),
                (" %s-h" % unit, DIM)])
    mean = series.mean(metric)
    if mean is not None:
        out.append([("avg ", DIM), (plot._figure(mean), bold(BRIGHT)), (" %ss in use" % unit, DIM)])
    util = series.gpu_util if metric == "gpu" else series.cpu_util
    if util is not None:
        out.append([(fmt.pct(util), bold(charts.ramp_style(util) if not scoped else BRIGHT)),
                    (" of cluster" if scoped else " of capacity", DIM)])
    if series.since > series.start + hd.DAY:  # history begins inside the range
        out.insert(1, [("data from ", DIM), (time.strftime("%b %d", time.localtime(
            series.since)), bold(WARN))])
    index, peak = series.peak(metric)
    if peak is not None and index is not None:
        when = series.edges[index]
        stamp = time.strftime("%H:%M" if series.unit in ("hour", "3hour") and name == "day"
                              else "%a %d %H:%M" if series.unit in ("hour", "3hour")
                              else "%b %d", time.localtime(when))
        # a bar is an average over its bin, so its peak is the busiest bin's average
        out.append([("peak %s avg " % plot.bin_word(series.unit), DIM),
                    (plot._figure(peak), bold(BRIGHT)), (" " + stamp, DIM)])
    if name != "all":
        before = series.previous.amount(metric) if series.previous is not None else None
        change = charts.delta(total, before, series.previous_complete)
        if before == 0 and series.previous_complete:
            out.append([("none in the previous " + RANGE_WORDS[name], DIM)])
        elif change and change[0][0] not in fmt.PLACEHOLDERS:
            out.append(change + [(" vs previous " + RANGE_WORDS[name], DIM)])
    other = "cpu" if metric == "gpu" else "gpu"
    amount = series.total.amount(other)
    if amount:
        out.append([(fmt.hours(amount / 3600.0), PLAIN),
                    (" %s-h" % ("CPU" if other == "cpu" else "GPU"), DIM)])
    return out


def scope_chart(ctx: Context, params: Dict[str, Any], name: str, metric: str,
                roles: Sequence[str] = (), previous: bool = True) -> Tuple[Entry, Any, Any]:
    """(entry, RangeSeries or None, plot.Chart or None) for a range and scope. The
    ceiling drawn is the cluster's capacity; for an account with a GPU limit, the
    limit; for a person (or an account without one), the cluster's, named so."""
    entry = range_entry(ctx, params, name, roles)
    series = value(entry)
    gpus, cpus = capacity(ctx)
    cap = gpus if metric == "gpu" else cpus  # type: Optional[float]
    cap_label = "cluster" if params else "capacity"
    if params.get("account") and not params.get("user") and metric == "gpu":
        found = _gpu_cap(ctx, params["account"])
        if found:
            cap, cap_label = found[1], "limit"
    chart = None
    if series is not None and series.seconds(metric) is not None:
        before = series.average(metric, previous=True) if name != "all" and previous \
            else None
        if before is not None and not any(before):
            before = None  # nothing then: no line, and no legend for it
        chart = plot.Chart(series.average(metric), series.edges, series.unit, before,
                           cap, series.mean(metric), _live_used(ctx, params, metric), ctx.now,
                           "GPUs" if metric == "gpu" else "CPUs", cap_label)
    return entry, series, chart


def chart_status(ctx: Context, entry: Entry, series: Any, chart: Any, name: str
                 ) -> List[Any]:
    """The chart's title notes: loading or updating. (The legend is drawn in the chart,
    under its time axis, where it cannot be cut off or scrolled away.)"""
    status = []  # type: List[Any]
    if series is not None and not series.complete:
        status.append([("loading " + (ctx.data.progress or "…"), WARN)])
    elif busy(entry) and series is not None:
        status.append([("updating…", DIM)])
    return status


def draw_chart(canvas: Any, row: int, rows: int, ctx: Context, chart: Any, entry: Entry,
               inner: int, what: str, name: str = "") -> None:
    if chart is None:
        if value(entry) is None:
            text = "computing…" if busy(entry) else "history " + missing(entry)
        else:
            text = "no %s usage recorded" % what.rstrip("s")
        canvas.put(row, 2, text, FAINT, limit=inner)
        return
    if rows < 3:
        return
    key = rows >= 5  # the legend row, under the time axis
    for parts in plot.area_chart(chart, inner, rows - 1 - key, ctx.ascii, ctx.color):
        canvas.segments(row, 2, parts, inner)
        row += 1
    if key:
        canvas.segments(row, 2, plot.legend(chart, RANGE_WORDS.get(name, name), ctx.ascii,
                                             inner, ctx.color), inner)


def overview_chart(ctx: Context, rows: int) -> Pane:
    """The chosen range's chart on the overview, when there is room; opens the detail."""
    name, metric = ctx.range, gpu_metric(ctx)
    entry, series, chart = scope_chart(ctx, {}, name, metric)
    what = "GPUs" if metric == "gpu" else "CPUs"
    target = range_target(name, {})

    def draw(canvas: Any, top: int, height: int, width: int, state: Any, focused: bool,
             shown: List[Line]) -> List[Tuple[int, ...]]:
        draw_chart(canvas, top, height, ctx, chart, entry, width - 4, what, name)
        for at in range(top - 1, top + height):
            ctx.app.click_targets.append((at, 0, width, target))
        return []

    title = "%s in use · %s" % (what, name if name != "all" else "all history")
    return Pane("chart", title, [Line([(title, PLAIN)], target, key=name)], grow=False,
                draw=draw, fixed=rows, focusable=False, min_lines=rows, max_lines=rows,
                status=chart_status(ctx, entry, series, chart, name))


def range_detail(ctx: Context, view: View) -> List[Pane]:
    params = {k: view.params.get(k) for k in SCOPE_ROLES if view.params.get(k)}
    name = ctx.range
    metric = view.extra.get("metric") or gpu_metric(ctx)
    scoped = bool(params)
    who = scope_name(params) or (
        ctx.snap.cluster.name if ctx.snap and ctx.snap.cluster.name else "cluster")
    view.title = "%s · %s" % (name.capitalize(), scope_name(params) or "cluster")
    roles = scope_roles(params)
    entry, series, chart = scope_chart(ctx, params, name, metric, roles,
                                       not view.extra.get("hide_previous"))
    gpus, _ = capacity(ctx)
    heads = _headline(series, metric, name, ctx, scoped, gpus) if series is not None else []
    what = "GPUs" if metric == "gpu" else "CPUs"

    # how the screen splits: the chart takes what the breakdown below does not need
    below = range_breakdown(ctx, params, name, series, entry, metric)
    most = 3 if ctx.height < 30 else max(3, min(8, (ctx.height - 10) // 4))
    shown = view.extra.get("tab:breakdown")  # the tab on show sizes the box
    rows_of = [len(p.lines) for p in below if p.id == shown] or [len(below[0].lines)]
    breakdown_rows = max(1, min(most, rows_of[0]))
    chart_body = max(4, ctx.height - 3 - 1 - 2 - breakdown_rows)

    def draw(canvas: Any, top: int, rows: int, width: int, state: Any, focused: bool,
             shown: List[Line]) -> List[Tuple[int, ...]]:
        inner = width - 4
        row = top
        # the range tabs: click one, or [ ] ← →
        col = 2
        for option in hd.RANGE_NAMES:
            label = " %s " % option.capitalize()
            end = canvas.put(row, col, label, TAB if option == name else DIM)
            ctx.app.click_targets.append((row, col, end, ("range", option)))
            col = end + 1
        if series is not None:
            fmt_start = ("%b %d %Y" if name in ("year", "all") else
                         "%a %b %d %H:%M" if name in ("day", "week") else "%b %d")
            span = "%s – %s" % (time.strftime(fmt_start, time.localtime(
                max(series.start, series.since))),
                                "now" if abs(ctx.now - series.end) < 7200
                                else time.strftime("%b %d %H:%M", time.localtime(series.end)))
            if col + 2 + fmt.width(span) <= inner + 2:
                canvas.put(row, col + 2, span, DIM)
        row += 1
        if rows < 3 or series is None:
            draw_chart(canvas, row, top + rows - row, ctx, None, entry, inner, what)
            return []
        # the headline figures, one line (two when there is room and they need it)
        lines_of = [[]]  # type: List[List[Tuple[str, str]]]
        sep = [("  ·  ", FAINT)]
        allowed = 2 if rows >= 14 else 1
        for item in heads:
            current = lines_of[-1]
            add = (sep if current else []) + item
            if charts.width(current) + charts.width(add) <= inner:
                current.extend(add)
            elif len(lines_of) < allowed:
                lines_of.append(list(item))
            else:
                break
        for parts in lines_of:
            canvas.segments(row, 2, parts, inner)
            row += 1
        row += 1 if rows - (row - top) > 10 else 0
        draw_chart(canvas, row, top + rows - row, ctx, chart, entry, inner, what, name)
        return []

    chart_pane = Pane("chart", "%s in use · %s" % (what, who), [text_line(name)],
                      grow=False, draw=draw, fixed=chart_body, focusable=False,
                      status=chart_status(ctx, entry, series, chart, name),
                      min_lines=chart_body, max_lines=chart_body)
    return [chart_pane] + below


def scope_roles(params: Dict[str, Any]) -> Tuple[str, ...]:
    """What a range detail breaks its usage down by: a person's accounts, an account's
    people, a QOS's or a partition's (or the cluster's) accounts and people."""
    if params.get("user"):
        return ("account",)
    if params.get("account"):
        return ("user",)
    return ("account", "user")


def range_breakdown(ctx: Context, params: Dict[str, Any], name: str, series: Any,
                    entry: Entry, metric: str) -> List[Pane]:
    """Who used the range: accounts and people for the cluster, people for an account,
    accounts for a person; and kinds of runs for a person or an account. Each drills
    into its own range detail (or a run type into its jobs)."""
    unit = "GPU-h" if metric == "gpu" else "CPU-h"
    columns = (
        Column("Name", 12, "<", 10, flex=True, most=24),
        Column(unit, 7, ">", 9),
        Column("share", 13, ">", 8, flex=True, most=36),
        Column("avg in use", 10, ">", 6),
        Column("vs prev", 7, ">", 7),
    )
    tabs = [(role, {"account": "Accounts", "user": "People"}[role])
            for role in scope_roles(params)]
    panes = []
    span = (series.end - series.since) if series is not None else 0  # as the headline
    for role, label in tabs:
        table = Table(columns, ctx.width - 2)
        lines = []
        if series is None:
            lines.append(text_line("history: " + ("computing…" if busy(entry)
                                                  else missing(entry)), FAINT))
        else:
            grand = series.total.amount(metric) or 0
            before = series.previous_groups.get(role, {})
            for key, stats in series.groups.get(role, ()):
                amount = stats.amount(metric)
                if not amount:
                    continue
                prev = before.get(key)
                change = charts.delta(amount, prev.amount(metric) if prev is not None else
                                      (0 if series.previous_complete else None),
                                      series.previous_complete) if name != "all" else \
                    [(fmt.NOT_APPLICABLE, FAINT)]
                share = amount / float(grand) if grand else None
                drill = dict(params)
                drill[role] = key
                if role == "account" and params.get("user"):
                    drill = {"account": key}
                target = range_target(name, drill)
                style = USER if role == "user" else RESOURCE
                cells = [(key, style), fmt.hours(amount / 3600.0), (lambda size, f=share: share_cell(f, ctx, max(3, size - 7))),
                         plot._figure(amount / float(span)) if span else fmt.NA, change]
                lines.append(table.row(cells, target, {"amount": amount, "name": key},
                                       key=key))
        status = "" if series is None or series.complete else "≥ partly loaded"
        panes.append(Pane("by_" + role, label, lines, table.header(),
                          sorts=(unit, "name"), natural=unit, group="breakdown", tab=label,
                          status=status))
    if params:
        filt = make_filter(params)
        types = ctx.hist(("runtypes", name, filt_key(params), "gpu_count"),
                         lambda r=name, f=filt: ctx.data.run_types(r, f, "gpu_count",
                                                                   medians=False))
        kinds = value(types)
        table = Table(columns[:3] + columns[4:5], ctx.width - 2)
        lines = []
        if kinds is None:
            lines.append(text_line("history: " + ("computing…" if busy(types)
                                                  else missing(types)), FAINT))
        else:
            for kind in kinds:  # each opens its breakdown (j there lists the jobs)
                target = runtype_target(params, "gpu_count", kind.key, kind.label)
                lines.append(table.row([kind.label, ("" if kind.complete else "≥")
                                        + fmt.hours(kind.gpu_hours),
                                        share_cell(kind.share, ctx, 6), ""], target,
                                       key=kind.key))
        panes.append(Pane("by_size", "Run sizes", lines, table.header(), group="breakdown",
                          tab="Run sizes"))
    return panes


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
    # the same query as Roles by account (one answer for both); the sparklines come from
    # the per-kind series, binned by the history layer
    entry = range_report(ctx, {}, (("account",), ("user", "account")))
    trends = kind_trends(ctx, {}, ctx.range, ("account",)) if ctx.data.has("kind_series") \
        else None
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
    # (not yet known counts as full: the column must not appear and vanish while loading)
    if ctx.app.no_history or all(v in ("full", fmt.NOT_YET) for v in seen.values()):
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
            if trends is not None:
                found = trends.get("account", {}).get(name)
                spark = list(found) if found else None
            elif report is not None and getattr(report, "series_by", None) == "account":
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
    panes = [Pane("people", "People" + (" · " + scope if scope else "") + " · " + ctx.range,
                  lines, table.header(), sorts=("GPU-h", "GPUs", "pending", "user"),
                  status=status, natural="GPU-h")]
    if params["account"] and not params["qos"] and not params["partition"]:
        # an account's own trend first: each range opens its range detail
        panes.insert(0, trend_pane(ctx, {"account": params["account"]}, person=True))
    return panes


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
                count = sum(1 for l in qos.limits if l.limit is not None)
                if count:
                    limits[qos.name] = count
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


# events about a whole role or period open a breakdown or a chart; events about a
# few attempts (a failure burst, a preemption run, a pause) open those attempts
_ROLE_EVENTS = {"first_account": "account", "first_qos": "qos"}
_CHART_EVENTS = ("record_day", "change", "no_baseline")


def event_target(event: Any, user: Optional[str]) -> Optional[Target]:
    title = event.text if len(event.text) <= 40 else event.text[:39] + "…"
    if event.job_id:
        return Target("job", {"job_id": event.job_id}, "job " + event.job_id)
    role = _ROLE_EVENTS.get(event.kind)
    where = event.where.items() if event.where is not None else ()
    values = {r: next(iter(v)) for r, v in where if len(v) == 1}
    if role is not None and values.get(role) and user:
        return runtype_target({"user": user}, role, values[role], values[role])
    if event.kind in _CHART_EVENTS and user:
        return range_target_current(user)
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


def range_target_current(user: str) -> Target:
    """The person's range detail for the range on show (push keeps the current range)."""
    return Target("range", {"user": user}, "Range · %s" % user)


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
    panes.append(Pane("live", "Now", lines, status=status, grow=False,
                      max_lines=4 if ctx.height >= 30 else 2))
    more = "more"  # one box at the bottom: your roles first, then events, jobs, activity
    range_name = ctx.range
    filt = make_filter(params)

    # where the GPU-hours went, by kind of run (g: GPUs, job name, outcome), and under
    # which roles: both answered on the page itself, each row opening its breakdown
    panes.append(gpu_breakdown_pane(ctx, view, params))
    panes.append(roles_breakdown_pane(ctx, params, group=more))

    # events (the tabs behind the one on show are worked out after what is on screen)
    shown = view.extra.get("tab:" + more, "roles")
    events_entry = ctx.hist(("events", range_name, user),
                            lambda r=range_name, f=filt: ctx.data.events(r, f),
                            later=shown != "events")
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

    # jobs list: a page at a time; a search asks history, not only the rows loaded
    jobs = paged_jobs(ctx, view, "Jobs · " + ctx.range, ("jobs", ctx.range, user, "gpu"), filt,
                      later=shown != "jobs")
    jobs.group, jobs.tab = more, jobs.title.replace(" · " + ctx.range, "")
    report_entry = range_report(ctx, params, (("account",), ("qos",), ("partition",)))
    activity = activity_pane(ctx, params, value(report_entry), report_entry)
    activity.group, activity.tab = more, "Activity"
    panes += [jobs, activity]
    return panes


# -- where the GPU-hours went --------------------------------------------------------

# the kinds of runs the Me/Person page splits GPU-hours by (g cycles them); accounts,
# QOS and partitions have their own pane beside it
PAGE_DIMENSIONS = ("gpu_count", "name_family", "outcome")
# one color per slice of a stacked bar, and for its row's name, largest first
SLICE_STYLES = ("heat5", ACCENT, GOOD, WARN, USER, RESOURCE, "heat3", BAD)
SLICE_GLYPHS = "█▓▒░"  # the same slices without color
BREAKDOWN_COLUMNS = (
    Column("Name", 12, "<", 10, flex=True, most=30),
    Column("GPU-h", 6, ">", 9),
    Column("share", 12, ">", 8, flex=True, most=30),
    Column("vs prev", 7, ">", 7),
    Column("trend", 8, ">", 6, flex=True, most=SPARK_MOST),
)
DIM_WORDS = {"gpu_count": "GPUs", "node_count": "nodes", "name_family": "job name",
             "outcome": "outcome", "account": "account", "qos": "QOS", "partition": "partition"}


def slice_style(index: int, ctx: Context) -> Tuple[str, str]:
    """(glyph, style) of the index-th slice: a color, or a shade without color."""
    if ctx.color:
        return ("#" if ctx.ascii else "█"), SLICE_STYLES[index % len(SLICE_STYLES)]
    glyphs = "#=+-" if ctx.ascii else SLICE_GLYPHS
    return glyphs[index % len(glyphs)], PLAIN


def stacked_bar(ctx: Context, shares: Sequence[float], size: int) -> List[Tuple[str, str]]:
    """One bar of ``size`` cells split by ``shares``, each slice in its row's color."""
    cells = breakdown.proportional(list(shares), size)
    out = []  # type: List[Tuple[str, str]]
    for index, count in enumerate(cells):
        if count:
            glyph, style = slice_style(index, ctx)
            out.append((glyph * count, style))
    used = sum(cells)
    if used < size:
        out.append((("." if ctx.ascii else "░") * (size - used), FAINT))
    return out


def scope_attempts(ctx: Context, params: Dict[str, Any], name: str) -> Entry:
    """The scope's attempts in the range (newest first, up to a page): trends per kind."""
    filt = make_filter(params)
    return ctx.hist(("attempts", name, filt_key(params)),
                    lambda r=name, f=filt: ctx.data.attempts(r, f))


def kind_trends(ctx: Context, params: Dict[str, Any], name: str, kinds: Tuple[str, ...]
                ) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """GPU-seconds per chart bin for each kind of run, from the history layer's series
    (exact, every attempt); None while loading or when it cannot tell (then the scope's
    attempts stand in)."""
    if not ctx.data.has("kind_series"):
        return None
    filt = make_filter(params)
    found = value(ctx.hist(("kind-trends", name, filt_key(params), kinds),
                           lambda r=name, f=filt, k=kinds: ctx.data.kind_trends(r, f, k)))
    return None if found is None else found[1]


def previous_types(ctx: Context, params: Dict[str, Any], name: str, dim: str) -> Entry:
    """GPU-hours per kind of run in the previous equal period, for the change column."""
    filt = make_filter(params)
    data = ctx.data

    def load(r: str = name, f: Any = filt, d: str = dim) -> Optional[Dict[str, float]]:
        win = hd.previous_window(r, data.clock())
        if win is None:
            return None
        if data.has("kind_series") and d != "name_family":  # GPU-seconds only: cheaper
            _, per = data.history.kind_series(win, f, kinds=(d,),
                                              cancel=getattr(data, "_cancel", None),
                                              make_bins=lambda a, b: [a, b] if b > a else [a])
            return {key: sum(values) / 3600.0 for key, values in per.get(d, {}).items()}
        if data.has("run_types"):
            found = data.history.run_types(win, f, by=d, cancel=getattr(data, "_cancel", None))
            return {item.key: (item.stats.gpu_hours or 0.0) for item in found}
        page = data.attempts(r, f, win=win)
        out = {}  # type: Dict[str, float]
        for kind in hd.run_types(page.rows, d):
            out[kind.key] = kind.gpu_hours or 0.0
        return out
    return ctx.hist(("runtypes-prev", name, filt_key(params), dim), load)


def range_bins(ctx: Context, name: str, rows: Sequence[Any]) -> List[int]:
    """The range's chart bins (for all of history, from the oldest attempt shown)."""
    return bins_at(name, rows, int(ctx.now))


def bins_at(name: str, rows: Sequence[Any], end: int) -> List[int]:
    start = hd.window(name, end).start
    if start is None:
        starts = [r.start for r in rows if r.start is not None]
        start = min(starts) if starts else end - 86400
    return hd.bin_edges(int(start), end, hd.bin_unit(name, int(start), end))


def family_trends(data: Any, name: str, filt: Any) -> Dict[Optional[str], List[float]]:
    """GPU-seconds per chart bin for each job-name family of the scope's attempts
    (newest first, up to a page): worked out off the UI thread, once per answer."""
    rows = data.attempts(name, filt).rows
    end = int(data.clock())
    return breakdown.held_per_bin(rows, bins_at(name, rows, end), end,
                                  lambda row: hd._key(row, "name_family"))


class KindView(NamedTuple):
    """What the run-type screen draws from its attempts, worked out once per answer
    (off the UI thread): how many there are, their breakdown and GPUs held per bin."""
    count: int
    info: Any  # breakdown.Detail
    edges: List[int]
    held: List[float]


def kind_view(rows: Sequence[Any], name: str, by_people: bool, end: int) -> KindView:
    edges = bins_at(name, rows, end)
    held = breakdown.held_per_bin(rows, edges, end).get(None) or [0.0] * (len(edges) - 1)
    return KindView(len(rows), breakdown.detail(rows, "user" if by_people else "family"),
                    edges, held)


def gpu_breakdown_pane(ctx: Context, view: View, params: Dict[str, Any]) -> Pane:
    """The scope's GPU-hours by kind of run: one stacked bar, then a row per kind with
    its GPU-hours, share, change on the previous period and a small trend."""
    dim = view.extra.get("dim", "gpu_count")
    if dim not in PAGE_DIMENSIONS:
        dim = view.extra["dim"] = "gpu_count"
    name = ctx.range
    filt = make_filter(params)
    types_entry = ctx.hist(("runtypes", name, filt_key(params), dim),
                           lambda r=name, f=filt, d=dim: ctx.data.run_types(r, f, d,
                                                                            medians=False))
    kinds = value(types_entry)
    before = value(previous_types(ctx, params, name, dim)) if name != "all" else None
    trends = {}  # type: Dict[Optional[str], List[float]]
    exact = kind_trends(ctx, params, name, (dim,)) if dim != "name_family" else None
    source = exact  # type: Any  # the answer the trends come from
    if exact is not None:
        trends = dict(exact.get(dim, {}))
    elif dim == "name_family":
        source = value(ctx.hist(("family-trends", name, filt_key(params)),
                                lambda r=name, f=filt: family_trends(ctx.data, r, f)))
        trends = source or {}
    elif not ctx.data.has("kind_series"):
        attempts = source = value(scope_attempts(ctx, params, name))
        if attempts is not None:
            rows = attempts.rows
            trends = breakdown.held_per_bin(rows, range_bins(ctx, name, rows), int(ctx.now),
                                            lambda row, d=dim: hd._key(row, d))
    table = Table(BREAKDOWN_COLUMNS, ctx.width - 2)
    if kinds is None:
        lines = [text_line("history: %s %s" % ("computing…" if busy(types_entry)
                                               else missing(types_entry),
                                               types_entry.message), DIM)]
    else:
        lines = kept_lines(ctx, ("runs", view.kind, name, filt_key(params), dim),
                           (kinds, before, source),
                           lambda: breakdown_lines(ctx, table, kinds, before, trends, name,
                                                   params, dim))
    status = [[("g", KEY), (" GPUs / job name / outcome", DIM)]]  # type: List[Any]
    if stale(types_entry):
        status.insert(0, "stale: %s %s" % (fmt.placeholder(types_entry.status),
                                           types_entry.message))
    who = "your" if view.kind == "me" else (params.get("user") or "") + "'s"
    rows_shown = max(4, min(10, (ctx.height - 18) // 2))
    return Pane("runs", "Where %s GPU-hours went · by %s · %s" % (who, DIM_WORDS[dim], name),
                lines, table.header(), sorts=("GPU-h", "name"), natural="GPU-h",
                status=status, grow=False, max_lines=rows_shown, min_lines=min(4, rows_shown))


def breakdown_lines(ctx: Context, table: Table, kinds: Sequence[Any], before: Any,
                    trends: Dict[Optional[str], List[float]], name: str,
                    params: Dict[str, Any], dim: str) -> List[Line]:
    """The stacked bar and a row per kind of run (see gpu_breakdown_pane)."""
    lines = []  # type: List[Line]
    kinds = [k for k in kinds if k.gpu_hours]
    total = sum(k.gpu_hours or 0 for k in kinds)
    if kinds:
        # the whole bar is the scope's GPU-hours: it opens their chart
        lines.append(Line(stacked_bar(ctx, [k.gpu_hours or 0 for k in kinds],
                                      max(10, ctx.width - 6)), range_target(name, params),
                          {"GPU-h": float("inf"), "name": ""}, key="(all)", pinned=True))
    else:
        lines.append(text_line("no GPU-hours in this range", FAINT))
    for index, kind in enumerate(kinds):
        share = (kind.gpu_hours or 0) / total if total else None
        glyph, style = slice_style(index, ctx)
        prev = before.get(kind.key, 0.0) if before is not None else None
        spark = trends.get(kind.key)
        # a swatch of its slice of the bar, so the bar reads without color too
        cells = [[(glyph, style), (" " + kind.label, style if ctx.color else PLAIN)],
                 ("" if kind.complete else "≥") + fmt.hours(kind.gpu_hours),
                 (lambda size, f=share: share_cell(f, ctx, max(3, size - 7))),
                 delta_cell(kind.gpu_hours, prev, before is not None)
                 if name != "all" else [(fmt.NOT_APPLICABLE, FAINT)],
                 (lambda size, data=spark: charts.spark(data, min(size, SPARK_MOST),
                                                        ctx.ascii, ctx.color))]
        lines.append(table.row(cells, runtype_target(params, dim, kind.key, kind.label),
                               {"GPU-h": kind.gpu_hours, "name": kind.label}, key=kind.key))
    return lines


ROLE_BREAKDOWN_COLUMNS = (Column("role", 9, "<", 9),) + BREAKDOWN_COLUMNS


def roles_breakdown_pane(ctx: Context, params: Dict[str, Any], group: Optional[str] = None
                         ) -> Pane:
    """The accounts, QOS and partitions the scope used in the range, by GPU-hours, with
    their share and change; each opens that role's breakdown."""
    name = ctx.range
    groups = (("account",), ("qos",), ("partition",))
    entry = range_report(ctx, params, groups)
    report = value(entry)
    filt = make_filter(params)
    prev = None
    if name != "all":
        prev = value(ctx.hist(("previous", name, filt_key(params), groups),
                              lambda r=name, f=filt: ctx.data.previous(r, f, groups)))
    by_role = {}  # type: Dict[Tuple[str, str], List[float]]
    exact = kind_trends(ctx, params, name, ("account", "qos", "partition"))
    if exact is not None:
        for role, per in exact.items():
            for key, bins in per.items():
                by_role[(role, key)] = bins
    elif not ctx.data.has("kind_series"):
        attempts = value(scope_attempts(ctx, params, name))
        trends = {}  # type: Dict[Optional[str], List[float]]
        if attempts is not None:
            rows = attempts.rows
            trends = breakdown.held_per_bin(rows, range_bins(ctx, name, rows), int(ctx.now),
                                            lambda row: "\0".join((row.account, row.qos,
                                                                     row.partition)))
        for joined, bins in trends.items():
            for role, part in zip(("account", "qos", "partition"), (joined or "").split("\0")):
                have = by_role.get((role, part))
                by_role[(role, part)] = bins if have is None else [a + b
                                                                   for a, b in zip(have, bins)]
    table = Table(ROLE_BREAKDOWN_COLUMNS, ctx.width - 2)
    lines = []
    if report is None:
        lines.append(text_line("history: %s" % ("computing…" if busy(entry)
                                                else missing(entry)), DIM))
    else:
        total = report.total.gpu_seconds or 0
        for role in ("account", "qos", "partition"):
            before = {key[0]: stats for key, stats in prev.group(role).items()} \
                if prev is not None and prev.complete else None
            first = True
            for key, stats in report.ranked(role):
                if not stats.gpu_seconds:
                    continue
                share = stats.gpu_seconds / total if total else None
                old = before.get(key[0]) if before is not None else None
                change = delta_cell(stats.gpu_hours, old.gpu_hours if old is not None else
                                    (0 if before is not None else None), before is not None) \
                    if name != "all" else [(fmt.NOT_APPLICABLE, FAINT)]
                cells = [(DIM_WORDS[role] if first else "", DIM), (key[0], RESOURCE),
                         fmt.hours(stats.gpu_hours),
                         (lambda size, f=share: share_cell(f, ctx, max(3, size - 7))), change,
                         (lambda size, data=by_role.get((role, key[0])): charts.spark(
                             data, min(size, SPARK_MOST), ctx.ascii, ctx.color))]
                first = False
                lines.append(table.row(cells, runtype_target(params, role, key[0], key[0]),
                                       {"GPU-h": stats.gpu_seconds, "name": key[0]},
                                       key=role + ":" + key[0]))
        if not lines:
            lines.append(text_line("no GPU-hours in this range", FAINT))
    return Pane("roles", "Roles · accounts, QOS, partitions · " + name, lines,
                table.header(), group=group, tab="Roles")


# -- one kind of run in detail ---------------------------------------------------------

def kind_words(dim: str, label: str) -> str:
    """'x4 runs', 'job name train_<n>', 'outcome failed', 'account vision'."""
    if dim == "gpu_count":
        return label + " runs"
    return "%s %s" % (DIM_WORDS.get(dim, dim), label)


def runtype_target(params: Dict[str, Any], dim: str, key: str, label: str) -> Target:
    scope = {k: params[k] for k in SCOPE_ROLES if params.get(k)}
    return Target("runtype", dict(scope, dim=dim, key=key, label=label),
                  "%s · %s" % (scope_name(scope) or "cluster", kind_words(dim, label)))


def runtype(ctx: Context, view: View) -> List[Pane]:
    """One kind of run (x4 GPU runs, a job-name family, an outcome, an account, a QOS or
    a partition) for a person or an account: its GPU-hours over the range, its share,
    outcomes, runtime and wait, and what it is made of. j lists its jobs."""
    p = view.params
    params = {k: p[k] for k in SCOPE_ROLES if p.get(k)}
    dim, key = p["dim"], p["key"]
    label = p.get("label") or hd.type_label(dim, key)
    name = ctx.range
    filt = make_filter(params)
    data = ctx.data
    kinds_entry = ctx.hist(("runtypes", name, filt_key(params), dim),
                           lambda r=name, f=filt, d=dim: data.run_types(r, f, d, medians=False))
    kinds = value(kinds_entry)
    kind = next((k for k in kinds or () if k.key == key), None)
    local = kind.rows if kind is not None else None
    by_people = bool(params.get("account")) and not params.get("user")

    def load(r: str = name, f: Any = filt, d: str = dim, k: str = key, ready: Any = local,
             people: bool = by_people) -> KindView:
        rows = ready  # computed from the attempts list (no history run types)
        if rows is None:
            win = hd.window(r, data.clock())
            more = data.uncounted() if hasattr(data, "uncounted") else {}
            rows = data.history.jobs(win, f, by="attempt", run_type=(d, k), sort="start",
                                     descending=True, limit=hd.ATTEMPT_LIMIT,
                                     cancel=getattr(data, "_cancel", None), **more).rows
        return kind_view(rows, r, people, int(data.clock()))
    rows_entry = ctx.hist(("runtype-view", name, filt_key(params), dim, key, by_people), load)
    found = value(rows_entry)
    rows = found  # None while loading
    info = found.info if found is not None else None
    chart = None
    if found is not None:
        edges, held = found.edges, found.held
        if any(held):
            values = [h / float(edges[i + 1] - edges[i]) if edges[i + 1] > edges[i] else 0.0
                      for i, h in enumerate(held)]
            span = float(max(1, edges[-1] - edges[0]))
            chart = plot.Chart(values, edges, hd.bin_unit(name, edges[0], edges[-1]), None,
                               None, sum(held) / span, None, ctx.now, "GPUs")
    who = scope_name(params) or "cluster"
    heads = []  # type: List[List[Tuple[str, str]]]
    if kind is not None:
        heads.append([(("" if kind.complete else "≥") + fmt.hours(kind.gpu_hours),
                       bold(BRIGHT)), (" GPU-h", DIM)])
        if kind.share is not None:
            heads.append([(fmt.pct(kind.share), bold(BRIGHT)), (" of %s's GPU-h" % who, DIM)])
        heads.append([(fmt.num(kind.attempts), PLAIN), (" attempts / ", DIM),
                      (fmt.num(kind.jobs), PLAIN), (" jobs", DIM)])
        if kind.failure is not None:
            heads.append([(fmt.pct(kind.failure), BAD if kind.failure >= 0.25 else PLAIN),
                          (" failed", DIM)])
    elif kinds is not None:
        heads.append([("none of these runs in this range", FAINT)])
    else:
        heads.append([("computing…" if busy(kinds_entry) else "history " + missing(kinds_entry),
                       FAINT)])

    def draw(canvas: Any, top: int, height: int, width: int, state: Any, focused: bool,
             shown: List[Line]) -> List[Tuple[int, ...]]:
        inner = width - 4
        row, col = top, 2
        for option in hd.RANGE_NAMES:
            text = " %s " % option.capitalize()
            end = canvas.put(row, col, text, TAB if option == name else DIM)
            ctx.app.click_targets.append((row, col, end, ("range", option)))
            col = end + 1
        row += 1
        line = []  # type: List[Tuple[str, str]]
        for item in heads:
            add = ([("  ·  ", FAINT)] if line else []) + item
            if charts.width(line) + charts.width(add) > inner:
                break
            line += add
        canvas.segments(row, 2, line, inner)
        row += 1
        if info is not None and info.attempts and height >= 10:
            split = breakdown.outcome_split(info.outcomes)
            total = float(sum(n for _, n in split)) or 1.0
            bar = max(10, min(40, inner // 3))
            seg = [(fmt.fit("outcomes", 10), DIM)]  # type: List[Tuple[str, str]]
            for (outcome, count), size in zip(split, breakdown.proportional(
                    [n for _, n in split], bar)):
                if size:
                    seg.append((("#" if ctx.ascii else "█") * size,
                                OUTCOME_STYLE.get(outcome, PLAIN)))
            seg.append(("  ", PLAIN))
            for outcome, count in split:
                seg += [(outcome, OUTCOME_STYLE.get(outcome, PLAIN)),
                        (" %s  " % fmt.pct(count / total), DIM)]
            canvas.segments(row, 2, seg, inner)
            row += 1
            canvas.segments(row, 2, [
                (fmt.fit("runtime", 10), DIM), ("median ", DIM),
                (fmt.duration(info.runtime[0]), PLAIN), ("  p90 ", DIM),
                (fmt.duration(info.runtime[1]), PLAIN), ("     wait  ", DIM), ("median ", DIM),
                (fmt.duration(info.wait[0]), PLAIN), ("  p90 ", DIM),
                (fmt.duration(info.wait[1]), PLAIN)], inner)
            row += 1
        if chart is None and rows is not None:
            canvas.put(row, 2, "no GPUs held by these runs in this range", FAINT, limit=inner)
            return []
        draw_chart(canvas, row, top + height - row, ctx, chart, rows_entry, inner, "GPUs", name)
        return []

    below = runtype_breakdown(ctx, params, dim, key, label, info, rows_entry, by_people)
    most = 3 if ctx.height < 30 else max(3, min(10, (ctx.height - 10) // 3))
    body = max(8, ctx.height - 3 - 1 - 2 - max(1, min(most, len(below.lines))))
    status = [[("j", KEY), (" its jobs", DIM)]]  # type: List[Any]
    if rows is None and busy(rows_entry):
        status.insert(0, [("computing…", DIM)])
    elif found is not None and found.count >= hd.ATTEMPT_LIMIT:
        status.insert(0, [("newest %s attempts" % fmt.num(found.count), WARN)])
    chart_pane = Pane("chart", "GPUs in use · %s · %s" % (who, kind_words(dim, label)),
                      [text_line(name)], grow=False, draw=draw, fixed=body, focusable=False,
                      status=status, min_lines=body, max_lines=body)
    return [chart_pane, below]


def runtype_jobs_target(view: View) -> Target:
    """``j`` on a run-type breakdown: its jobs, as a list."""
    p = view.params
    scope = {k: p[k] for k in SCOPE_ROLES if p.get(k)}
    return Target("jobs", dict(scope, run_type=(p["dim"], p["key"])),
                  "%s · jobs" % view.title)


def runtype_breakdown(ctx: Context, params: Dict[str, Any], dim: str, key: str, label: str,
                      info: Any, entry: Entry, by_people: bool) -> Pane:
    """What the kind of run is made of: job-name families (or, for an account, people),
    largest first; each opens its own breakdown."""
    columns = (Column("Name", 14, "<", 10, flex=True, most=40), Column("GPU-h", 6, ">", 9),
               Column("share", 12, ">", 8, flex=True, most=30), Column("att", 5, ">", 5),
               Column("fail", 5, ">", 6))
    table = Table(columns, ctx.width - 2)
    if info is None:
        lines = [text_line("history: %s" % ("computing…" if busy(entry) else missing(entry)),
                           DIM)]
    else:
        lines = kept_lines(ctx, ("parts", filt_key(params), dim, key, by_people), (info,),
                           lambda: parts_lines(ctx, table, params, dim, key, label, info,
                                               by_people))
    return Pane("parts", "People" if by_people else "Job names", lines, table.header(),
                sorts=("GPU-h", "attempts", "name"), natural="GPU-h")


def parts_lines(ctx: Context, table: Table, params: Dict[str, Any], dim: str, key: str,
                label: str, info: Any, by_people: bool) -> List[Line]:
    """runtype_breakdown's rows: the job-name families (or people) of one kind of run."""
    lines = []  # type: List[Line]
    total = info.gpu_seconds
    for group in info.groups:
        share = group.gpu_seconds / total if total else None
        if by_people:  # this person's runs of the same kind
            target = runtype_target(dict(params, user=group.key), dim, key, label)
            name = (group.key, USER)
        else:
            target = runtype_target(params, "name_family", group.key, group.key)
            name = (group.key, PLAIN)
        fail = group.failure
        lines.append(table.row(
            [name, fmt.hours(group.gpu_seconds / 3600.0),
             (lambda size, f=share: share_cell(f, ctx, max(3, size - 7))),
             fmt.num(group.attempts),
             (fmt.pct(fail), BAD if fail >= 0.25 else PLAIN) if fail is not None
             else (fmt.NOT_APPLICABLE, FAINT)], target,
            {"GPU-h": group.gpu_seconds, "name": group.key, "attempts": group.attempts},
            key=group.key))
    if not lines:
        lines.append(text_line("no attempts in this range", FAINT))
    return lines


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
    if rows is None:
        return Pane(pane_id, title, [text_line("history: %s %s" % (
            missing(entry), entry.message if entry else ""), DIM)])
    # laid out once per answer; the ages ("3 h ago") again every half minute
    lines = kept_lines(ctx, ("jobs", pane_id, title, hours, int(ctx.now // 30)), (rows,),
                       lambda: job_lines(ctx, table, rows))
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


def job_lines(ctx: Context, table: Table, rows: Sequence[Any]) -> List[Line]:
    lines = []  # type: List[Line]
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
    return lines


def paged_jobs(ctx: Context, view: View, title: str, key: Tuple[Any, ...], filt: Any,
               win: Any = None, run_type: Any = None, pane_id: str = "jobs",
               later: bool = False) -> Pane:
    """The first JOB_PAGE jobs, more as the cursor reaches the end (app.py raises
    view.extra["jobs_limit"]); the pane's / filter is also sent to history as a search."""
    state = view.panes.get(pane_id)
    match = (state.filter if state is not None else "").strip()
    limit = view.extra.get("jobs_limit", hd.JOB_PAGE)
    entry = ctx.hist(key, lambda r=ctx.range, f=filt, l=limit, m=match: ctx.data.jobs(
        r, f, limit=l, win=win, run_type=run_type, match=m), version=(limit, match),
        later=later)
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

LANE_WORDS = {"user": "person", "account": "account", "qos": "QOS", "partition": "partition"}


class Lanes:
    """The timeline's data: per lane (a person, account, QOS or partition), the GPUs
    (or CPUs) it held on average in each column, and four shade levels from the data."""

    def __init__(self, start: int, end: int, lanes: List[Tuple[str, List[float], float, float,
                                                                int]],
                 bounds: List[float]) -> None:
        self.start, self.end = start, end
        self.lanes = lanes  # (key, average held per column, held-seconds, peak, peak column)
        self.bounds = bounds  # upper bounds of shade levels 1-3; level 4 is above


def lane_bounds(values: Sequence[float]) -> List[float]:
    """Three increasing bounds that split what the lanes held into four shades: the
    quartiles of every non-empty cell, rounded (whole GPUs from 1 up)."""
    held = sorted(v for v in values if v > 0)
    if not held:
        return [1.0, 2.0, 4.0]
    out = []  # type: List[float]
    for fraction in (0.25, 0.5, 0.75):
        value = held[min(len(held) - 1, int(fraction * len(held)))]
        value = float(max(1, int(round(value)))) if value >= 0.75 else round(value, 1) or 0.1
        if out and value <= out[-1]:
            value = out[-1] + (1 if out[-1] >= 1 else 0.1)
        out.append(value)
    return out


def lane_level(value: float, bounds: Sequence[float]) -> int:
    """0 (nothing held), else 1-4."""
    if value <= 0:
        return 0
    for level, bound in enumerate(bounds, 1):
        if value <= bound:
            return level
    return 4


def compute_lanes(report: Any, cols: int) -> Lanes:
    """Hourly per-key usage (``report.keyed_values``, seconds held per hour) averaged
    into ``cols`` columns over the report's span."""
    start, end = report.start, report.end
    span = float(max(1, end - start))
    size = span / cols
    edges = report.edges
    lanes = []
    every = []  # type: List[float]
    for key, seconds in (report.keyed_values or {}).items():
        held = [0.0] * cols  # seconds held, per column
        for hour, amount in enumerate(seconds):
            if not amount:
                continue
            a, b = max(edges[hour], start), min(edges[hour + 1], end)
            if b <= a:
                continue
            rate = amount / float(edges[hour + 1] - edges[hour])
            first = max(0, int((a - start) / size))
            last = min(cols - 1, int((b - start - 1e-9) / size))
            for col in range(first, last + 1):
                lo, hi = start + col * size, start + (col + 1) * size
                overlap = min(b, hi) - max(a, lo)
                if overlap > 0:
                    held[col] += rate * overlap
        average = [h / size for h in held]
        total = sum(held)
        if total <= 0:
            continue
        peak_col = max(range(cols), key=lambda c: (average[c], c))
        lanes.append((key, average, total, average[peak_col], peak_col))
        every += average
    lanes.sort(key=lambda lane: (-lane[2], lane[0]))
    return Lanes(start, end, lanes, lane_bounds(every))


def lanes_entry(ctx: Context, range_name: str, by: str, metric: str, cols: int) -> Entry:
    data = ctx.data

    def load(name: str = range_name, b: str = by, m: str = metric, c: int = cols) -> Lanes:
        report = data.history.query(hd.window(name, data.clock()), None, series_by=b,
                                    series_metric=m, efficiency=False,
                                    cancel=getattr(data, "_cancel", None))
        return compute_lanes(report, c)
    return ctx.hist(("lanes", range_name, by, metric, cols), load)


def _held(value: float) -> str:
    return plot._figure(round(value, 1) if value < 10 else round(value))


def lane_legend(lanes: Lanes, what: str, ascii_only: bool, color: bool) -> List[Any]:
    """'░ ≤4 ▒ ≤16 ▓ ≤32 █ >32 GPUs': what each shade means, from the data."""
    shades = charts.ASCII_SHADES if ascii_only else charts.SHADES
    out = []  # type: List[Tuple[str, str]]
    for level in range(1, 5):
        if out:
            out.append((" ", PLAIN))
        bound = lanes.bounds[level - 1] if level < 4 else lanes.bounds[-1]
        text = ("≤" if level < 4 else ">") + _held(bound)
        out += [(shades[level - 1], LANE_STYLES[level] if color else PLAIN), (text, DIM)]
    return out + [(" " + what + " held on average", DIM)]


LANE_STYLES = {1: "heat2", 2: "heat3", 3: "heat4", 4: "heat5"}


def timeline(ctx: Context, view: View) -> List[Pane]:
    """Who held the GPUs when: one lane per person, account, QOS or partition (``g``),
    each cell shaded by how many GPUs (or CPUs) the lane held on average then. A lane
    opens its range detail: its chart, figures and breakdown."""
    by = view.extra.get("lanes", "user")
    gpus_now = ctx.snap.totals.gpus.total if ctx.snap is not None else None
    metric = view.extra.get("metric") or ("gpu" if gpus_now != 0 else "cpu")
    what = "GPUs" if metric == "gpu" else "CPUs"
    label_width = min(16, max(8, ctx.width // 6))
    cols = max(10, ctx.width - label_width - 4)
    title = ("%s in use per %s over time · %s" if ctx.width >= 120 else "%s per %s · %s") % (
        what, LANE_WORDS[by], ctx.range)  # short when narrow: the shade scale must fit
    entry = lanes_entry(ctx, ctx.range, by, metric, cols)
    found = value(entry)
    if found is None:
        return [Pane("timeline", title, [text_line("history: %s %s" % (
            missing(entry), entry.message), DIM)])]
    shown = found.end - found.start
    size = shown / float(cols)
    axis = timeline_axis(found.start, size, cols, shown)
    header = Line([(fmt.fit(LANE_WORDS[by], label_width) + " ", DIM), ("".join(axis), DIM)])
    shades = charts.ASCII_SHADES if ctx.ascii else charts.SHADES
    style = USER if by == "user" else RESOURCE
    lines = []
    for key, average, total, peak, peak_col in found.lanes:
        cells = []  # type: List[Tuple[str, str]]
        for held in average:
            level = lane_level(held, found.bounds)
            if not level:
                cells.append((" ", PLAIN))
            else:
                cells.append((shades[level - 1], LANE_STYLES[level] if ctx.color else PLAIN))
        parts = [(fmt.fit(key, label_width) + " ", style)] + charts._join(cells)
        params = {by: key}
        lines.append(Line(parts, range_target(ctx.range, params),
                          {"busy": total, "name": key, "peak": peak,
                           "peak_at": found.start + (peak_col + 0.5) * size}, key=key))
    if not lines:
        lines.append(text_line("no %s held in this range" % what, FAINT))
    # the status names the selected lane: what it held on average and at its busiest
    state = view.panes.get("timeline") if isinstance(view.panes, dict) else None
    cursor = state.cursor if state is not None else 0
    # the selected lane in the title row; the shade scale on a row of its own under the
    # lanes, so a long title never pushes it out
    status = []  # type: List[Any]
    selected = [l for l in lines if l.sort]
    if selected:
        chosen = selected[max(0, min(len(selected) - 1, cursor))]
        info = chosen.sort
        status.append([(info["name"], style), (" avg ", DIM),
                       (_held(info["busy"] / float(max(1, shown))), PLAIN),
                       (" · peak ", DIM), (_held(info["peak"]), PLAIN),
                       (" " + time.strftime("%b %d %H:%M", time.localtime(info["peak_at"])), DIM),
                       ("  ↵ chart", DIM)])
    scale = Line([(fmt.fit("shade =", label_width) + " ", DIM)]
                 + lane_legend(found, what, ctx.ascii, ctx.color))
    return [Pane("timeline", title, lines, header, sorts=("busy", "name"),
                 status=status, select_width=label_width + 1, click_opens=True,
                 footer=scale)]


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
    ("  ← →             previous / next tab", PLAIN),
    ("  ↑ ↓ PgUp PgDn   move in the focused pane (PgUp/PgDn a page);  Home/End  first/last", PLAIN),
    ("  Tab Shift-Tab   next / previous pane", PLAIN),
    ("  Enter           drill into the row (›);  Esc / Backspace  back up the breadcrumb", PLAIN),
    ("  /  s  S         filter the pane as you type (Ctrl-U clears; a jobs list searches all of", PLAIN),
    ("                  history);  cycle sort column (▼▲ in its header);  reverse sort", PLAIN),
    ("  [ ]  1-5        range: day, week, month, year, all — shared by every view (or click one", PLAIN),
    ("                  in the header)", PLAIN),
    ("  ↵ on a range    Overview tiles, Trend rows: that range in detail — GPUs in use over time", PLAIN),
    ("                  against capacity and the previous period, who used it;  ← → [ ] step", PLAIN),
    ("                  through the ranges there, g GPUs or CPUs, v previous period on/off", PLAIN),
    ("  g               Roles: account/QOS/partition;  Person: run types by GPUs, nodes, job name,", PLAIN),
    ("                  account, QOS, partition, outcome;", PLAIN),
    ("                  Timeline: lanes by person/account/QOS/partition; ↵ or a click on a", PLAIN),
    ("                  lane opens its range detail (chart, figures, who used it)", PLAIN),
    ("  c               Timeline: GPU or CPU lanes", PLAIN),
    ("  l e b u         Job: stdout tail, stderr tail, batch script, submit line (on request only)", PLAIN),
    ("  E               load efficiency (step usage) for the current range and person", PLAIN),
    ("  r  ?  q         refresh now, this help, quit", PLAIN),
    ("  mouse           click a tab, a range or a tile; on a row a click selects and a second", PLAIN),
    ("                  click opens; the wheel scrolls the pane under the pointer", PLAIN),
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
    ("  lanes                 Timeline: each cell is a slice of the range, shaded by the GPUs (or", PLAIN),
    ("                        CPUs) the lane held on average then; the scale on the border is", PLAIN),
    ("                        the quartiles of all cells, so each shade is about as common", PLAIN),
    ("  paths                 Job: as Slurm recorded them; checked with os.stat, never read", PLAIN),
    ("  share                 part of the cluster's allocated GPU-hours in the same range;", PLAIN),
    ("                        of acct / of QOS / of part / of user: part of that one's GPU-hours", PLAIN),
    ("  members               Overview: users with an association or usage in the account", PLAIN),
    ("  GPUs in use           Range detail: allocated GPU-hours per bin ÷ the bin's hours (the", PLAIN),
    ("                        average held); bins: day hourly, week 3-hourly, month daily, year", PLAIN),
    ("                        weekly; the amber line is the previous period at the same offsets;", PLAIN),
    ("                        peak = the busiest bin's average; now = GPUs held this moment", PLAIN),
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
        user = ctx.app.user or ctx.app.invoker_name()  # known before the first snapshot
    view.params["user"] = user
    return person(ctx, view)


SCREENS = {
    "overview": overview, "me": me, "person": person, "nodes": nodes, "queue": queue,
    "people": people, "roles": roles, "timeline": timeline, "job": job, "jobs": jobs_list,
    "help": help_screen, "range": range_detail, "runtype": runtype,
}

FOOTERS = {
    "overview": [("↵", "open")],
    "range": [("←→", "range"), ("g", "GPU/CPU"), ("v", "prev")],
    "me": [("g", "GPUs/name/outcome"), ("E", "efficiency")],
    "person": [("g", "GPUs/name/outcome"), ("E", "efficiency")],
    "runtype": [("j", "jobs"), ("←→", "range")],
    "roles": [("g", "group")],
    "timeline": [("↵", "lane chart"), ("g", "lanes"), ("c", "GPU/CPU")],
    "job": [("l e b u", "fetch")],
}
