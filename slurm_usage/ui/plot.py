"""Large figures and the range chart, as styled segments.

``big`` draws a number three rows tall (half blocks, or seven-segment strokes in
ASCII). ``area_chart`` draws usage over a range: one bar per bin when bins are
wide enough to read one by one (a day by the hour, a month by the day), else a
filled area; the previous equal period as a dashed line over it, the capacity,
the mean and "now" as labelled rows, and a time axis under it. Everything is
plain data in, rows of ``(text, style)`` out: no clock, no curses.
"""

import datetime
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

from . import fmt
from .canvas import ACCENT, BRIGHT, DIM, FAINT, GOOD, PLAIN, WARN, bold
from .charts import EIGHTHS, _join, heat_style

Segment = Tuple[str, str]

# -- big figures -------------------------------------------------------------------

# 3x5 pixel glyphs, drawn with half blocks: three rows of two pixel rows each
_PIXELS = {
    "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"), "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"),
    "/": ("001", "001", "010", "100", "100"), "%": ("101", "001", "010", "100", "101"),
    ".": ("000", "000", "000", "000", "010"), " ": ("0", "0", "0", "0", "0"),
    "k": ("100", "101", "110", "101", "101"), "M": ("101", "111", "111", "101", "101"),
    "-": ("000", "000", "111", "000", "000"), "?": ("111", "001", "011", "000", "010"),
}
# a slash and a percent sign from the 3x5 grid turn into a smear of half blocks: these
# two are drawn with a true diagonal instead, the percent's circles as solid quarters
_DRAWN = {"/": ("  ╱", " ╱ ", "╱  "), "%": ("▀ ╱", " ╱ ", "╱ ▀")}
_HALF = {("1", "1"): "█", ("1", "0"): "▀", ("0", "1"): "▄", ("0", "0"): " "}
# seven segments in ASCII
_SEGMENTS = {
    "0": (" _ ", "| |", "|_|"), "1": ("   ", "  |", "  |"), "2": (" _ ", " _|", "|_ "),
    "3": (" _ ", " _|", " _|"), "4": ("   ", "|_|", "  |"), "5": (" _ ", "|_ ", " _|"),
    "6": (" _ ", "|_ ", "|_|"), "7": (" _ ", "  |", "  |"), "8": (" _ ", "|_|", "|_|"),
    "9": (" _ ", "|_|", " _|"), "/": ("  /", " / ", "/  "), "%": ("o /", " / ", "/ o"),
    ".": ("   ", "   ", " . "), " ": (" ", " ", " "), "k": ("|  ", "|/ ", "|\\ "),
    "M": ("   ", "|v|", "| |"), "-": ("   ", " _ ", "   "), "?": (" _ ", "  )", " . "),
}


def big_rows(text: str, ascii_only: bool = False) -> List[str]:
    """``text`` three rows tall; characters without a glyph are left out."""
    rows = ["", "", ""]
    for number, char in enumerate(text):
        if ascii_only:
            glyph = _SEGMENTS.get(char)
            if glyph is None:
                continue
            cells = list(glyph)
        else:
            drawn = _DRAWN.get(char)
            pixels = _PIXELS.get(char)
            if drawn is not None:
                cells = list(drawn)
                for row in range(3):
                    rows[row] += (" " if number else "") + cells[row]
                continue
            if pixels is None:
                continue
            pairs = ((pixels[0], pixels[1]), (pixels[2], pixels[3]),
                     (pixels[4], "0" * len(pixels[4])))
            cells = ["".join(_HALF[(a, b)] for a, b in zip(top, low)) for top, low in pairs]
        for row in range(3):
            rows[row] += (" " if number and char not in " " else "") + cells[row]
    return rows


def big_width(text: str, ascii_only: bool = False) -> int:
    return fmt.width(big_rows(text, ascii_only)[0])


# -- scales ---------------------------------------------------------------------------

def nice_step(peak: float, ticks: int = 4) -> float:
    """A round step that splits ``peak`` into at most ``ticks`` steps."""
    if peak <= 0:
        return 1.0
    raw = peak / max(1, ticks)
    power = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        if mult * power >= raw:
            step = mult * power
            return step if step >= 1 or peak < 4 else 1.0
    return 10 * power


def nice_top(peak: float, ticks: int = 4) -> Tuple[float, float]:
    """(step, top): the top a round multiple of the step at or above ``peak``."""
    step = nice_step(peak, ticks)
    top = step * max(1, math.ceil(peak / step - 1e-9))
    return step, top


# -- the time axis ---------------------------------------------------------------------

def _local(t: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(t)


def _midnights(start: int, end: int, every: int = 1) -> List[int]:
    out = []
    day = _local(start).date()
    while True:
        day += datetime.timedelta(days=1)
        at = int(time.mktime((day.year, day.month, day.day, 0, 0, 0, 0, 1, -1)))
        if at >= end:
            return out
        if day.toordinal() % every == 0:
            out.append(at)


def _months(start: int, end: int, every: int = 1) -> List[int]:
    out = []
    when = _local(start)
    year, month = when.year, when.month
    while True:
        month += 1
        if month > 12:
            year, month = year + 1, 1
        at = int(time.mktime((year, month, 1, 0, 0, 0, 0, 1, -1)))
        if at >= end:
            return out
        if (month - 1) % every == 0:
            out.append(at)


def _hours(start: int, end: int, every: int) -> List[int]:
    out = []
    at = start - start % 3600
    while True:
        at += 3600
        if at >= end:
            return out
        if _local(at).hour % every == 0:
            out.append(at)


def _hour_label(t: int) -> str:
    when = _local(t)
    return when.strftime("%a %d") if when.hour == 0 else when.strftime("%H:%M")


def label_levels(start: int, end: int, unit: str) -> List[List[Tuple[int, str]]]:
    """Candidate axis labels, finest first; the chart uses the finest level that fits."""
    span = end - start
    levels = []  # type: List[List[Tuple[int, str]]]
    if span <= 2 * 86400:
        for every in (2, 3, 4, 6, 12):
            levels.append([(t, _hour_label(t)) for t in _hours(start, end, every)])
    if span <= 10 * 86400:
        for every in (1, 2):
            levels.append([(t, _local(t).strftime("%a %d")) for t in _midnights(start, end, every)])
    if 2 * 86400 < span <= 75 * 86400:
        for every in (1, 2, 3, 7, 14):
            levels.append([(t, _local(t).strftime("%b %d")) for t in _midnights(start, end, every)])
    if span > 40 * 86400:
        def month_label(t: int, year_too: bool) -> str:
            when = _local(t)
            if when.month == 1:
                return when.strftime("%Y")
            return when.strftime("%b %Y" if year_too else "%b")
        long = span > 400 * 86400
        for every in (1, 2, 3, 6, 12):
            levels.append([(t, month_label(t, long and every < 12))
                           for t in _months(start, end, every)])
    return [level for level in levels if level] or [[]]


def place_labels(cols: int, at: Sequence[Tuple[float, str]], gap: int = 2
                 ) -> Optional[List[Tuple[int, str]]]:
    """Labels centred on their columns, clamped inside; None if any two would touch."""
    placed = []  # type: List[Tuple[int, str]]
    last_end = -gap
    for col, text in at:
        size = fmt.width(text)
        begin = int(round(col - size / 2.0))
        begin = max(0, min(cols - size, begin))
        if begin < last_end + gap or size > cols:
            return None
        placed.append((begin, text))
        last_end = begin + size
    return placed


def time_axis(cols: int, start: int, end: int, unit: str, now: Optional[float] = None,
              right: Optional[str] = None) -> str:
    """The axis under a chart of [start, end) ``cols`` wide: the finest labels that fit,
    and the right end named (``now``, or its date)."""
    span = float(max(1, end - start))
    end_text = right
    if end_text is None:
        end_text = "now" if now is not None and abs(now - end) < 2 * 3600 else \
            _local(end).strftime("%b %d")
    reserve = fmt.width(end_text) + 2
    chosen = []  # type: List[Tuple[int, str]]
    for level in label_levels(start, end, unit):
        marks = [((t - start) / span * cols, text) for t, text in level]
        marks = [(c, t) for c, t in marks if c + fmt.width(t) / 2.0 < cols - reserve]
        placed = place_labels(cols - reserve, marks)
        if placed is not None:
            chosen = placed
            break
    line = [" "] * cols
    for begin, text in chosen:
        for offset, char in enumerate(text):
            if 0 <= begin + offset < cols:
                line[begin + offset] = char
    tail = cols - fmt.width(end_text)
    if tail >= 0:
        line[tail:] = list(end_text)
    return "".join(line)


# -- the chart -----------------------------------------------------------------------

CAP = "┈"
MEAN = "┄"
PREV = "╌"
ASCII_PARTS = "..::++##"
ASCII_CAP, ASCII_MEAN, ASCII_PREV = "=", "~", "-"


class Chart:
    """What area_chart draws; values are in the chart's unit (GPUs, CPUs)."""

    def __init__(self, values: Sequence[float], edges: Sequence[int], unit: str = "day",
                 previous: Optional[Sequence[float]] = None, capacity: Optional[float] = None,
                 mean: Optional[float] = None, now: Optional[float] = None,
                 clock: Optional[float] = None, what: str = "GPUs") -> None:
        self.values = list(values)
        self.edges = list(edges)
        self.unit = unit
        self.previous = list(previous) if previous is not None else None
        self.capacity = capacity
        self.mean = mean
        self.now = now
        self.clock = clock
        self.what = what


def _columns(values: Sequence[float], edges: Sequence[int], cols: int
             ) -> Tuple[List[Optional[float]], bool, List[int]]:
    """Per column: the value (None for a gap between bars), whether drawn as bars, and
    which bin each column shows (-1 for gaps)."""
    start, end = edges[0], edges[-1]
    span = float(max(1, end - start))
    count = len(values)
    out = [None] * cols  # type: List[Optional[float]]
    owner = [-1] * cols
    if not count or cols <= 0:
        return out, False, owner
    bars = cols / float(count) >= 3
    if bars:
        for index, value in enumerate(values):
            left = int(round((edges[index] - start) / span * cols))
            right = int(round((edges[index + 1] - start) / span * cols))
            for col in range(left, max(left + 1, right - 1)):  # the last column is the gap
                if 0 <= col < cols:
                    out[col] = value
                    owner[col] = index
        return out, True, owner
    for col in range(cols):
        low = start + span * col / cols
        high = start + span * (col + 1) / cols
        total = weight = 0.0
        for index in range(count):
            a, b = max(low, edges[index]), min(high, edges[index + 1])
            if b > a:
                total += values[index] * (b - a)
                weight += b - a
                if owner[col] < 0 or b - a > (high - low) / 2:
                    owner[col] = index
        out[col] = total / weight if weight else 0.0
    return out, False, owner


def _line(values: Sequence[float], edges: Sequence[int], cols: int) -> List[Optional[float]]:
    """A series as a line: every column, time-weighted (never gapped)."""
    if not values:
        return [None] * cols
    start, end = edges[0], edges[-1]
    span = float(max(1, end - start))
    out = []  # type: List[Optional[float]]
    for col in range(cols):
        low = start + span * col / cols
        high = start + span * (col + 1) / cols
        total = weight = 0.0
        for index, value in enumerate(values):
            a, b = max(low, edges[index]), min(high, edges[index + 1])
            if b > a:
                total += value * (b - a)
                weight += b - a
        out.append(total / weight if weight else None)
    return out


def area_chart(chart: Chart, width: int, rows: int, ascii_only: bool = False,
               color: bool = True) -> List[List[Segment]]:
    """``rows`` rows of chart and one of time axis, each ``width`` cells: a y-axis in
    the chart's unit at the left, labels for capacity, the mean and now at the right."""
    values = chart.values
    peak = max(values) if values else 0.0
    prev_peak = max(chart.previous) if chart.previous else 0.0
    highest = max(peak, prev_peak, chart.capacity or 0.0, chart.now or 0.0)
    step, top = nice_top(highest * 1.04 if highest else 1.0, 4 if rows >= 6 else 2)
    labels = [fmt.num(step * k, 1 if step < 1 else 0) for k in range(int(round(top / step)) + 1)]
    gutter = max(fmt.width(t) for t in labels) + 1
    # the right margin names the rows: "capacity 56", "avg 31.6", "now 44"
    names = []  # type: List[Tuple[float, str, str, int]]
    if chart.now is not None:
        names.append((chart.now, "now " + _figure(chart.now), bold(GOOD), 0))
    if chart.capacity:
        names.append((chart.capacity, "capacity " + _figure(chart.capacity), DIM, 1))
    if chart.mean is not None and peak > 0:
        names.append((chart.mean, "avg " + _figure(chart.mean), ACCENT, 2))
    margin = max([fmt.width(text) for _, text, _, _ in names] + [0])
    margin = margin + 2 if margin and width - gutter - margin - 2 >= 30 else 0
    cols = max(1, width - gutter - margin)
    rows = max(1, rows)

    def level_of(value: float) -> int:
        """The row (0 = top) that holds ``value``."""
        level = int(value / top * rows - 1e-9) if top > 0 else 0
        return rows - 1 - max(0, min(rows - 1, level))

    filled, bars, owner = _columns(values, chart.edges, cols)
    previous = _line(chart.previous, chart.edges, cols) if chart.previous else [None] * cols
    peak_col = None
    if values and peak > 0:
        best = max(range(len(values)), key=lambda i: (values[i], i))
        inside = [c for c in range(cols) if owner[c] == best]
        peak_col = inside[len(inside) // 2] if inside else None
    grid = [[(" ", PLAIN)] * cols for _ in range(rows)]
    area = set()  # type: set
    parts = ASCII_PARTS if ascii_only else EIGHTHS
    for col, value in enumerate(filled):
        if value is None or value <= 0:
            continue
        height = value / top * rows * 8 if top > 0 else 0
        for row in range(rows):
            level = rows - 1 - row
            amount = height - level * 8
            if amount <= 0:
                continue
            glyph = parts[-1] if amount >= 8 else parts[max(0, min(7, int(round(amount)) - 1))]
            # the body in one tone, its top edge brighter: an area with a line on it
            crest = amount <= 8
            tone = ("heat5" if crest else "heat3") if color else (bold(PLAIN) if crest else PLAIN)
            grid[row][col] = (glyph, tone)
            area.add((row, col))

    def free(row: int, col: int) -> bool:
        return grid[row][col][0] == " "

    lines = {}  # type: Dict[int, Tuple[str, str]]
    if chart.mean is not None and peak > 0:
        lines[level_of(chart.mean)] = (ASCII_MEAN if ascii_only else MEAN, ACCENT)
    if chart.capacity:
        lines[level_of(chart.capacity)] = (ASCII_CAP if ascii_only else CAP, DIM)
    for row, (glyph, style) in lines.items():
        for col in range(cols):
            if free(row, col):
                grid[row][col] = (glyph, style)
    for col, value in enumerate(previous):
        if value is None or value <= 0:
            continue  # nothing then: no line along the floor
        row = level_of(value)
        if (row, col) in area:
            continue  # under the area: this period covers it
        grid[row][col] = (ASCII_PREV if ascii_only else PREV, WARN)
    if peak_col is not None:
        row = level_of(peak) - 1
        if row >= 0:
            label = ("v" if ascii_only else "▾") + " peak " + _figure(peak)
            begin = max(0, min(cols - fmt.width(label), peak_col))
            if begin + fmt.width(label) <= cols:
                for offset, char in enumerate(label):
                    grid[row][begin + offset] = (char, BRIGHT if offset else bold(BRIGHT))
    if chart.now is not None:
        grid[level_of(chart.now)][cols - 1] = ("*" if ascii_only else "●", bold(GOOD))

    # y-axis: round values on the rows that hold them; 0 on the bottom row
    ticks = {}  # type: Dict[int, str]
    for k, text in enumerate(labels):  # on the row that holds the value, as every line is
        row = level_of(step * k) if k else rows - 1
        if row not in ticks:
            ticks[row] = text
    margin_text = {}  # type: Dict[int, Tuple[str, str]]
    if margin:
        for value, text, style, _ in sorted(names, key=lambda n: n[3]):
            want = level_of(value)
            for row in (want, want - 1, want + 1, want - 2, want + 2):
                if 0 <= row < rows and row not in margin_text:
                    margin_text[row] = (text, style)
                    break
    out = []  # type: List[List[Segment]]
    bar_v, bar_t = ("|", "+") if ascii_only else ("│", "┤")
    for row in range(rows):
        tick = ticks.get(row)
        line = [((tick or "").rjust(gutter - 1), DIM), (bar_t if tick else bar_v, FAINT)]
        line += grid[row]
        if margin:
            text, style = margin_text.get(row, ("", PLAIN))
            line.append(("  " + text.ljust(margin - 2), style))
        out.append(_join(line))
    axis = time_axis(cols, chart.edges[0], chart.edges[-1], chart.unit, chart.clock)
    out.append([(" " * gutter, PLAIN), (axis, DIM), (" " * margin, PLAIN)])
    return out


def _figure(value: float) -> str:
    if value < 10 and value != int(value):
        return "%.1f" % value
    return fmt.num(value)


def legend(chart: Chart, range_word: str, ascii_only: bool = False) -> List[Segment]:
    """What the marks mean, for the chart's title row."""
    block = "#" if ascii_only else "█"
    out = [(block, "heat4"), (" " + chart.what + " in use", DIM)]
    if chart.previous is not None:
        out += [("  ", PLAIN), ((ASCII_PREV if ascii_only else PREV) * 2, WARN),
                (" previous " + range_word, DIM)]
    if chart.mean is not None:
        out += [("  ", PLAIN), ((ASCII_MEAN if ascii_only else MEAN) * 2, ACCENT),
                (" average", DIM)]
    if chart.capacity:
        out += [("  ", PLAIN), ((ASCII_CAP if ascii_only else CAP) * 2, DIM),
                (" capacity", DIM)]
    return out
