"""Panes of lines, tables that fit the width, and stacking panes vertically."""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from . import fmt
from .canvas import DIM, HEAD, PLAIN, SELECTED, Canvas, Segment


class Target(NamedTuple):
    """Where Enter goes: a screen kind and its parameters."""

    kind: str
    params: Dict[str, Any]
    title: str


class Line:
    """One row of a pane. ``action`` makes it drillable; ``sort`` holds sort keys."""

    __slots__ = ("parts", "action", "sort", "key", "search", "pinned")

    def __init__(self, parts: Sequence[Segment], action: Optional[Target] = None,
                 sort: Optional[Dict[str, Any]] = None, key: Optional[str] = None,
                 pinned: bool = False) -> None:
        self.parts = list(parts)
        self.action = action
        self.sort = sort or {}
        self.key = key
        self.pinned = pinned  # stays first whatever the sort (the viewer's own row)
        self.search = "".join(text for text, _ in self.parts).lower()

    @property
    def text(self) -> str:
        return "".join(text for text, _ in self.parts)


def text_line(text: str, style: str = PLAIN, action: Optional[Target] = None) -> Line:
    return Line([(text, style)], action)


class Pane:
    def __init__(self, pane_id: str, title: str, lines: Optional[List[Line]] = None,
                 header: Optional[Line] = None, sorts: Sequence[str] = (),
                 grow: bool = True, status: str = "", focusable: bool = True,
                 min_lines: int = 1, max_lines: Optional[int] = None, natural: str = "",
                 follow: bool = False, more: bool = False,
                 select_width: Optional[int] = None) -> None:
        self.id = pane_id
        self.title = title
        self.lines = lines or []
        self.header = header
        self.sorts = tuple(sorts)
        self.grow = grow
        # short note at the right of the title; a sequence is parts by priority, the
        # last dropped first when the title row is short
        self.status = status
        self.focusable = focusable
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.natural = natural  # the sort name the unsorted order already follows (▼ shown)
        self.follow = follow  # opens at the last line and stays there until moved (a log tail)
        self.more = more  # only the first rows are loaded; reaching the end loads more
        # the selected row is marked over this many leading columns only (None = all), so
        # what the rest of the row draws (a timeline lane's bars) stays visible
        self.select_width = select_width


class PaneState:
    __slots__ = ("cursor", "scroll", "filter", "sort", "descending", "follow")

    def __init__(self) -> None:
        self.cursor = 0
        self.scroll = 0
        self.filter = ""
        self.sort = -1  # index into Pane.sorts; -1 = natural order
        self.descending = True
        self.follow = None  # type: Optional[bool]  # None until the pane is first drawn


def visible_lines(pane: Pane, state: PaneState) -> List[Line]:
    lines = pane.lines
    if state.filter:
        needle = state.filter.lower()
        lines = [line for line in lines if needle in line.search]
    if 0 <= state.sort < len(pane.sorts):
        name = pane.sorts[state.sort]
        known = [line for line in lines if line.sort.get(name) is not None]
        unknown = [line for line in lines if line.sort.get(name) is None]
        known.sort(key=lambda line: line.sort[name], reverse=state.descending)
        lines = known + unknown
    if any(line.pinned for line in lines):
        lines = [line for line in lines if line.pinned] + [l for l in lines if not l.pinned]
    return lines


# -- tables ------------------------------------------------------------------

class Column(NamedTuple):
    name: str
    width: int
    align: str = "<"  # < or >
    priority: int = 0  # higher survives longer when space is short
    flex: bool = False  # takes the spare width
    most: int = 0  # a flex column grows to at most this width (0 = no limit)


Cell = Tuple[str, str]


def fit_columns(columns: Sequence[Column], space: int) -> List[Tuple[int, Column, int]]:
    """(index, column, width) of the columns that fit, in their original order."""
    chosen = list(range(len(columns)))

    def needed(indices: List[int]) -> int:
        return sum(columns[i].width for i in indices) + max(0, len(indices) - 1)

    while len(chosen) > 1 and needed(chosen) > space:
        drop = min(chosen, key=lambda i: (columns[i].priority, -i))
        chosen.remove(drop)
    spare = max(0, space - needed(chosen))
    flex = [i for i in chosen if columns[i].flex]
    extra = {i: 0 for i in flex}
    # flex columns grow in order up to their limit; an unlimited one takes what is left,
    # so one long name column cannot take the whole width from the others
    for i in flex:
        if columns[i].most:
            extra[i] = min(spare, max(0, columns[i].most - columns[i].width))
            spare -= extra[i]
    unlimited = [i for i in flex if not columns[i].most]
    for number, i in enumerate(unlimited):
        give = spare // (len(unlimited) - number)
        extra[i] += give
        spare -= give
    out = []
    for i in chosen:
        out.append((i, columns[i], columns[i].width + extra.get(i, 0)))
    if out and needed(chosen) > space:  # a single column wider than the screen
        index, column, size = out[-1]
        out[-1] = (index, column, max(1, size - (needed(chosen) - space)))
    return out


class Table:
    def __init__(self, columns: Sequence[Column], space: int) -> None:
        self.columns = list(columns)
        self.layout = fit_columns(self.columns, space)

    def header(self) -> Line:
        parts = []
        for number, (_, column, size) in enumerate(self.layout):
            if number:
                parts.append((" ", HEAD))
            parts.append((fmt.fit(column.name, size, column.align), HEAD))
        return Line(parts)

    def row(self, cells: Sequence[Any], action: Optional[Target] = None,
            sort: Optional[Dict[str, Any]] = None, key: Optional[str] = None) -> Line:
        parts = []
        for number, (index, column, size) in enumerate(self.layout):
            if number:
                parts.append((" ", PLAIN))
            cell = cells[index] if index < len(cells) else ""
            text, style = cell if isinstance(cell, tuple) else (cell, PLAIN)
            if text in fmt.PLACEHOLDERS and style == PLAIN:
                style = DIM
            parts.append((fmt.fit(text, size, column.align), style))
        return Line(parts, action, sort, key)


# -- layout ------------------------------------------------------------------

def _chrome(pane: Pane) -> int:
    return 1 + (1 if pane.header is not None else 0)


def allocate(panes: List[Pane], counts: List[int], height: int, focus: int) -> List[int]:
    """Rows of body per pane (0 = title only, -1 = hidden)."""
    n = len(panes)
    want = []
    for pane, count in zip(panes, counts):
        cap = count if pane.max_lines is None else min(count, pane.max_lines)
        want.append(max(cap, 1))
    small = [min(want[i], max(1, panes[i].min_lines)) for i in range(n)]
    body = [0] * n
    used = sum(_chrome(p) for p in panes)
    if used > height:  # not even every title fits: keep the focused pane and neighbours
        shown = [-1] * n
        order = sorted(range(n), key=lambda i: (abs(i - focus), i))
        room = height
        for i in order:
            need = _chrome(panes[i]) + (1 if i == focus else 0)
            if need <= room:
                shown[i] = 1 if i == focus else 0
                room -= need
        if focus < n and shown[focus] == 1:
            shown[focus] += max(0, room)
        return shown
    room = height - used
    for i in sorted(range(n), key=lambda i: (i != focus, i)):
        give = min(small[i], room)
        body[i] += give
        room -= give
    for i in sorted(range(n), key=lambda i: (i != focus, i)):
        give = min(want[i] - body[i], room)
        if give > 0:
            body[i] += give
            room -= give
    if room > 0:
        growers = [i for i in range(n) if panes[i].grow] or [n - 1]
        target = focus if focus in growers else growers[-1]
        body[target] += room
    return body


def draw_pane(canvas: Canvas, top: int, height: int, pane: Pane, state: PaneState,
              focused: bool, lines: List[Line]) -> List[Tuple[int, int]]:
    """Draw; returns (screen row, line index) for mouse hit-testing."""
    width = canvas.width
    title_style = HEAD
    mark = "▌" if focused else " "
    info = []
    if state.filter:
        info.append("/" + state.filter)
    if 0 <= state.sort < len(pane.sorts):
        info.append("sort " + pane.sorts[state.sort] + ("▼" if state.descending else "▲"))
    if lines and height > _chrome(pane) and pane.focusable:
        info.append("%d/%d" % (min(state.cursor + 1, len(lines)), len(lines)))
    status = [pane.status] if isinstance(pane.status, str) else list(pane.status or ())
    status = [part for part in status if part]
    col = canvas.put(top, 0, mark, title_style if focused else DIM)
    col = canvas.put(top, col, pane.title + " ", title_style)
    room = width - col - 2  # one rule cell before the note, one cell spare at the end
    while len(status) > 1 and fmt.width("  ".join(info + status)) > room:
        status.pop()  # the least important part goes first
    right = "  ".join(info + status)
    if fmt.width(right) > room:
        right = fmt.fit(right, max(0, room)).rstrip() if room > 1 else ""
    rule_end = width - fmt.width(right) - (2 if right else 0)
    for at in range(col, max(col, rule_end)):
        canvas.put(top, at, "─", DIM)
    if right:
        canvas.put(top, max(col, rule_end) + 1, right, DIM)
    row = top + 1
    if pane.header is not None and height > 1:
        canvas.segments(row, 1, sort_marked(pane, state), width - 1)
        row += 1
    body = top + height - row
    hits = []
    if body <= 0:
        return hits
    if not lines:
        canvas.put(row, 1, "(nothing to show)" if not state.filter else "(no match)", DIM)
        return hits
    if state.follow is None:
        state.follow = pane.follow
    if state.follow:
        state.cursor = len(lines) - 1
    state.cursor = max(0, min(state.cursor, len(lines) - 1))
    if state.cursor < state.scroll:
        state.scroll = state.cursor
    if state.cursor >= state.scroll + body:
        state.scroll = state.cursor - body + 1
    state.scroll = max(0, min(state.scroll, max(0, len(lines) - body)))
    for offset in range(body):
        index = state.scroll + offset
        if index >= len(lines):
            break
        line = lines[index]
        selected = focused and pane.focusable and index == state.cursor
        canvas.segments(row + offset, 1, line.parts, width - 1)
        if selected:
            canvas.restyle(row + offset, 0, width if pane.select_width is None
                           else min(width, pane.select_width), SELECTED)
            canvas.put(row + offset, 0, "›" if line.action else " ", SELECTED)
        hits.append((row + offset, index))
    if len(lines) > body:  # a scroll hint at the right edge
        canvas.put(row, width - 1, "▲" if state.scroll else " ", DIM)
        canvas.put(row + body - 1, width - 1,
                   "▼" if state.scroll + body < len(lines) else " ", DIM)
    return hits


# sort names that label a column with another header
_SORT_ALIASES = {"name": ("account", "node", "user", "run type", "job"),
                 "attempts": ("att",), "gpus free": ("gpu",), "cpus free": ("cpu",),
                 "time": ("start",), "gpu-h": ("gpu-h in bin",)}


def sort_marked(pane: Pane, state: PaneState) -> List[Segment]:
    """The header with ▼/▲ in the column the rows are sorted by (or the natural order's)."""
    parts = list(pane.header.parts) if pane.header is not None else []
    if 0 <= state.sort < len(pane.sorts):
        name, arrow = pane.sorts[state.sort], "▼" if state.descending else "▲"
    elif pane.natural:
        name, arrow = pane.natural, "▼"
    else:
        return parts
    wanted = (name.lower(),) + _SORT_ALIASES.get(name.lower(), ())
    for number, (text, style) in enumerate(parts):
        core = text.strip()
        if not core or core.lower() not in wanted:
            continue
        size = fmt.width(text)
        marked = core + arrow
        if fmt.width(marked) > size:
            marked = core[:max(0, size - 1)] + arrow
        right = text.startswith(" ") and not text.endswith(" ")
        pad = " " * max(0, size - fmt.width(marked))
        parts[number] = (pad + marked if right else marked + pad, style)
        break
    return parts


def wrap(text: str, size: int, indent: Optional[int] = None) -> List[str]:
    """Greedy word wrap by display width; continuation lines keep ``indent`` spaces
    (default: the text's own leading spaces)."""
    lead = len(text) - len(text.lstrip(" "))
    if indent is None:
        indent = lead
    if fmt.width(text) <= size:
        return [text]
    words = text[lead:].split(" ")
    out, current = [], " " * lead
    first = True
    for word in words:
        candidate = current + (" " if current.strip() else "") + word if current.strip() else current + word
        if fmt.width(candidate) <= size:
            current = candidate
            continue
        if current.strip():
            out.append(current)
        current = " " * min(indent, size // 2) + word
        while fmt.width(current) > size > 0:
            out.append(current[:size])
            current = current[size:]
    if current.strip() or not out:
        out.append(current)
    return out


Builder = Callable[..., List[Pane]]
