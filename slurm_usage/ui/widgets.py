"""Panes of lines, tables that fit the width, and stacking panes vertically."""

import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from . import fmt
from . import theme
from .canvas import ACCENT, DIM, FAINT, HEAD, KEY, PLAIN, SEL, Canvas, Segment, bold, on


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
                 select_width: Optional[int] = None, group: Optional[str] = None,
                 tab: Optional[str] = None, draw: Optional[Callable[..., Any]] = None,
                 fixed: int = 0, frame: bool = True, click_opens: bool = False,
                 range_cursor: bool = False, footer: Optional[Line] = None) -> None:
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
        # panes of one group share a box: one is shown, the others are tabs in its title
        self.group = group
        self.tab = tab  # the label among the group's tabs (default: the title)
        # a pane that draws itself: draw(canvas, top, rows, width, state, focused, lines)
        # returns hits like draw_pane; ``fixed`` is the rows it wants
        self.draw = draw
        self.fixed = fixed
        self.frame = frame  # False: no box (the overview's cards)
        self.click_opens = click_opens  # one click opens a row (tiles are buttons)
        self.range_cursor = range_cursor  # the cursor is the range: arrows change it
        # a row under the lines that never scrolls or takes the cursor (a legend)
        self.footer = footer


class PaneState:
    __slots__ = ("cursor", "scroll", "filter", "sort", "descending", "follow", "mark",
                 "chosen")

    def __init__(self) -> None:
        self.cursor = 0
        self.scroll = 0
        self.filter = ""
        self.sort = -1  # index into Pane.sorts; -1 = natural order
        self.descending = True
        self.follow = None  # type: Optional[bool]  # None until the pane is first drawn
        # (cursor, key) of the selected row when last drawn: a refresh that reorders the
        # rows keeps the same row selected
        self.mark = None  # type: Optional[Tuple[int, Any]]
        # the viewer picked the row (a key, a click, the wheel): only then does a reorder
        # carry the cursor along; else it stays on the first row as the rows load
        self.chosen = False


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
    # width no flex column may take (every one has a limit) spreads over the fixed
    # columns: the data gets the room, not a gap at the right
    # (a little each; the rest as padding, so the figures stay together)
    fixed = [i for i in chosen[1:] if not columns[i].flex]
    for i in fixed:
        give = min(2, spare)
        extra[i] = extra.get(i, 0) + give
        spare -= give
    if spare and chosen:  # past its limit the last flex column pads (its content keeps
        # to the limit and its alignment), else the first column does
        target = flex[-1] if flex else chosen[0]
        extra[target] = extra.get(target, 0) + spare
    out = []
    for i in chosen:
        out.append((i, columns[i], columns[i].width + extra.get(i, 0)))
    if out and needed(chosen) > space:  # a single column wider than the screen
        index, column, size = out[-1]
        out[-1] = (index, column, max(1, size - (needed(chosen) - space)))
    return out


def _blank(cell: Any) -> bool:
    """A cell that only says "not known" (a placeholder, or nothing)."""
    if callable(cell):
        return False
    if isinstance(cell, list):
        text = "".join(t for t, _ in cell).strip()
    else:
        text = (cell[0] if isinstance(cell, tuple) else str(cell)).strip()
    return text in fmt.PLACEHOLDERS or text == ""


class Table:
    """Columns fitted to the width. ``prune``: column names left out when no row has a
    value in them (efficiency before it is loaded): their rows are kept until the
    header is asked for, and then laid out. A cell is text, (text, style), styled parts,
    or a function of the column's width returning parts (a chart)."""

    def __init__(self, columns: Sequence[Column], space: int,
                 prune: Sequence[str] = ()) -> None:
        self.columns = list(columns)
        self.space = space
        self.layout = fit_columns(self.columns, space)
        self.prune = tuple(prune)
        self._pending = []  # type: List[Tuple[Line, Sequence[Any]]]

    def _finish(self) -> None:
        if not self._pending and not self.prune:
            return
        pending, self._pending = self._pending, []
        empty = [index for index, column in enumerate(self.columns)
                 if column.name in self.prune and pending
                 and all(_blank(cells[index]) for _, cells in pending if index < len(cells))]
        if empty:
            keep = [i for i in range(len(self.columns)) if i not in empty]
            self.layout = [(i, c, size) for i, c, size in fit_columns(
                [self.columns[i] for i in keep], self.space)]
            self.layout = [(keep[i], c, size) for i, c, size in self.layout]
        self.prune = ()
        for line, cells in pending:
            line.parts = self._parts(cells)
            line.search = "".join(text for text, _ in line.parts).lower()

    def header(self) -> Line:
        self._finish()
        parts = []
        for number, (_, column, size) in enumerate(self.layout):
            if number:
                parts.append((" ", DIM))
            parts.append((fmt.fit(column.name, size, column.align), DIM))
        return Line(parts)

    def has(self, name: str) -> bool:
        """Whether a column made it into the layout."""
        return any(column.name == name for _, column, _ in self.layout)

    def size(self, name: str, default: int = 0) -> int:
        return next((size for _, column, size in self.layout if column.name == name), default)

    def row(self, cells: Sequence[Any], action: Optional[Target] = None,
            sort: Optional[Dict[str, Any]] = None, key: Optional[str] = None) -> Line:
        if self.prune:
            line = Line([], action, sort, key)
            self._pending.append((line, list(cells)))
            return line
        return Line(self._parts(cells), action, sort, key)

    def _parts(self, cells: Sequence[Any]) -> List[Segment]:
        parts = []  # type: List[Segment]
        for number, (index, column, size) in enumerate(self.layout):
            if number:
                parts.append((" ", PLAIN))
            cell = cells[index] if index < len(cells) else ""
            if callable(cell):
                cell = cell(size)
            if isinstance(cell, list):
                parts.extend(fit_parts(cell, size, column.align))
                continue
            text, style = cell if isinstance(cell, tuple) else (cell, PLAIN)
            if text in fmt.PLACEHOLDERS and style in (PLAIN, DIM):
                style = FAINT
            if style == PLAIN and column.align == ">":
                parts.extend(fit_parts(number_parts(text), size, ">"))
                continue
            parts.append((fmt.fit(text, size, column.align), style))
        return parts


_NUMBER = re.compile(r"^(≥?)([+-]?[0-9][0-9.,]*)([kMGTKB%]?)$")


def number_parts(text: str) -> List[Segment]:
    """'≥72.5k' -> ≥ and k muted, the figure plain: units recede, the number reads first."""
    found = _NUMBER.match(text)
    if not found:
        return [(text, PLAIN)]
    mark, figure, unit = found.groups()
    return [(t, style) for t, style in ((mark, DIM), (figure, PLAIN), (unit, DIM)) if t]


def fit_parts(parts: Sequence[Segment], size: int, align: str = "<") -> List[Segment]:
    """Styled parts cut (with …) or padded to exactly ``size`` cells."""
    total = sum(fmt.width(t) for t, _ in parts)
    if total > size:
        out, used = [], 0
        for number, (text, style) in enumerate(parts):
            step = fmt.width(text)
            last = number == len(parts) - 1
            if used + step < size or (last and used + step == size):
                out.append((text, style))
                used += step
                continue
            out.append((fmt.fit(text, size - used), style))
            break
        return out
    pad = " " * (size - total)
    if align == ">":
        return [(pad, PLAIN)] + list(parts) if pad else list(parts)
    return list(parts) + [(pad, PLAIN)] if pad else list(parts)


# -- layout ------------------------------------------------------------------

def _chrome(pane: Pane) -> int:
    if not pane.frame:
        return 0
    return 1 + (1 if pane.header is not None else 0) + (1 if pane.footer is not None else 0)


def chrome(pane: Pane) -> int:
    return _chrome(pane)


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
    # what is left stays below the panes: a box is never drawn around empty rows
    return body


def title_bar(canvas: Canvas, row: int, left: List[Segment], right: List[Segment],
              glyphs: theme.Box, edge: str, focused: bool,
              hits: Optional[List[Tuple[int, int, Any]]] = None) -> None:
    """``╭─ Title ───── note ─╮`` (``├`` ``┤`` when joining the pane above).
    ``left`` parts may carry a hit tag as a third item (a tab label)."""
    width = canvas.width
    border = ACCENT if focused else FAINT
    first = glyphs.tl if edge == "top" else glyphs.lt
    last = glyphs.tr if edge == "top" else glyphs.rt
    canvas.put(row, 0, first + glyphs.h, border)
    col = 3
    room = width - 3 - 3  # one rule cell and the corner, at least one rule cell between
    right_width = sum(fmt.width(t) for t, _ in right)
    left_room = max(0, room - (right_width + 2 if right else 0))
    for part in left:
        text, style = part[0], part[1]
        if col - 3 >= left_room:
            break
        end = canvas.put(row, col, fmt.fit(text, left_room - (col - 3))
                         if fmt.width(text) > left_room - (col - 3) else text, style)
        if hits is not None and len(part) > 2:
            hits.append((col, end, part[2]))
        col = end
    canvas.put(row, col, " ", PLAIN)
    col += 1
    rule_end = width - 2 - (right_width + 2 if right else 0)
    if col < rule_end:
        canvas.put(row, col, glyphs.h * (rule_end - col), border)
    if right:
        at = max(col, rule_end) + 1
        for text, style in right:
            at = canvas.put(row, at, text, style, limit=width - 3 - at + 1)
        canvas.put(row, at, " ", PLAIN)
    canvas.put(row, width - 2, glyphs.h, border)
    canvas.put(row, width - 1, last, border)


def notes(pane: Pane, state: PaneState, lines: List[Line], height: int,
          room: int) -> List[Segment]:
    """The right of a pane's title: filter, sort, position, then the pane's status (its
    least important parts dropped first when short). A status is a string, a sequence of
    strings by priority, or of styled part lists."""
    info = []  # type: List[List[Segment]]
    if state.filter:
        info.append([("/" + state.filter, ACCENT)])
    if 0 <= state.sort < len(pane.sorts):
        info.append([("sort " + pane.sorts[state.sort] + ("▼" if state.descending else "▲"),
                      DIM)])
    if lines and height > _chrome(pane) and pane.focusable and len(lines) > 1:
        info.append([("%d/%d" % (min(state.cursor + 1, len(lines)), len(lines)), FAINT)])
    status = [pane.status] if isinstance(pane.status, str) else list(pane.status or ())
    items = info + [[(part, DIM)] if isinstance(part, str) else list(part)
                    for part in status if part]

    def size(chosen: List[List[Segment]]) -> int:
        return sum(fmt.width(t) for item in chosen for t, _ in item) + 2 * max(0, len(chosen) - 1)

    while len(items) > len(info) + 1 and size(items) > room:
        items.pop()  # the least important note goes first
    out = []  # type: List[Segment]
    for number, item in enumerate(items):
        if number:
            out.append(("  ", PLAIN))
        out.extend(item)
    if size(items) > room:
        if room <= 1:
            return []
        return fit_parts(out, room)
    return out


def draw_pane(canvas: Canvas, top: int, height: int, pane: Pane, state: PaneState,
              focused: bool, lines: List[Line], edge: str = "top",
              glyphs: Optional[theme.Box] = None,
              tabs: Optional[Sequence[Tuple[Pane, bool]]] = None,
              tab_hits: Optional[List[Tuple[int, int, Any]]] = None) -> List[Tuple[int, int]]:
    """Draw; returns (screen row, line index) for mouse hit-testing. ``tabs``: the
    panes sharing this box, (pane, shown) in order, listed in the title."""
    glyphs = glyphs or theme.ROUNDED
    width = canvas.width
    if pane.draw is not None and not pane.frame:
        if height <= 0:  # no rows given (a short screen): nothing may draw over others
            return []
        return pane.draw(canvas, top, height, width, state, focused, lines)
    border = ACCENT if focused else FAINT
    title_style = KEY if focused else HEAD
    if tabs and len(tabs) > 1:
        left = []  # type: List[Any]
        for number, (other, shown) in enumerate(tabs):
            if number:
                left.append(("  ", PLAIN))
            label = other.tab or other.title
            left.append((label, title_style if shown else DIM, other.id))
    else:
        left = [(pane.title, title_style)]
    room = max(0, width - 8 - min(sum(fmt.width(p[0]) for p in left), width // 2))
    right = notes(pane, state, lines, height, room)
    title_bar(canvas, top, left, right, glyphs, edge, focused, tab_hits)
    row = top + 1
    inner = width - 4

    def sides(at: int) -> None:
        canvas.put(at, 0, glyphs.v, border)
        canvas.put(at, width - 1, glyphs.v, border)

    if pane.header is not None and height > 1:
        sides(row)
        canvas.segments(row, 2, sort_marked(pane, state), inner)
        canvas.overlay(row, 1, width - 1, "band")
        row += 1
    body = top + height - row
    for at in range(row, top + height):
        sides(at)
    if pane.footer is not None and body > 1:  # the last row: fixed, under the lines
        canvas.segments(top + height - 1, 2, pane.footer.parts, inner)
        body -= 1
    hits = []  # type: List[Tuple[int, int]]
    if body <= 0:
        return hits
    if pane.draw is not None:
        return pane.draw(canvas, row, body, width, state, focused, lines)
    if not lines:
        canvas.put(row, 2, "(nothing to show)" if not state.filter else "(no match)", FAINT)
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
        canvas.segments(row + offset, 2, line.parts, inner)
        if selected:
            end = width - 1 if pane.select_width is None else min(width - 1,
                                                                 pane.select_width + 2)
            canvas.overlay(row + offset, 1, end, SEL)
            canvas.put(row + offset, 1, "›" if line.action else " ", on(bold(ACCENT), SEL))
        hits.append((row + offset, index))
    if len(lines) > body:  # a scroll hint in the right padding
        canvas.put(row, width - 2, "▲" if state.scroll else " ", FAINT)
        canvas.put(row + body - 1, width - 2,
                   "▼" if state.scroll + body < len(lines) else " ", FAINT)
    return hits


def close_box(canvas: Canvas, row: int, glyphs: theme.Box, focused: bool) -> None:
    border = ACCENT if focused else FAINT
    canvas.put(row, 0, glyphs.bl + glyphs.h * max(0, canvas.width - 2) + glyphs.br, border)


# sort names that label a column with another header
_SORT_ALIASES = {"name": ("account", "node", "user", "run type", "job"),
                 "attempts": ("att",), "gpus free": ("gpu", "gpus"), "cpus free": ("cpu", "cpus"),
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
