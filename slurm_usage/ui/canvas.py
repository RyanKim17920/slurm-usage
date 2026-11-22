"""An in-memory screen. Screens draw here; curses (or a test) reads it back.

Every string is passed through ``sanitize.clean`` on the way in, so nothing a
user controls can reach the terminal as an escape sequence.
"""

from typing import Callable, List, Optional, Tuple

from ..sanitize import clean
from .fmt import char_width

# Styles by meaning, not by color. The curses backend maps them to attributes.
PLAIN = ""
RESOURCE = "resource"    # cyan: a node, partition, account, QOS, GPU type
USER = "user"            # magenta: a person
GOOD = "good"            # green: productive
WARN = "warn"            # amber: partial, waiting
BAD = "bad"              # red: needs attention
DIM = "dim"              # gray: context, unavailable
HEAD = "head"            # bold: titles and column headers
SELECTED = "selected"    # reverse: the cursor row
TAB = "tab"              # the active tab
STYLES = (PLAIN, RESOURCE, USER, GOOD, WARN, BAD, DIM, HEAD, SELECTED, TAB)

Segment = Tuple[str, str]  # (text, style)
_WIDE_TAIL = ""  # placeholder cell after a double-width character


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = max(0, width), max(0, height)
        self.cells = [[(" ", PLAIN)] * self.width for _ in range(self.height)]

    def put(self, row: int, col: int, text: object, style: str = PLAIN,
            limit: Optional[int] = None) -> int:
        """Write at (row, col), clipped; returns the column after the text."""
        if not 0 <= row < self.height:
            return col
        end = self.width if limit is None else min(self.width, col + max(0, limit))
        line = self.cells[row]
        for char in clean(text):
            step = char_width(char)
            if step == 0:
                continue
            if col + step > end:
                break
            if col >= 0:
                line[col] = (char, style)
                if step == 2:
                    line[col + 1] = (_WIDE_TAIL, style)
            col += step
        return col

    def segments(self, row: int, col: int, parts: List[Segment],
                 limit: Optional[int] = None, fill: Optional[str] = None) -> int:
        end = self.width if limit is None else min(self.width, col + limit)
        start = col
        for text, style in parts:
            if col >= end:
                break
            col = self.put(row, col, text, style, end - col)
        if fill is not None:
            for at in range(start, end):
                char, style = self.cells[row][at] if 0 <= row < self.height else (" ", "")
                if 0 <= row < self.height:
                    self.cells[row][at] = (char, fill if style in (PLAIN,) else style)
        return col

    def restyle(self, row: int, start: int, end: int, style: str) -> None:
        if not 0 <= row < self.height:
            return
        for at in range(max(0, start), min(self.width, end)):
            self.cells[row][at] = (self.cells[row][at][0], style)

    def map_cells(self, fn: Callable[[str], str]) -> None:
        """Replace every cell's character (each result must be one cell wide)."""
        for line in self.cells:
            for at, (char, style) in enumerate(line):
                line[at] = (fn(char), style)

    def text(self) -> str:
        """The screen as plain text, one line per row (trailing spaces kept)."""
        return "\n".join("".join(char for char, _ in line) for line in self.cells)

    def row_text(self, row: int) -> str:
        return "".join(char for char, _ in self.cells[row])

    def runs(self, row: int) -> List[Segment]:
        """Consecutive cells of one style, for a backend that writes strings."""
        out = []  # type: List[List[str]]
        for char, style in self.cells[row]:
            if out and out[-1][1] == style:
                out[-1][0] += char
            else:
                out.append([char, style])
        return [(text, style) for text, style in out]
