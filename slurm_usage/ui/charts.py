"""Small charts as styled segments: sparklines, gauges, heat cells, deltas, pips.

Every function returns ``[(text, style), ...]`` exactly ``size`` cells wide (or the
natural width where there is no size), so a table cell or a card can hold one. With
``color`` off the glyphs carry the meaning alone (shades for intensity); with
``ascii_only`` only ASCII is drawn.
"""

from typing import List, Optional, Sequence, Tuple

from . import fmt
from .canvas import BAD, DIM, FAINT, GOOD, PLAIN, WARN

Segment = Tuple[str, str]
EIGHTHS = "▁▂▃▄▅▆▇█"
PARTIAL = "▏▎▍▌▋▊▉"
HEAT = ("heat1", "heat2", "heat3", "heat4", "heat5")
SHADES = "░▒▓█"
ASCII_SPARKS = "._-~=+*#"  # calmer than fmt's: no @ walls
ASCII_SHADES = ".:+#"


def heat_style(fraction: float) -> str:
    """One of the five ramp steps for 0..1 (anything above 0 is at least the first)."""
    if fraction <= 0:
        return FAINT
    return HEAT[min(4, int(fraction * 5 - 1e-9))]


def ramp_style(fraction: Optional[float]) -> str:
    """good below 70%, warn below 90%, bad from 90% (for how full something is)."""
    if fraction is None:
        return DIM
    return GOOD if fraction < 0.7 else WARN if fraction < 0.9 else BAD


def _join(cells: List[Segment]) -> List[Segment]:
    """Merge neighbours of one style."""
    out = []  # type: List[List[str]]
    for text, style in cells:
        if out and out[-1][1] == style:
            out[-1][0] += text
        else:
            out.append([text, style])
    return [(text, style) for text, style in out]


def spark(values: Optional[Sequence[float]], size: int, ascii_only: bool = False,
          color: bool = True, style: Optional[str] = None) -> List[Segment]:
    """A one-row sparkline scaled to its own peak, right-aligned in ``size`` cells.
    Each bar is shaded on the ramp by its height; a zero is left blank."""
    if size <= 0:
        return []
    if values is None:
        return [(fmt.fit(fmt.NA, size, ">"), FAINT)]
    data = fmt.resample(list(values), size)
    if not data:
        return [(fmt.fit(fmt.NOT_APPLICABLE, size, ">"), FAINT)]
    marks = ASCII_SPARKS if ascii_only else EIGHTHS
    top = max(data)
    cells = [(" ", PLAIN)] * (size - len(data))  # type: List[Segment]
    for value in data:
        if value <= 0 or top <= 0:
            cells.append((" ", PLAIN))  # nothing allocated: left blank
            continue
        level = min(len(marks) - 1, int(value / top * (len(marks) - 1) + 0.5))
        # the middle of the ramp: the palest step is kept for the heat views
        bar_style = style or (HEAT[1 + min(2, level * 3 // len(marks))] if color else PLAIN)
        cells.append((marks[level], bar_style))
    return _join(cells)


# braille dots, bottom up: the left column, then the right
_LEFT = (0x40, 0x04, 0x02, 0x01)
_RIGHT = (0x80, 0x20, 0x10, 0x08)


def braille(values: Optional[Sequence[float]], size: int, ascii_only: bool = False,
            color: bool = True, style: Optional[str] = None) -> List[Segment]:
    """A finer sparkline: two samples per cell, four dot rows, filled from the bottom
    (eighth blocks in ASCII mode). Zeros are left blank."""
    if ascii_only or size <= 0:
        return spark(values, size, ascii_only, color, style)
    if values is None:
        return [(fmt.fit(fmt.NA, size, ">"), FAINT)]
    data = fmt.resample(list(values), size * 2)
    if not data:
        return [(fmt.fit(fmt.NOT_APPLICABLE, size, ">"), FAINT)]
    if len(data) % 2:
        data = [0.0] + data
    top = max(data)

    def level(value: float) -> int:
        if value <= 0 or top <= 0:
            return 0
        return max(1, min(4, int(value / top * 4 + 0.5)))

    cells = [(" ", PLAIN)] * (size - len(data) // 2)  # type: List[Segment]
    for at in range(0, len(data), 2):
        left, right = level(data[at]), level(data[at + 1])
        if not left and not right:
            cells.append((" ", PLAIN))
            continue
        bits = 0
        for row in range(left):
            bits |= _LEFT[row]
        for row in range(right):
            bits |= _RIGHT[row]
        tone = style or (HEAT[min(4, max(left, right))] if color else PLAIN)
        cells.append((chr(0x2800 + bits), tone))
    return _join(cells)


def gauge(fraction: Optional[float], size: int, ascii_only: bool = False,
          style: Optional[str] = None) -> List[Segment]:
    """A bar with eighth-cell precision on a faint track; colored by ``ramp_style``
    unless a style is given."""
    if size <= 0:
        return []
    if fraction is None:
        return [("░" * size if not ascii_only else "." * size, FAINT)]
    fraction = max(0.0, min(1.0, fraction))
    fill = style or ramp_style(fraction)
    if ascii_only:
        full = int(round(fraction * size))
        return _join([("#" * full, fill), ("." * (size - full), FAINT)])
    eighths = int(round(fraction * size * 8))
    full, part = divmod(eighths, 8)
    out = [("█" * full, fill)]
    used = full
    if part and used < size:
        out.append((PARTIAL[part - 1], fill))
        used += 1
    out.append(("░" * (size - used), FAINT))
    return [s for s in out if s[0]]


def stacked(parts: Sequence[Tuple[float, str]], size: int,
            ascii_only: bool = False, color: bool = True) -> List[Segment]:
    """Several shares of one whole side by side ((fraction, style), ...), rounded so
    a share above zero gets at least one cell."""
    if size <= 0:
        return []
    cells = []
    left = size
    for number, (fraction, style) in enumerate(parts):
        count = int(round(max(0.0, fraction) * size))
        if fraction > 0:
            count = max(1, count)
        count = min(left, count)
        # without color the parts differ by glyph: solid, then shaded
        glyph = ("#=*"[number % 3] if ascii_only else
                 "█" if color or number == 0 else "▒")
        cells.append(glyph * count)
        left -= count
    out = [(text, style) for text, (_, style) in zip(cells, parts) if text]
    if left:
        out.append(("." * left if ascii_only else "░" * left, FAINT))
    return out


def pips(used: Optional[float], total: Optional[float], most: int = 8,
         ascii_only: bool = False) -> List[Segment]:
    """■■■□□ for small counts (a node's GPUs); scaled when there are more than ``most``."""
    if not total or used is None:
        return []
    count = int(min(total, most))
    filled = int(round(used / total * count)) if total else 0
    if used and not filled:
        filled = 1
    on, off = ("#", ".") if ascii_only else ("■", "□")
    return _join([(on * filled, GOOD), (off * (count - filled), FAINT)])


def delta(now: Optional[float], before: Optional[float], loaded: bool = True) -> List[Segment]:
    """▲12% / ▼9%: the arrow colored, the figure plain; placeholders faint."""
    if not loaded:
        return [(fmt.NOT_YET, FAINT)]
    text = fmt.delta(now, before)
    if text in fmt.PLACEHOLDERS:
        return [(text, FAINT)]
    if text.startswith("+"):
        return [("▲", GOOD), (text[1:], PLAIN)]
    if text.startswith("-"):
        return [("▼", WARN), (text[1:], PLAIN)]
    return [(text, DIM)]


def heat_cells(fractions: Sequence[float], ascii_only: bool = False, color: bool = True,
               glyph: str = "▆") -> List[Segment]:
    """One cell per fraction (0..1): the glyph on the ramp with color, shades without;
    empty where zero."""
    cells = []
    for fraction in fractions:
        if fraction <= 0:
            cells.append((" ", PLAIN))
        elif ascii_only:
            shades = ASCII_SHADES
            cells.append((shades[min(3, int(fraction * 4 - 1e-9))], heat_style(fraction)
                          if color else PLAIN))
        elif color:
            cells.append((glyph, heat_style(fraction)))
        else:
            cells.append((SHADES[min(3, int(fraction * 4 - 1e-9))], PLAIN))
    return _join(cells)


def legend(ascii_only: bool = False, color: bool = True, glyph: str = "▆") -> List[Segment]:
    """The ramp from low to full, for a pane's border."""
    steps = (0.1, 0.3, 0.5, 0.7, 0.95)
    return [("low ", DIM)] + heat_cells(steps, ascii_only, color, glyph) + [(" full", DIM)]


def width(parts: Sequence[Segment]) -> int:
    return sum(fmt.width(text) for text, _ in parts)
