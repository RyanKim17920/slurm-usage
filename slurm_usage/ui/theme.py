"""The palette and glyph sets, in one place.

Screens style text by meaning ("good", "user", "dim", ...), never by color. A style
string is ``role[+b][@background]``: ``"good+b"`` is a bold good figure,
``"user@sel"`` a person's name on the selected row. ``resolve`` turns one into a
``Look`` (foreground, background, attributes); the curses backend turns a Look into
attributes for the depth the terminal has:

    truecolor  the RGB values themselves (a terminfo with direct color, COLORS >= 2**24)
    256        the nearest xterm-256 entry, chosen by hand for each color
    16 / 8     the ANSI colors, with bold/dim where the palette has no room
    mono       no color at all (C, NO_COLOR): bold, dim and reverse only

ncurses addresses colors through the terminfo entry, so a terminal that says
COLORTERM=truecolor but runs with TERM=xterm-256color gets the 256-color mapping,
which is chosen to match the RGB palette closely.
"""

import os
from typing import Dict, Mapping, NamedTuple, Optional, Tuple

RGB = Tuple[int, int, int]


class Color(NamedTuple):
    rgb: RGB
    x256: int
    x16: int  # 0-7 normal, 8-15 bright
    x8: int   # 0-7
    dim8: bool = False  # in 8 colors, also dim (gray is white + dim)


# -- the palette ---------------------------------------------------------------
MUTED = Color((138, 145, 155), 245, 8, 7, True)
FAINT = Color((88, 95, 106), 240, 8, 7, True)
ACCENT = Color((95, 175, 255), 75, 12, 4)
RESOURCE = Color((90, 200, 210), 80, 6, 6)
USER = Color((215, 140, 220), 176, 13, 5)
GOOD = Color((130, 210, 140), 114, 2, 2)
WARN = Color((222, 178, 96), 179, 3, 3)
BAD = Color((255, 105, 105), 203, 9, 1)
BRIGHT = Color((235, 240, 245), 255, 15, 7)
INK = Color((16, 19, 25), 234, 0, 0)  # text on an accent pill
SEL_BG = Color((38, 50, 72), 237, 4, 4)
BAND_BG = Color((26, 31, 40), 235, 0, 0)  # a table's column header row
# a single-hue ramp (dark teal-blue to pale cyan), rising in lightness
HEAT = (Color((22, 66, 98), 24, 8, 4), Color((26, 106, 152), 31, 4, 4),
        Color((38, 146, 196), 38, 6, 6), Color((96, 176, 220), 74, 14, 6),
        Color((150, 210, 250), 117, 15, 7))


class Role(NamedTuple):
    fg: Optional[Color] = None
    bold: bool = False
    dim: bool = False        # dim when there is no color for it (mono)
    underline: bool = False  # only in mono (red is not there to say it)


ROLES = {
    "": Role(),
    "head": Role(bold=True),
    "title": Role(bold=True),
    "dim": Role(MUTED, dim=True),
    "faint": Role(FAINT, dim=True),
    "accent": Role(ACCENT),
    "key": Role(ACCENT, bold=True),
    "bright": Role(BRIGHT, bold=True),
    "resource": Role(RESOURCE),
    "user": Role(USER),
    "good": Role(GOOD),
    "warn": Role(WARN, bold=False),
    "bad": Role(BAD, bold=True, underline=True),
    "selected": Role(),
    "tab": Role(INK, bold=True),
    "heat1": Role(HEAT[0], dim=True), "heat2": Role(HEAT[1], dim=True), "heat3": Role(HEAT[2]),
    "heat4": Role(HEAT[3]), "heat5": Role(HEAT[4], bold=True),
}  # type: Dict[str, Role]

BACKGROUNDS = {"sel": SEL_BG, "pill": ACCENT, "band": BAND_BG}
# styles that carry their own background
IMPLIED_BG = {"selected": "sel", "tab": "pill"}
HEAT_STYLES = ("heat1", "heat2", "heat3", "heat4", "heat5")


class Look(NamedTuple):
    fg: Optional[Color]
    bg: Optional[str]  # a BACKGROUNDS key
    bold: bool
    dim: bool
    underline: bool


def resolve(style: str) -> Look:
    """'good+b@sel' -> Look(GOOD, 'sel', bold, ...). Unknown roles read as plain text."""
    base, _, bg = (style or "").partition("@")
    role_name, _, mods = base.partition("+")
    role = ROLES.get(role_name, ROLES[""])
    bold = role.bold or "b" in mods
    bg_name = bg or IMPLIED_BG.get(role_name)
    return Look(role.fg, bg_name if bg_name in BACKGROUNDS else None, bold, role.dim,
                role.underline)


def with_bg(style: str, background: str) -> str:
    """The same style over another background (the selected row keeps its colors)."""
    return (style or "").partition("@")[0] + "@" + background


def bold(style: str) -> str:
    base, at, bg = (style or "").partition("@")
    return (base if "+" in base else base + "+") + "b" + at + bg


# -- depth -------------------------------------------------------------------------
DEPTHS = ("mono", "8", "16", "256", "truecolor")


def detect_depth(colors: int, use_color: bool = True,
                 environ: Optional[Mapping[str, str]] = None) -> str:
    """The richest depth the terminal can address. ``colors`` is curses.COLORS."""
    environ = os.environ if environ is None else environ
    if not use_color or "NO_COLOR" in environ or colors < 8:
        return "mono"
    if colors >= 1 << 24:
        return "truecolor"
    if colors >= 256:
        return "256"
    return "16" if colors >= 16 else "8"


def color_number(color: Color, depth: str) -> Tuple[int, bool]:
    """(curses color number, also dim) for a color at a depth."""
    if depth == "truecolor":
        r, g, b = color.rgb
        return (r << 16) | (g << 8) | b, False
    if depth == "256":
        return color.x256, False
    if depth == "16":
        return color.x16, False
    return color.x8, color.dim8


def backgrounds_shown(depth: str) -> bool:
    """Whether background colors are used (256 and truecolor); below that the
    selected row and the active tab are drawn in reverse video."""
    return depth in ("256", "truecolor")


# -- glyphs ------------------------------------------------------------------------
class Box(NamedTuple):
    tl: str
    tr: str
    bl: str
    br: str
    h: str
    v: str
    lt: str  # ├
    rt: str  # ┤


ROUNDED = Box("╭", "╮", "╰", "╯", "─", "│", "├", "┤")
SQUARE = Box("┌", "┐", "└", "┘", "─", "│", "├", "┤")
ASCII_BOX = Box("+", "+", "+", "+", "-", "|", "+", "+")


def box(ascii_only: bool = False, environ: Optional[Mapping[str, str]] = None) -> Box:
    """Rounded corners, square ones on the Linux console (its font has no arcs), or
    ASCII."""
    if ascii_only:
        return ASCII_BOX
    environ = os.environ if environ is None else environ
    return SQUARE if environ.get("TERM", "") in ("linux", "cons25") else ROUNDED
