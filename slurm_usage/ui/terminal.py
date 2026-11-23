"""curses front end: key decoding, colors, mouse, resize, and the draw loop."""

import curses
import os
import time
from typing import Any, Dict, Optional

from .app import App
from .fmt import width
from . import theme
from .canvas import Canvas

_KEYS = {
    curses.KEY_UP: "UP", curses.KEY_DOWN: "DOWN", curses.KEY_LEFT: "LEFT",
    curses.KEY_RIGHT: "RIGHT", curses.KEY_PPAGE: "PGUP", curses.KEY_NPAGE: "PGDN",
    curses.KEY_HOME: "HOME", curses.KEY_END: "END", curses.KEY_ENTER: "ENTER",
    curses.KEY_BTAB: "BTAB", curses.KEY_BACKSPACE: "BACKSPACE", curses.KEY_RESIZE: "RESIZE",
    10: "ENTER", 13: "ENTER", 27: "ESC", 9: "TAB", 127: "BACKSPACE", 8: "BACKSPACE",
    21: "CTRL_U",
}


class Attrs:
    """curses attributes for style strings, made on first use (one color pair per
    foreground/background combination the screens actually draw)."""

    def __init__(self, use_color: bool) -> None:
        self.depth = "mono"
        self._cache = {}  # type: Dict[str, int]
        self._pairs = {}  # type: Dict[Any, int]
        if not use_color:
            return
        try:
            curses.start_color()
            curses.use_default_colors()
        except curses.error:
            return
        self.depth = theme.detect_depth(getattr(curses, "COLORS", 0), True)

    def get(self, style: str) -> int:
        found = self._cache.get(style)
        if found is None:
            found = self._cache[style] = self._make(style)
        return found

    def _make(self, style: str) -> int:
        look = theme.resolve(style)
        if look.bg == "band" and not theme.backgrounds_shown(self.depth):
            look = look._replace(bg=None)  # a tint only: nothing to say without it
        attr = curses.A_BOLD if look.bold else curses.A_NORMAL
        if self.depth == "mono" or (look.bg and not theme.backgrounds_shown(self.depth)):
            if look.dim:
                attr |= curses.A_DIM
            if look.underline and self.depth == "mono":
                attr |= curses.A_UNDERLINE
            if look.bg == "sel":
                return curses.A_REVERSE  # the selected row: reverse video, plain
            if look.bg:
                return attr | curses.A_REVERSE | curses.A_BOLD
            if self.depth == "mono":
                return attr
        fg, dim = (-1, False) if look.fg is None else theme.color_number(look.fg, self.depth)
        bg = -1
        if look.bg and theme.backgrounds_shown(self.depth):
            bg = theme.color_number(theme.BACKGROUNDS[look.bg], self.depth)[0]
        if dim:
            attr |= curses.A_DIM
        if fg == -1 and bg == -1:
            return attr
        return attr | self._pair(fg, bg)

    def _pair(self, fg: int, bg: int) -> int:
        number = self._pairs.get((fg, bg))
        if number is None:
            number = len(self._pairs) + 1
            try:
                if number >= getattr(curses, "COLOR_PAIRS", 64):
                    raise curses.error("out of color pairs")
                curses.init_pair(number, fg, bg)
            except (curses.error, ValueError, OverflowError):
                if self.depth == "truecolor":  # no extended pairs here: use the 256 mapping
                    self.depth = "256"
                    self._cache.clear()
                return 0
            self._pairs[(fg, bg)] = number
        return curses.color_pair(number)


def _attrs(use_color: bool) -> "Attrs":
    return Attrs(use_color)


def _blit(screen: Any, canvas: Canvas, attrs: Attrs) -> None:
    screen.erase()
    for row in range(canvas.height):
        col = 0
        for text, style in canvas.runs(row):
            size = width(text)
            if row == canvas.height - 1 and col + size >= canvas.width:
                text = text[: max(0, canvas.width - col - 1)]  # the last cell scrolls the screen
            try:
                screen.addstr(row, col, text, attrs.get(style))
            except curses.error:
                pass
            col += size
    screen.noutrefresh()
    curses.doupdate()


def _main(screen: Any, app: App) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    colored = app.color
    attrs = _attrs(colored)
    screen.keypad(True)
    screen.timeout(250)
    try:
        # no REPORT_MOUSE_POSITION: motion reports would wake the loop on every move
        curses.mousemask(curses.ALL_MOUSE_EVENTS)
        curses.mouseinterval(0)
    except curses.error:
        pass
    app.start()
    dirty, last_draw, waiting, drawn = True, 0.0, 250, -1
    while True:
        app.poll()
        if app.color != colored:  # C switched colors
            colored = app.color
            attrs = _attrs(colored)
            dirty = True
        now = time.time()
        # the last answer lands when nothing is busy any more: drawn at once, not on the
        # next one-second tick
        finished = getattr(app.loader, "finished", 0)
        if dirty or now - last_draw > 1.0 or app.loader.busy or finished != drawn:
            height, width = screen.getmaxyx()
            canvas = Canvas(width, height)
            app.render(canvas)
            _blit(screen, canvas, attrs)
            dirty, last_draw, drawn = False, now, finished
        # answers land while work is in flight: look again sooner then
        wait = 100 if app.loader.busy else 250
        if wait != waiting:
            screen.timeout(wait)
            waiting = wait
        try:
            code = screen.get_wch()
        except curses.error:
            continue
        key = _decode(code)
        if key == "MOUSE":
            try:
                _, x, y, _, state = curses.getmouse()
            except curses.error:
                continue
            action = mouse_action(state)
            if action == "up":
                app.wheel(-3, x, y)
            elif action == "down":
                app.wheel(3, x, y)
            elif action == "click":
                app.click(x, y)
            else:
                continue  # a release or a motion report: nothing changed, nothing to draw
            dirty = True
            continue
        if key is None:
            continue
        if key == "RESIZE":
            curses.update_lines_cols() if hasattr(curses, "update_lines_cols") else None
        if not app.key(key):
            return
        dirty = True


# ncurses before mouse version 2 (older distributions) has no BUTTON5 names: the wheel
# turning down arrives as bit 0x200000 there (and REPORT_MOUSE_POSITION on some builds)
_WHEEL_UP = getattr(curses, "BUTTON4_PRESSED", 0x80000)
_WHEEL_DOWN = getattr(curses, "BUTTON5_PRESSED", 0x200000)
_CLICK = curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED


def mouse_action(state: int) -> Optional[str]:
    """'up', 'down' (the wheel), 'click', or None for anything else."""
    if state & _WHEEL_UP:
        return "up"
    if state & _WHEEL_DOWN or (not hasattr(curses, "BUTTON5_PRESSED")
                               and state & getattr(curses, "REPORT_MOUSE_POSITION", 0)):
        return "down"
    if state & _CLICK:
        return "click"
    return None


def _decode(code: Any) -> Optional[str]:
    if isinstance(code, str):
        if code in ("\n", "\r"):
            return "ENTER"
        if code == "\x1b":
            return "ESC"
        if code == "\t":
            return "TAB"
        if code in ("\x7f", "\b"):
            return "BACKSPACE"
        if code == "\x15":
            return "CTRL_U"
        return code if code.isprintable() else None
    if code == curses.KEY_MOUSE:
        return "MOUSE"
    return _KEYS.get(code)


def run(app: App, use_color: Optional[bool] = None) -> None:
    """Run until the user quits.

    Not curses.wrapper: it calls start_color() whenever the terminal has colors, which
    makes ncurses paint every cell white on black (ESC[37m ESC[40m) even with NO_COLOR.
    Without colors start_color() is never called, so the terminal's own colors stay."""
    os.environ.setdefault("ESCDELAY", "25")
    if use_color is not None:
        app.color = use_color
    screen = None
    try:
        screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        screen.keypad(True)
        if app.color:
            try:
                curses.start_color()
            except curses.error:
                pass
        _main(screen, app)
    finally:
        if screen is not None:
            try:
                screen.keypad(False)
                curses.echo()
                curses.nocbreak()
            except curses.error:
                pass
            curses.endwin()
        app.stop()
