"""curses front end: key decoding, colors, mouse, resize, and the draw loop."""

import curses
import os
import time
from typing import Any, Dict, Optional

from .app import App
from .fmt import width
from .canvas import BAD, DIM, GOOD, HEAD, PLAIN, RESOURCE, SELECTED, TAB, USER, WARN, Canvas

_KEYS = {
    curses.KEY_UP: "UP", curses.KEY_DOWN: "DOWN", curses.KEY_LEFT: "LEFT",
    curses.KEY_RIGHT: "RIGHT", curses.KEY_PPAGE: "PGUP", curses.KEY_NPAGE: "PGDN",
    curses.KEY_HOME: "HOME", curses.KEY_END: "END", curses.KEY_ENTER: "ENTER",
    curses.KEY_BTAB: "BTAB", curses.KEY_BACKSPACE: "BACKSPACE", curses.KEY_RESIZE: "RESIZE",
    10: "ENTER", 13: "ENTER", 27: "ESC", 9: "TAB", 127: "BACKSPACE", 8: "BACKSPACE",
}


def _attrs(use_color: bool) -> Dict[str, int]:
    base = {PLAIN: curses.A_NORMAL, HEAD: curses.A_BOLD, SELECTED: curses.A_REVERSE,
            TAB: curses.A_REVERSE | curses.A_BOLD, DIM: curses.A_DIM, RESOURCE: curses.A_NORMAL,
            USER: curses.A_NORMAL, GOOD: curses.A_NORMAL, WARN: curses.A_BOLD,
            BAD: curses.A_BOLD | curses.A_UNDERLINE}
    if not use_color:
        return base
    try:
        curses.start_color()
        curses.use_default_colors()
        background = -1
    except curses.error:
        return base
    amber = 214 if curses.COLORS >= 256 else curses.COLOR_YELLOW
    gray = 244 if curses.COLORS >= 256 else curses.COLOR_WHITE
    pairs = {RESOURCE: curses.COLOR_CYAN, USER: curses.COLOR_MAGENTA, GOOD: curses.COLOR_GREEN,
             WARN: amber, BAD: curses.COLOR_RED, DIM: gray}
    for number, (style, color) in enumerate(pairs.items(), start=1):
        try:
            curses.init_pair(number, color, background)
            base[style] = curses.color_pair(number) | (curses.A_BOLD if style == BAD else 0)
            if style == DIM and curses.COLORS < 256:
                base[style] |= curses.A_DIM
        except curses.error:
            pass
    return base


def _blit(screen: Any, canvas: Canvas, attrs: Dict[str, int]) -> None:
    screen.erase()
    for row in range(canvas.height):
        col = 0
        for text, style in canvas.runs(row):
            size = width(text)
            if row == canvas.height - 1 and col + size >= canvas.width:
                text = text[: max(0, canvas.width - col - 1)]  # the last cell scrolls the screen
            try:
                screen.addstr(row, col, text, attrs.get(style, 0))
            except curses.error:
                pass
            col += size
    screen.noutrefresh()
    curses.doupdate()


def _main(screen: Any, app: App, use_color: bool) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    attrs = _attrs(use_color)
    screen.keypad(True)
    screen.timeout(250)
    try:
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
        curses.mouseinterval(0)
    except curses.error:
        pass
    app.start()
    dirty, last_draw = True, 0.0
    while True:
        app.poll()
        now = time.time()
        if dirty or now - last_draw > 1.0 or app.loader.busy:
            height, width = screen.getmaxyx()
            canvas = Canvas(width, height)
            app.render(canvas)
            _blit(screen, canvas, attrs)
            dirty, last_draw = False, now
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
            if state & getattr(curses, "BUTTON4_PRESSED", 0):
                app.wheel(-3)
            elif state & getattr(curses, "BUTTON5_PRESSED", 0):
                app.wheel(3)
            elif state & (curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED):
                app.click(x, y)
            dirty = True
            continue
        if key is None:
            continue
        if key == "RESIZE":
            curses.update_lines_cols() if hasattr(curses, "update_lines_cols") else None
        if not app.key(key):
            return
        dirty = True


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
        return code if code.isprintable() else None
    if code == curses.KEY_MOUSE:
        return "MOUSE"
    return _KEYS.get(code)


def run(app: App, use_color: bool = True) -> None:
    """Run until the user quits."""
    os.environ.setdefault("ESCDELAY", "25")
    try:
        curses.wrapper(_main, app, use_color)
    finally:
        app.stop()
