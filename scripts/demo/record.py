#!/usr/bin/env python3
"""Record the README video: docs/media/demo.gif and docs/media/demo.mp4.

Drives the real TUI against the invented demo cluster (run_demo.py) in a private
tmux server, captures the screen after each scripted key, renders the frames
with Pillow and encodes them. Nothing here touches a real cluster.

Needs tmux, and in some Python environment: Pillow, pyte, imageio-ffmpeg.

    uv venv /tmp/rec && VIRTUAL_ENV=/tmp/rec uv pip install pillow pyte imageio-ffmpeg
    /tmp/rec/bin/python scripts/demo/record.py            # writes docs/media/
    /tmp/rec/bin/python scripts/demo/record.py --frames /tmp/frames   # also keep PNGs

The first run loads about four months of invented history (a minute or two)
before recording starts.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Optional, Tuple

import pyte
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
COLS, ROWS = 120, 30
CELL_W, CELL_H = 10, 20
FONT_SIZE = 16
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FPS = 15
REPO = "github.com/RyanKim17920/slurm-usage"
INSTALL = "uv tool install git+https://" + REPO

# canvas
PAD = 14  # inside the window, around the terminal
BAR = 34  # window title bar
WIN_W = COLS * CELL_W + 2 * PAD
WIN_H = ROWS * CELL_H + 2 * PAD + BAR
MARGIN_X = 26
CANVAS_W = WIN_W + 2 * MARGIN_X
CAPTION_H = 64
CANVAS_H = 22 + WIN_H + CAPTION_H
CANVAS_W += CANVAS_W % 2
CANVAS_H += CANVAS_H % 2

# a dark theme; curses colors map onto it
BG = (13, 17, 23)
FG = (201, 209, 217)
BRIGHT = (240, 246, 252)
PALETTE = {
    "black": (72, 79, 88), "red": (255, 123, 114), "green": (86, 211, 100),
    "brown": (227, 179, 65), "yellow": (227, 179, 65), "blue": (121, 192, 255),
    "magenta": (210, 168, 255), "cyan": (57, 197, 207), "white": (201, 209, 217),
    "brightblack": (110, 118, 129), "brightred": (255, 161, 152),
    "brightgreen": (126, 231, 135), "brightyellow": (242, 204, 96),
    "brightblue": (165, 214, 255), "brightmagenta": (226, 197, 255),
    "brightcyan": (86, 212, 221), "brightwhite": BRIGHT,
}
SELECTED_BG = (31, 51, 82)
TAB_BG = (56, 139, 253)
WINDOW_BG = BG
CANVAS_TOP = (24, 29, 45)
CANVAS_BOTTOM = (11, 13, 19)
ACCENT = (56, 139, 253)


def rgb(value: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if value == "default":
        return default
    if value in PALETTE:
        return PALETTE[value]
    if len(value) == 6:
        try:
            return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore
        except ValueError:
            pass
    return default


def blend(a, b, t: float):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


# --------------------------------------------------------------------------- fonts
class Fonts:
    def __init__(self) -> None:
        self.regular = ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSansMono.ttf"), FONT_SIZE)
        self.bold = ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSansMono-Bold.ttf"), FONT_SIZE)
        self.ui = lambda size, bold=False: ImageFont.truetype(
            os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
        self.mono = lambda size, bold=False: ImageFont.truetype(
            os.path.join(FONT_DIR, "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"),
            size)


# --------------------------------------------------------------------------- terminal
def parse(ansi: str) -> pyte.Screen:
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.Stream(screen)
    lines = ansi.split("\n")
    for index, line in enumerate(lines[:ROWS]):
        stream.feed("\x1b[%d;1H\x1b[0m" % (index + 1) + line)
    return screen


# up, down, left, right, rounded
BOX = {0x2500: (0, 0, 1, 1, 0), 0x2502: (1, 1, 0, 0, 0), 0x256D: (0, 1, 0, 1, 1),
       0x256E: (0, 1, 1, 0, 1), 0x2570: (1, 0, 0, 1, 1), 0x256F: (1, 0, 1, 0, 1),
       0x250C: (0, 1, 0, 1, 0), 0x2510: (0, 1, 1, 0, 0), 0x2514: (1, 0, 0, 1, 0),
       0x2518: (1, 0, 1, 0, 0), 0x251C: (1, 1, 0, 1, 0), 0x2524: (1, 1, 1, 0, 0)}


def draw_cell(draw: ImageDraw.ImageDraw, fonts: Fonts, x: int, y: int, char: str,
              fg, bold: bool) -> None:
    code = ord(char) if len(char) == 1 else 0
    w, h = CELL_W, CELL_H
    if 0x2581 <= code <= 0x2588:  # lower eighths and full block
        top = y + h - round(h * (code - 0x2580) / 8)
        draw.rectangle([x, top, x + w - 1, y + h - 1], fill=fg)
    elif 0x2589 <= code <= 0x258F:  # left eighths
        width = round(w * (0x2590 - code) / 8)
        draw.rectangle([x, y, x + width - 1, y + h - 1], fill=fg)
    elif code == 0x2580:  # upper half
        draw.rectangle([x, y, x + w - 1, y + h // 2 - 1], fill=fg)
    elif code == 0x2590:  # right half
        draw.rectangle([x + w // 2, y, x + w - 1, y + h - 1], fill=fg)
    elif code in (0x2591, 0x2592, 0x2593):  # shades
        share = {0x2591: 0.28, 0x2592: 0.5, 0x2593: 0.75}[code]
        draw.rectangle([x, y + 1, x + w - 1, y + h - 2], fill=blend(BG, fg, share))
    elif code in BOX:  # lines and corners, joined across cells (fonts leave gaps)
        cx, cy = x + w // 2, y + h // 2
        up, down, left, right, arc = BOX[code]
        r = w // 2
        if arc:
            ox = cx + r if right else cx - r
            oy = cy + r if down else cy - r
            start = {(True, True): 180, (False, True): 270, (True, False): 90,
                     (False, False): 0}[(right, down)]
            draw.arc([ox - r, oy - r, ox + r, oy + r], start, start + 90, fill=fg)
            if down:
                draw.line([cx, oy, cx, y + h - 1], fill=fg)
            else:
                draw.line([cx, y, cx, oy], fill=fg)
        else:
            if left or right:
                draw.line([x if left else cx, cy, x + w - 1 if right else cx, cy], fill=fg)
            if up or down:
                draw.line([cx, y if up else cy, cx, y + h - 1 if down else cy], fill=fg)
    elif 0x2800 < code <= 0x28FF:  # braille: dots on a 2x4 grid
        bits = code - 0x2800
        for bit, (col, row) in enumerate(((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
                                          (0, 3), (1, 3))):
            if bits & (1 << bit):
                px = x + (w * (1 + 2 * col * 1.0)) / 4.0  # at a quarter and three quarters
                py = y + h * (row + 0.75) / 4.4
                draw.ellipse([px - 1.6, py - 1.6, px + 1.6, py + 1.6], fill=fg)
    elif char.strip():
        font = fonts.bold if bold else fonts.regular
        draw.text((x + w / 2, y + h / 2 + 1), char, font=font, fill=fg, anchor="mm")


def render_terminal(screen: pyte.Screen, fonts: Fonts) -> Image.Image:
    image = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H), BG)
    draw = ImageDraw.Draw(image)
    glyphs = []
    for row in range(ROWS):  # every background first, so no glyph is painted over
        line = screen.buffer[row]
        for col in range(COLS):
            cell = line[col]
            fg = rgb(cell.fg, FG)
            bg = rgb(cell.bg, BG)
            if cell.bold and cell.fg == "default":
                fg = BRIGHT
            if cell.reverse:
                if cell.bold:  # the active tab
                    bg, fg = TAB_BG, BRIGHT
                else:  # the selected row
                    bg = SELECTED_BG
                    if cell.fg == "default":
                        blocky = 0x2580 <= ord(cell.data[:1] or " ") <= 0x259F
                        fg = PALETTE["brightblue"] if blocky else BRIGHT
            x, y = col * CELL_W, row * CELL_H
            if bg != BG:
                draw.rectangle([x, y, x + CELL_W - 1, y + CELL_H - 1], fill=bg)
            glyphs.append((x, y, cell.data, fg, cell.bold))
    for x, y, char, fg, bold in glyphs:
        draw_cell(draw, fonts, x, y, char, fg, bold)
    return image


# --------------------------------------------------------------------------- canvas
def background() -> Image.Image:
    image = Image.new("RGB", (CANVAS_W, CANVAS_H))
    draw = ImageDraw.Draw(image)
    for y in range(CANVAS_H):
        draw.line([0, y, CANVAS_W, y], fill=blend(CANVAS_TOP, CANVAS_BOTTOM, y / CANVAS_H))
    return image


def window(term: Image.Image, fonts: Fonts, title: str) -> Image.Image:
    base = background()
    x0, y0 = (CANVAS_W - WIN_W) // 2, 16
    shadow = Image.new("L", (CANVAS_W, CANVAS_H), 0)
    ImageDraw.Draw(shadow).rounded_rectangle([x0 + 4, y0 + 10, x0 + WIN_W + 4, y0 + WIN_H + 10],
                                             12, fill=150)
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    base.paste((0, 0, 0), mask=shadow)
    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle([x0, y0, x0 + WIN_W - 1, y0 + WIN_H - 1], 12, fill=WINDOW_BG,
                           outline=(48, 54, 61))
    draw.rounded_rectangle([x0, y0, x0 + WIN_W - 1, y0 + BAR], 12, fill=(22, 27, 34))
    draw.rectangle([x0 + 1, y0 + BAR - 12, x0 + WIN_W - 2, y0 + BAR], fill=(22, 27, 34))
    draw.line([x0, y0 + BAR, x0 + WIN_W - 1, y0 + BAR], fill=(48, 54, 61))
    for i, color in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        cx, cy = x0 + 20 + i * 20, y0 + BAR // 2
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=color)
    draw.text((CANVAS_W // 2, y0 + BAR // 2), title, font=fonts.ui(14), fill=(139, 148, 158),
              anchor="mm")
    base.paste(term, (x0 + PAD, y0 + BAR + PAD))
    return base


def caption(image: Image.Image, fonts: Fonts, text: str, key: Optional[str],
            alpha: float = 1.0) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    top = 16 + WIN_H
    cy = top + CAPTION_H // 2 + 2
    x = (CANVAS_W - WIN_W) // 2 + 4
    if text:
        draw.rectangle([x, cy - 13, x + 4, cy + 13], fill=blend(CANVAS_BOTTOM, ACCENT, alpha))
        draw.text((x + 18, cy), text, font=fonts.ui(23, True),
                  fill=blend(CANVAS_BOTTOM, BRIGHT, alpha), anchor="lm")
    if key:
        font = fonts.mono(19, True)
        width = int(draw.textlength(key, font=font)) + 26
        right = (CANVAS_W + WIN_W) // 2 - 4
        draw.rounded_rectangle([right - width, cy - 18, right, cy + 18], 8, fill=(33, 38, 45),
                               outline=(88, 96, 105), width=2)
        draw.text((right - width / 2, cy + 1), key, font=font, fill=BRIGHT, anchor="mm")
    return image


def card(fonts: Fonts, lines: List[Tuple[str, int, bool, Tuple[int, int, int], bool]]) -> Image.Image:
    """lines: (text, size, bold, color, monospace)."""
    image = background()
    draw = ImageDraw.Draw(image)
    heights = [size * 1.55 for _, size, *_ in lines]
    y = (CANVAS_H - sum(heights)) / 2
    for (text, size, bold, color, mono), height in zip(lines, heights):
        font = fonts.mono(size, bold) if mono else fonts.ui(size, bold)
        if mono and text:
            width = draw.textlength(text, font=font)
            draw.rounded_rectangle([CANVAS_W / 2 - width / 2 - 18, y + height / 2 - size * 0.95,
                                    CANVAS_W / 2 + width / 2 + 18, y + height / 2 + size * 0.95],
                                   10, fill=(22, 27, 34), outline=(48, 54, 61))
        draw.text((CANVAS_W / 2, y + height / 2), text, font=font, fill=color, anchor="mm")
        y += height
    return image


# --------------------------------------------------------------------------- tmux
class Tmux:
    def __init__(self, socket: str) -> None:
        self.base = ["tmux", "-L", socket, "-f", "/dev/null"]

    def __call__(self, *args: str, check: bool = True) -> str:
        return subprocess.run(self.base + list(args), check=check, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True).stdout

    def start(self, name: str, command: str) -> None:
        self("new-session", "-d", "-s", name, "-x", str(COLS), "-y", str(ROWS), command)

    def capture(self, name: str) -> str:
        return self("capture-pane", "-t", name, "-e", "-p")

    def text(self, name: str) -> str:
        return self("capture-pane", "-t", name, "-p")

    def keys(self, name: str, *keys: str) -> None:
        self("send-keys", "-t", name, *keys)

    def kill(self) -> None:
        self("kill-server", check=False)


BUSY = ("checking", "attempts: ?", "history: ?", "loading", "fetching")


def has(text: str, ready) -> bool:
    """``ready`` is None, a substring, or a predicate on the screen text."""
    if ready is None:
        return True
    return ready(text) if callable(ready) else ready in text


def efficiency_shown(text: str) -> bool:
    # efficiency figures in, the load finished, and no figure partial or unknown
    return ("%·" in text and "loading efficiency" not in text and "≥" not in text
            and not re.search(r"^\W*day +\S+ +\?", text, re.M))


def settle(tmux: Tmux, name: str, least: float = 0.6, most: float = 12.0, quiet: float = 1.0,
           ready=None) -> str:
    """The screen once it has stopped changing for ``quiet`` seconds (the header's clock
    aside), shows nothing still loading, and contains ``ready`` if given."""
    time.sleep(least)
    deadline = time.time() + most
    last, since = None, time.time()
    while True:
        now = tmux.capture(name)
        text = tmux.text(name)
        body = now.split("\n", 1)[1] if "\n" in now else now
        if body != last:
            last, since = body, time.time()
        # the header (first line) and the status line (last) may say "loading" for longer
        middle = "\n".join(text.split("\n")[1:-2])
        busy = any(word in middle for word in BUSY)
        done = not busy and has(text, ready)
        if (done and time.time() - since >= quiet) or time.time() > deadline:
            return now
        time.sleep(0.2)


def wait_loaded(tmux: Tmux, name: str, timeout: float = 600) -> None:
    """Until every range is loaded: no partial (≥) figures and no loading note."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        text = tmux.text(name)
        header = text.split("\n", 1)[0]
        if ("in use" in text and "loading" not in text and "≥" not in text
                and "history" not in header and "…" not in header):
            return
        time.sleep(2)
    raise SystemExit("history did not finish loading")


# --------------------------------------------------------------------------- script
CAPTIONS = {
    "now": "GPUs in use right now, and every range at a glance",
    "ranges": "Day · week · month · year · all time, against the period before",
    "detail": "Click a range: GPUs in use over it, against capacity and the period before",
    "step": "Step through the ranges in place",
    "drill": "Who used it: drill into an account or a person",
    "people": "Every GPU-hour, by person and account",
    "person": "One person: trend, running now, kinds of runs, events",
    "anon": "A anonymizes everything at once, for demos and screenshots",
    "me": "Your own page, with measured GPU and CPU efficiency",
    "job": "Drill down to one job: attempts, steps, a live log tail",
    "timeline": "Timeline: who held the cluster, and when",
    "roles": "Accounts, QOS and partitions, with their limits",
    "nodes": "Every node and every queued job, live",
}

# (caption id, keys to send, keycap label, seconds to hold, text to wait for).
# With a text to wait for, the keys are sent again (up to 8 times) until it shows:
# that is how ``g`` cycles to a chosen grouping. A key ("click", text, rows below, n)
# clicks the mouse n times on the first place ``text`` shows, ``rows below`` it.
SCRIPT = [
    ("now", None, None, 3.2, "Accounts"),
    ("ranges", "3", "3  month", 1.2, "Accounts · month"),
    ("ranges", "5", "5  all", 1.2, "Accounts · all"),
    ("ranges", "2", "2  week", 0.6, "Accounts · week"),
    ("detail", ("click", "─ Month ", 2, 1), "click  Month", 3.4, "GPUs in use · orion"),
    ("step", "]", "]  year", 1.8, "Year · cluster"),
    ("step", "Left", "←  month", 1.0, "Month · cluster"),
    ("step", "[", "[  week", 1.8, "Week · cluster"),
    ("drill", ("click", "physics ", 0, 2), "click  physics", 2.6, "GPUs in use · physics"),
    ("drill", "Escape", "Esc", 0.4, "GPUs in use · orion"),
    ("people", "p", "p  People", 1.3, "People · week"),
    ("people", "Down", "↓", 0.3, None),
    ("people", "Down", "↓", 0.4, None),
    ("person", "Enter", "Enter", 2.2, "What kinds of runs"),
    ("anon", "A", "A  anonymize", 2.2, "ANON"),
    ("anon", "A", "A  off", 0.6, lambda text: "ANON" not in text.split("\n", 1)[0]),
    ("me", "m", "m  Me", 1.0, "Trend · alice"),
    ("me", "E", "E  efficiency", 1.8, efficiency_shown),
    ("job", "Tab", "Tab", 0.4, None),
    ("job", "Enter", "Enter", 2.0, "Attempts (2)"),
    ("job", "l", "l  log tail", 2.4, "stdout tail"),
    ("timeline", "t", "t  Timeline", 1.6, "lanes by user"),
    ("roles", "a", "a  Roles", 1.2, "Roles: Accounts"),
    ("nodes", "n", "n  Nodes", 1.6, "Nodes ("),
    ("nodes", "j", "j  Queue", 1.4, "Queue"),
]


def click(tmux: "Tmux", name: str, needle: str, below: int, times: int, ready=None) -> None:
    """Mouse clicks (SGR encoding, as xterm sends them) where ``needle`` shows, up to
    ``times`` of them, stopping once ``ready`` shows (a first click may only select)."""
    for row, line in enumerate(tmux.text(name).split("\n")):
        col = line.find(needle)
        if col >= 0:
            x, y = col + 2, row + below + 1
            for _ in range(times):
                for end in ("M", "m"):
                    tmux("send-keys", "-t", name, "-l", "\x1b[<0;%d;%d%s" % (x, y, end))
                time.sleep(0.6)
                if ready is not None and has(tmux.text(name), ready):
                    return
            return
    raise SystemExit("nothing to click: %r" % needle)


def run_sessions(tmux: Tmux, work: str) -> List[Tuple[str, str, Optional[str], float]]:
    """Play SCRIPT; returns (ansi screen, caption, keycap, seconds) per shot."""
    cache = os.path.join(work, "cache")
    os.makedirs(cache, mode=0o700, exist_ok=True)
    os.chmod(cache, 0o700)
    tmux.start("main", "cd %s && TERM=xterm-256color LANG=C.UTF-8 SLURM_USAGE_CACHE=%s %s %s; "
               "sleep 2" % (ROOT, cache, sys.executable, os.path.join(HERE, "run_demo.py")))
    print("loading the demo history...")
    wait_loaded(tmux, "main")
    tmux.keys("main", "o")
    settle(tmux, "main", 1.0)
    shots = []
    for caption_id, key, label, hold, ready in SCRIPT:
        quick = key in ("Down", "Tab")
        for attempt in range(8 if ready and key == "g" else 1):
            if isinstance(key, tuple):
                click(tmux, "main", *key[1:], ready=ready)
            elif key:
                tmux.keys("main", key)
            screen = settle(tmux, "main", 0.25 if quick else 0.6, quiet=0.4 if quick else 1.0,
                            most=3.0 if key == "g" else 12.0, ready=ready)
            if has(tmux.text("main"), ready):
                break
        shots.append((screen, CAPTIONS[caption_id], label, hold))
    return shots


# --------------------------------------------------------------------------- output
def assemble(shots, fonts: Fonts) -> List[Tuple[Image.Image, float]]:
    frames = []  # (image, seconds)
    intro = card(fonts, [
        ("slurm-usage", 64, True, BRIGHT, False),
        ("", 10, False, FG, False),
        ("A read-only terminal monitor and usage explorer", 26, False, FG, False),
        ("for any Slurm cluster", 26, False, FG, False),
    ])
    outro = card(fonts, [
        ("Read-only.  Any Slurm cluster.  Standard library only.", 26, True, BRIGHT, False),
        ("", 12, False, FG, False),
        (INSTALL, 21, False, (126, 231, 135), True),
        ("", 12, False, FG, False),
        (REPO, 24, False, (139, 148, 158), False),
    ])
    frames.append((intro, 2.0))
    previous = intro
    last_caption = None
    for index, (screen, text, key, hold) in enumerate(shots):
        term = render_terminal(parse(screen), fonts)
        base = window(term, fonts, "slurm-usage — invented demo cluster")
        if previous is intro:  # fade in from the title card
            final = caption(base, fonts, text, key)
            for step in range(1, 6):
                frames.append((Image.blend(intro, final, step / 6), 1 / FPS))
        if text != last_caption and last_caption is not None:
            for step in range(1, 4):
                frames.append((caption(base, fonts, text, key, step / 4), 1 / FPS))
        frames.append((caption(base, fonts, text, key), hold))
        last_caption = text
        previous = base
    last = frames[-1][0]
    for step in range(1, 7):
        frames.append((Image.blend(last, outro, step / 7), 1 / FPS))
    frames.append((outro, 3.4))
    return frames


def write_mp4(frames, path: str) -> None:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "%dx%d" % (CANVAS_W, CANVAS_H), "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-tune", "animation",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    carry = 0.0
    for image, seconds in frames:
        carry += seconds * FPS
        count = int(round(carry))
        carry -= count
        data = image.convert("RGB").tobytes()
        for _ in range(count):
            proc.stdin.write(data)
    proc.stdin.close()
    if proc.wait():
        raise SystemExit("ffmpeg failed")


def write_gif(frames, path: str, scale: float = 1.0) -> None:
    images, durations = [], []
    palette_source = None
    for image, seconds in frames:
        if scale != 1.0:
            image = image.resize((int(CANVAS_W * scale), int(CANVAS_H * scale)), Image.LANCZOS)
        images.append(image)
        durations.append(max(20, int(round(seconds * 1000 / 10)) * 10))
    # one shared palette from a few representative frames keeps colors stable
    strip = Image.new("RGB", (images[0].width, images[0].height * 4))
    picks = [images[len(images) * i // 4] for i in range(4)]
    for i, image in enumerate(picks):
        strip.paste(image, (0, i * images[0].height))
    palette_source = strip.quantize(colors=128, method=Image.MEDIANCUT)
    quantized = [image.quantize(palette=palette_source, dither=Image.NONE) for image in images]
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=durations,
                      loop=0, optimize=False, disposal=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "docs", "media"))
    parser.add_argument("--frames", help="also write every distinct frame as PNG here")
    parser.add_argument("--work", help="scratch directory (default: a temporary one)")
    parser.add_argument("--gif-scale", type=float, default=1.0)
    args = parser.parse_args()
    if shutil.which("tmux") is None:
        raise SystemExit("tmux is needed")
    work = args.work or tempfile.mkdtemp(prefix="slurm-usage-record-")
    os.makedirs(work, exist_ok=True)
    fonts = Fonts()
    tmux = Tmux("slurm-usage-record-%d" % os.getpid())
    try:
        shots = run_sessions(tmux, work)
    finally:
        tmux.kill()
    frames = assemble(shots, fonts)
    if args.frames:
        os.makedirs(args.frames, exist_ok=True)
        for index, (image, seconds) in enumerate(frames):
            if seconds > 0.2:
                image.save(os.path.join(args.frames, "frame-%03d.png" % index))
    os.makedirs(args.out, exist_ok=True)
    write_mp4(frames, os.path.join(args.out, "demo.mp4"))
    write_gif(frames, os.path.join(args.out, "demo.gif"), args.gif_scale)
    total = sum(seconds for _, seconds in frames)
    for name in ("demo.mp4", "demo.gif"):
        size = os.path.getsize(os.path.join(args.out, name))
        print("%s: %.1f MB, %.1f s, %dx%d" % (name, size / 1e6, total, CANVAS_W, CANVAS_H))
    if not args.work:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
