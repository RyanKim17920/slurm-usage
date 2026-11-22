"""Pure text formatting: numbers, durations, placeholders, widths, sparklines.

Placeholders mean one thing each, everywhere:

    n/a      the source is unavailable here
    —        not applicable
    ?        not sampled (loaded) yet
    denied   Slurm refused this invoker
    not stored  Slurm answered, but keeps no copy (e.g. a batch script)
    ERR      the query failed
"""

import time
import unicodedata
from typing import Iterable, List, Optional, Sequence

from ..model import Status
from ..sanitize import clean

NA, NOT_APPLICABLE, NOT_YET, DENIED, ERR = "n/a", "—", "?", "denied", "ERR"
NOT_STORED = "not stored"
PLACEHOLDERS = (NA, NOT_APPLICABLE, NOT_YET, DENIED, ERR, NOT_STORED)
SPARKS = "▁▂▃▄▅▆▇█"
SHADES = " ░▒▓█"
ASCII_SPARKS = "_.-=+*#@"


def placeholder(status: Optional[str]) -> str:
    """The placeholder for a source status that produced no value."""
    if status == Status.DENIED:
        return DENIED
    if status in (Status.ERROR, Status.TIMEOUT):
        return ERR
    if status in (Status.UNKNOWN, "pending", None):
        return NOT_YET
    if status == Status.EMPTY:
        return NOT_APPLICABLE
    if status == Status.NOT_STORED:
        return NOT_STORED
    return NA


# -- width -------------------------------------------------------------------

def char_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def width(text: str) -> int:
    if text.isascii():
        return len(text)
    return sum(char_width(char) for char in text)


def fit(text: object, size: int, align: str = "<") -> str:
    """Clean, then cut (with …) or pad to exactly ``size`` display cells."""
    text = clean(text)
    if size <= 0:
        return ""
    used = width(text)
    if used > size:
        out, total = [], 0
        for char in text:
            step = char_width(char)
            if total + step > size - 1:
                break
            out.append(char)
            total += step
        text = "".join(out) + "…"
        used = total + 1
    pad = " " * (size - used)
    if align == ">":
        return pad + text
    if align == "^":
        half = len(pad) // 2
        return pad[:half] + text + pad[half:]
    return text + pad


# -- numbers -----------------------------------------------------------------

def num(value: Optional[float], digits: int = 0) -> str:
    """Compact human number: 950, 12.3k, 4.1M."""
    if value is None:
        return NA
    sign = "-" if value < 0 else ""
    value = abs(value)
    for limit, divisor, suffix in ((1e12, 1e12, "T"), (1e9, 1e9, "G"), (1e6, 1e6, "M"),
                                   (1e4, 1e3, "k")):
        if value >= limit:
            scaled = value / divisor
            return "%s%.*f%s" % (sign, 0 if scaled >= 100 else 1, scaled, suffix)
    if digits and value < 100:
        return "%s%.*f" % (sign, digits, value)
    return "%s%d" % (sign, round(value))


def hours(value: Optional[float]) -> str:
    if value is None:
        return NA
    if 0 < value < 10:
        return "%.1f" % value
    return num(value)


def pct(fraction: Optional[float], digits: int = 0) -> str:
    if fraction is None:
        return NA
    return "%.*f%%" % (digits, fraction * 100)


def delta(now: Optional[float], before: Optional[float]) -> str:
    """Change vs the previous period."""
    if now is None or before is None:
        return NA
    if before == 0:
        return NOT_APPLICABLE if now == 0 else "new"
    change = (now - before) / before
    if abs(change) < 0.005:
        return "±0%"
    return "%+.0f%%" % (change * 100) if abs(change) < 10 else "%+.0fx" % (now / before)


def mem(value: Optional[float]) -> str:
    if value is None:
        return NA
    for unit, size in (("T", 1 << 40), ("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if value >= size:
            scaled = value / size
            return "%.*f%s" % (0 if scaled >= 10 else 1, scaled, unit)
    return "%dB" % value


def duration(seconds: Optional[float]) -> str:
    """3d04h, 5h12m, 7m03s, 42s."""
    if seconds is None:
        return NA
    seconds = int(seconds)
    if seconds < 0:
        return "-" + duration(-seconds)
    days, rest = divmod(seconds, 86400)
    hrs, rest = divmod(rest, 3600)
    mins, secs = divmod(rest, 60)
    if days >= 100:
        return "%dd" % days
    if days:
        return "%dd%02dh" % (days, hrs)
    if hrs:
        return "%dh%02dm" % (hrs, mins)
    if mins:
        return "%dm%02ds" % (mins, secs)
    return "%ds" % secs


def age(seconds: Optional[float]) -> str:
    if seconds is None:
        return NA
    if seconds < 5:
        return "now"
    return duration(seconds) + " ago"


def when(epoch: Optional[float], now: Optional[float] = None) -> str:
    """Local time: HH:MM today, else Mon DD HH:MM, else YYYY-MM-DD."""
    if epoch is None:
        return NA
    now = time.time() if now is None else now
    if abs(now - epoch) < 20 * 3600:
        return time.strftime("%H:%M", time.localtime(epoch))
    if abs(now - epoch) < 300 * 86400:
        return time.strftime("%b %d %H:%M", time.localtime(epoch))
    return time.strftime("%Y-%m-%d", time.localtime(epoch))


def day(epoch: Optional[float]) -> str:
    if epoch is None:
        return NA
    return time.strftime("%a %b %d", time.localtime(epoch))


def ratio(used: Optional[float], total: Optional[float]) -> str:
    if used is None or total is None:
        return NA
    return "%s/%s" % (num(used), num(total))


# -- small charts ------------------------------------------------------------

def resample(values: Sequence[float], count: int) -> List[float]:
    """Sum ``values`` into ``count`` nearly equal bins (or fewer, if fewer values)."""
    values = list(values)
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return [float(v) for v in values]
    out = []
    for index in range(count):
        low = index * len(values) // count
        high = (index + 1) * len(values) // count
        out.append(float(sum(values[low:high])))
    return out


def sparkline(values: Optional[Iterable[float]], size: int, ascii_only: bool = False) -> str:
    if values is None:
        return fit(NA, size)
    data = resample(list(values), size)
    if not data:
        return fit(NOT_APPLICABLE, size)
    marks = ASCII_SPARKS if ascii_only else SPARKS
    top = max(data)
    if top <= 0:
        return (marks[0] * len(data)).rjust(size)
    out = []
    for value in data:
        if value <= 0:
            out.append(" ")
        else:
            out.append(marks[min(len(marks) - 1, int(value / top * (len(marks) - 1) + 0.5))])
    return "".join(out).rjust(size)


def bar(fraction: Optional[float], size: int, ascii_only: bool = False) -> str:
    if fraction is None or size <= 0:
        return " " * max(0, size)
    fraction = max(0.0, min(1.0, fraction))
    if ascii_only:
        filled = int(round(fraction * size))
        return "#" * filled + "." * (size - filled)
    eighths = int(round(fraction * size * 8))
    full, part = divmod(eighths, 8)
    text = "█" * full + ("▏▎▍▌▋▊▉"[part - 1] if part else "")
    return text + " " * (size - len(text))


def shade(fraction: float, ascii_only: bool = False) -> str:
    marks = " .:+#" if ascii_only else SHADES
    if fraction <= 0:
        return marks[0]
    return marks[min(len(marks) - 1, 1 + int(fraction * (len(marks) - 1)))]


def median(values: Sequence[float]) -> Optional[float]:
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    middle = len(data) // 2
    return data[middle] if len(data) % 2 else (data[middle - 1] + data[middle]) / 2.0
