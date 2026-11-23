"""Command line: ``slurm-usage`` (TUI), ``--once`` (text), ``--json``."""

import argparse
import locale
import os
import sys
from typing import Any, List, Optional

from . import __version__
from .model import Status

HISTORY_MODES = ("full", "recent", "off")
CACHE_ENV = "SLURM_USAGE_CACHE"
# Options that were removed, and what replaced each.
REMOVED = {
    "--no-history": "use --history off (or press H in the app)",
    "--no-full-history": "use --history recent (or press H in the app)",
    "--no-color": "set NO_COLOR=1 (or press C in the app)",
    "--ascii": "ASCII is chosen from the terminal's encoding; press U in the app to switch",
    "--cache-dir": "set %s=<dir> instead" % CACHE_ENV,
}
_LEGACY_TERMS = ("vt52", "vt100", "vt102", "vt220", "ansi", "cons25")

EPILOG = """\
environment:
  NO_COLOR=1               start without colors
  %s=<dir>  history cache directory (must be private to you)

keys in the app: ? help, O settings, A anonymize, C colors, U Unicode/ASCII,
H history full/recent/off
""" % CACHE_ENV


def _removed(argv: List[str]) -> Optional[str]:
    for arg in argv:
        if arg == "--":
            return None
        name = arg.split("=", 1)[0]
        if name in REMOVED:
            return "%s was removed: %s" % (name, REMOVED[name])
    return None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="slurm-usage", allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG,
        description="Read-only Slurm monitor and usage explorer. Nothing it does changes "
                    "anything in Slurm.")
    out = parser.add_argument_group("output (default: the full-screen app on a terminal)")
    out.add_argument("--once", action="store_true", help="print a text summary and exit")
    out.add_argument("--json", action="store_true", help="print the summary as JSON and exit")
    scope = parser.add_argument_group("what to show")
    scope.add_argument("--range", choices=("day", "week", "month", "year", "all"),
                       default="week", help="usage window (default: week)")
    scope.add_argument("--user", help="open on this person")
    scope.add_argument("--account", help="open on this account's people")
    scope.add_argument("--history", choices=HISTORY_MODES, default="full",
                       help="full: load all of history in the background (default); "
                            "recent: only the chosen range and shorter; off: live view only")
    scope.add_argument("--anonymize", action="store_true",
                       help="pseudonyms for users, accounts, nodes and job names; "
                            "no paths, scripts or logs (A toggles it in the app)")
    parser.add_argument("--version", action="version", version="slurm-usage " + __version__)
    problem = _removed(argv)
    if problem:
        parser.print_usage(sys.stderr)
        parser.exit(2, "slurm-usage: %s\n" % problem)
    return parser.parse_args(argv)


def cache_dir() -> Optional[str]:
    """The history cache directory from the environment, or None for the default."""
    found = os.environ.get(CACHE_ENV, "").strip()
    return os.path.abspath(os.path.expanduser(found)) if found else None


def wants_color() -> bool:
    return "NO_COLOR" not in os.environ and os.environ.get("TERM") not in ("dumb", "")


def wants_ascii(stream: Any = None) -> bool:
    """ASCII glyphs when the output (or the locale) cannot show the Unicode ones, or the
    terminal is a legacy one that has none."""
    from .ui import text
    stream = stream if stream is not None else sys.stdout
    encoding = getattr(stream, "encoding", None)
    if encoding is None:
        try:
            encoding = locale.getpreferredencoding(False)
        except Exception:
            encoding = None
    if encoding is not None and not text.can_encode(encoding):
        return True
    return os.environ.get("TERM", "").split("-")[0] in _LEGACY_TERMS


def _history(args: argparse.Namespace) -> Any:
    from .ui import data as hd
    try:
        from .history import History
        history = History(cache_dir=cache_dir())  # type: Any  # opens nothing until asked
    except Exception as error:  # the history layer failing must not take the live view down
        history = hd.NoHistory(Status.ERROR, "history unavailable: %s" % error)
    return hd.HistoryData(history, full_history=args.history == "full")


def build(args: argparse.Namespace) -> Any:
    from .live import LiveCollector
    from .ui import data as hd
    collector = getattr(args, "collector", None) or LiveCollector()  # type: Any
    data = _history(args)
    anon = None
    if args.anonymize:
        from .ui.anonymize import AnonCollector, AnonHistory, Anonymizer
        anon = Anonymizer()
        collector = AnonCollector(collector, anon)
        if not isinstance(data.history, hd.NoHistory):
            data.history = AnonHistory(data.history, anon)
    return collector, data, anon


_BUILD = build


def run_once(args: argparse.Namespace, out: Any = None) -> int:
    # the live snapshot is mostly waiting for Slurm: start it first, then import and
    # load history while it runs
    import threading
    live = {}  # type: dict
    raw = None  # type: Any
    if not args.anonymize and build is _BUILD:  # (tests put in their own build())
        from .live import LiveCollector
        raw = LiveCollector()

    def take(collector: Any) -> None:
        try:
            live["snap"] = collector.snapshot()
        except Exception as error:
            live["error"] = error
    taking = None  # type: Any
    if raw is not None:
        taking = threading.Thread(target=take, args=(raw,), name="snapshot", daemon=True)
        taking.start()
    from .ui import text
    out = out or sys.stdout
    args.collector = raw  # the one already taking a snapshot
    collector, data, anon = build(args)
    if taking is None:
        taking = threading.Thread(target=take, args=(collector,), name="snapshot", daemon=True)
        taking.start()
    if anon is None and hasattr(hd_module(), "ResultCache"):
        # an overview kept by an earlier run (the app's, say) of the same data is reused
        data.results = hd_module().ResultCache(data.history)
    user, account = args.user, args.account
    if anon is not None:  # flags name real people; outputs use pseudonyms
        user = anon.name("user", user) if user else None
        account = anon.name("account", account) if account else None
    info = None
    no_history = args.history == "off"
    if not no_history:
        quiet = not sys.stderr.isatty()

        def progress(event: Any) -> None:
            if not quiet:
                sys.stderr.write("\rloading history: %s %s   " % (event.phase, event.done))
                sys.stderr.flush()
        data._progress = progress  # type: ignore
        try:
            if not _fresh(data, args.range):
                data.load(args.range)
        except Exception:
            pass
        if not quiet:
            sys.stderr.write("\r" + " " * 50 + "\r")
        try:
            info = data.info()
        except Exception:
            info = None
        try:
            data.overview(None, None)  # the slow part of the summary, while live is taken
        except Exception:
            pass
    taking.join()
    snap = live.get("snap")
    if "error" in live:
        print("live: %s" % live["error"], file=sys.stderr)
    summary = text.collect(snap, None if no_history else data, args.range, user, account, info)
    if anon is not None:
        summary["anonymized"] = True
    if args.json:
        write(out, text.to_json(summary) + "\n")
    else:
        write(out, text.render(summary), ascii_only=wants_ascii(out))
    if getattr(data, "results", None) is not None:
        out.flush()
        data.results.flush()  # after the output: kept for the next run
    return 0


def hd_module() -> Any:
    from .ui import data as hd
    return hd


FRESH = 60.0  # history refreshed this recently (by the app, say) is used as it is


def _fresh(data: Any, range_name: str) -> bool:
    """Whether the cache already holds ``range_name`` up to a minute ago."""
    import time
    from .ui import data as hd
    try:
        info = data.history.info()
    except Exception:
        return False
    covered, as_of = getattr(info, "covered_from", None), getattr(info, "as_of", None)
    if covered is None or as_of is None or time.time() - as_of > FRESH:
        return False
    start = hd.window(range_name).start
    return covered == 0 or (start is not None and covered <= start)


def write(out: Any, body: str, ascii_only: bool = False) -> None:
    """Never fails on the output's encoding: ASCII glyphs where Unicode cannot be shown,
    and anything still unencodable replaced."""
    encoding = getattr(out, "encoding", None)
    from .ui import text
    if ascii_only or (encoding is not None and not text.can_encode(encoding)):
        body = text.to_ascii(body)
    if encoding is not None:
        body = body.encode(encoding, "replace").decode(encoding, "replace")
    out.write(body)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    dumb = os.environ.get("TERM", "dumb") in ("", "dumb", "unknown")
    if args.json or args.once or dumb or not (sys.stdout.isatty() and sys.stdin.isatty()):
        try:
            return run_once(args)
        except BrokenPipeError:
            return 0
    from .ui.app import App
    from .ui.terminal import run
    collector, data, anon = build(args)
    user = args.user
    if anon is not None and user:
        user = anon.name("user", user)
    account = args.account
    if anon is not None and account:
        account = anon.name("account", account)
    start = "person" if user else "overview"
    app = App(collector, data, args.range, user=user, account=account,
              ascii_only=wants_ascii(), anonymized=anon is not None, history=args.history,
              start=start, color=wants_color())
    if start == "person":
        app.stack[0].title = user
    try:
        run(app)
    except KeyboardInterrupt:
        pass
    return 0
