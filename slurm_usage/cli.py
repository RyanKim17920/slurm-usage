"""Command line: ``slurm-usage`` (TUI), ``--once`` (text), ``--json``."""

import argparse
import os
import sys
from typing import Any, List, Optional

from . import __version__
from .model import Status


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="slurm-usage",
        description="Read-only Slurm monitor and usage explorer. Nothing it does changes "
                    "anything in Slurm.")
    parser.add_argument("--once", action="store_true",
                        help="print a plain-text summary and exit (default when not a terminal)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON and exit")
    parser.add_argument("--range", choices=("day", "week", "month", "year", "all"),
                        default="week", help="usage window (default: week)")
    parser.add_argument("--user", help="open on this person (with --once: add their page)")
    parser.add_argument("--account", help="open on this account's people (with --once: only it)")
    parser.add_argument("--no-history", action="store_true",
                        help="live view only; do not read Slurm accounting")
    parser.add_argument("--no-full-history", action="store_true",
                        help="do not load all-time history in the background (it is still "
                             "loaded if you pick the 'all' range)")
    parser.add_argument("--anonymize", action="store_true",
                        help="stable pseudonyms for users, accounts, nodes and job names; "
                             "hide paths, scripts, submit lines and logs")
    parser.add_argument("--no-color", action="store_true", help="no colors (also: NO_COLOR)")
    parser.add_argument("--ascii", action="store_true", help="ASCII-only charts")
    parser.add_argument("--cache-dir", help="where history is cached (default: per-user cache)")
    parser.add_argument("--version", action="version", version="slurm-usage " + __version__)
    return parser.parse_args(argv)


def _history(args: argparse.Namespace) -> Any:
    from .ui import data as hd
    if args.no_history:
        return hd.HistoryData(hd.NoHistory(Status.UNAVAILABLE, "history is off (--no-history)"))
    try:
        from .history import History
        history = History(cache_dir=args.cache_dir)  # type: Any
    except Exception as error:  # the history layer failing must not take the live view down
        history = hd.NoHistory(Status.ERROR, "history unavailable: %s" % error)
    return hd.HistoryData(history, full_history=not args.no_full_history)


def build(args: argparse.Namespace) -> Any:
    from .live import LiveCollector
    from .ui import data as hd
    from .ui.anonymize import AnonCollector, AnonHistory, Anonymizer
    collector = LiveCollector()  # type: Any
    data = _history(args)
    anon = None
    if args.anonymize:
        anon = Anonymizer()
        collector = AnonCollector(collector, anon)
        if not isinstance(data.history, hd.NoHistory):
            data.history = AnonHistory(data.history, anon)
    return collector, data, anon


def run_once(args: argparse.Namespace, out: Any = None) -> int:
    from .ui import text
    out = out or sys.stdout
    collector, data, anon = build(args)
    try:
        snap = collector.snapshot()
    except Exception as error:
        snap = None
        print("live: %s" % error, file=sys.stderr)
    user, account = args.user, args.account
    if anon is not None:  # flags name real people; outputs use pseudonyms
        user = anon.name("user", user) if user else None
        account = anon.name("account", account) if account else None
    info = None
    if not args.no_history:
        quiet = not sys.stderr.isatty()

        def progress(event: Any) -> None:
            if not quiet:
                sys.stderr.write("\rloading history: %s %s   " % (event.phase, event.done))
                sys.stderr.flush()
        data._progress = progress  # type: ignore
        try:
            data.load(args.range)
        except Exception:
            pass
        if not quiet:
            sys.stderr.write("\r" + " " * 50 + "\r")
        try:
            info = data.info()
        except Exception:
            info = None
    summary = text.collect(snap, None if args.no_history else data, args.range, user, account,
                           info)
    if anon is not None:
        summary["anonymized"] = True
    write(out, text.to_json(summary) + "\n" if args.json else text.render(summary),
          ascii_only=args.ascii)
    return 0


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
    color = not args.no_color and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
    start = "person" if user else "overview"
    app = App(collector, data, args.range, user=user, account=account,
              ascii_only=args.ascii, anonymized=anon is not None, no_history=args.no_history,
              start=start)
    if start == "person":
        app.stack[0].title = user
    try:
        run(app, use_color=color)
    except KeyboardInterrupt:
        pass
    return 0
