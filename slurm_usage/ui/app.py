"""The application state machine: navigation stack, keys, refresh, drawing.

It knows nothing about curses. ``terminal.run`` feeds it key names and blits
its Canvas; tests do the same with an in-memory canvas.

Key names: single characters, or UP DOWN LEFT RIGHT PGUP PGDN HOME END ENTER
ESC TAB BTAB BACKSPACE RESIZE.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..model import Status
from . import data as hd
from . import fmt
from .canvas import BAD, DIM, GOOD, HEAD, PLAIN, RESOURCE, TAB, WARN, Canvas
from .filestat import check as stat_path
from .filestat import script_from_submit_line
from .loader import Loader
from .screens import FOOTERS, LANE_GROUPS, ROLE_GROUPS, SCREENS, TABS, Context, View
from .widgets import PaneState, allocate, draw_pane, visible_lines

ACTIVE_REFRESH = 30.0
MAX_REFRESH = 300.0
IDLE_AFTER = 120.0
HISTORY_UPDATE = 300.0
TAB_KINDS = tuple(kind for kind, _, _, _ in TABS)
TAB_KEYS = {key: kind for kind, _, _, key in TABS}
FETCH_KEYS = {"l": "stdout", "e": "stderr", "b": "script", "u": "submit"}


class App:
    def __init__(self, collector: Any, data: hd.HistoryData, range_name: str = "week",
                 user: Optional[str] = None, account: Optional[str] = None,
                 ascii_only: bool = False, background: bool = True, anonymized: bool = False,
                 no_history: bool = False, start: str = "overview") -> None:
        self.collector = collector
        self.data = data
        self.range = range_name
        self.user = user
        self.ascii = ascii_only
        self.anonymized = anonymized
        self.no_history = no_history
        self.background = background
        self.loader = Loader(workers=3 if background else 0)
        self.snapshot = None  # type: Any
        self.snapshot_entry = None  # type: Any
        self.snap_generation = 0
        self.refreshes = 0
        self.history_info = None  # type: Any
        self.last_input = time.time()
        self.message = ""
        self.input = None  # type: Optional[List[str]]  # filter being typed
        self.hits = []  # type: List[Tuple[int, int, int]]  # (row, pane index, line index)
        self.tab_hits = []  # type: List[Tuple[int, int, str]]
        self._last_click = (0.0, -1, -1)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._history_thread = None  # type: Optional[threading.Thread]
        root = View(start, {"user": user} if start in ("me", "person") and user else {},
                    dict((k, t) for k, t, _, _ in TABS).get(start, start))
        if account and start == "overview":
            self.stack = [root, View("people", {"account": account}, "People in " + account)]
        else:
            self.stack = [root]

    # -- data ------------------------------------------------------------------

    def interval(self) -> float:
        idle = time.time() - self.last_input
        if idle < IDLE_AFTER:
            return ACTIVE_REFRESH
        return min(MAX_REFRESH, ACTIVE_REFRESH * 2 ** int(idle / IDLE_AFTER))

    def start(self) -> None:
        if self.background and not self.no_history:
            self._history_thread = threading.Thread(target=self._history_loop, daemon=True,
                                                    name="history")
            self._history_thread.start()
        elif not self.no_history:
            self.data.load_in_order(self.range)

    def _history_loop(self) -> None:
        last_update = time.time()
        while not self._stop.is_set():
            auto = self.data.auto_ranges() if hasattr(self.data, "auto_ranges") else hd.RANGE_NAMES
            pending = [n for n in [self.range] + list(auto) if n not in self.data.loaded]
            if pending:
                try:
                    first = not self.data.loaded
                    if not self.data.load(pending[0], update=first):
                        continue  # stopped for a range the viewer asked for since
                    last_update = time.time() if first else last_update
                except Exception:
                    failure = self.data.failure
                    if failure and failure[0] in (Status.UNAVAILABLE, Status.DENIED):
                        return
                    self._wake.wait(60)
                    self._wake.clear()
                continue
            wait = HISTORY_UPDATE * max(1.0, self.interval() / ACTIVE_REFRESH)
            self._wake.wait(max(1.0, wait - (time.time() - last_update)))
            self._wake.clear()
            if self._stop.is_set():
                return
            if time.time() - last_update >= HISTORY_UPDATE or self._forced:
                self._forced = False
                try:
                    self.data.update()
                except Exception:
                    pass
                last_update = time.time()

    _forced = False

    def poll(self) -> None:
        """Ask for whatever is due. Never blocks when running in the background."""
        entry = self.loader.get(("snap",), self.collector.snapshot, max_age=self.interval(),
                                generation=self.refreshes)
        self.snapshot_entry = entry
        if entry.value is not None and entry.value is not self.snapshot:
            self.snapshot = entry.value
            self.snap_generation += 1
        if not self.no_history:
            self.history_info = self.loader.get(("info",), self.data.info, max_age=30,
                                                generation=self.data.generation)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self.data.stop()
        self.loader.close()

    # -- navigation --------------------------------------------------------------

    @property
    def view(self) -> View:
        return self.stack[-1]

    def go_tab(self, kind: str) -> None:
        title = dict((k, t) for k, t, _, _ in TABS)[kind]
        self.stack = [View(kind, {}, title)]

    def push(self, kind: str, params: Dict[str, Any], title: str) -> None:
        if len(self.stack) > 40:
            del self.stack[1:2]
        self.stack.append(View(kind, params, title))

    def back(self) -> None:
        if len(self.stack) > 1:
            self.stack.pop()

    def set_range(self, name: str) -> None:
        if name != self.range:
            self.range = name
            self.message = "range: " + name
            prioritize = getattr(self.data, "prioritize", None)
            if prioritize is not None and not self.no_history:
                prioritize(name)  # a longer load in progress yields to this one
            self._wake.set()

    # -- drawing -------------------------------------------------------------------

    def panes(self, width: int) -> Tuple[Context, List[Any]]:
        ctx = Context(self, width)
        builder = SCREENS.get(self.view.kind, SCREENS["overview"])
        return ctx, builder(ctx, self.view)

    def render(self, canvas: Canvas) -> None:
        width, height = canvas.width, canvas.height
        if width < 20 or height < 5:
            canvas.put(0, 0, "too small" if width < 18 else "terminal too small", BAD)
            return
        ctx, panes = self.panes(width)
        self._header(canvas, ctx)
        view = self.view
        focusable = [i for i, p in enumerate(panes) if p.focusable] or [0]
        if view.focus not in focusable:
            view.focus = focusable[0] if view.focus >= len(panes) or view.focus < 0 else min(
                focusable, key=lambda i: abs(i - view.focus))
        states = [view.panes.setdefault(p.id, PaneState()) for p in panes]
        shown = [visible_lines(p, s) for p, s in zip(panes, states)]
        body = allocate(panes, [len(lines) for lines in shown], height - 2, view.focus)
        row = 1
        self.hits = []
        for index, (pane, state, lines, size) in enumerate(zip(panes, states, shown, body)):
            if size < 0:
                continue
            chrome = 1 + (1 if pane.header is not None else 0)
            hits = draw_pane(canvas, row, chrome + size, pane, state, index == view.focus, lines)
            self.hits.extend((r, index, i) for r, i in hits)
            row += chrome + size
        self._footer(canvas)
        self._last = (panes, states, shown)
        self._last_view, self._width = view, width

    def _header(self, canvas: Canvas, ctx: Context) -> None:
        width = canvas.width
        name = ctx.snap.cluster.name if ctx.snap and ctx.snap.cluster.name else "slurm"
        col = canvas.put(0, 0, name + " ", RESOURCE)
        if self.anonymized:
            col = canvas.put(0, col, "ANON", BAD) + 1
        long_tabs = width >= 110
        self.tab_hits = []
        root = self.stack[0].kind
        for kind, title, short, _ in TABS:
            label = title if long_tabs else short
            start = col
            col = canvas.put(0, col, label, TAB if kind == root else DIM)
            self.tab_hits.append((start, col, kind))
            col += 1
        age = None
        if self.snapshot is not None:
            age = time.time() - self.snapshot.taken_at
        col = canvas.put(0, col, "│ ", DIM)
        col = canvas.put(0, col, "%s · %s" % (self.range, fmt.age(age) if age is not None
                                               else "?"), PLAIN)
        if self.data.progress:
            col = canvas.put(0, col, " · history ", DIM)
            if self.data.fraction is not None:
                col = canvas.put(0, col, fmt.bar(self.data.fraction, 6, self.ascii), WARN)
                col = canvas.put(0, col, " ", DIM)
            col = canvas.put(0, col, self.data.progress, WARN)
        elif self.loader.busy:
            col = canvas.put(0, col, " …", DIM)
        crumbs = " › ".join(v.title for v in self.stack)
        room = width - col - 3
        if room > 4 and len(self.stack) > 1:
            if fmt.width(crumbs) > room:
                crumbs = "…" + crumbs[-(room - 1):]
            canvas.put(0, col, " │ ", DIM)
            canvas.put(0, col + 3, crumbs, HEAD)

    def _footer(self, canvas: Canvas) -> None:
        row = canvas.height - 1
        if self.input is not None:
            col = canvas.put(row, 0, "filter: ", HEAD)
            col = canvas.put(row, col, "".join(self.input) + "▏", PLAIN)
            canvas.put(row, col + 2, "Enter apply · Esc cancel", DIM)
            return
        specific = FOOTERS.get(self.view.kind, "")
        items = ["↵ open", "esc back", "tab pane", "/ filter", "s sort", "[ ] range",
                 "r refresh", "? help", "q quit"]
        if specific and specific != "↵ open":
            items[3:3] = specific.split(" · ")
        must = {"? help", "q quit", "esc back"}
        room = canvas.width - 1 - (fmt.width(self.message) + 3 if self.message else 0)
        chosen = list(items)
        while chosen and fmt.width(" · ".join(chosen)) > room:
            drop = next((i for i in range(len(chosen) - 1, -1, -1) if chosen[i] not in must),
                        None)
            if drop is None:
                break
            del chosen[drop]
        col = 0
        if self.message:
            col = canvas.put(row, 0, self.message, WARN) + 1
            col = canvas.put(row, col, "│ ", DIM)
        canvas.put(row, col, " · ".join(chosen), DIM)

    # -- input -------------------------------------------------------------------

    _last_view = None  # type: Optional[View]
    _width = 80

    def _focused(self) -> Tuple[Any, PaneState, List[Any]]:
        if self._last_view is not self.view:  # keys arrived before the next draw
            _, panes = self.panes(self._width)
            states = [self.view.panes.setdefault(p.id, PaneState()) for p in panes]
            self._last = (panes, states, [visible_lines(p, st) for p, st in zip(panes, states)])
            self._last_view = self.view
        panes, states, shown = getattr(self, "_last", ([], [], []))
        index = self.view.focus
        if 0 <= index < len(panes):
            return panes[index], states[index], shown[index]
        return None, PaneState(), []

    def key(self, key: str) -> bool:
        """Handle one key; False means quit."""
        self.last_input = time.time()
        if key == "RESIZE":
            return True
        if self.input is not None:
            return self._typing(key)
        self.message = ""
        view = self.view
        pane, state, lines = self._focused()
        if key in ("q", "Q"):
            return False
        if key in ("ESC", "BACKSPACE"):
            if state.filter:
                state.filter = ""
            else:
                self.back()
        elif key in TAB_KEYS and view.kind != "help":
            self.go_tab(TAB_KEYS[key])
        elif key == "?":
            if view.kind == "help":
                self.back()
            else:
                self.push("help", {}, "Help")
        elif key in ("[", "]"):
            index = hd.RANGE_NAMES.index(self.range) + (1 if key == "]" else -1)
            self.set_range(hd.RANGE_NAMES[max(0, min(len(hd.RANGE_NAMES) - 1, index))])
        elif key in "12345" and len(key) == 1:
            self.set_range(hd.RANGE_NAMES[int(key) - 1])
        elif key == "r":
            self.refreshes += 1
            self.loader.invalidate()
            self._forced = True
            self._wake.set()
            if hasattr(self.collector, "refresh"):
                try:
                    self.collector.refresh()
                except Exception:
                    pass
            self.message = "refreshing"
        elif key in ("TAB", "BTAB"):
            self._cycle_focus(1 if key == "TAB" else -1)
        elif key in ("UP", "DOWN", "PGUP", "PGDN", "HOME", "END", "k", "J"):
            step = {"UP": -1, "k": -1, "DOWN": 1, "J": 1, "PGUP": -10, "PGDN": 10,
                    "HOME": -10 ** 9, "END": 10 ** 9}[key]
            state.cursor = max(0, min(len(lines) - 1, state.cursor + step)) if lines else 0
        elif key == "ENTER":
            self._enter(lines, state)
        elif key == "/":
            self.input = list(state.filter)
        elif key == "s" and pane is not None and pane.sorts:
            state.sort = state.sort + 1 if state.sort + 1 < len(pane.sorts) else -1
            state.cursor = 0
        elif key == "S":
            state.descending = not state.descending
        elif key == "g":
            self._group()
        elif key in ("LEFT", "RIGHT"):
            if view.kind == "timeline":
                view.extra["bin"] = view.extra.get("bin", 10 ** 6) + (1 if key == "RIGHT" else -1)
            else:
                root = TAB_KINDS.index(self.stack[0].kind) if self.stack[0].kind in TAB_KINDS else 0
                self.go_tab(TAB_KINDS[(root + (1 if key == "RIGHT" else -1)) % len(TAB_KINDS)])
        elif key in ("+", "-", "=") and view.kind == "timeline":
            zoom = view.extra.get("zoom", 1)
            view.extra["zoom"] = max(1, min(256, zoom * 2 if key in ("+", "=") else zoom // 2))
        elif key in (",", ".", "<", ">") and view.kind == "timeline":
            pan = view.extra.get("pan", 0.0) + (0.5 if key in (",", "<") else -0.5)
            view.extra["pan"] = max(0.0, min(view.extra.get("zoom", 1) - 1.0, pan))
        elif key == "E":
            self._efficiency()
        elif key in FETCH_KEYS and view.kind == "job":
            self._fetch(FETCH_KEYS[key])
        return True

    def _typing(self, key: str) -> bool:
        _, state, _ = self._focused()
        if key == "ENTER":
            state.filter = "".join(self.input or [])
            state.cursor = 0
            self.input = None
        elif key == "ESC":
            self.input = None
        elif key == "BACKSPACE":
            if self.input:
                self.input.pop()
        elif len(key) == 1 and key.isprintable() and len(self.input or []) < 60:
            self.input.append(key)
        return True

    def _cycle_focus(self, step: int) -> None:
        self._focused()
        panes = getattr(self, "_last", ([],))[0]
        focusable = [i for i, p in enumerate(panes) if p.focusable]
        if not focusable:
            return
        view = self.view
        at = focusable.index(view.focus) if view.focus in focusable else 0
        view.focus = focusable[(at + step) % len(focusable)]

    def _enter(self, lines: List[Any], state: PaneState) -> None:
        if not lines:
            return
        line = lines[max(0, min(state.cursor, len(lines) - 1))]
        if line.action is not None:
            target = line.action
            self.push(target.kind, dict(target.params), target.title)

    def _group(self) -> None:
        view = self.view
        if view.kind == "roles":
            current = view.extra.get("group", "account")
            view.extra["group"] = ROLE_GROUPS[(ROLE_GROUPS.index(current) + 1) % len(ROLE_GROUPS)]
            view.panes.clear()
        elif view.kind in ("person", "me"):
            names = [d for d, _ in hd.DIMENSIONS]
            current = view.extra.get("dim", "gpu_count")
            view.extra["dim"] = names[(names.index(current) + 1) % len(names)]
            view.panes.pop("runs", None)
        elif view.kind == "timeline":
            current = view.extra.get("lanes", "user")
            view.extra["lanes"] = LANE_GROUPS[(LANE_GROUPS.index(current) + 1) % len(LANE_GROUPS)]
            view.panes.clear()

    def _efficiency(self) -> None:
        if self.no_history:
            self.message = "history is off"
            return
        user = self.view.params.get("user") if self.view.kind in ("person", "me") else None
        name, data = self.range, self.data
        filt = hd.where(user=user) if user else None
        self.message = "loading efficiency for %s%s" % (name, " · " + user if user else "")
        self.loader.submit(lambda: data.load_efficiency(name, filt))

    # -- job detail: on-request text and file checks ----------------------------------

    def _fetch(self, kind: str) -> None:
        view = self.view
        job_id = str(view.params.get("job_id"))
        view.extra["show"] = kind
        live = self.snapshot.job(job_id) if self.snapshot is not None else None
        paths = view.extra.get("paths")
        row = view.params.get("row")
        collector, data, anonymized = self.collector, self.data, self.anonymized

        def run() -> Tuple[str, List[str], str]:
            if anonymized:
                return Status.UNAVAILABLE, [], "hidden by --anonymize"
            if kind in ("stdout", "stderr"):
                raw = None
                if paths:
                    raw = paths[1] if kind == "stdout" else (paths[3] or paths[1])
                if not raw:
                    return Status.EMPTY, [], "no log path known"
                if live is not None:
                    tail = collector.tail_log(live, raw)
                else:
                    from ..live import tail_file
                    tail = tail_file(raw, getattr(row, "uid", None))
                head = ["… earlier lines not shown"] if tail.truncated else []
                return tail.status, head + tail.lines, tail.detail
            if kind == "script":
                got = collector.batch_script(live) if live is not None else data.batch_script(job_id)
                status = got.status
                text = got.text
                detail = getattr(got, "detail", None) or getattr(got, "message", "")
                if status in Status.USABLE and text:
                    return status, ["# stored at submit (Slurm's copy; the file on disk may "
                                    "differ)"] + text.split("\n"), detail
                if status in Status.USABLE:
                    return status, [], detail or "Slurm returned an empty script"
                return status, [], detail
            got = collector.submit_line(live) if live is not None else data.submit_line(job_id)
            text = got.text
            detail = getattr(got, "detail", None) or getattr(got, "message", "")
            if got.status in Status.USABLE and text:
                workdir = (live.raw_workdir if live is not None
                           else view.extra.get("raw_workdir"))
                script = script_from_submit_line(text.split("\n")[0], workdir)
                if script:
                    view.extra["script_path"] = script
                return got.status, text.split("\n"), detail
            return got.status, [], detail

        self.loader.forget(("o", "fetch", job_id, kind))
        self.loader.get(("o", "fetch", job_id, kind), run, generation=self.refreshes)

    def path_state(self, raw: Optional[str], started: Optional[float],
                   directory: bool = False) -> Any:
        """os.stat off the UI thread (a dead network mount must not freeze the screen)."""
        if self.anonymized:
            return None
        entry = self.loader.get(("o", "stat", raw, started, directory),
                                lambda: stat_path(raw, started, directory),
                                generation=self.refreshes)
        return entry.value

    # -- mouse -------------------------------------------------------------------

    def click(self, x: int, y: int) -> None:
        self.last_input = time.time()
        if y == 0:
            for start, end, kind in self.tab_hits:
                if start <= x < end:
                    self.go_tab(kind)
            return
        now = time.time()
        for row, pane, line in self.hits:
            if row == y:
                view = self.view
                view.focus = pane
                panes, states, shown = self._last
                states[pane].cursor = line
                last_time, last_pane, last_line = self._last_click
                if now - last_time < 0.4 and (last_pane, last_line) == (pane, line):
                    self._enter(shown[pane], states[pane])
                self._last_click = (now, pane, line)
                return

    def wheel(self, step: int) -> None:
        _, state, lines = self._focused()
        if lines:
            state.cursor = max(0, min(len(lines) - 1, state.cursor + step))
