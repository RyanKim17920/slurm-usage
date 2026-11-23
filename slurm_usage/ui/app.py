"""The application state machine: navigation stack, keys, refresh, drawing.

It knows nothing about curses. ``terminal.run`` feeds it key names and blits
its Canvas; tests do the same with an in-memory canvas.

Key names: single characters, or UP DOWN LEFT RIGHT PGUP PGDN HOME END ENTER
ESC TAB BTAB BACKSPACE RESIZE.
"""

import copy
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..model import Status
from . import data as hd
from . import fmt
from . import text as textmod
from .anonymize import AnonCollector, AnonHistory, Anonymizer
from . import charts
from . import theme
from .canvas import (ACCENT, BAD, DIM, FAINT, GOOD, HEAD, KEY, PLAIN, RESOURCE, SEL, TAB,
                     WARN, Canvas, bold, on)
from .filestat import check as stat_path
from .filestat import script_from_submit_line
from .loader import Loader
from . import screens
from .screens import FOOTERS, LANE_GROUPS, ROLE_GROUPS, SCREENS, TABS, Context, View
from .widgets import chrome as pane_chrome
from .widgets import PaneState, Target, allocate, close_box, draw_pane, visible_lines

ACTIVE_REFRESH = 30.0
MAX_REFRESH = 300.0
IDLE_AFTER = 120.0
HISTORY_UPDATE = 300.0
MESSAGE_SECONDS = 8.0  # how long a completion message stays in the footer
TAB_KINDS = tuple(kind for kind, _, _, _ in TABS)
TAB_KEYS = {key: kind for kind, _, _, key in TABS}
FETCH_KEYS = {"l": "stdout", "e": "stderr", "b": "script", "u": "submit"}
HISTORY_MODES = ("full", "recent", "off")
# Settings, the same on every screen: key -> what it switches
TOGGLE_KEYS = {"A": "anonymize", "C": "colors", "U": "glyphs", "H": "history"}
SETTINGS_KEY = "O"
CLEAR_CACHE_KEY = "X"
_PERSON_PARAMS = {"user": "user", "account": "account", "node": "node"}
_VOCAB_PARAMS = ("qos", "partition")


class App:
    def __init__(self, collector: Any, data: hd.HistoryData, range_name: str = "week",
                 user: Optional[str] = None, account: Optional[str] = None,
                 ascii_only: bool = False, background: bool = True, anonymized: bool = False,
                 no_history: bool = False, start: str = "overview",
                 history: Optional[str] = None, color: bool = True) -> None:
        # Anonymizing is switchable: keep the real collector and history, and one
        # Anonymizer for the whole run so a pseudonym never changes.
        raw_collector, anon = collector, None
        if isinstance(collector, AnonCollector):
            raw_collector, anon = collector._collector, collector._anon
        raw_history = data.history
        if isinstance(raw_history, AnonHistory):
            anon = anon or raw_history._anon
            raw_history = raw_history._history
        self.anon = anon or Anonymizer()
        self._raw_collector, self._raw_history = raw_collector, raw_history
        self._anon_collector = AnonCollector(raw_collector, self.anon)
        self._raw_snapshot = None  # type: Any
        self.anonymized = bool(anonymized)
        self.collector = self._anon_collector if self.anonymized else raw_collector
        self.data = data
        data.history = AnonHistory(raw_history, self.anon) if self.anonymized else raw_history
        self.range = range_name
        self.user = user
        self.ascii = ascii_only
        self.color = color
        self.history_mode = history or ("off" if no_history else
                                        "full" if getattr(data, "full_history", True) else
                                        "recent")
        data.full_history = self.history_mode == "full"
        if hasattr(data, "pause"):
            data.pause(self.history_mode == "off")
        self.overlay = None  # type: Optional[str]  # "settings" while the box is open
        self.confirm = None  # type: Optional[str]  # an action waiting for y
        self.background = background
        self.loader = Loader(workers=3 if background else 0)
        self.snapshot = None  # type: Any
        self.snapshot_entry = None  # type: Any
        self.snap_generation = 0
        self.refreshes = 0
        self.history_info = None  # type: Any
        self.last_input = time.time()
        self.message = ""
        self._flash = None  # type: Optional[Tuple[str, float]]  # message that expires
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

    @property
    def no_history(self) -> bool:
        return self.history_mode == "off"

    def start(self) -> None:
        if self.background:  # also when off: H can switch history on later
            self._history_thread = threading.Thread(target=self._history_loop, daemon=True,
                                                    name="history")
            self._history_thread.start()
        elif not self.no_history:
            self.data.load_in_order(self.range)

    def _history_loop(self) -> None:
        last_update = time.time()
        while not self._stop.is_set():
            if self.no_history:  # switched off: wait to be switched on (or stopped)
                self._wake.wait()
                self._wake.clear()
                continue
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
        # always the real snapshot: the pseudonymous one is derived from it, so switching
        # anonymizing on or off needs no new query
        entry = self.loader.get(("snap",), self._raw_collector.snapshot,
                                max_age=self.interval(), generation=self.refreshes)
        self.snapshot_entry = entry
        if entry.value is not None and entry.value is not self._raw_snapshot:
            self._raw_snapshot = entry.value
            self._derive_snapshot()

    def _derive_snapshot(self) -> None:
        raw = self._raw_snapshot
        if raw is None:
            self.snapshot = None
        elif self.anonymized:
            self.snapshot = self.anon.snapshot(copy.deepcopy(raw))
        else:
            self.snapshot = raw
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
        if kind == "range" and params.get("range") in hd.RANGE_NAMES:
            self.set_range(params["range"])  # range detail opens on the range clicked
        self.stack.append(View(kind, params, title))

    def step_range(self, step: int) -> None:
        index = hd.RANGE_NAMES.index(self.range) + step
        self.set_range(hd.RANGE_NAMES[max(0, min(len(hd.RANGE_NAMES) - 1, index))])

    def back(self) -> None:
        if len(self.stack) > 1:
            self.stack.pop()

    def set_range(self, name: str) -> None:
        if name != self.range:
            self.range = name
            view = self.stack[-1] if self.stack else None
            if view is not None and view.kind == "range" and " · " in view.title:
                view.title = name.capitalize() + " · " + view.title.split(" · ", 1)[1]
            self.message = "range: " + name
            prioritize = getattr(self.data, "prioritize", None)
            if prioritize is not None and not self.no_history:
                prioritize(name)  # a longer load in progress yields to this one
            self._wake.set()

    # -- drawing -------------------------------------------------------------------

    def panes(self, width: int, height: Optional[int] = None) -> Tuple[Context, List[Any]]:
        ctx = Context(self, width, height if height is not None else self._height)
        builder = SCREENS.get(self.view.kind, SCREENS["overview"])
        return ctx, builder(ctx, self.view)

    def render(self, canvas: Canvas) -> None:
        width, height = canvas.width, canvas.height
        if width < 20 or height < 5:
            canvas.put(0, 0, "too small" if width < 18 else "terminal too small", BAD)
            return
        self._height = height
        self.click_targets = []  # (row, start, end, Target or ("range", name)) drawn this frame
        ctx, panes = self.panes(width, height)
        self._header(canvas, ctx)
        view = self.view
        focusable = [i for i, p in enumerate(panes) if p.focusable] or [0]
        wanted = view.extra.pop("focus_pane", None)  # e.g. a log tail just asked for
        if wanted is not None:
            view.focus = next((i for i, p in enumerate(panes) if p.id == wanted), view.focus)
        if view.focus not in focusable:
            view.focus = focusable[0] if view.focus >= len(panes) or view.focus < 0 else min(
                focusable, key=lambda i: abs(i - view.focus))
        states = [view.panes.setdefault(p.id, PaneState()) for p in panes]
        shown = [visible_lines(p, s) for p, s in zip(panes, states)]
        self._reselect(view, panes, states, shown)
        for pane, state, lines in zip(panes, states, shown):
            keep_selection(pane, state, lines)
        # a group shows one of its panes: the focused one, else the one last focused
        visible = []  # type: List[int]
        members = {}  # type: Dict[str, List[int]]
        for index, pane in enumerate(panes):
            if pane.group:
                members.setdefault(pane.group, []).append(index)
        for group, indices in members.items():
            if view.focus in indices:
                view.extra["tab:" + group] = panes[view.focus].id
        for index, pane in enumerate(panes):
            if pane.group:
                indices = members[pane.group]
                active = view.extra.get("tab:" + pane.group)
                chosen = next((i for i in indices if panes[i].id == active), indices[0])
                if index != chosen:
                    continue
            visible.append(index)
        framed = any(panes[i].frame for i in visible)
        focus_at = visible.index(view.focus) if view.focus in visible else 0
        counts = [panes[i].fixed or len(shown[i]) if panes[i].draw is not None
                  else len(shown[i]) for i in visible]
        room = height - 2 - (1 if framed else 0)
        body = allocate([panes[i] for i in visible], counts, room, focus_at)
        self._page = max(1, body[focus_at] if 0 <= focus_at < len(body) else 10)
        glyphs = theme.box(self.ascii)
        row = 1
        self.hits = []
        self.group_hits = []  # type: List[Tuple[int, int, int, int]]
        edge = "top"
        last_framed = None  # type: Optional[int]
        for slot, (index, size) in enumerate(zip(visible, body)):
            pane, state, lines = panes[index], states[index], shown[index]
            if size < 0:
                continue
            chrome = pane_chrome(pane)
            tabs = None
            tab_hits = []  # type: List[Tuple[int, int, Any]]
            if pane.group and len(members[pane.group]) > 1:
                tabs = [(panes[i], i == index) for i in members[pane.group]]
            hits = draw_pane(canvas, row, chrome + size, pane, state, index == view.focus,
                             lines, edge, glyphs, tabs, tab_hits)
            for start, end, pane_id in tab_hits:
                target = next((i for i in members.get(pane.group or "", ()) if
                               panes[i].id == pane_id), None)
                if target is not None:
                    self.group_hits.append((row, start, end, target))
            for hit in hits:
                self.hits.append((hit[0], index, hit[1]) + tuple(hit[2:]))
            if pane.frame:
                edge = "join"
                last_framed = index
            else:
                edge = "top"
            row += chrome + size
        if framed and row < height - 1:
            close_box(canvas, row, glyphs, last_framed == view.focus)
        for state, lines in zip(states, shown):
            at = state.cursor
            state.mark = (at, lines[at].key) if 0 <= at < len(lines) else None
        self._footer(canvas)
        self._last = (panes, states, shown)
        self._last_view, self._width = view, width
        if self.overlay == "settings":
            self._settings_box(canvas)
        if self.ascii:
            canvas.map_cells(textmod.ascii_cell)

    def _header(self, canvas: Canvas, ctx: Context) -> None:
        width = canvas.width
        name = ctx.snap.cluster.name if ctx.snap and ctx.snap.cluster.name else "slurm"
        root = self.stack[0].kind
        crumbs = [v.title for v in self.stack[1:]]
        age = None
        if self.snapshot is not None:
            age = time.time() - self.snapshot.taken_at

        # the right side, richest first: every range with the chosen one marked, then
        # only the chosen one
        def right(full: bool, clock: bool = True) -> List[Tuple[Any, ...]]:
            parts = []  # type: List[Tuple[Any, ...]]
            if self.data.progress and not self.no_history:
                parts.append(("history ", DIM))
                if self.data.fraction is not None:
                    parts += charts.gauge(self.data.fraction, 6, self.ascii, ACCENT)
                    parts.append((" ", PLAIN))
                parts += [(self.data.progress, DIM), ("  ", PLAIN)]
            elif self.loader.busy:
                parts += [("…", DIM), ("  ", PLAIN)]
            if self.history_mode != "full":
                parts += [("history " + self.history_mode, WARN), ("  ", PLAIN)]
            if full:
                for number, option in enumerate(hd.RANGE_NAMES):
                    if number:
                        parts.append((" ", PLAIN))
                    parts.append((option, KEY if option == self.range else FAINT, option))
            else:
                parts.append((self.range, KEY, self.range))
            if clock:
                parts += [("  ", PLAIN), (fmt.age(age) if age is not None else "?", DIM)]
            return parts + [(" ", PLAIN)]

        def tabs(long: bool) -> List[Tuple[str, str, str]]:
            out = []
            for kind, title, short, _ in TABS:
                out.append((" %s " % (title if long else short),
                            TAB if kind == root else DIM, kind))
            return out

        lead = [(" ◆ ", ACCENT), (name, bold(RESOURCE)), (" ", PLAIN)]
        if self.anonymized:
            lead += [("ANON", bold(BAD)), (" ", PLAIN)]
        lead.append((" ", PLAIN))
        trail = [(" › " + " › ".join(crumbs), HEAD)] if crumbs else []
        # what gives way first: long tab names, the age, then the full list of ranges
        options = [(True, True, True), (False, True, True), (False, True, False),
                   (True, False, True), (False, False, True), (False, False, False)]
        last = (" › " + crumbs[-1]) if crumbs else ""
        for long_tabs, full_range, clock in options:
            tab_parts = tabs(long_tabs)
            used = (charts.width(lead) + sum(fmt.width(t) for t, _, _ in tab_parts)
                    + sum(fmt.width(p[0]) for p in right(full_range, clock)))
            need = min(24, fmt.width(last)) if crumbs else 0
            if width - used - 1 >= need:
                break
        col = canvas.segments(0, 0, lead)
        self.tab_hits = []
        for text, style, kind in tab_parts:
            start = col
            col = canvas.put(0, col, text, style)
            self.tab_hits.append((start, col, kind))
        right_parts = right(full_range, clock)
        right_width = sum(fmt.width(p[0]) for p in right_parts)
        if crumbs:
            room = width - col - right_width - 1
            text = trail[0][0]
            if fmt.width(text) > room:  # the deepest crumb is the one that says where
                text = last
            if room > 4:
                if fmt.width(text) > room:
                    text = fmt.fit(text, room)
                canvas.put(0, col, text, HEAD)
        col = max(col, width - right_width)
        self.range_hits = []  # type: List[Tuple[int, int, str]]
        for part in right_parts:
            start = col
            col = canvas.put(0, col, part[0], part[1])
            if len(part) > 2 and col > start:
                self.range_hits.append((start, col, part[2]))

    def _footer(self, canvas: Canvas) -> None:
        row = canvas.height - 1
        if self.input is not None:
            col = canvas.put(row, 0, " / ", KEY)
            col = canvas.put(row, col, "filter ", DIM)
            text = "".join(self.input)
            # the previous filter is selected: typing replaces it
            col = canvas.put(row, col, text, on(PLAIN, SEL) if self._replace and text
                             else bold(PLAIN))
            col = canvas.put(row, col, "▏", ACCENT)
            canvas.segments(row, col + 2, [("filters as you type  ", DIM), ("↵", KEY),
                                           (" keep  ", DIM), ("esc", KEY), (" cancel  ", DIM),
                                           ("^U", KEY), (" clear", DIM)])
            return
        flash = self._flash
        if flash is not None and (self.message != flash[0] or time.time() >= flash[1]):
            if self.message == flash[0]:
                self.message = ""
            self._flash = None
        items = [("↵", "open"), ("esc", "back"), ("tab", "pane"), ("/", "filter"),
                 ("s", "sort"), ("[ ]", "range"), ("r", "refresh"), ("O", "settings"),
                 ("?", "help"), ("q", "quit")]
        specific = [item for item in FOOTERS.get(self.view.kind, ()) if item[0] != "↵"]
        if any(what == "range" for _, what in specific):
            items = [item for item in items if item[1] != "range"]
        items[3:3] = specific
        must = {"?", "q", "esc"}
        room = canvas.width - 1 - (fmt.width(self.message) + 5 if self.message else 0)

        def size(chosen: List[Tuple[str, str]]) -> int:
            return sum(fmt.width(k) + 1 + fmt.width(d) for k, d in chosen) + 2 * len(chosen)

        chosen = list(items)
        while chosen and size(chosen) > room:
            drop = next((i for i in range(len(chosen) - 1, -1, -1) if chosen[i][0] not in must),
                        None)
            if drop is None:
                break
            del chosen[drop]
        col = 0
        if self.message:
            col = canvas.put(row, 1, "● ", WARN)
            col = canvas.put(row, col, self.message, WARN) + 1
            col = canvas.put(row, col, "│", FAINT)
        col += 1
        for key, what in chosen:
            col = canvas.put(row, col, key, KEY)
            col = canvas.put(row, col, " " + what + "  ", DIM)

    # -- input -------------------------------------------------------------------

    _last_view = None  # type: Optional[View]
    _width = 80
    _height = 40
    click_targets = []  # type: List[Tuple[int, int, int, Any]]
    range_hits = []  # type: List[Tuple[int, int, str]]

    def _focused(self) -> Tuple[Any, PaneState, List[Any]]:
        if self._last_view is not self.view:  # keys arrived before the next draw
            _, panes = self.panes(self._width, self._height)
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
        if self.confirm is not None:
            action, self.confirm = self.confirm, None
            if key in ("y", "Y") and action == "clear_cache":
                self.clear_cache()
            else:
                self.message = "cache kept"
            return True
        if key in TOGGLE_KEYS:
            self.toggle(TOGGLE_KEYS[key])
            return True
        if self.overlay is not None:
            if key in (SETTINGS_KEY, "ESC", "BACKSPACE", "q", "Q", "ENTER"):
                self.overlay = None
            elif key in (CLEAR_CACHE_KEY, CLEAR_CACHE_KEY.lower()):
                self.confirm = "clear_cache"
            return True
        if key == SETTINGS_KEY:
            self.overlay = "settings"
            return True
        view = self.view
        pane, state, lines = self._focused()
        if key in ("q", "Q"):
            return False
        if key in ("ESC", "BACKSPACE"):
            if state.filter:
                state.filter = ""
            else:
                self.back()
        elif key in TAB_KEYS and view.kind != "help" and not (
                key == "j" and view.kind == "runtype"):  # j there: its jobs
            self.go_tab(TAB_KEYS[key])
        elif key == "?":
            if view.kind == "help":
                self.back()
            else:
                self.push("help", {}, "Help")
        elif key in ("[", "]"):
            self.step_range(1 if key == "]" else -1)
        elif key in ("UP", "DOWN", "LEFT", "RIGHT") and pane is not None and pane.range_cursor:
            self.step_range(1 if key in ("DOWN", "RIGHT") else -1)  # the tiles: cursor = range
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
            page = max(1, getattr(self, "_page", 11) - 1)  # a page, keeping one line
            step = {"UP": -1, "k": -1, "DOWN": 1, "J": 1, "PGUP": -page, "PGDN": page,
                    "HOME": -10 ** 9, "END": 10 ** 9}[key]
            state.cursor = max(0, min(len(lines) - 1, state.cursor + step)) if lines else 0
            state.chosen = True
            # a log tail keeps following its end until moved away from it
            state.follow = bool(pane is not None and pane.follow and key == "END")
            if pane is not None and pane.more and lines and state.cursor >= len(lines) - 1 \
                    and step > 0:
                view.extra["jobs_limit"] = view.extra.get("jobs_limit", hd.JOB_PAGE) + hd.JOB_PAGE
                self.message = "loading more jobs"
        elif key == "ENTER":
            self._enter(lines, state, pane)
        elif key == "/":
            self.input = list(state.filter)
            self._replace = True  # the old filter is selected: typing replaces it
            self._before = state.filter
        elif key == "s" and pane is not None and pane.sorts:
            state.sort = state.sort + 1 if state.sort + 1 < len(pane.sorts) else -1
            state.cursor = 0
        elif key == "S":
            state.descending = not state.descending
        elif key == "g":
            self._group()
        elif key == "j" and view.kind == "runtype":
            target = screens.runtype_jobs_target(view)
            self.push(target.kind, dict(target.params), target.title)
        elif key in ("LEFT", "RIGHT") and view.kind in ("range", "runtype"):
            self.step_range(1 if key == "RIGHT" else -1)
        elif key in ("LEFT", "RIGHT"):
            root = TAB_KINDS.index(self.stack[0].kind) if self.stack[0].kind in TAB_KINDS else 0
            self.go_tab(TAB_KINDS[(root + (1 if key == "RIGHT" else -1)) % len(TAB_KINDS)])
        elif key == "c" and view.kind == "timeline":
            metric = view.extra.get("metric") or "gpu"
            view.extra["metric"] = "cpu" if metric == "gpu" else "gpu"
            view.panes.clear()
        elif key == "v" and view.kind == "range":
            view.extra["hide_previous"] = not view.extra.get("hide_previous")
            self.message = "previous period " + ("hidden" if view.extra["hide_previous"]
                                                 else "shown")
        elif key == "E":
            self._efficiency()
        elif key in FETCH_KEYS and view.kind == "job":
            self._fetch(FETCH_KEYS[key])
        return True

    _replace = False
    _before = ""

    def _typing(self, key: str) -> bool:
        """The filter applies as it is typed (rows are filtered in memory; a jobs list
        also asks history). Esc puts the previous filter back."""
        _, state, _ = self._focused()
        text = self.input if self.input is not None else []
        if key == "ENTER":
            state.filter = "".join(text)
            state.cursor = 0
            self.input = None
            return True
        if key == "ESC":
            state.filter = self._before
            self.input = None
            return True
        if key in ("CTRL_U",) or (key == "BACKSPACE" and self._replace):
            del text[:]
        elif key == "BACKSPACE":
            if text:
                text.pop()
        elif len(key) == 1 and key.isprintable():
            if self._replace:
                del text[:]
            if len(text) < 60:
                text.append(key)
        else:
            return True
        self._replace = False
        state.filter = "".join(text)
        state.cursor = 0
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

    def _enter(self, lines: List[Any], state: PaneState, pane: Any = None) -> None:
        if not lines:
            return
        line = lines[max(0, min(state.cursor, len(lines) - 1))]
        if pane is not None and pane.range_cursor:  # the tiles: the range is the cursor
            line = next((l for l in lines if l.key == self.range), line)
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
            names = list(screens.PAGE_DIMENSIONS)
            current = view.extra.get("dim", "gpu_count")
            view.extra["dim"] = names[(names.index(current) + 1) % len(names)]
            view.panes.pop("runs", None)
        elif view.kind == "range":
            metric = view.extra.get("metric") or "gpu"
            view.extra["metric"] = "cpu" if metric == "gpu" else "gpu"
            self.message = "showing " + view.extra["metric"].upper() + "s"
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
        scope = "%s%s" % (name, " · " + user if user else "")
        loading = "loading efficiency for " + scope
        self.message = loading

        def work() -> None:
            try:
                count = data.load_efficiency(name, filt)
                done = ("efficiency loaded for %s: %s jobs read" % (scope, fmt.num(count))
                        if count else "efficiency already loaded for " + scope)
            except Exception as error:
                done = "efficiency for %s: %s" % (scope, getattr(error, "message", None)
                                                  or error)
            if self.message == loading:  # not replaced by something newer meanwhile
                self.flash(done)
        self.loader.submit(work)

    def flash(self, text: str, seconds: float = MESSAGE_SECONDS) -> None:
        """A footer message that clears itself after ``seconds``."""
        self.message = text
        self._flash = (text, time.time() + seconds)

    # -- job detail: on-request text and file checks ----------------------------------

    def _fetch(self, kind: str) -> None:
        view = self.view
        job_id = str(view.params.get("job_id"))
        view.extra["show"] = kind
        live = self.snapshot.job(job_id) if self.snapshot is not None else None
        paths = view.extra.get("paths")
        row = view.params.get("row")
        collector, data, anonymized = self.collector, self.data, self.anonymized
        loader = self.loader

        def is_batch() -> bool:
            """Known to be an sbatch job: a batch step, or an sbatch submit line."""
            for key in (("o", "detail", job_id), ("l", "steps", job_id)):
                found = loader.peek(key)
                got = found.value if found is not None else None
                for att in getattr(got, "attempts", None) or [got]:
                    if any(getattr(s, "step_id", None) == "batch"
                           for s in getattr(att, "steps", None) or ()):
                        return True
            submit = loader.peek(("o", "fetch", job_id, "submit"))
            if submit is not None and submit.value is not None:
                lines = submit.value[1]
                return bool(lines) and lines[0].lstrip().split(" ")[0].endswith("sbatch")
            return False

        def run() -> Tuple[str, List[str], str]:
            if anonymized:
                return Status.UNAVAILABLE, [], "hidden while anonymized"
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
                if status == Status.NOT_STORED and is_batch():
                    from ..parse import SCRIPT_NOT_STORED_BATCH
                    detail = SCRIPT_NOT_STORED_BATCH  # never "not a batch job" for an sbatch job
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
        view.panes.pop("output", None)  # a new text opens fresh (a tail at its end)
        view.extra["focus_pane"] = "output"
        self.loader.get(("o", "fetch", job_id, kind), run, generation=self.refreshes)

    def path_state(self, raw: Optional[str], since: Optional[float],
                   directory: bool = False, running: bool = False,
                   what: str = "the job ended") -> Any:
        """os.stat off the UI thread (a dead network mount must not freeze the screen)."""
        if self.anonymized:
            return None
        entry = self.loader.get(("o", "stat", raw, since, directory, running, what),
                                lambda: stat_path(raw, since, directory, running, what),
                                generation=self.refreshes)
        return entry.value

    # -- settings: anonymize, colors, glyphs, history ------------------------------

    def toggle(self, what: str) -> None:
        if what == "anonymize":
            self.set_anonymized(not self.anonymized)
        elif what == "colors":
            self.color = not self.color
            self.message = "colors " + ("on" if self.color else "off")
        elif what == "glyphs":
            self.ascii = not self.ascii
            self.message = "glyphs: " + ("ASCII" if self.ascii else "Unicode")
        elif what == "history":
            at = HISTORY_MODES.index(self.history_mode)
            self.set_history(HISTORY_MODES[(at + 1) % len(HISTORY_MODES)])

    def set_history(self, mode: str) -> None:
        if mode not in HISTORY_MODES or mode == self.history_mode:
            return
        self.history_mode = mode
        data = self.data
        data.full_history = mode == "full"
        if mode == "off":
            data.pause(True)  # stops the load in progress; what it committed stays cached
            self.message = "history off: live view only (H to switch on)"
        else:
            data.pause(False)
            if mode == "recent" and getattr(data, "_loading", None) == "all" \
                    and self.range != "all":
                data.stop_current()  # all of history is no longer wanted in the background
            self.message = "history: " + ("all of it in the background" if mode == "full"
                                          else "the chosen range and shorter")
            if not self.background:
                data.load_in_order(self.range)
        self.loader.invalidate()
        self._wake.set()

    def set_anonymized(self, on: bool) -> None:
        """Switch pseudonyms on or off at once: every cached answer is dropped, open file
        text is closed, and the breadcrumbs, filters and selections are mapped."""
        if on == self.anonymized:
            return
        selected = self._selection()
        self.anonymized = on
        self.collector = self._anon_collector if on else self._raw_collector
        self.data.history = (AnonHistory(self._raw_history, self.anon) if on
                             else self._raw_history)
        self.loader.clear()  # nothing computed in the other mode may be shown
        self.history_info = None
        with self.data._lock:
            self.data.generation += 1
        self._derive_snapshot()  # first: it makes every name on screen known
        for view in self.stack:
            self._remap_view(view, on)
        if self.user:
            self.user = (self.anon.name("user", self.user) if on
                         else self.anon.real("user", self.user))
        if selected is not None:
            view, pane_id, params = selected
            view.extra["reselect"] = (pane_id, self._map_params(params, on))
        self._last_view = None
        self.confirm = None
        self.message = ("anonymize on: pseudonyms; paths, scripts and logs hidden" if on
                        else "anonymize off")

    def _map_params(self, params: Dict[str, Any], on: bool) -> Dict[str, Any]:
        anon, out = self.anon, {}
        for key, value in params.items():
            if value is None:
                pass
            elif key in _PERSON_PARAMS and isinstance(value, str):
                kind = _PERSON_PARAMS[key]
                value = anon.name(kind, value) if on else anon.real(kind, value)
            elif key in _VOCAB_PARAMS and isinstance(value, str):
                value = anon.vocab(value) if on else anon.vocab_real(value)
            elif key == "row":
                value = anon.row(value) if on else anon.row_real(value)
            elif key == "rows":
                value = [anon.row(r) if on else anon.row_real(r) for r in value]
            elif key == "run_type" and isinstance(value, tuple) and len(value) == 2:
                value = (value[0], anon.run_type(value[0], value[1]) if on
                         else anon.run_type_real(value[0], value[1]))
            out[key] = value
        return out

    def _map_filter(self, text: str, on: bool) -> str:
        """A typed filter: a whole name maps to its pseudonym (and back); a job id stays;
        any other text typed against real names is dropped when anonymizing."""
        if not text:
            return text
        word = text.strip()
        anon = self.anon
        if on:
            for kind in ("user", "account", "node", "cluster", "reservation"):
                if word in anon.forward[kind]:
                    return anon.forward[kind][word]
            if word.isdigit() or any(word in anon.backward[k] for k in anon.backward):
                return text
            family = hd.name_family(word)
            if family in anon.forward["job"]:
                return anon.forward["job"][family]
            return ""
        for kind in ("user", "account", "node", "cluster", "reservation"):
            if word in anon.backward[kind]:
                return anon.backward[kind][word]
        return text

    def _remap_view(self, view: View, on: bool) -> None:
        for key in ("paths", "raw_workdir", "script_path", "show"):  # file text is closed
            view.extra.pop(key, None)
        view.panes.pop("output", None)
        if on:
            mapped = self._map_params(view.params, True)
            filters = {}
            for pane_id, state in view.panes.items():
                new = self._map_filter(state.filter, True)
                filters[pane_id] = (state.filter, new)
                state.filter = new
            view.extra["plain"] = (dict(view.params), view.title, filters)
            view.params = mapped
            view.title = self.anon.title(view.title) or view.title
            return
        plain = view.extra.pop("plain", None)
        if plain is not None:
            params, title, filters = plain
            if view.kind in ("me",):
                params = self._map_params(view.params, False)
            view.params, view.title = params, title
            for pane_id, state in view.panes.items():
                old, new = filters.get(pane_id, (None, None))
                state.filter = old if old is not None and state.filter == new \
                    else self._map_filter(state.filter, False)
        else:  # opened while anonymized
            view.params = self._map_params(view.params, False)
            view.title = self.anon.title_real(view.title) or view.title
            for state in view.panes.values():
                state.filter = self._map_filter(state.filter, False)

    def _selection(self) -> Optional[Tuple[View, str, Dict[str, Any]]]:
        """The focused row of the current screen, by what it opens (it survives a change
        of names and order)."""
        if self._last_view is not self.view:
            return None
        panes, states, shown = getattr(self, "_last", ([], [], []))
        index = self.view.focus
        if not 0 <= index < len(panes) or not shown[index]:
            return None
        line = shown[index][max(0, min(states[index].cursor, len(shown[index]) - 1))]
        if line.action is None:
            return None
        return self.view, panes[index].id, dict(line.action.params, _kind=line.action.kind)

    def _reselect(self, view: View, panes: List[Any], states: List[PaneState],
                  shown: List[List[Any]]) -> None:
        wanted = view.extra.get("reselect")
        if wanted is None:
            return
        pane_id, params = wanted
        for pane, state, lines in zip(panes, states, shown):
            if pane.id != pane_id or not lines:
                continue
            for number, line in enumerate(lines):
                action = line.action
                if action is not None and dict(action.params, _kind=action.kind) == params:
                    state.cursor = number
                    state.chosen = True
                    break
            view.extra.pop("reselect", None)

    def cache_location(self) -> str:
        if self.anonymized:
            return "hidden while anonymized"
        info = self.history_info.value if self.history_info is not None else None
        if info is None:
            return "not known yet" if not self.no_history else "not known (history is off)"
        path = getattr(info, "cache_path", None)
        return path or "none: history is kept in memory for this run"

    def clear_cache(self) -> None:
        """Empty the history cache (it is rebuilt from Slurm), off the UI thread."""
        raw, data = self._raw_history, self.data
        rebuild = getattr(raw, "rebuild", None)
        if rebuild is None:
            self.message = "there is no history cache to clear"
            return
        data.pause(True)  # a load in progress stops before the cache is emptied

        def work() -> None:
            try:
                rebuild()
                self.message = "history cache cleared; reloading"
            except Exception as error:
                self.message = "could not clear the cache: %s" % (
                    getattr(error, "message", None) or error)
            finally:
                data.forget()
                data.pause(self.no_history)
                self.loader.clear()
                self.history_info = None
                self._wake.set()
                if not self.background and not self.no_history:
                    data.load_in_order(self.range)
        self.message = "clearing the history cache"
        self.loader.submit(work)

    def settings_rows(self) -> List[Tuple[str, str, str]]:
        """(key, setting, state) for every setting."""
        return [
            ("A", "anonymize", "on" if self.anonymized else "off"),
            ("C", "colors", "on" if self.color else "off"),
            ("U", "glyphs", "ASCII" if self.ascii else "Unicode"),
            ("H", "history", {"full": "full (all of it, in the background)",
                              "recent": "recent (the chosen range and shorter)",
                              "off": "off (live view only)"}[self.history_mode]),
        ]

    def _settings_box(self, canvas: Canvas) -> None:
        rows = [("Settings", HEAD)]
        for key, name, state in self.settings_rows():
            rows.append(("%s  %-10s %s" % (key, name, state), PLAIN))
        rows.append(("", PLAIN))
        where = self.cache_location()
        room = max(10, canvas.width - 8 - 14)
        if fmt.width(where) > room:  # the end of a long path says the most
            where = "…" + where[-(room - 1):]
        rows.append(("   cache      " + where, DIM))
        rows.append(("   (moved with $SLURM_USAGE_CACHE)", DIM))
        if self.confirm == "clear_cache":
            rows.append(("X  clear cache? y = yes, any other key = no", WARN))
        else:
            rows.append(("X  clear cache (asks first)", PLAIN))
        rows.append(("", PLAIN))
        rows.append(("O or Esc closes", DIM))
        inner = min(canvas.width - 4, max(fmt.width(t) for t, _ in rows[1:]) + 4)
        height = min(canvas.height - 2, len(rows) + 1)
        if inner < 10 or height < 4:
            return
        top = max(1, (canvas.height - height) // 2)
        left = max(0, (canvas.width - inner - 2) // 2)
        glyphs = theme.box(self.ascii)
        title = " Settings "
        canvas.put(top, left, glyphs.tl + glyphs.h + " " * len(title)
                   + glyphs.h * max(0, inner - 1 - len(title)) + glyphs.tr, ACCENT)
        canvas.put(top, left + 2, title, KEY)
        for number in range(height - 2):
            row = top + 1 + number
            text, style = rows[number + 1] if number + 1 < len(rows) else ("", PLAIN)
            canvas.put(row, left, glyphs.v + " " * inner + glyphs.v, ACCENT)
            canvas.restyle(row, left + 1, left + 1 + inner, PLAIN)
            if len(text) > 3 and text[0] != " " and text[1:3] == "  ":
                canvas.put(row, left + 2, text[0], KEY)
                canvas.put(row, left + 3, text[1:], style, limit=inner - 3)
            else:
                canvas.put(row, left + 2, text, style, limit=inner - 2)
        canvas.put(top + height - 1, left, glyphs.bl + glyphs.h * inner + glyphs.br, ACCENT)

    # -- mouse -------------------------------------------------------------------

    def click(self, x: int, y: int) -> None:
        """A click: a tab, a range in the header, a range tab or tile, the hero; on a
        row it selects, and a second click on the selected row (or a double click) opens
        it. Tiles open on one click."""
        self.last_input = time.time()
        if self.overlay is not None:  # a click outside the box closes it
            self.overlay = None
            return
        if y == 0:
            for start, end, kind in self.tab_hits:
                if start <= x < end:
                    self.go_tab(kind)
                    return
            for start, end, name in self.range_hits:
                if start <= x < end:
                    self.set_range(name)
                    return
            return
        for row, start, end, what in self.click_targets:
            if row == y and start <= x < end:
                if isinstance(what, tuple) and len(what) == 2 and what[0] == "range":
                    self.set_range(what[1])
                elif isinstance(what, Target):
                    self.push(what.kind, dict(what.params), what.title)
                return
        for row, start, end, pane in getattr(self, "group_hits", ()):
            if row == y and start <= x < end:
                self.view.focus = pane
                return
        now = time.time()
        for hit in self.hits:
            row, pane, line = hit[:3]
            if len(hit) > 3 and not hit[3] <= x < hit[4]:
                continue
            if row == y:
                view = self.view
                panes, states, shown = self._last
                # a second click opens: on a row the viewer selected (by a click or a key),
                # not on the row the cursor merely starts on, so one click never opens
                # one row and selects another
                was = (view.focus == pane and states[pane].cursor == line
                       and states[pane].chosen)
                view.focus = pane
                states[pane].cursor = line
                states[pane].chosen = True
                last_time, last_pane, last_line = self._last_click
                double = now - last_time < 0.4 and (last_pane, last_line) == (pane, line)
                if panes[pane].click_opens or was or double:
                    self._last_click = (0.0, -1, -1)
                    self._enter(shown[pane], states[pane])
                    return
                self._last_click = (now, pane, line)
                return

    def wheel(self, step: int, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Scroll the pane under the pointer (else the focused one)."""
        self.last_input = time.time()
        if y is not None:
            for hit in self.hits:
                if hit[0] == y and (len(hit) <= 3 or x is None or hit[3] <= x < hit[4]):
                    self.view.focus = hit[1]
                    break
        pane, state, lines = self._focused()
        if pane is not None and pane.range_cursor:
            self.step_range(1 if step > 0 else -1)
            return
        if lines:
            state.cursor = max(0, min(len(lines) - 1, state.cursor + step))
            state.chosen = True


def keep_selection(pane: Any, state: PaneState, lines: List[Any]) -> None:
    """Rows reordered or added since the last frame (a refresh): the cursor follows the
    row it was on, found by its key, unless the cursor itself moved since."""
    mark = state.mark
    if mark is None or pane.range_cursor or not lines or not state.chosen:
        return
    cursor, key = mark
    if key is None or state.cursor != cursor:
        return
    if 0 <= cursor < len(lines) and lines[cursor].key == key:
        return
    for index, line in enumerate(lines):
        if line.key == key:
            state.cursor = index
            return
