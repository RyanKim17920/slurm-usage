"""The trimmed command line and the in-app switches (A C U H O), with invented data."""

import io
import os
import unittest
from types import SimpleNamespace
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from slurm_usage import cli
from slurm_usage.ui import data as hd
from slurm_usage.ui.anonymize import AnonCollector, AnonHistory, Anonymizer
from slurm_usage.ui.app import App, FETCH_KEYS, TAB_KEYS, TOGGLE_KEYS
from slurm_usage.ui.canvas import Canvas
from slurm_usage.ui.screens import HELP
from tests import ui_fakes as uf


def draw(app, width=160, height=45):
    canvas = Canvas(width, height)
    app.render(canvas)
    return canvas.text()


def make_app(**kw):
    history, fake = uf.real_history(days=30)
    data = hd.HistoryData(history, clock=lambda: fake.now)
    app = App(uf.FakeCollector(uf.make_snapshot(fake.now)), data, "month", background=False,
              **kw)
    app.start()
    app.poll()
    return app


def leaks(screen):
    return [real for real in uf.REAL_IDENTIFIERS if real in screen]


def open_person(app, user="carol"):
    """People tab, cursor on ``user``'s row, Enter."""
    app.key("p")
    draw(app)
    panes, states, shown = app._last
    for index, lines in enumerate(shown):
        for number, line in enumerate(lines):
            if line.action is not None and line.action.params.get("user") == user:
                app.view.focus = index
                states[index].cursor = number
                app._last_view = app.view
                app.key("ENTER")
                return
    raise AssertionError("no row for %s" % user)


def open_job(app):
    app.key("j")
    draw(app)
    app.key("ENTER")
    assert app.view.kind == "job", app.view.kind


class CommandLineTest(unittest.TestCase):
    def parse_fails(self, argv):
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            cli.parse_args(argv)
        self.assertEqual(caught.exception.code, 2)
        return err.getvalue()

    def test_history_choice(self):
        self.assertEqual(cli.parse_args([]).history, "full")
        for mode in ("full", "recent", "off"):
            self.assertEqual(cli.parse_args(["--history", mode]).history, mode)
        self.assertIn("invalid choice", self.parse_fails(["--history", "some"]))

    def test_removed_flags_say_what_replaced_them(self):
        expected = {"--no-history": "--history off", "--no-full-history": "--history recent",
                    "--no-color": "NO_COLOR", "--ascii": "press U",
                    "--cache-dir": "SLURM_USAGE_CACHE"}
        for flag, hint in expected.items():
            message = self.parse_fails(["--once", flag])
            self.assertIn("%s was removed" % flag, message)
            self.assertIn(hint, message)
            self.assertNotIn("unrecognized arguments", message)
        self.assertIn("SLURM_USAGE_CACHE", self.parse_fails(["--cache-dir=/somewhere"]))
        self.assertIn("SLURM_USAGE_CACHE", self.parse_fails(["--cache-dir", "/somewhere"]))
        # after "--" nothing is an option
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli.parse_args(["--", "--ascii"])
        self.assertEqual(caught.exception.code, 2)

    def test_help_is_short_and_grouped(self):
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            cli.parse_args(["--help"])
        text = out.getvalue()
        for needle in ("output", "what to show", "--history {full,recent,off}", "NO_COLOR",
                       "SLURM_USAGE_CACHE", "O settings", "H history"):
            self.assertIn(needle, text)
        for gone in ("--no-history", "--no-color", "--ascii", "--cache-dir"):
            self.assertNotIn(gone, text)
        self.assertLess(len(text.splitlines()), 35)

    def test_cache_directory_comes_from_the_environment(self):
        with mock.patch.dict(os.environ, {"SLURM_USAGE_CACHE": "~/somewhere/cache"}):
            self.assertEqual(cli.cache_dir(), os.path.expanduser("~/somewhere/cache"))
            with mock.patch("slurm_usage.history.History") as history:
                cli._history(cli.parse_args([]))
            history.assert_called_once_with(cache_dir=os.path.expanduser("~/somewhere/cache"))
        with mock.patch.dict(os.environ, {"SLURM_USAGE_CACHE": ""}):
            self.assertIsNone(cli.cache_dir())

    def test_history_modes_reach_the_loader(self):
        with mock.patch("slurm_usage.history.History"):
            self.assertTrue(cli._history(cli.parse_args([])).full_history)
            self.assertFalse(cli._history(cli.parse_args(["--history", "recent"])).full_history)

    def test_color_and_ascii_are_detected(self):
        with mock.patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=True):
            self.assertTrue(cli.wants_color())
            self.assertFalse(cli.wants_ascii(io.TextIOWrapper(io.BytesIO(), encoding="utf-8")))
            self.assertTrue(cli.wants_ascii(io.TextIOWrapper(io.BytesIO(), encoding="ascii")))
        with mock.patch.dict(os.environ, {"TERM": "xterm", "NO_COLOR": "1"}, clear=True):
            self.assertFalse(cli.wants_color())
        with mock.patch.dict(os.environ, {"TERM": "vt100"}, clear=True):
            self.assertTrue(cli.wants_ascii(io.TextIOWrapper(io.BytesIO(), encoding="utf-8")))

    def test_main_hands_the_detected_settings_to_the_app(self):
        made = {}

        class FakeApp:
            def __init__(self, *args, **kwargs):
                made.update(kwargs)
                self.stack = [mock.Mock()]
        with mock.patch.dict(os.environ, {"TERM": "xterm", "NO_COLOR": "1"}, clear=True), \
                mock.patch("sys.stdout.isatty", return_value=True), \
                mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(cli, "build", return_value=(None, None, None)), \
                mock.patch("slurm_usage.ui.app.App", FakeApp), \
                mock.patch("slurm_usage.ui.terminal.run") as run:
            self.assertEqual(cli.main(["--history", "recent"]), 0)
        self.assertEqual((made["color"], made["history"]), (False, "recent"))
        self.assertIn("ascii_only", made)
        run.assert_called_once()


class KeysTest(unittest.TestCase):
    def test_no_collisions_and_help_lists_every_switch(self):
        taken = set(TAB_KEYS) | set(FETCH_KEYS) | set("qQ?[]12345rsSgEkJc+-=,.<>/")
        for key in list(TOGGLE_KEYS) + ["O"]:
            self.assertNotIn(key, taken, key)
        help_text = "\n".join(text for text, _ in HELP)
        for key in list(TOGGLE_KEYS) + ["O"]:
            self.assertRegex(help_text, r"\n  (?:\S+  )*%s\b" % key)

    def test_existing_keys_still_work(self):
        app = make_app()
        app.key("O")
        app.key("O")
        app.key("o")
        self.assertEqual(app.stack[0].kind, "overview")
        open_job(app)
        app.key("u")  # the submit line, not the glyph switch
        self.assertEqual(app.view.extra.get("show"), "submit")
        self.assertFalse(app.ascii)

    def test_footer_offers_settings(self):
        self.assertIn("O settings", draw(make_app()).splitlines()[-1])


class ToggleTest(unittest.TestCase):
    def test_colors(self):
        app = make_app(color=True)
        app.key("C")
        self.assertFalse(app.color)
        self.assertIn("colors off", draw(app))
        app.key("C")
        self.assertTrue(app.color)

    def test_glyphs(self):
        app = make_app()
        self.assertTrue(any(ord(c) > 127 for c in draw(app)))
        app.key("U")
        self.assertTrue(app.ascii)
        screen = draw(app)
        self.assertTrue(all(ord(c) < 128 for c in screen), [c for c in screen if ord(c) > 127])
        app.key("U")
        self.assertFalse(app.ascii)
        self.assertTrue(any(ord(c) > 127 for c in draw(app)))

    def test_history_cycles_and_off_stops_loading(self):
        app = make_app()
        self.assertEqual(app.history_mode, "full")
        app.key("H")
        self.assertEqual((app.history_mode, app.data.full_history), ("recent", False))
        self.assertIn("history recent", draw(app).splitlines()[0])
        app.data._loading = "year"  # pretend a load is running
        app.key("H")
        self.assertEqual(app.history_mode, "off")
        self.assertTrue(app.no_history and app.data.paused and app.data._preempt.is_set())
        app.data._loading = None
        self.assertFalse(app.data.load("week"))  # nothing loads while off
        self.assertIn("history off", draw(app).splitlines()[0])
        app.key("p")
        self.assertIn("history off", draw(app))
        app.data.loaded.discard("month")
        app.key("H")
        self.assertEqual((app.history_mode, app.data.full_history, app.data.paused),
                         ("full", True, False))
        self.assertIn("month", app.data.loaded)  # loading resumed (inline in tests)

    def test_started_with_history_off_can_switch_on(self):
        app = make_app(history="off")
        self.assertEqual(app.data.loaded, set())
        app.key("H")
        self.assertEqual(app.history_mode, "full")
        self.assertIn("month", app.data.loaded)


class AnonymizeToggleTest(unittest.TestCase):
    def test_person_page(self):
        app = make_app()
        open_person(app, "carol")
        screen = draw(app)
        self.assertIn("carol", screen)
        app.key("A")
        screen = draw(app)
        self.assertEqual(leaks(screen), [])
        self.assertIn("ANON", screen.splitlines()[0])
        pseudonym = app.anon.name("user", "carol")
        self.assertEqual((app.view.params["user"], app.view.title), (pseudonym, pseudonym))
        self.assertIn(pseudonym, screen)
        app.key("ESC")  # the breadcrumbs stay valid: back to the people list, pseudonymous
        self.assertEqual(leaks(draw(app)), [])
        app.key("A")
        screen = draw(app)
        self.assertNotIn("ANON", screen.splitlines()[0])
        self.assertIn("carol", screen)

    def test_selection_and_filters_follow_the_pseudonyms(self):
        app = make_app()
        open_person(app, "carol")
        app.key("ESC")
        draw(app)
        before = app._focused()
        app.key("A")
        draw(app)
        pane, state, lines = app._focused()
        line = lines[state.cursor]
        self.assertEqual(line.action.params.get("user"), app.anon.name("user", "carol"))
        app.key("A")
        draw(app)
        pane, state, lines = app._focused()
        self.assertEqual(lines[state.cursor].action.params.get("user"), "carol")
        # a whole name maps to its pseudonym and back; part of a real name is dropped
        state.filter = "carol"
        app.key("A")
        self.assertEqual(app._focused()[1].filter, app.anon.name("user", "carol"))
        app.key("A")
        self.assertEqual(app._focused()[1].filter, "carol")
        app._focused()[1].filter = "caro"
        app.key("A")
        self.assertEqual(app._focused()[1].filter, "")
        self.assertEqual(leaks(draw(app)), [])
        app.key("A")
        self.assertEqual(app._focused()[1].filter, "caro")
        self.assertIsNotNone(before)

    def test_open_job_detail_leaves_nothing_real(self):
        app = make_app()
        open_job(app)
        app.key("l")
        screen = draw(app)
        self.assertIn("/home/x/out", screen)
        self.assertIn("step 1 loss", screen)
        job_id = app.view.params["job_id"]
        app.key("A")
        screen = draw(app)
        self.assertEqual(leaks(screen), [])
        for gone in ("/home/", "step 1 loss", "stdout tail", "out-"):
            self.assertNotIn(gone, screen)
        self.assertNotIn("show", app.view.extra)
        self.assertNotIn("output", app.view.panes)
        self.assertEqual(app.view.params["job_id"], job_id)
        # every stacked screen and the text a fetch would show stay hidden too
        for key in FETCH_KEYS:
            app.key(key)
            screen = draw(app)
            self.assertEqual(leaks(screen), [], key)
            self.assertNotIn("/home/", screen)
            self.assertIn("hidden while anonymized", screen)
        app.key("A")
        screen = draw(app)
        self.assertIn("/home/x/out", screen)
        self.assertNotIn("Submit line", screen)  # the open text was closed, not restored

    def focus_pane(self, app, pane_id):
        for _ in range(10):
            pane = app._focused()[0]
            if pane is not None and pane.id == pane_id:
                return
            app.key("TAB")
            draw(app)
        raise AssertionError("no pane " + pane_id)

    def test_history_job_rows_and_run_types_are_mapped(self):
        app = make_app()
        open_person(app, "alice")
        draw(app)
        app.key("g")
        app.key("g")  # run types by job name
        self.assertEqual(app.view.extra["dim"], "name_family")
        draw(app)
        self.focus_pane(app, "runs")
        app.key("ENTER")  # the jobs of one job-name family
        draw(app)
        app.key("ESC")
        draw(app)
        self.focus_pane(app, "jobs")
        app.key("ENTER")  # a job from history, carrying its row
        self.assertIn("row", app.view.params)
        draw(app)
        app.key("A")
        app.key("ESC")
        app.key("ENTER")
        app.key("ESC")  # walk the breadcrumbs while anonymized
        for view in app.stack:
            blob = repr(view.params) + view.title
            for real in uf.REAL_IDENTIFIERS + ("eval", "sweep", "train"):
                self.assertNotIn(real, blob, (view.kind, real))
        self.assertEqual(leaks(draw(app)), [])
        app.key("A")
        self.assertEqual(app.stack[1].title, "alice")
        self.assertIn("alice", draw(app))

    def test_run_type_jobs_page(self):
        app = make_app()
        open_person(app, "alice")
        draw(app)
        app.key("g")
        app.key("g")
        draw(app)
        self.focus_pane(app, "runs")
        app.key("ENTER")
        title = app.view.title
        self.assertIn("job name", title)
        draw(app)
        app.key("A")
        screen = draw(app)
        self.assertEqual(leaks(screen), [])
        self.assertNotEqual(app.view.title, title)
        self.assertRegex(app.view.title, r"user-\d+ · job name job-\d+")
        app.key("A")
        self.assertEqual(app.view.title, title)

    def test_an_app_started_anonymized_can_switch_back(self):
        history, fake = uf.real_history(days=20)
        anon = Anonymizer()
        data = hd.HistoryData(AnonHistory(history, anon), clock=lambda: fake.now)
        collector = AnonCollector(uf.FakeCollector(uf.make_snapshot(fake.now)), anon)
        app = App(collector, data, "month", background=False, anonymized=True)
        app.start()
        app.poll()
        app.key("p")
        self.assertEqual(leaks(draw(app)), [])
        app.key("A")
        app.poll()
        self.assertIn("carol", draw(app))
        app.key("A")
        self.assertEqual(leaks(draw(app)), [])


class SettingsOverlayTest(unittest.TestCase):
    def test_lists_every_switch_and_the_cache(self):
        app = make_app()
        app.history_info = SimpleNamespace(value=SimpleNamespace(
            cache_path="/cache/dir/history.sqlite", notes=[]), at=0, state="ok")
        app.key("O")
        screen = draw(app)
        for needle in ("Settings", "A  anonymize  off", "C  colors", "U  glyphs",
                       "H  history", "/cache/dir/history.sqlite", "X  clear cache"):
            self.assertIn(needle, screen)
        app.key("U")  # switches work inside the box, which stays open
        self.assertIn("glyphs     ASCII", draw(app))
        app.key("A")
        screen = draw(app)
        self.assertIn("hidden while anonymized", screen)
        self.assertNotIn("/cache/dir", screen)
        app.key("ESC")
        self.assertIsNone(app.overlay)
        self.assertNotIn("Settings", draw(app))

    def test_clear_cache_asks_first(self):
        app = make_app()
        calls = []
        real_rebuild = app._raw_history.rebuild
        app._raw_history.rebuild = lambda: calls.append(1) or real_rebuild()
        app.key("O")
        app.key("X")
        self.assertIn("clear cache? y = yes", draw(app))
        app.key("n")
        self.assertEqual(calls, [])
        self.assertIn("cache kept", draw(app))
        app.key("X")
        app.key("y")
        self.assertEqual(calls, [1])
        self.assertIn("month", app.data.loaded)  # reloaded from Slurm
        self.assertIn("cleared", app.message)


if __name__ == "__main__":
    unittest.main()
