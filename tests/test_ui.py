"""The terminal UI, rendered to an in-memory screen with invented data."""

import io
import json
import os
import tempfile
import time
import unittest

from slurm_usage.model import Status
from slurm_usage.ui import data as hd
from slurm_usage.ui import fmt
from slurm_usage.ui.anonymize import AnonCollector, AnonHistory, Anonymizer
from slurm_usage.ui.app import App
from slurm_usage.ui.canvas import Canvas
from slurm_usage.ui.filestat import check, script_from_submit_line
from slurm_usage.ui.loader import Loader
from slurm_usage.ui import text
from slurm_usage.ui.widgets import Column, Pane, allocate, fit_columns, wrap
from tests import ui_fakes as uf

SIZES = ((80, 24), (200, 50))
TABS = ("o", "m", "n", "j", "p", "a", "t")
BAD_CHARS = ("\x1b", "\x07", "\x9b", "\x9d", "\x00")


def make_app(history=None, collector=None, range_name="week", **kw):
    if history is None:
        history, fake = uf.real_history(days=40)
        data = hd.HistoryData(history, clock=lambda: fake.now)
        snap = uf.make_snapshot(fake.now)
    else:
        data = hd.HistoryData(history)
        snap = uf.make_snapshot()
    app = App(collector or uf.FakeCollector(snap), data, range_name, background=False, **kw)
    app.start()
    app.poll()
    return app


def draw(app, width=80, height=24):
    canvas = Canvas(width, height)
    app.render(canvas)
    return canvas


def assert_clean(case, canvas):
    screen = canvas.text()
    for bad in BAD_CHARS:
        case.assertNotIn(bad, screen)
    for row in range(canvas.height):
        case.assertLessEqual(fmt.width(canvas.row_text(row)), canvas.width)


def every_view(app):
    """Yield (label, app) after visiting each tab, a drill from each, and the job/help views."""
    for key in TABS:
        app.key(key)
        yield key, app
        app.key("ENTER")
        yield key + "+enter", app
        app.key("ESC")
    app.key("j")
    app.key("ENTER")
    for fetch in "lebu":
        app.key(fetch)
        yield "job+" + fetch, app
    app.key("?")
    yield "help", app


class LayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = make_app()

    def test_every_view_renders_at_both_sizes(self):
        for width, height in SIZES:
            self.app.go_tab("overview")
            for label, app in every_view(self.app):
                for range_key in ("1", "5", "2"):
                    app.key(range_key)
                    app.poll()
                    canvas = draw(app, width, height)
                    assert_clean(self, canvas)
                    self.assertIn("orion", canvas.row_text(0), label)
                    self.assertTrue(canvas.row_text(height - 1).strip(), label)

    def test_overview_content_80x24(self):
        self.app.go_tab("overview")
        screen = draw(self.app).text()
        for needle in ("GPUs", "CPUs", "Nodes", "Queue", "Trend", "day", "week", "month",
                       "year", "all", "Accounts", "physics", "biolab", "Warnings",
                       "capacity differs", "drain", "JobArray"):
            self.assertIn(needle, screen)
        self.assertIn("12/16", screen)  # the GrpTRES gpu cap joined from Snapshot.limits

    def test_totals_first_no_job_names_on_landing_tabs(self):
        for tab in ("o", "n", "j", "a", "p", "t"):
            self.app.key(tab)
            screen = draw(self.app, 200, 50).text()
            self.assertNotIn("train_lr3_0042", screen, tab)
            self.assertNotIn("eval_ckpt", screen, tab)
        self.app.go_tab("people")
        self.app.push("person", {"user": "alice"}, "alice")
        self.assertIn("train_lr3_0042", draw(self.app, 200, 50).text())

    def test_tiny_terminal(self):
        canvas = draw(self.app, 10, 3)
        self.assertIn("too", canvas.text())
        assert_clean(self, draw(self.app, 30, 6))


class NavigationTest(unittest.TestCase):
    def setUp(self):
        self.app = make_app()
        draw(self.app)

    def test_drill_and_back(self):
        app = self.app
        app.key("p")
        self.assertEqual(app.view.kind, "people")
        app.key("ENTER")
        self.assertEqual(app.view.kind, "person")
        self.assertEqual(len(app.stack), 2)
        draw(app)
        app.key("TAB")
        app.key("TAB")  # runs pane
        app.key("ENTER")
        self.assertEqual(app.view.kind, "jobs")
        draw(app)
        app.key("ENTER")
        self.assertEqual(app.view.kind, "job")
        screen = draw(app).text()
        self.assertIn("› job ", screen.split("\n")[0])
        self.assertEqual([v.kind for v in app.stack], ["people", "person", "jobs", "job"])
        for _ in range(3):
            app.key("ESC")
        self.assertEqual([v.kind for v in app.stack], ["people"])
        app.key("ESC")
        self.assertEqual(len(app.stack), 1)

    def test_tab_resets_stack_and_range_is_shared(self):
        app = self.app
        app.key("p")
        app.key("ENTER")
        app.key("]")
        self.assertEqual(app.range, "month")
        app.key("o")
        self.assertEqual([v.kind for v in app.stack], ["overview"])
        self.assertEqual(app.range, "month")
        app.key("1")
        self.assertEqual(app.range, "day")
        app.key("[")
        self.assertEqual(app.range, "day")
        app.key("5")
        app.key("]")
        self.assertEqual(app.range, "all")

    def test_filter_sort_group_help_quit(self):
        app = self.app
        app.key("n")
        draw(app)
        for ch in "/gpu-b":
            app.key(ch)
        self.assertIn("filter", draw(app).row_text(23))
        app.key("ENTER")
        screen = draw(app).text()
        self.assertIn("gpu-b01", screen)
        self.assertNotIn("gpu-a01", screen)
        app.key("ESC")  # clears the filter first
        self.assertIn("gpu-a01", draw(app).text())
        app.key("s")
        self.assertIn("sort name", draw(app).text())
        app.key("a")
        app.key("g")
        self.assertIn("QOS", draw(app).text())
        app.key("g")
        self.assertIn("Partitions", draw(app).text())
        app.key("?")
        self.assertIn("Glossary", draw(app, 120, 80).text())
        app.key("?")
        self.assertNotEqual(app.view.kind, "help")
        self.assertFalse(app.key("q"))

    def test_person_run_types_cycle_and_timeline_keys(self):
        app = self.app
        app.key("m")
        for _ in range(len(hd.DIMENSIONS)):
            screen = draw(app, 200, 50).text()
            self.assertIn("What kinds of runs", screen)
            app.key("g")
        app.key("t")
        draw(app, 120, 30)
        for key in ("+", "+", ",", "LEFT", "g", "-", "."):
            app.key(key)
            assert_clean(self, draw(app, 120, 30))
        app.key("g")
        app.key("g")
        app.key("g")
        draw(app, 120, 30)
        app.key("ENTER")
        self.assertEqual(app.view.kind, "jobs")
        assert_clean(self, draw(app, 120, 30))

    def test_mouse(self):
        app = self.app
        app.key("n")
        draw(app)
        app.click(1, 3)
        app.click(1, 3)  # double click opens
        self.assertEqual(app.view.kind, "queue")
        app.wheel(3)
        app.click(app.tab_hits[0][0], 0)
        self.assertEqual(app.view.kind, "overview")

    def test_job_fetch_shows_denied_plainly(self):
        app = self.app
        app.key("j")
        draw(app)
        app.key("ENTER")
        app.key("b")
        self.assertIn("denied", draw(app).text())
        app.key("u")
        self.assertIn("sbatch", draw(app, 100, 40).text())
        app.key("l")
        screen = draw(app, 100, 40).text()
        self.assertIn("loss", screen)
        self.assertNotIn("\x1b", screen)

    def test_job_script_not_stored_is_not_denied(self):
        from slurm_usage import parse
        from slurm_usage.live import Fetched
        app = self.app
        app.collector.batch_script = lambda job: Fetched(Status.NOT_STORED,
                                                         detail=parse.SCRIPT_NOT_STORED)
        app.key("j")
        draw(app)
        app.key("ENTER")
        app.key("b")
        screen = draw(app, 160, 40).text()
        self.assertIn("not stored: Slurm keeps no copy", screen)
        self.assertNotIn("denied:", screen)


class RefreshTest(unittest.TestCase):
    def test_adaptive_interval(self):
        app = make_app(uf.FakeHistory())
        self.assertEqual(app.interval(), 30.0)
        app.last_input = time.time() - 1000
        self.assertGreater(app.interval(), 30.0)
        self.assertLessEqual(app.interval(), 300.0)

    def test_refresh_key_revalidates(self):
        app = make_app(uf.FakeHistory())
        calls = len(app.collector.calls)
        app.key("r")
        app.poll()
        self.assertGreater(len(app.collector.calls), calls)


class HostileAndMissingTest(unittest.TestCase):
    def test_hostile_names_never_reach_the_terminal(self):
        snap = uf.make_snapshot()
        for job in snap.jobs:  # as if a parser had let raw text through
            job.name = uf.HOSTILE_NAME
            job.reason = "\x1b]52;c;ZXZpbA==\x07"
        snap.nodes[0].reason = "\x9b31m" + uf.HOSTILE_NAME
        app = make_app(uf.FakeHistory(), uf.FakeCollector(snap))
        for width, height in SIZES:
            app.go_tab("overview")
            for _, a in every_view(app):
                assert_clean(self, draw(a, width, height))
        canvas = Canvas(40, 2)
        canvas.put(0, 0, "a\x1b[2Jb\x1b]0;t\x07c‮d​e")
        self.assertEqual(canvas.row_text(0).strip(), "abcde")

    def test_missing_everything(self):
        snap = uf.make_snapshot()
        snap.nodes = snap.jobs = snap.roles = snap.limits = snap.partitions = None
        snap.capabilities["jobs"].status = Status.DENIED
        for history in (hd.NoHistory(), uf.FailingHistory()):
            app = make_app(history, uf.FakeCollector(snap), no_history=isinstance(
                history, hd.NoHistory))
            for width, height in SIZES:
                app.go_tab("overview")
                for _, a in every_view(app):
                    assert_clean(self, draw(a, width, height))
            screen = draw(app, 200, 50)
            app.go_tab("overview")
            text_ = draw(app, 200, 50).text()
            self.assertTrue("n/a" in text_ or "denied" in text_)

    def test_live_failure_still_draws(self):
        class Broken(uf.FakeCollector):
            def snapshot(self):
                raise RuntimeError("scontrol exploded")
        app = make_app(uf.FakeHistory(), Broken())
        for _, a in every_view(app):
            assert_clean(self, draw(a))
        app.go_tab("overview")
        self.assertIn("ERR", draw(app).text())


class AnonymizeTest(unittest.TestCase):
    def test_no_real_identifier_anywhere(self):
        history, fake = uf.real_history(days=40)
        anon = Anonymizer()
        data = hd.HistoryData(AnonHistory(history, anon), clock=lambda: fake.now)
        collector = AnonCollector(uf.FakeCollector(uf.make_snapshot(fake.now)), anon)
        app = App(collector, data, "month", background=False, anonymized=True)
        app.start()
        app.poll()
        screens = []
        for width, height in SIZES:
            app.go_tab("overview")
            for _, a in every_view(app):
                a.poll()
                screens.append(draw(a, width, height).text())
            app.go_tab("people")
            draw(app, width, height)
            app.key("ENTER")
            for key in ("TAB", "TAB", "ENTER"):
                app.key(key)
                screens.append(draw(app, width, height).text())
        self.assertIn("ANON", screens[0])
        self.assertTrue(any("user-0" in s for s in screens))
        self.assertTrue(any("acct-0" in s for s in screens))
        summary = text.collect(collector.snapshot(), data, "month", "user-01", None, data.info())
        blob = "\n".join(screens) + text.to_json(summary) + text.render(summary)
        for real in uf.REAL_IDENTIFIERS:
            self.assertNotIn(real, blob, real)
        self.assertNotIn("/home/", blob)
        self.assertIn("hidden while anonymized", "\n".join(screens))

    def test_stable_pseudonyms_and_reverse_filters(self):
        anon = Anonymizer()
        self.assertEqual(anon.name("user", "alice"), "user-01")
        self.assertEqual(anon.name("user", "bob"), "user-02")
        self.assertEqual(anon.name("user", "alice"), "user-01")
        self.assertEqual(anon.name("user", "user-01"), "user-01")  # idempotent
        self.assertEqual(anon.real("user", "user-02"), "bob")
        # job names: an opaque pseudonym per family, none of the text
        self.assertEqual(anon.job_name("alice_sweep_0042"), "job-01")
        self.assertEqual(anon.job_name("alice_sweep_0043"), "job-01")
        self.assertEqual(anon.job_name("cancer_atlas_v2"), "job-02")
        self.assertNotIn("/", anon.scrub("see /home/alice/x and a@b.org") or "")

    def test_scrub_whole_tokens_short_names_and_tres(self):
        anon = Anonymizer()
        anon.name("user", "al")     # shorter than 3 characters: still replaced
        anon.name("account", "bio")
        text = anon.scrub("al ran out of bio quota; biology and also unchanged, (al)")
        self.assertEqual(text, "user-01 ran out of acct-01 quota; biology and also "
                               "unchanged, (user-01)")
        self.assertEqual(anon.scrub("node-al.x and bio-2 kept"), "node-al.x and bio-2 kept")
        warning = "gres/gpu capacity differs: nodes have 20"
        self.assertEqual(anon.scrub(warning), warning)

    def test_events_register_names_before_scrubbing(self):
        from slurm_usage.history.events import Event
        from slurm_usage.history.select import Filter

        class History:
            def events(self, window=None, where=None, **kwargs):
                return [Event("first_account", 0, None, "First use of account zeta",
                              {"account": "zeta"}, Filter(account="zeta"))]
        anon = Anonymizer()
        found = AnonHistory(History(), anon).events()
        self.assertNotIn("zeta", found[0].sentence)
        self.assertIn("acct-01", found[0].sentence)


class DataTest(unittest.TestCase):
    def test_name_family(self):
        self.assertEqual(hd.name_family("train_lr3_0042"), hd.name_family("train_lr7_0099"))
        self.assertEqual(hd.name_family(""), "(no name)")
        self.assertEqual(hd._local_family("run_12_a3f9c2d1"), "run_<n>_<hex>")

    def test_local_run_types_sum_and_drill(self):
        history = uf.FakeHistory()
        data = hd.HistoryData(history)
        page = data.attempts("month", hd.where(user="alice"))
        for dim, _ in hd.DIMENSIONS:
            kinds = hd.run_types(page.rows, dim)
            self.assertEqual(sum(k.attempts for k in kinds), len(page.rows))
            shares = [k.share for k in kinds if k.share is not None]
            if shares:
                self.assertAlmostEqual(sum(shares), 1.0, places=6)

    def test_history_run_types_match_attempts(self):
        history, fake = uf.real_history(days=20)
        data = hd.HistoryData(history, clock=lambda: fake.now)
        data.load_in_order("month")
        filt = hd.where(user="carol")
        kinds = data.run_types("month", filt, "outcome")
        self.assertTrue(kinds)
        page = data.jobs("month", filt, run_type=("outcome", kinds[0].key))
        self.assertTrue(all(row.outcome == kinds[0].key for row in page.rows))
        self.assertTrue(data.events("month", filt) is not None)
        ov = data.overview(20, 384)
        self.assertEqual([r.name for r in ov.ranges], list(hd.RANGE_NAMES))

    def test_events_are_sentences(self):
        history = uf.FakeHistory()
        data = hd.HistoryData(history)
        found = data.events("year", hd.where(user="alice"))
        kinds = {e.kind for e in found}
        self.assertIn("longest", kinds)
        self.assertTrue(kinds & {"big_run", "preempted", "failures", "record_day"})
        for event in found:
            self.assertTrue(any(ch.isdigit() for ch in event.text), event.text)


class SlowHistory(uf.FakeHistory):
    """Loading anything longer than a day runs until it is cancelled."""

    def __init__(self):
        super().__init__()
        import threading
        self.started = threading.Event()

    def refresh(self, window=None, progress=None, cancel=None, update=True):
        self.refreshed.append(window)
        if window is None or window.start < self.now - 2 * 86400:
            self.started.set()
            deadline = time.time() + 10
            while not cancel() and time.time() < deadline:
                time.sleep(0.005)
            from slurm_usage.history.errors import Cancelled
            raise Cancelled()
        return self.info()


class PrioritizeTest(unittest.TestCase):
    def test_shorter_range_preempts_a_longer_load(self):
        import threading
        history = SlowHistory()
        data = hd.HistoryData(history, clock=lambda: history.now)
        outcome = []
        worker = threading.Thread(target=lambda: outcome.append(data.load("year")))
        worker.start()
        self.assertTrue(history.started.wait(5))
        self.assertFalse(data.prioritize("all"))  # longer: the load in progress is on the way
        self.assertTrue(data.prioritize("day"))
        worker.join(5)
        self.assertEqual(outcome, [False])
        self.assertIsNone(data.failure)  # yielding is not a failure
        self.assertTrue(data.load("day"))
        self.assertIn("day", data.loaded)
        self.assertNotIn("year", data.loaded)

    def test_app_range_change_prioritizes(self):
        app = make_app(uf.FakeHistory())
        calls = []
        app.data.prioritize = lambda name: calls.append(name) or False
        app.set_range("year" if app.range != "year" else "day")
        self.assertEqual(len(calls), 1)


class BackgroundLoadTest(unittest.TestCase):
    def loaded(self, history, **kw):
        data = hd.HistoryData(history, clock=lambda: history.now, **kw)
        data.load_in_order("week")
        return data, [w for w in history.refreshed]

    def test_full_history_is_loaded_by_default_when_cached(self):
        history = uf.FakeHistory()
        data, windows = self.loaded(history)
        self.assertIn("all", data.loaded)
        self.assertIn(None, windows)  # None = all of history
        self.assertEqual(data.limit_note(), "")

    def test_no_full_history(self):
        history = uf.FakeHistory()
        data, windows = self.loaded(history, full_history=False)
        self.assertEqual(data.loaded, {"day", "week", "month", "year"})
        self.assertNotIn(None, windows)
        self.assertIn("recent history only", data.limit_note())
        args = __import__("slurm_usage.cli", fromlist=["x"]).parse_args(["--history", "recent"])
        self.assertEqual(args.history, "recent")

    def test_memory_only_history_stops_at_a_month_and_says_so(self):
        history = uf.FakeHistory()
        info = history.info

        def memory_info():
            found = info()
            found.cache_path = None
            return found
        history.info = memory_info
        data, windows = self.loaded(history)
        self.assertEqual(data.loaded, {"day", "week", "month"})
        self.assertIn("month", data.limit_note())
        self.assertTrue(any(n.code == "background_limit" for n in data.info().notes))
        data.load("year")  # still loaded when asked for
        self.assertIn("year", data.loaded)


class PartsTest(unittest.TestCase):
    def test_fmt(self):
        self.assertEqual(fmt.num(12345), "12.3k")
        self.assertEqual(fmt.num(950), "950")
        self.assertEqual(fmt.duration(90061), "1d01h")
        self.assertEqual(fmt.delta(150, 100), "+50%")
        self.assertEqual(fmt.delta(0, 0), "—")
        self.assertEqual(fmt.width(fmt.fit("日本語テキスト", 7)), 7)
        self.assertEqual(len(fmt.sparkline([1, 2, 3, 0], 4)), 4)
        self.assertEqual(fmt.placeholder(Status.DENIED), "denied")
        self.assertEqual(fmt.placeholder(Status.ERROR), "ERR")

    def test_canvas_wide_chars_and_clipping(self):
        canvas = Canvas(6, 1)
        canvas.put(0, 0, "日本語テ")
        self.assertEqual(fmt.width(canvas.row_text(0)), 6)
        canvas.put(0, 4, "abc")
        self.assertEqual(len(canvas.cells[0]), 6)

    def test_columns_and_allocation(self):
        cols = [Column("a", 10, priority=9), Column("b", 30, priority=1), Column("c", 10,
                                                                                priority=5)]
        kept = [c.name for _, c, _ in fit_columns(cols, 25)]
        self.assertEqual(kept, ["a", "c"])
        panes = [Pane(str(i), "p") for i in range(6)]
        for height in (3, 8, 23, 48):
            sizes = allocate(panes, [30] * 6, height, 2)
            used = sum(1 + s for s in sizes if s >= 0)
            self.assertLessEqual(used, height)
            self.assertGreaterEqual(sizes[2], 0)
        self.assertEqual(wrap("  a b c d e f", 6, 2)[1][:2], "  ")

    def test_loader_threaded_and_failure(self):
        loader = Loader(workers=2)
        entry = loader.get("x", lambda: 42)
        deadline = time.time() + 5
        while entry.at is None and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(entry.value, 42)
        self.assertFalse(entry.busy)

        def boom():
            raise hd.HistoryError(Status.DENIED, "no")
        bad = Loader(workers=0).get("y", boom)
        self.assertEqual((bad.state, bad.status), ("failed", Status.DENIED))
        loader.close()

    def test_filestat(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "out.log")
            with open(path, "w") as handle:
                handle.write("x")
            started = time.time() - 1000
            self.assertEqual(check(path, started).note, "exists · changed after the job ended")
            self.assertFalse(check(path, time.time() + 100).changed)
            self.assertFalse(check(path + ".gone", started).exists)
            self.assertIn("not a regular file", check(folder, started).note)
            self.assertEqual(check(folder, started, directory=True).note, "exists")
            self.assertIsNone(check("/proc/self/environ", started).exists)
            self.assertIsNone(check("relative/path", started).exists)
        self.assertEqual(script_from_submit_line("sbatch -p gpu --gres=gpu:2 job.sh --x 1",
                                                 "/w"), "/w/job.sh")
        self.assertIsNone(script_from_submit_line("srun python x.py", "/w"))


class TextOutputTest(unittest.TestCase):
    def test_once_and_json(self):
        history, fake = uf.real_history(days=30)
        data = hd.HistoryData(history, clock=lambda: fake.now)
        data.load_in_order("week")
        snap = uf.make_snapshot(fake.now)
        summary = text.collect(snap, data, "week", "alice", None, data.info())
        rendered = text.render(summary)
        for needle in ("GPU", "Range", "week", "Account", "physics", "Person alice"):
            self.assertIn(needle, rendered)
        parsed = json.loads(text.to_json(summary))
        self.assertEqual([t["range"] for t in parsed["trend"]], list(hd.RANGE_NAMES))
        self.assertIn("gpu_count", parsed["person"]["run_types"])
        self.assertNotIn("\x1b", rendered)

    def test_no_history_summary(self):
        summary = text.collect(uf.make_snapshot(), None, "day")
        self.assertEqual(summary["history"]["status"], "off")
        self.assertIn("Queue", text.render(summary))


if __name__ == "__main__":
    unittest.main()
