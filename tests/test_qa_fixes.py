"""Regressions from a hands-on QA pass: each test names the bug it pins down.
Invented data only (tests/ui_fakes.py)."""

import os
import tempfile
import time
import unittest
from collections import namedtuple

from slurm_usage.history import HistoryError, JobPage, Window
from slurm_usage.model import Job, Node, Status, Step
from slurm_usage.ui import data as hd
from slurm_usage.ui import fmt, screens, text
from slurm_usage.ui.anonymize import AnonCollector, AnonHistory, Anonymizer
from slurm_usage.ui.app import App
from slurm_usage.ui.canvas import Canvas
from slurm_usage.ui.filestat import GRACE, check
from slurm_usage.ui.loader import Loader
from slurm_usage.ui.widgets import (Column, Line, Pane, PaneState, draw_pane, fit_columns,
                                    sort_marked)
from tests import ui_fakes as uf
from tests.test_ui import draw, make_app


class DeferredLoader(Loader):
    """Runs nothing until flush(): what a worker thread does, after the screen moved on."""

    def __init__(self) -> None:
        super().__init__(workers=0)
        self.queue = []

    def _run(self, entry, fn):
        self.queue.append((entry, fn))

    def flush(self) -> None:
        while self.queue:
            entry, fn = self.queue.pop(0)
            Loader._run(self, entry, fn)


def big_history():
    """Over two years of invented jobs: alice has more than a page of them."""
    now = int(time.time())
    return uf.FakeHistory(now, uf.make_attempts(now, days=900))


def deferred_app(history=None, range_name="week"):
    app = make_app(history, range_name=range_name)
    app.loader = DeferredLoader()
    app.poll()
    app.loader.flush()
    app.poll()
    return app


def settle(app, width=160, height=60, rounds=4):
    for _ in range(rounds):
        draw(app, width, height)
        app.loader.flush()
        app.poll()
    return draw(app, width, height).text()


class PersonRunTypesTest(unittest.TestCase):
    """1. `g` modes showed `history: ERR 'other'` in the TUI only: the run-types and events
    lambdas read a variable the outcome loop later rebound, and ran on a worker after."""

    def test_every_dimension_loads_when_run_later(self):
        app = deferred_app()
        app.key("m")
        for dim in screens.PAGE_DIMENSIONS:
            screen = settle(app)
            self.assertIn("by %s" % screens.DIM_WORDS[dim], screen)
            self.assertNotIn("ERR", screen, dim)
            self.assertNotIn("'other'", screen, dim)
            entry = app.loader.peek(("h", "runtypes", "week", (("user", "alice"),), dim))
            self.assertEqual(entry.state, "ok", (dim, entry.message))
            app.key("g")
        events = app.loader.peek(("h", "events", "week", "alice"))
        self.assertEqual(events.state, "ok", events.message)

    def test_run_types_recompute_when_history_grows(self):
        app = deferred_app()
        app.key("m")
        settle(app)
        key = ("h", "runtypes", "week", (("user", "alice"),), "gpu_count")
        first = app.loader.peek(key).value
        app.data.generation += 1  # more history loaded, or E measured efficiency
        settle(app)
        entry = app.loader.peek(key)
        self.assertEqual(entry.state, "ok")
        self.assertIsNot(entry.value, first)

    def test_stale_value_is_marked(self):
        app = deferred_app()
        app.key("m")
        settle(app)
        app.data.run_types = lambda *a, **k: (_ for _ in ()).throw(HistoryError(Status.ERROR,
                                                                                  "boom"))
        app.data.generation += 1
        screen = settle(app)
        self.assertIn("stale: ERR boom", screen)

    def test_history_run_types_before_the_store_opens(self):
        history, _ = uf.real_history(days=3)
        with self.assertRaises(HistoryError):  # was AttributeError: 'NoneType' ... 'lock'
            history.run_types()


class ShareTest(unittest.TestCase):
    """2. Person share went over 100% while a partial person total was divided by a
    cluster total that was further behind."""

    Row = namedtuple("Row", "gpu_hours complete")

    def test_share_only_from_equally_loaded_figures(self):
        both = screens.person_share(self.Row(50.0, True), self.Row(200.0, True))
        self.assertEqual(both[0], "25.0%")
        lower = screens.person_share(self.Row(50.0, False), self.Row(200.0, True))
        self.assertEqual(lower[0], "≥25.0%")
        behind = screens.person_share(self.Row(300.0, False), self.Row(200.0, False))
        self.assertEqual(behind[0], fmt.NOT_YET)
        self.assertEqual(screens.person_share(self.Row(300.0, True), self.Row(200.0, True))[0],
                         "100.0%")


class FileStatTest(unittest.TestCase):
    """3. Every log was "changed since job ran": its mtime was compared with the start."""

    def test_logs_against_the_end_with_grace(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "out.log")
            with open(path, "w") as handle:
                handle.write("x")
            now = time.time()
            self.assertEqual(check(path, None, running=True).note, "exists · being written")
            self.assertFalse(check(path, now - GRACE + 30).changed)  # its own last flush
            self.assertTrue(check(path, now - 3 * GRACE).changed)
            self.assertIn("job started", check(path, now - 3 * GRACE, what="the job started").note)


class EventsRangeTest(unittest.TestCase):
    """4. "Events · week" listed events from months before the week."""

    def test_events_outside_the_range_are_dropped(self):
        from slurm_usage.history.events import Event as HEvent
        now = 2_000_000_000
        old = HEvent("preemption_run", now - 60 * 86400, now - 50 * 86400, "old")
        spans = HEvent("preemption_run", now - 20 * 86400, now - 86400, "spans")
        inside = HEvent("longest_run", now - 3600, now, "inside")

        class Stub:
            def events(self, window, where, cancel=None):
                return [old, spans, inside]

            def info(self):
                return None

        data = hd.HistoryData(Stub(), clock=lambda: now)
        texts = [e.text for e in data.events("week", None)]
        self.assertEqual(sorted(texts), ["inside", "spans"])

    def test_preemption_run_counts_only_the_window(self):
        from tests import test_history_support as hs
        now = hs.T0 + 40 * 86400
        jobs, index = [], 0

        def attempt(job_id, start, state, number):
            nonlocal index
            index += 1
            jobs.append(hs.job(job_id, start=int(start), end=int(start) + 3600, state=state,
                               index=index, restarts=number))

        for job_id in range(100, 120):  # the usual: two preemptions, long before
            for number, state in enumerate(("PREEMPTED", "PREEMPTED", "COMPLETED")):
                attempt(job_id, now - 35 * 86400 + job_id * 600 + number * 7200, state, number)
        at = now - 22 * 86400  # 12 preemptions over three weeks, 4 of them in the last week
        for number in range(12):
            attempt(77, at, "PREEMPTED", number)
            at += (2 if number < 8 else 0.5) * 86400
        attempt(77, at, "COMPLETED", 12)
        fake = hs.FakeSlurm(jobs, now=now, live=[])
        history = hs.make_history(fake)
        history.refresh(None)
        week = Window(now - 7 * 86400, None)
        found = [e for e in history.events(week) if e.kind == "preemption_run"]
        self.assertEqual(len(found), 1)
        event = found[0]
        self.assertGreaterEqual(event.time, week.start)  # dated inside the week
        self.assertEqual(event.values["preempted"], 4)
        self.assertIn("4 times (12 since it was submitted)", event.sentence)


class AccountsAgreeTest(unittest.TestCase):
    """5. Overview accounts read the hourly rollup and disagreed with Roles."""

    def test_overview_accounts_equal_roles(self):
        app = make_app()
        draw(app, 160, 50)
        ctx = screens.Context(app, 160)
        overview = {l.key: l.sort["GPU-h"] for l in screens.accounts_pane(ctx).lines}
        app.go_tab("roles")
        roles = {l.key: l.sort["GPU-h"] / 3600.0
                 for l in screens.roles(ctx, app.view)[0].lines}
        self.assertTrue(overview)
        for name, hours in overview.items():
            self.assertAlmostEqual(hours, roles.get(name, 0), places=6, msg=name)


class LogTailTest(unittest.TestCase):
    """6. A log tail opened at its top."""

    def test_tail_opens_at_the_end_and_focused(self):
        app = make_app(uf.FakeHistory())
        app.collector.tail_log = lambda job, raw, max_bytes=0: __import__(
            "slurm_usage.live", fromlist=["LogTail"]).LogTail(
                Status.OK, ["line %d" % i for i in range(200)])
        app.key("j")
        draw(app)
        app.key("ENTER")
        app.key("l")
        screen = draw(app, 120, 40).text()
        self.assertIn("line 199", screen)
        self.assertNotIn("line 0\n", screen)
        state = app.view.panes["output"]
        self.assertEqual(state.cursor, 199)
        app.key("UP")  # moving away stops following
        draw(app, 120, 40)
        self.assertEqual(app.view.panes["output"].cursor, 198)


class JobsPagingTest(unittest.TestCase):
    """7. Jobs panes stopped at 500 rows and / searched only those."""

    def setUp(self):
        self.history = big_history()
        self.app = make_app(self.history, range_name="all")

    def jobs_pane(self):
        ctx, panes = self.app.panes(160)
        return next(p for p in panes if p.id == "jobs")

    def test_count_paging_and_server_side_search(self):
        app = self.app
        app.key("m")
        draw(app, 160, 60)
        pane = self.jobs_pane()
        total = len({r.job_id for r in self.history.rows if r.user == "alice"})
        self.assertGreater(total, 500)
        self.assertIn("(500 of %s)" % fmt.num(total), pane.title)
        self.assertTrue(pane.more)
        app.view.focus = next(i for i, p in enumerate(app.panes(160)[1]) if p.id == "jobs")
        draw(app, 160, 60)
        app.key("END")
        self.assertEqual(app.view.extra["jobs_limit"], 1000)
        self.assertIn("(1000 of", self.jobs_pane().title)
        # a job far down the list, found by history, not among the rows loaded
        rows = [r for r in self.history.rows if r.user == "alice"]
        far = min(rows, key=lambda r: (r.gpus or 0) * (r.elapsed or 0)).job_id
        app.key("/")
        for char in far:
            app.key(char)
        app.key("ENTER")
        pane = self.jobs_pane()
        self.assertTrue(any(far in l.text for l in pane.lines))
        self.assertIn("matching", pane.status)


class RunningJobTest(unittest.TestCase):
    """8. A running job: `n/a mem` though memory was requested; header and attempt elapsed
    disagreed; running steps had no name, state or elapsed; raw gpuutil (645) shown."""

    def make(self):
        history = uf.FakeHistory()
        now = history.now
        snap = uf.make_snapshot(now)
        job = snap.jobs[0]
        job.mem_alloc = None  # AllocTRES without mem
        app = make_app(history, collector=uf.FakeCollector(snap))
        app.collector.steps = lambda live: __import__(
            "slurm_usage.live", fromlist=["StepReport"]).StepReport(Status.OK, [
                Step(live.job_id, "batch", total_cpu=100.0, max_rss=1 << 30,
                     gpu_util_max=645.0, tres_alloc={"cpu": 8, "gres/gpu": 16}),
                Step(live.job_id, "0", name="python")])
        app.push("job", {"job_id": job.job_id}, "job")
        return app, job

    def test_running_job_summary_and_steps(self):
        app, job = self.make()
        # each line without its box border, so the content starts as it did
        screen = "\n".join(line[1:] for line in draw(app, 160, 50).text().split("\n"))
        self.assertIn("%s mem (requested)" % fmt.mem(job.mem_req), screen)
        self.assertRegex(screen, r"elapsed +%s" % fmt.duration(job.elapsed))
        self.assertIn("4 GPUs", screen)
        batch = next(l for l in screen.split("\n") if l.startswith(" batch"))
        self.assertIn("RUNNING", batch)
        self.assertIn("batch", batch[10:])
        self.assertIn(fmt.duration(job.elapsed), batch)
        self.assertIn("40%", batch)  # 645 over 16 GPUs, not "645"
        self.assertNotIn("645", screen)
        numbered = next(l for l in screen.split("\n") if l.startswith(" 0 "))
        self.assertIn("running", numbered)

    def test_gpu_util_normalized_and_capped(self):
        step = Step("1", "0", gpu_util_max=930.0, tres_alloc={"gres/gpu": 8})
        self.assertEqual(screens.step_gpu_util(step), 1.0)  # 930 over 8 GPUs, capped
        self.assertAlmostEqual(screens.step_gpu_util(Step("1", "0", gpu_util_max=50.0), 1), 0.5)
        self.assertIsNone(screens.step_gpu_util(Step("1", "0", gpu_util_max=50.0)))


class NoColorTest(unittest.TestCase):
    """9. NO_COLOR still painted white on black (ESC[37m ESC[40m)."""

    def test_no_color_sequences(self):
        import fcntl
        import pty
        import select
        import signal
        import struct
        import termios
        pid, fd = pty.fork()
        if pid == 0:  # child: draw and quit on q
            os.environ["TERM"] = "xterm-256color"
            try:
                fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 120, 0, 0))
                from slurm_usage.ui import terminal
                history = uf.FakeHistory()
                app = App(uf.FakeCollector(uf.make_snapshot(history.now)),
                          hd.HistoryData(history), "week", background=False)
                terminal.run(app, use_color=False)
            finally:
                os._exit(0)
        out = b""
        start = time.time()
        sent = False
        while time.time() - start < 20:
            ready, _, _ = select.select([fd], [], [], 0.3)
            if ready:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                out += chunk
            if not sent and (b"in use" in out or time.time() - start > 8):
                os.write(fd, b"q")
                sent = True
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        os.waitpid(pid, 0)
        os.close(fd)
        self.assertIn(b"in use", out)
        for code in (b"\x1b[37m", b"\x1b[40m", b"\x1b[39;49m\x1b[37m"):
            self.assertFalse(code in out, code)
        import re
        self.assertFalse(re.search(rb"\x1b\[(?:[0-9;]*;)?(?:3[0-7]|4[0-7]|38;5;\d+)m", out))


class CoverageTest(unittest.TestCase):
    """10. GPU util* showed a value measured over ~0% of the allocation, and 0 for rows
    with nothing to measure; --once left coverage out."""

    def test_cells(self):
        self.assertEqual(screens.eff_cell((0.96, 0.001))[0], fmt.NOT_YET)
        self.assertEqual(screens.eff_cell((0.96, 0.5))[0], "96%·50%")
        self.assertEqual(text._eff_text({"value": 0.93, "coverage": 0.62}), "93%·62%")
        self.assertEqual(text._eff_text({"value": 0.93, "coverage": 0.01}), fmt.NOT_YET)

    def test_roles_row_without_usage_is_not_a_zero_util(self):
        app = make_app()
        app.snapshot.roles.append(type(app.snapshot.roles[0])(account="dormant",
                                                               lineage=["root"]))
        app.go_tab("roles")
        line = next(l for l in draw(app, 160, 40).text().split("\n") if "dormant" in l)
        self.assertEqual(line.split()[-2], fmt.NOT_APPLICABLE)


class ConsistencyTest(unittest.TestCase):
    """11. Assorted inconsistencies."""

    def test_fail_counts_timeouts_and_help_says_so(self):
        Row = namedtuple("Row", "gpus nodes name account qos partition outcome gpu_seconds "
                                "job_id elapsed wait time_limit_used gpu_peak_utilization "
                                "cpu_efficiency cpu_seconds")
        rows = [Row(1, "a", "x", "p", "n", "g", outcome, 3600, str(i), 10, 1, None, None,
                    None, 0) for i, outcome in enumerate(("completed", "timeout"))]
        self.assertEqual(hd.run_types(rows, "gpu_count")[0].failure, 0.5)
        self.assertTrue(any("failed, node_fail or timeout" in t for t, _ in screens.HELP))

    def test_labels(self):
        app = make_app()
        overview = draw(app, 160, 50).text()
        self.assertIn("members", overview)
        app.go_tab("roles")
        self.assertIn("active", draw(app, 160, 50).text())
        app.go_tab("people")
        app.push("people", {"account": "physics"}, "People in physics")
        self.assertIn("of acct", draw(app, 160, 50).text())

    def test_completing_elapsed(self):
        now = time.time()
        job = Job(job_id="9", state="COMPLETING", start=int(now - 600), end=int(now - 60),
                  elapsed=0)
        self.assertEqual(screens.live_elapsed(job, now), 540)

    def test_timeline_lane_opens_its_range_detail(self):
        """A lane is an aggregate: it opens a chart of that lane, not a list of jobs."""
        app = make_app()
        app.key("t")
        for group in ("user", "account", "qos", "partition"):
            self.assertEqual(app.view.extra.get("lanes", "user"), group)
            draw(app, 160, 40)
            _, state, lines = app._focused()
            key = lines[state.cursor].key
            app.key("ENTER")
            self.assertEqual(app.view.kind, "range")
            self.assertIn(key, app.view.title)
            screen = draw(app, 160, 40).text()
            self.assertIn("in use · ", screen)
            self.assertNotIn("history ?", screen)
            app.key("ESC")
            app.key("g")

    def test_batch_script_message_for_an_sbatch_job(self):
        from slurm_usage import parse
        from slurm_usage.live import Fetched
        app = make_app(uf.FakeHistory())
        app.collector.batch_script = lambda job: Fetched(Status.NOT_STORED,
                                                         detail=parse.SCRIPT_NOT_STORED)
        app.collector.submit_line = lambda job: Fetched(
            Status.OK, "sbatch --partition=gpu --gres=gpu:4 " + "--export=" + "X" * 300
            + " run_the_long_experiment.sh")
        app.key("j")
        draw(app)
        app.key("ENTER")
        app.key("u")
        screen = draw(app, 100, 50).text()
        self.assertIn("run_the_long_experiment.sh", screen)  # wrapped, not cut off
        app.key("b")
        screen = draw(app, 100, 50).text()
        self.assertIn("this cluster does not", screen)
        self.assertNotIn("not a batch job", screen)

    def test_efficiency_label_and_steps_fallback(self):
        history = uf.FakeHistory()
        row = next(r for r in history.rows if r.outcome == "completed" and r.cpus)
        row.cpu_efficiency = None
        app = make_app(history)
        app.push("job", {"job_id": row.job_id}, "job")
        screen = draw(app, 160, 50).text()
        self.assertIn("efficiency CPU ", screen)
        self.assertIn("(from steps)", screen)

    def test_plural(self):
        self.assertEqual(fmt.plural(1, "GPU"), "1 GPU")
        self.assertEqual(fmt.plural(4, "GPU"), "4 GPUs")

    def test_anonymize_keeps_step_names_and_hides_role_like_qos(self):
        anon = Anonymizer()
        self.assertEqual(anon.step_name("batch", "batch"), "batch")
        self.assertEqual(anon.step_name("extern", "extern"), "extern")
        self.assertEqual(anon.step_name("3", "3"), "3")
        self.assertTrue(anon.step_name("train_secret", "0").startswith("job-"))
        snap = uf.make_snapshot()
        snap.qos.append(type(snap.qos[0])("physics", 1))  # a QOS named like an account
        snap.jobs[0].qos = "physics"
        anon.snapshot(snap)
        self.assertNotIn("physics", [q.name for q in snap.qos])
        self.assertEqual(snap.jobs[0].qos, anon.name("account", "physics"))
        self.assertEqual(snap.jobs[1].qos, "normal")  # ordinary QOS names stay


class CoordinatorFindingsTest(unittest.TestCase):
    def test_drained_and_not_responding_nodes_are_counted(self):
        from slurm_usage.live import summary
        nodes = [Node(name="a", state="IDLE", state_flags=["DRAIN"]),
                 Node(name="b", state="IDLE", state_flags=["NOT_RESPONDING"]),
                 Node(name="c", state="DOWN", state_flags=["DRAIN"]),
                 Node(name="d", state="MIXED")]
        states = summary.health(nodes, []).node_states
        self.assertEqual(fmt.down_drain(states), (2, 1))

    def test_anonymized_history_keeps_its_cache(self):
        history = uf.FakeHistory()
        data = hd.HistoryData(AnonHistory(history, Anonymizer()))
        self.assertIsNone(data.history.info().cache_path)
        self.assertTrue(data.persistent())
        self.assertEqual(data.limit_note(), "")
        self.assertIn("all", data.auto_ranges())

    def test_person_jobs_count_is_not_the_page_size(self):
        history = big_history()
        data = hd.HistoryData(history, clock=lambda: history.now)
        row = data.trend("all", hd.where(user="alice"))
        self.assertGreater(row.jobs, 500)


class UxTest(unittest.TestCase):
    def test_progress_never_goes_back_or_says_history_twice(self):
        from slurm_usage.history.errors import Progress
        data = hd.HistoryData(uf.FakeHistory())
        data._loading = "week"
        seen = []
        for event in (Progress("history", 50, 100), Progress("refresh", 10, 100),
                      Progress("rollup", 0, 10), Progress("rollup", 10, 10)):
            data._progress(event)
            seen.append(data.fraction)
        self.assertEqual(seen, sorted(seen))
        self.assertTrue(data.progress.startswith("week "))
        self.assertNotIn("history", data.progress)

    def test_people_and_roles_use_the_summary_while_computing(self):
        app = deferred_app()
        overview = settle(app)
        self.assertIn("physics", overview)
        app.go_tab("people")
        draw(app, 160, 40)  # the exact report is queued, not run
        screen = draw(app, 160, 40).text()
        self.assertIn("hourly summary", screen)
        line = next(l for l in screen.split("\n") if "alice" in l)
        self.assertNotIn(" ? ", line.split("alice", 1)[1][:60])

    def test_columns_share_the_width(self):
        columns = [Column("Name", 10, flex=True, most=24), Column("state", 8, flex=True,
                                                                   most=20),
                   Column("reason", 10, flex=True)]
        layout = fit_columns(columns, 160)
        self.assertEqual(layout[0][2], 24)
        self.assertEqual(layout[1][2], 20)
        self.assertEqual(sum(size for _, _, size in layout) + 2, 160)

    def test_nodes_mem_and_gpu_type(self):
        self.assertEqual(fmt.gpu_model("nvidia_h100_80gb_hbm3"), "h100")
        self.assertEqual(fmt.gpu_model("a100"), "a100")
        app = make_app(uf.FakeHistory())
        app.snapshot.nodes[0].mem_alloc = None
        app.snapshot.nodes[0].gpu_types = {"nvidia_h100_80gb_hbm3": 8}
        app.go_tab("nodes")
        screen = draw(app, 160, 30).text()
        self.assertIn("n/a/", screen)  # not tracked: the size only (see help: mem n/a/size)
        self.assertIn(" h100 ", screen)

    def test_sort_marker_in_header(self):
        pane = Pane("p", "P", [], Line([("Name ", ""), (" ", ""), ("GPU-h", "")]),
                    sorts=("GPU-h", "name"), natural="GPU-h")
        state = PaneState()
        self.assertIn("▼", "".join(t for t, _ in sort_marked(pane, state)[2:]))
        state.sort, state.descending = 1, False
        self.assertIn("Name▲", "".join(t for t, _ in sort_marked(pane, state)))

    def test_you_stay_pinned(self):
        app = make_app(uf.FakeHistory())
        app.go_tab("people")
        for _ in range(4):
            lines = [l for l in draw(app, 160, 40).text().split("\n") if "(you)" in l]
            self.assertTrue(lines)
            rows = draw(app, 160, 40).text().split("\n")
            self.assertIn("(you)", rows[3])  # first row under title and header
            app.key("s")

    def test_timeline_axis_legend_and_gpu_lanes(self):
        import re
        for size, shown in ((3600.0, 7 * 86400), (300.0, 86400 / 2), (86400.0, 365 * 86400)):
            axis = "".join(screens.timeline_axis(1_700_000_000, size, 140, shown)).strip()
            labels = re.split(r"\s{2,}", axis)
            self.assertEqual(len(labels), len(set(labels)), axis)  # no "Nov 16  Nov 16"
        app = make_app()
        app.key("t")
        screen = draw(app, 200, 40).text()
        self.assertIn("GPUs in use per person over time", screen)
        self.assertIn("GPUs held on average", screen)  # the scale of the lane shading
        self.assertRegex(screen, r"░≤[0-9.]+ ▒≤[0-9.]+ ▓≤[0-9.]+ █>[0-9.]+")
        app.key("c")
        self.assertIn("CPUs in use per person", draw(app, 200, 40).text())

        def held(metric):
            report = app.data.history.query(hd.window("week", app.data.clock()), None,
                                            series_by="user", series_metric=metric)
            lanes = screens.compute_lanes(report, 100)
            return {key: total for key, _, total, _, _ in lanes.lanes}
        gpu, cpu = held("gpu"), held("cpu")
        self.assertLessEqual(set(gpu), set(cpu))  # CPU-only jobs are not GPU lanes
        self.assertLess(sum(gpu.values()), sum(cpu.values()))

    def test_filter_select_all_ctrl_u_and_live(self):
        app = make_app(uf.FakeHistory())
        app.key("j")
        draw(app)
        app.key("/")
        app.key("1")
        app.key("0")
        app.key("3")
        state = app.view.panes["queue"]
        self.assertEqual(state.filter, "103")  # applied while typing
        app.key("ENTER")
        app.key("/")
        app.key("x")  # the old filter was selected: replaced
        self.assertEqual(state.filter, "x")
        app.key("CTRL_U")
        self.assertEqual(state.filter, "")
        app.key("ESC")  # cancel puts the previous one back
        self.assertEqual(state.filter, "103")

    def test_help_page_down(self):
        app = make_app(uf.FakeHistory())
        app.key("?")
        draw(app, 100, 20)
        app.key("PGDN")
        self.assertGreater(app.view.panes["help"].cursor, 10)

    def test_overview_accounts_sorted_as_labelled(self):
        app = make_app()
        draw(app, 160, 50)
        pane = screens.accounts_pane(screens.Context(app, 160))
        hours = [l.sort["GPU-h"] for l in pane.lines]
        self.assertEqual(hours, sorted(hours, reverse=True))
        self.assertEqual(pane.natural, "GPU-h")
        self.assertIn("GPU-h▼", draw(app, 160, 50).text())


class EfficiencyRollupTest(unittest.TestCase):
    """E (load efficiency) left the overview partial (month/year '≥…', Δ '?', person share
    '?') until r: the step usage it writes marks its hours dirty in the rollup, and only
    a later history load rebuilt them."""

    def _history(self):
        from tests.test_history_support import H, T0, FakeSlurm, job, make_history, step
        now = T0 + 40 * 86400
        jobs = [job(i, start=now - (i + 2) * 86400, end=now - (i + 2) * 86400 + 4 * H,
                    gpus=4, cpus=8, user="erin", uid=7, TotalCPU="16:00:00",
                    steps=[step("batch", TotalCPU="16:00:00",
                                TRESUsageInTot="cpu=16:00:00,mem=32G,gres/gpuutil=200")])
                for i in range(1, 30)]
        fake = FakeSlurm(jobs, now=now, live=[])
        history = make_history(fake)
        history.refresh()
        history.build_rollup()
        return history, fake

    def test_raw_load_marks_rollup_dirty(self):
        history, _ = self._history()
        self.assertTrue(all(r.complete for r in history.overview(8, 64).ranges))
        history.load_efficiency(where=hd.where(user="erin"))
        self.assertIsNotNone(history.overview(8, 64).pending_rollup)  # the mechanism

    def test_ui_load_efficiency_leaves_overview_complete(self):
        history, fake = self._history()
        data = hd.HistoryData(history, clock=lambda: fake.now)
        before = data.generation
        self.assertGreater(data.load_efficiency("year", hd.where(user="erin")), 0)
        self.assertGreater(data.generation, before)
        found = history.overview(8, 64)
        self.assertIsNone(found.pending_rollup)
        self.assertTrue(all(r.complete for r in found.ranges), [r.name for r in found.ranges])
        gpu = next(r for r in found.ranges if r.name == "month").total.efficiency("gpu")
        self.assertAlmostEqual(gpu[0], 0.5)  # the measured usage reached the rollup
        self.assertEqual(data.progress, "")


class EfficiencyMessageTest(unittest.TestCase):
    """'loading efficiency for <range> · <user>' stayed in the footer after the load
    finished, until the next key."""

    def test_completion_message_replaces_and_expires(self):
        app = make_app()
        app.key("m")
        draw(app, 160, 40)
        app.key("E")  # no worker threads here: the load runs before key() returns
        footer = draw(app, 160, 40).row_text(39)
        self.assertNotIn("loading efficiency", footer)
        self.assertIn("efficiency loaded for week", footer)
        app._flash = (app.message, time.time() - 1)  # its time is up
        footer = draw(app, 160, 40).row_text(39)
        self.assertNotIn("efficiency loaded", footer)
        self.assertEqual(app.message, "")

    def test_newer_message_is_not_overwritten_or_cleared(self):
        app = make_app()
        queued = []
        app.loader.submit = queued.append  # what a worker thread runs later
        app.key("m")
        app.key("E")
        self.assertIn("loading efficiency", app.message)
        app.message = "range: month"  # something newer while it loads
        queued.pop()()
        self.assertEqual(app.message, "range: month")
        draw(app, 160, 40)
        self.assertEqual(app.message, "range: month")


class TimelineHeaderAndSelectionTest(unittest.TestCase):
    def test_header_note_drops_whole_parts_never_mid_word(self):
        """At 120 columns the note was cut mid-word ('...with GPUs hel')."""
        app = make_app()
        app.key("t")
        for width in (80, 120, 160, 240):
            title = draw(app, width, 30).row_text(1).rstrip()
            self.assertLessEqual(fmt.width(title), width)
            self.assertTrue(title.endswith("╮"), title)  # the box closes: nothing cut it
            self.assertIn(" avg ", title)  # the selected lane, in the title row
            # the shade scale: a row of its own under the lanes, whole at every width
            screen = draw(app, width, 30).text().split("\n")
            scale = [row for row in screen if "shade =" in row]
            self.assertEqual(len(scale), 1, width)
            self.assertRegex(scale[0], r"GPUs held on average +│$")

    def test_a_single_long_note_is_ellipsized(self):
        canvas = Canvas(40, 3)
        pane = Pane("p", "Title", [Line([("x", "")])], status="a very long note about many things")
        draw_pane(canvas, 0, 3, pane, PaneState(), True, pane.lines)
        title = canvas.row_text(0).rstrip()
        self.assertTrue(title.endswith("… ─╮"), title)
        self.assertLessEqual(fmt.width(title), 40)

    def test_selected_lane_keeps_its_bars(self):
        """The selected lane was painted over in the selection color, bars and all."""
        app = make_app()
        app.key("t")
        canvas = draw(app, 120, 30)
        selected = lambda style: style.endswith("@sel")
        row = next(r for r in range(canvas.height)
                   if any(selected(style) for _, style in canvas.cells[r]))
        styles = [style for _, style in canvas.cells[row]]
        label = min(16, max(8, 120 // 6)) + 3  # the border, the padding, the label, a space
        self.assertTrue(all(selected(style) for style in styles[1:label]))
        self.assertFalse(any(selected(style) for style in styles[label:]))
        self.assertTrue(any(style.startswith("heat") for style in styles[label:]))


class TrendJobCountTest(unittest.TestCase):
    def test_trend_jobs_is_the_distinct_count_not_a_page(self):
        """The trend's 'jobs' (512 for both year and all for the demo's erin, whose
        history is 120 days) is every distinct started job, not a page of the list."""
        history, fake = uf.real_history(days=120)
        data = hd.HistoryData(history, clock=lambda: fake.now)
        data.load("all")
        started = {int(j["JobIDRaw"]) for j in fake.jobs
                   if j.get("Start") is not None and j["Start"] <= fake.now}
        self.assertGreater(len(started), hd.JOB_PAGE)
        for name in ("year", "all"):
            self.assertEqual(data.trend(name).jobs, len(started), name)
        self.assertEqual(data.jobs("all", limit=10 ** 6).total, len(started))


if __name__ == "__main__":
    unittest.main()
