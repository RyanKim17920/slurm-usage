"""Aggregates open charts and breakdowns; job lists are one key away.

The Me/Person page answers "where did my GPU-hours go, under which roles" without a
keypress; a kind of run or a role opens its breakdown screen; a Timeline lane opens
its range detail; Range detail's previous-period overlay toggles with v."""

import unittest
from collections import namedtuple

from slurm_usage.ui import breakdown, screens
from slurm_usage.ui import data as hd
from tests.test_range_detail import find
from tests.test_ui import assert_clean, draw, make_app

Row = namedtuple("Row", "job_id start end gpus gpu_seconds elapsed wait outcome name user")


def focus(app, pane_id, width=120, height=34):
    for _ in range(10):
        draw(app, width, height)
        pane = app._focused()[0]
        if pane is not None and pane.id == pane_id:
            return pane
        app.key("TAB")
    raise AssertionError("no pane " + pane_id)


class PureTest(unittest.TestCase):
    def test_held_per_bin_splits_an_attempt_across_bins(self):
        rows = [Row("1", 0, 3600, 2, 7200, 3600, 0, "completed", "a", "x"),
                Row("2", 1800, None, 4, 0, None, 0, "running", "b", "x"),
                Row("3", 100, 200, None, None, 100, 0, "completed", "c", "x")]
        out = breakdown.held_per_bin(rows, [0, 1800, 3600, 5400], now=5400)
        self.assertEqual(out[None], [2 * 1800, 2 * 1800 + 4 * 1800, 4 * 1800])
        keyed = breakdown.held_per_bin(rows, [0, 5400], 5400, lambda r: r.name)
        self.assertEqual(keyed, {"a": [7200.0], "b": [4 * 3600.0]})

    def test_detail_quantiles_outcomes_and_groups(self):
        rows = [Row(str(i), 0, 10, 1, 3600 * (i + 1), 60 * (i + 1), 10 * i,
                    "failed" if i % 4 == 0 else "completed", "train_%d" % i if i < 6 else "eval",
                    "u%d" % (i % 2)) for i in range(10)]
        info = breakdown.detail(rows)
        self.assertEqual((info.attempts, info.jobs), (10, 10))
        self.assertEqual(info.runtime, (300, 540))  # nearest rank median and p90
        self.assertEqual(info.wait, (40, 80))
        self.assertAlmostEqual(info.failure, 0.3)
        self.assertEqual([g.key for g in info.groups], ["eval", "train_<n>"])
        people = breakdown.detail(rows, "user")
        self.assertEqual(sorted(g.key for g in people.groups), ["u0", "u1"])
        self.assertEqual(breakdown.outcome_split(info.outcomes),
                         [("completed", 7), ("failed", 3)])

    def test_proportional_cells_sum_and_keep_small_parts(self):
        for counts in ([1, 1, 1], [1000, 1], [5, 0, 3], [0.2, 0.3, 0.5]):
            for size in (3, 10, 37):
                cells = breakdown.proportional(counts, size)
                self.assertEqual(sum(cells), size, (counts, size))
                for count, cell in zip(counts, cells):
                    if count:
                        self.assertGreaterEqual(cell, 1)

    def test_lane_bounds_increase_and_levels(self):
        for values in ([0.1] * 50, [1, 2, 3, 4, 50, 60], list(range(1, 200)), [], [0.3, 0.3]):
            bounds = screens.lane_bounds(values)
            self.assertEqual(len(bounds), 3)
            self.assertTrue(bounds[0] < bounds[1] < bounds[2], (values, bounds))
        bounds = [4.0, 16.0, 32.0]
        self.assertEqual([screens.lane_level(v, bounds) for v in (0, 0.2, 4, 5, 32, 33)],
                         [0, 1, 1, 2, 3, 4])


class MePageTest(unittest.TestCase):
    def setUp(self):
        self.app = make_app()
        self.app.key("m")

    def test_gpu_hours_and_roles_are_on_the_page_without_a_key(self):
        for width, height in ((80, 24), (120, 34), (80, 60)):
            canvas = draw(self.app, width, height)
            screen = canvas.text()
            assert_clean(self, canvas)
            self.assertIn("Where your GPU-hours went", screen)
            self.assertIn("Roles", screen)
            self.assertIn("account", screen)  # the first role row, not behind a tab
            self.assertNotIn("What kinds of runs", screen)
            # the four boxes: Trend, Now, the GPU-hour breakdown, Roles (+ tabs)
            panes = self.app._last[0]
            shown = [p.id for p in panes if not p.group or p.id == "roles"]
            self.assertEqual(shown[:4], ["trend", "live", "runs", "roles"])

    def test_g_cycles_gpus_job_name_and_outcome(self):
        seen = []
        for _ in range(3):
            seen.append(draw(self.app, 120, 34).text())
            self.app.key("g")
        for screen, word in zip(seen, ("by GPUs", "by job name", "by outcome")):
            self.assertIn(word, screen)
        self.assertIn("by GPUs", draw(self.app, 120, 34).text())

    def test_a_kind_of_run_opens_its_breakdown_and_j_its_jobs(self):
        focus(self.app, "runs")
        self.app.key("DOWN")
        self.app.key("ENTER")
        view = self.app.view
        self.assertEqual(view.kind, "runtype")
        self.assertEqual(view.params["dim"], "gpu_count")
        screen = draw(self.app, 120, 34).text()
        for word in ("GPU-h", "of alice's GPU-h", "attempts /", "outcomes", "runtime",
                     "median", "p90", "wait", "Job names", "GPUs in use"):
            self.assertIn(word, screen)
        self.app.key("j")
        self.assertEqual(self.app.view.kind, "jobs")
        self.assertEqual(self.app.view.params["run_type"][0], "gpu_count")
        assert_clean(self, draw(self.app, 120, 34))
        self.app.key("ESC")
        # a job-name family in the breakdown drills further
        focus(self.app, "parts")
        self.app.key("ENTER")
        self.assertEqual((self.app.view.kind, self.app.view.params["dim"]),
                         ("runtype", "name_family"))
        assert_clean(self, draw(self.app, 120, 34))
        self.app.key("RIGHT")  # ranges step in place
        self.assertEqual((self.app.view.kind, self.app.range), ("runtype", "month"))

    def test_the_stacked_bar_opens_the_persons_chart(self):
        focus(self.app, "runs")
        self.app.key("HOME")
        self.app.key("ENTER")
        self.assertEqual(self.app.view.kind, "range")
        self.assertEqual(self.app.view.params.get("user"), "alice")

    def test_a_role_opens_its_breakdown(self):
        focus(self.app, "roles")
        self.app.key("ENTER")
        view = self.app.view
        self.assertEqual((view.kind, view.params["dim"]), ("runtype", "account"))
        self.assertIn("account", view.title)
        screen = draw(self.app, 120, 34).text()
        self.assertIn("GPUs in use · alice · account", screen)


class OtherRoutesTest(unittest.TestCase):
    def setUp(self):
        self.app = make_app()

    def test_range_detail_run_sizes_open_breakdowns(self):
        self.app.push("range", {"account": "physics"}, "Week · physics")
        draw(self.app, 120, 34)
        for _ in range(3):
            if self.app._focused()[0].id == "by_size":
                break
            self.app.key("TAB")
            draw(self.app, 120, 34)
        pane = self.app._focused()[0]
        if pane.id != "by_size":  # a tab of the breakdown box
            self.app.view.focus = next(i for i, p in enumerate(self.app._last[0])
                                       if p.id == "by_size")
        self.app.key("ENTER")
        self.assertEqual(self.app.view.kind, "runtype")
        screen = draw(self.app, 120, 34).text()
        self.assertIn("People", screen)  # an account's kind of run: by person
        focus(self.app, "parts")
        self.app.key("ENTER")
        self.assertEqual(self.app.view.kind, "runtype")
        self.assertIn("user", self.app.view.params)

    def test_timeline_lane_click_opens_its_chart(self):
        app = self.app
        app.key("t")
        app.key("g")  # by account
        canvas = draw(app, 120, 34)
        _, _, lines = app._focused()
        name = lines[1].key
        row, col = find(canvas, name, range(2, 34))
        app.click(col + 1, row)  # one click: lanes are buttons
        self.assertEqual((app.view.kind, app.view.params.get("account")), ("range", name))
        self.assertIn("GPUs in use · " + name, draw(app, 120, 34).text())
        app.key("ESC")
        self.assertEqual(app.view.kind, "timeline")

    def test_qos_and_partition_range_details(self):
        for role, value in (("qos", "normal"), ("partition", "gpu")):
            self.app.push("range", {role: value}, "x")
            canvas = draw(self.app, 120, 34)
            assert_clean(self, canvas)
            screen = canvas.text()
            self.assertIn("GPUs in use · %s %s" % ("QOS" if role == "qos" else role, value),
                          screen)
            self.assertIn("cluster", screen)  # the ceiling drawn is the cluster's
            self.app.key("ESC")

    def test_v_hides_and_shows_the_previous_period(self):
        self.app.push("range", {"range": "week"}, "Week · cluster")
        legend = "─╮ previous week"  # the overlay's entry in the legend under the chart
        self.assertIn(legend, draw(self.app, 160, 40).text())
        self.app.key("v")
        self.assertNotIn(legend, draw(self.app, 160, 40).text())
        self.assertIn("hidden", self.app.message)
        self.app.key("v")
        self.assertIn(legend, draw(self.app, 160, 40).text())

    def test_role_events_open_a_breakdown(self):
        Event = namedtuple("Event", "at kind text style rows job_id where window")
        first = Event(0, "first_account", "First use of account x", "plain", [], None,
                      {"account": {"vision"}}, None)
        target = screens.event_target(first, "alice")
        self.assertEqual((target.kind, target.params["dim"], target.params["key"]),
                         ("runtype", "account", "vision"))
        record = Event(0, "record_day", "Record day", "good", [], None, None,
                       hd.Window(0, 86400))
        self.assertEqual(screens.event_target(record, "alice").kind, "range")


if __name__ == "__main__":
    unittest.main()
