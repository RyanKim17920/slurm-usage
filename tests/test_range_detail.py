"""Range detail: the binned series, the overview's hero and tiles, and every way into a
range (mouse and keys) at the sizes people use."""

import itertools
import time
import unittest

from slurm_usage.ui import data as hd
from slurm_usage.ui import fmt, plot
from slurm_usage.ui import text as textmod
from slurm_usage.ui.canvas import Canvas
from slurm_usage.ui.widgets import Line, Pane, PaneState
from slurm_usage.ui.app import keep_selection
from tests import ui_fakes as uf
from tests.test_ui import assert_clean, draw, make_app

SIZES = ((80, 24), (120, 34), (200, 50))
HOUR, DAY = 3600, 86400


def find(canvas, needle, rows=None):
    """(row, col) of the first ``needle`` on screen (cells, not characters)."""
    for row in rows if rows is not None else range(canvas.height):
        cells = canvas.cells[row]
        for col in range(canvas.width):
            text = "".join(c for c, _ in cells[col:col + len(needle) * 2])
            if text.startswith(needle):
                return row, col
    raise AssertionError("%r not on screen:\n%s" % (needle, canvas.text()))


class SeriesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        history, cls.fake = uf.real_history(days=40)
        cls.data = hd.HistoryData(history, clock=lambda: cls.fake.now)
        cls.data.load_in_order("all")

    def test_series_sum_to_the_trend_totals(self):
        for filt in (None, hd.where(user="alice"), hd.where(account="physics")):
            for name in hd.RANGE_NAMES:
                series = self.data.range_series(name, filt, ("account", "user"), None, 20, 384)
                trend = self.data.trend(name, filt, 20, 384)
                self.assertEqual(sum(series.gpu), series.total.gpu_seconds, name)
                self.assertEqual(sum(series.cpu), series.total.cpu_seconds, name)
                self.assertAlmostEqual(series.total.gpu_hours, trend.gpu_hours, 6)
                self.assertAlmostEqual(series.total.cpu_hours, trend.cpu_hours, 6)
                self.assertAlmostEqual(series.gpu_util or 0, trend.gpu_util or 0, 9)
                if name == "all":
                    self.assertIsNone(series.prev_gpu)
                else:
                    self.assertEqual(sum(series.prev_gpu), series.previous.gpu_seconds, name)
                    self.assertAlmostEqual(series.previous.gpu_hours, trend.gpu_delta, 6)
                # the groups split the total exactly
                for role in ("account", "user"):
                    self.assertEqual(sum(s.gpu_seconds for _, s in series.groups[role]),
                                     series.total.gpu_seconds)
                # bins tile the window: contiguous, increasing, from start to end
                edges = series.edges
                self.assertEqual((edges[0], edges[-1]),
                                 (max(series.start, series.since), series.end))
                self.assertTrue(all(a < b for a, b in zip(edges, edges[1:])), name)
                self.assertEqual(len(edges) - 1, len(series.gpu))
                average = series.average("gpu")
                self.assertAlmostEqual(
                    sum(v * (b - a) for v, a, b in zip(average, edges, edges[1:])),
                    series.total.gpu_seconds, 3)

    def test_cluster_detail_is_the_tiles_figures(self):
        """The cluster's long ranges come from the rollup; every figure matches the
        overview's (the tiles) and the exact query over the same window."""
        overview = self.data.overview(20, 384)
        for summary in overview.ranges:
            name = summary.name
            series = self.data.rollup_series(name, 20, 384)
            self.assertEqual(sum(series.gpu), summary.total.gpu_seconds, name)
            self.assertEqual(sum(series.cpu), summary.total.cpu_seconds, name)
            self.assertEqual(series.gpu_util, summary.gpu_utilization)
            exact = self.data.report(name, None, group_by=(("account",), ("user",)),
                                     win=hd.Window(None if name == "all" else summary.start,
                                                   summary.end))
            self.assertEqual(series.total.gpu_seconds, exact.total.gpu_seconds, name)
            got = {k: v.gpu_seconds for k, v in series.groups["user"]}
            want = {k[0]: v.gpu_seconds for k, v in exact.group("user").items()}
            self.assertEqual({k: v for k, v in got.items() if v},
                             {k: v for k, v in want.items() if v}, name)
            if name in ("month", "year") and series.prev_gpu is not None:
                self.assertEqual(sum(series.prev_gpu), summary.previous.gpu_seconds, name)

    def test_average_over_capacity_is_the_utilization(self):
        """One span for both: where history begins inside a range, the average and the
        utilization are both over the covered span, and the chart starts there."""
        cases = [("cluster", lambda n: self.data.rollup_series(n, 20, 384))]
        for filt in (None, hd.where(user="alice"), hd.where(account="physics")):
            cases.append((str(filt), lambda n, f=filt: self.data.range_series(
                n, f, (), None, 20, 384)))
        for label, make in cases:
            for name in hd.RANGE_NAMES:
                series = make(name)
                for metric, cap in (("gpu", 20), ("cpu", 384)):
                    util = series.gpu_util if metric == "gpu" else series.cpu_util
                    self.assertAlmostEqual(series.mean(metric) / cap, util, 6,
                                           (label, name, metric))
                self.assertEqual(series.edges[0], max(series.start, series.since),
                                 (label, name))
                self.assertEqual(sum(series.gpu), series.total.gpu_seconds, (label, name))
        # 40 days of history: a year begins at the data, drawn by the day
        year = self.data.rollup_series("year", 20, 384)
        self.assertGreater(year.since, year.start + 300 * DAY)
        self.assertEqual(year.unit, "day")

    def test_bins_per_range(self):
        # (a year of a 40-day history is drawn by the day: it starts at the data)
        expected = {"day": ("hour", 24, 26), "week": ("3hour", 56, 58),
                    "month": ("day", 30, 32), "year": ("day", 40, 42)}
        for name, (unit, low, high) in expected.items():
            series = self.data.range_series(name, None, (), None, 20, 384)
            self.assertEqual(series.unit, unit, name)
            self.assertTrue(low - 1 <= len(series.gpu) <= high, (name, len(series.gpu)))
        self.assertEqual(hd.chart_unit("year", 0, 365 * DAY, 0), "week")
        self.assertEqual(hd.chart_unit("year", 0, 365 * DAY, 340 * DAY), "day")
        self.assertEqual(hd.bin_unit("all", 0, 40 * DAY), "day")
        self.assertEqual(hd.bin_unit("all", 0, 400 * DAY), "week")
        self.assertEqual(hd.bin_unit("all", 0, 4 * 365 * DAY), "month")

    def test_bin_edges_follow_the_clock_and_rebin_keeps_every_second(self):
        start = int(time.mktime((2026, 3, 3, 10, 17, 0, 0, 1, -1)))
        end = start + 30 * DAY
        for unit in ("hour", "3hour", "day", "week", "month"):
            edges = hd.bin_edges(start, end, unit)
            inner = edges[1:-1]
            for at in inner:
                local = time.localtime(at)
                self.assertEqual((local.tm_min, local.tm_sec), (0, 0), unit)
                if unit in ("day", "week", "month"):
                    self.assertEqual(local.tm_hour, 0, unit)
                if unit == "week":
                    self.assertEqual(local.tm_wday, 0)
                if unit == "month":
                    self.assertEqual(local.tm_mday, 1)
        hours = list(range(start - start % HOUR, end + HOUR, HOUR))
        values = [i % 7 for i in range(len(hours) - 1)]
        for unit in ("hour", "3hour", "day", "week"):
            edges = hd.bin_edges(hours[0], hours[-1], unit)
            self.assertEqual(sum(hd.rebin(hours, values, edges)), sum(values), unit)
            shifted = hd.rebin(hours, values, [e + 7 * DAY for e in edges], 7 * DAY)
            self.assertEqual(sum(shifted), sum(values), unit)


class ChartTest(unittest.TestCase):
    def chart(self, **kw):
        edges = [i * HOUR for i in range(25)]
        values = [float(i % 9) for i in range(24)]
        args = dict(previous=[3.0] * 24, capacity=12.0, mean=4.0, now=6.0, clock=24 * HOUR)
        args.update(kw)
        return plot.Chart(values, edges, "hour", **args)

    def test_every_row_is_exactly_the_width(self):
        for width in (20, 33, 60, 117, 196):
            for rows in (1, 3, 6, 12, 30):
                for ascii_only in (False, True):
                    for color in (False, True):
                        out = plot.area_chart(self.chart(), width, rows, ascii_only, color)
                        self.assertEqual(len(out), rows + 1)
                        for line in out:
                            self.assertEqual(sum(fmt.width(t) for t, _ in line), width,
                                             (width, rows, ascii_only))

    def test_ascii_and_no_color(self):
        out = plot.area_chart(self.chart(), 90, 10, ascii_only=True, color=False)
        text = "\n".join("".join(t for t, _ in line) for line in out)
        self.assertTrue(all(ord(c) < 128 for c in text), text)
        for word in ("capacity 12", "avg 4", "now 6 (live)", "peak hour avg 8"):
            self.assertIn(word, text)
        self.assertIn("=", text)  # capacity
        self.assertIn("#", text)  # the area
        styles = {s for line in out for _, s in line}
        self.assertFalse(any(s.startswith("heat") for s in styles), styles)
        colored = plot.area_chart(self.chart(), 90, 10, color=True)
        self.assertTrue(any(s.startswith("heat") for line in colored for _, s in line))

    def test_bars_for_wide_bins_and_area_for_narrow_ones(self):
        bars = plot.area_chart(self.chart(previous=None, mean=None, now=None,
                                          capacity=None), 100, 6, ascii_only=True)
        bottom = "".join(t for t, _ in bars[-2])
        self.assertIn("# ", bottom.strip())  # gaps between bars
        edges = [i * HOUR for i in range(201)]
        area = plot.area_chart(plot.Chart([5.0] * 200, edges, "hour"), 60, 6, ascii_only=True)
        self.assertNotIn(" ", "".join(t for t, _ in area[-2])[4:].strip())

    def test_axis_labels_and_scale(self):
        out = plot.area_chart(self.chart(), 100, 12)
        gutter = ["".join(t for t, _ in line).split("┤")[0].strip() for line in out[:-1]]
        labels = [g for g in gutter if g]
        self.assertEqual(labels[-1], "0")
        self.assertGreaterEqual(float(labels[0]), 12)  # the capacity fits under the top
        self.assertTrue(out[-1][1][0].rstrip().endswith("now"))
        self.assertEqual(plot.nice_top(56 * 1.04), (20, 60))
        self.assertEqual(plot.nice_top(0.4)[1], 0.4)

    def test_previous_period_is_one_connected_line(self):
        """The overlay is a line that steps between levels, never scattered dashes."""
        edges = [i * DAY for i in range(31)]
        previous = [float((i * 7) % 23 + 2) for i in range(30)]
        values = [float((i * 5) % 31) for i in range(30)]  # bars above and below it
        for (width, rows), bars in itertools.product(((120, 30), (80, 50), (60, 12), (200, 8)),
                                                     ([0.0] * 30, values)):
            chart = plot.Chart(bars, edges, "day", previous=previous, clock=30 * DAY)
            out = plot.area_chart(chart, width, rows)
            grid = []
            for line in out[:-1]:
                cells = []
                for text, style in line:
                    cells += [(char, style) for char in text]
                grid.append(cells)
            past = lambda style: style.startswith("past")
            cols = [c for c in range(width) if any(past(row[c][1]) for row in grid)]
            self.assertTrue(cols)
            self.assertEqual(cols, list(range(cols[0], cols[-1] + 1)), "a gap in the line")
            for col in cols:
                rows_here = [r for r in range(rows) if past(grid[r][col][1])]
                self.assertEqual(rows_here, list(range(rows_here[0], rows_here[-1] + 1)))
                glyphs = "".join(grid[r][col][0] for r in rows_here)
                if len(rows_here) == 1:
                    self.assertEqual(glyphs, "─", (width, col))
                else:  # a step: a corner at each end, a stroke between
                    self.assertIn(glyphs[0], "╭╮")
                    self.assertIn(glyphs[-1], "╯╰")
                    self.assertTrue(set(glyphs[1:-1]) <= {"│"}, glyphs)
                if col + 1 in cols:  # it leaves this column on the row the next one starts
                    leave = rows_here[0] if grid[rows_here[0]][col][0] in "╭─" and \
                        grid[rows_here[-1]][col][0] in "╯─" else rows_here[-1]
                    nxt = [r for r in range(rows) if past(grid[r][col + 1][1])]
                    starts = {r for r in nxt if grid[r][col + 1][0] in "─╯╮"}
                    self.assertIn(leave, starts, (width, rows, col))

    def test_previous_line_is_drawn_over_the_bars(self):
        """Last period lower than this one: the line runs through the bars, in bright
        amber on the bar's own color, and is never hidden behind them."""
        edges = [i * DAY for i in range(31)]
        chart = plot.Chart([40.0] * 30, edges, "day", previous=[20.0] * 30, clock=30 * DAY)
        out = plot.area_chart(chart, 100, 20)
        over = [(t, s) for line in out for t, s in line if s.startswith("pastover")]
        text = "".join(t for t, _ in over)
        self.assertGreaterEqual(len(text), 60)  # along the whole width, bars and gaps
        self.assertTrue(set(text) <= set("─╭╮╰╯│"), text)
        self.assertTrue(any(s == "pastover@bar" for _, s in over))  # the bar stays behind
        mono = plot.area_chart(chart, 100, 20, color=False)  # bold across the bars
        self.assertTrue(any("─" in t for line in mono for t, s in line if s == "+b"))

    def test_previous_line_steps_in_the_gaps_between_bars(self):
        edges = [i * DAY for i in range(11)]
        chart = plot.Chart([1.0] * 10, edges, "day", previous=[5.0, 9.0] * 5, clock=10 * DAY)
        text = "\n".join("".join(t for t, _ in line) for line in plot.area_chart(chart, 80, 20))
        for glyph in "─╭╮╰╯│":
            self.assertIn(glyph, text)
        self.assertNotIn("╌", text)
        ascii_text = "\n".join("".join(t for t, _ in line)
                               for line in plot.area_chart(chart, 80, 20, ascii_only=True))
        self.assertTrue(all(ord(c) < 128 for c in ascii_text))

    def test_axis_holds_capacity_now_and_previous(self):
        edges = [i * DAY for i in range(31)]
        values = [float(20 + i % 29) for i in range(30)]  # peak 48
        for capacity, now, prev in ((64, 52, 30), (64, 70, 30), (30, 20, 90), (56, 40, 40)):
            chart = plot.Chart(values, edges, "day", previous=[float(prev)] * 30,
                               capacity=float(capacity), now=float(now), clock=30 * DAY)
            for rows in (6, 12, 30, 55):
                step, top = plot.axis_top(chart, rows)
                self.assertGreaterEqual(top, max(capacity, now, prev, 48))
                self.assertEqual(top % step, 0)
                self.assertIn(step, (1, 2, 2.5, 5, 10, 20, 25, 50, 100))
                out = plot.area_chart(chart, 100, rows)
                text = "\n".join("".join(t for t, _ in line) for line in out)
                ticks = [l.split("┤")[0].strip() for l in text.split("\n") if "┤" in l]
                self.assertGreaterEqual(float(ticks[0]), capacity)
                if capacity > 48:  # above the bars: the capacity line is in sight
                    self.assertIn("┈", text)
        # 64 GPUs: the tall chart's top is a round 70, not 60 and not a wasteful 80
        chart = plot.Chart(values, edges, "day", capacity=64.0, now=52.0)
        self.assertEqual(plot.axis_top(chart, 47)[1], 70)

    def test_now_and_peak_labels_say_what_they_are(self):
        edges = [i * DAY for i in range(31)]
        values = [float(20 + i % 29) for i in range(30)]
        chart = plot.Chart(values, edges, "day", capacity=64.0, now=52.0, mean=23.0,
                           clock=30 * DAY)
        text = "\n".join("".join(t for t, _ in line) for line in plot.area_chart(chart, 100, 30))
        self.assertIn("now 52 (live)", text)
        self.assertIn("peak day avg 48", text)
        self.assertIn("capacity 64", text)
        self.assertNotIn("▾ peak 48", text)
        for width in (24, 40, 60, 80, 120, 200):
            row = plot.legend(chart, "month", width=width)
            self.assertEqual(sum(fmt.width(t) for t, _ in row), width)
        full = "".join(t for t, _ in plot.legend(chart, "month", width=120))
        self.assertIn("GPUs in use · daily average", full)
        self.assertIn("now (live)", full)
        week = plot.Chart(values, edges, "3hour", previous=values, capacity=64.0)
        narrow = "".join(t for t, _ in plot.legend(week, "week", width=50))
        self.assertIn("prev", narrow)
        self.assertIn("3-hourly", narrow)

    def test_big_slash_and_percent_are_drawn_not_smeared(self):
        rows = plot.big_rows("/%")
        self.assertEqual([r.count("╱") for r in rows], [2, 2, 2])
        self.assertFalse(any("▄▀" in r or "▀▄" in r for r in rows), rows)
        ascii_rows = plot.big_rows("/%", True)
        self.assertEqual(ascii_rows, ["  / o /", " /   / ", "/   / o"])

    def test_big_figures(self):
        for ascii_only in (False, True):
            rows = plot.big_rows("40/56 71%", ascii_only)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({fmt.width(r) for r in rows}), 1)
            self.assertEqual(plot.big_width("40/56", ascii_only), fmt.width(rows[0][:19]))
        self.assertTrue(all(ord(c) < 128 for r in plot.big_rows("0123456789/%", True) for c in r))


class RoutingTest(unittest.TestCase):
    def setUp(self):
        self.app = make_app()

    def test_tiles_open_their_range_by_click_and_enter(self):
        for width, height in SIZES:
            for name in hd.RANGE_NAMES:
                self.app.go_tab("overview")
                canvas = draw(self.app, width, height)
                row, col = find(canvas, name.capitalize(), range(5, 8))
                self.app.click(col + 1, row + 2)  # inside the tile, not on its title
                self.assertEqual((self.app.view.kind, self.app.range), ("range", name),
                                 (width, height, name))
                screen = draw(self.app, width, height).text()
                self.assertIn("in use · orion", screen)
                self.app.key("ESC")
                self.assertEqual(self.app.view.kind, "overview")
            # keys: the tiles are focused first; arrows pick the range, Enter opens it
            self.app.go_tab("overview")
            draw(self.app, width, height)
            self.app.key("1")
            for name in hd.RANGE_NAMES:
                self.assertEqual(self.app.range, name)
                self.app.key("ENTER")
                self.assertEqual((self.app.view.kind, self.app.range), ("range", name))
                self.app.key("ESC")
                draw(self.app, width, height)
                self.app.key("RIGHT")

    def test_roles_rows_open_the_row_clicked_in_every_grouping(self):
        """A click selects, a second click on the same row opens it: the same for the
        first row as for any other, in each grouping, and Esc comes back to it."""
        for width, height in ((120, 34), (80, 60)):
            for groups in (0, 1, 2):
                self.app.go_tab("roles")
                for _ in range(groups):
                    self.app.key("g")
                for index in (0, 1):
                    canvas = draw(self.app, width, height)
                    pane, state, lines = self.app._focused()
                    name = lines[index].key
                    row, col = find(canvas, name, range(2, height))
                    self.app.click(col + 1, row)
                    self.assertEqual(self.app.view.kind, "roles", (width, groups, index))
                    draw(self.app, width, height)
                    self.app.click(col + 1, row)
                    self.assertEqual(self.app.view.kind, "people", (width, groups, name))
                    self.assertIn(name, self.app.view.title)
                    self.app.key("ESC")
                    draw(self.app, width, height)
                    _, state, lines = self.app._focused()
                    self.assertEqual(lines[state.cursor].key, name)
                    self.app.view.panes.clear()  # nothing picked: a fresh pane
                self.app.view.extra.pop("group", None)

    def test_header_ranges_are_clickable_everywhere(self):
        for width, height in ((80, 24), (120, 34)):
            for tab in ("o", "m", "p", "a", "t"):
                self.app.key(tab)
                for name in ("month", "year", "day", "all", "week"):
                    canvas = draw(self.app, width, height)
                    row, col = find(canvas, name, [0])
                    self.app.click(col + 1, 0)
                    self.assertEqual(self.app.range, name, (width, tab, name))

    def test_range_detail_tabs_keys_and_metric(self):
        for width, height in ((80, 24), (120, 34)):
            self.app.go_tab("overview")
            draw(self.app, width, height)
            self.app.key("3")
            self.app.key("ENTER")
            self.assertEqual((self.app.view.kind, self.app.range), ("range", "month"))
            for name in ("Year", "Day", "All", "Week", "Month"):
                canvas = draw(self.app, width, height)
                row, col = find(canvas, " %s " % name, [2])
                self.app.click(col + 1, row)
                self.assertEqual(self.app.range, name.lower())
                self.assertEqual(self.app.view.kind, "range")
                self.assertIn(name + " · cluster", self.app.view.title)
            for key, want in (("]", "year"), ("RIGHT", "all"), ("RIGHT", "all"), ("[", "year"),
                              ("LEFT", "month")):
                self.app.key(key)
                self.assertEqual(self.app.range, want, key)
            self.assertIn("GPU-h", draw(self.app, width, height).text())
            self.app.key("g")
            screen = draw(self.app, width, height).text()
            self.assertIn("CPUs in use", screen)
            self.assertIn("CPU-h", screen)
            self.app.key("g")
            self.app.key("ESC")
            self.assertEqual(self.app.view.kind, "overview")

    def test_person_and_account_trend_rows_open_their_range(self):
        cases = (("person", {"user": "alice"}, "alice"),
                 ("people", {"account": "physics"}, "physics"))
        for kind, params, who in cases:
            for width, height in ((80, 24), (120, 34)):
                for index, name in enumerate(hd.RANGE_NAMES):
                    self.app.go_tab("people")
                    self.app.push(kind, params, who)
                    canvas = draw(self.app, width, height)
                    # Enter on the row
                    panes = self.app._last[0]
                    self.assertEqual(panes[0].id, "trend")
                    self.app.view.focus = 0
                    self.app.view.panes["trend"].cursor = index
                    self.app.key("ENTER")
                    self.assertEqual((self.app.view.kind, self.app.range), ("range", name))
                    self.assertEqual({k: v for k, v in self.app.view.params.items()
                                      if k != "range"}, params)
                    screen = draw(self.app, width, height).text()
                    self.assertIn("in use · " + who, screen)
                    self.app.key("ESC")
                    # a click on the row, then a second click on it, opens it too
                    canvas = draw(self.app, width, height)
                    row, col = find(canvas, name + " ", range(2, 9))
                    self.app.click(col, row)
                    if self.app.view.kind != "range":
                        self.app.click(col, row)
                    self.assertEqual((self.app.view.kind, self.app.range), ("range", name),
                                     (kind, width, name))

    def test_breakdown_rows_drill_further(self):
        self.app.push("range", {"range": "month"}, "Month · cluster")
        canvas = draw(self.app, 120, 34)
        row, col = find(canvas, "physics")
        self.app.click(col, row)
        self.app.click(col, row)
        self.assertEqual(self.app.view.kind, "range")
        self.assertEqual(self.app.view.params.get("account"), "physics")
        screen = draw(self.app, 120, 34).text()
        self.assertIn("in use · physics", screen)
        self.assertIn("People", screen)
        self.app.key("ESC")
        self.app.key("TAB")  # the people tab of the breakdown
        draw(self.app, 120, 34)
        pane, _, lines = self.app._focused()
        self.assertEqual(pane.id, "by_user")
        self.assertTrue(lines and lines[0].action.params.get("user"))
        self.app.key("ENTER")
        self.assertEqual(self.app.view.params.get("user"), lines[0].action.params["user"])

    def test_tiles_and_headline_say_where_history_begins(self):
        self.app.data._persistent = True
        self.app.data.load_in_order("week")
        for width in (80, 120, 160):
            self.app.go_tab("overview")
            screen = draw(self.app, width, 34).text()
            self.assertEqual(screen.count("from ") + screen.count("since "), 2, width)
        self.app.push("range", {"range": "year"}, "Year · cluster")
        screen = draw(self.app, 120, 34).text()
        self.assertIn("data from", screen)

    def test_hero_opens_the_day(self):
        self.app.key("4")
        canvas = draw(self.app, 120, 34)
        row, col = find(canvas, "GPUs in use", [1])
        self.app.click(col + 5, 3)
        self.assertEqual((self.app.view.kind, self.app.range), ("range", "day"))

    def test_wheel_scrolls_the_pane_under_the_pointer(self):
        self.app.go_tab("overview")
        canvas = draw(self.app, 120, 34)
        row, col = find(canvas, "biolab")
        self.app.wheel(3, col, row)
        accounts = [i for i, p in enumerate(self.app._last[0]) if p.id == "accounts"][0]
        self.assertEqual(self.app.view.focus, accounts)
        self.assertEqual(self.app.view.panes["accounts"].cursor, 2)
        # over the tiles the wheel moves through the ranges
        row, col = find(canvas, "Month", range(5, 8))
        before = self.app.range
        self.app.wheel(3, col, row + 1)
        self.assertNotEqual(self.app.range, before)


class RobustTest(unittest.TestCase):
    def test_every_size_and_mode(self):
        app = make_app()
        screens = [("overview", {}), ("range", {"range": "month"}),
                   ("range", {"range": "year", "user": "alice"}),
                   ("range", {"range": "all", "account": "physics"})]
        for kind, params in screens:
            app.go_tab("overview")
            if kind == "range":
                app.push(kind, params, "x")
            for ascii_only, color in ((False, True), (True, True), (False, False)):
                app.ascii, app.color = ascii_only, color
                for width, height in ((20, 5), (30, 8), (45, 12), (80, 24), (100, 30),
                                      (120, 34), (200, 50), (250, 70), (80, 60), (240, 20)):
                    canvas = draw(app, width, height)
                    assert_clean(self, canvas)
                    if ascii_only:
                        text = canvas.text()
                        self.assertTrue(all(ord(c) < 128 for c in text), (kind, width))
            app.ascii, app.color = False, True

    def test_partial_frames_keep_their_height(self):
        """Before the live state answers, the hero and tiles take the rows they will
        have, so nothing jumps when it arrives."""
        app = make_app()
        before = draw(app, 120, 34).text().split("\n")
        app.snapshot = None
        empty = draw(app, 120, 34).text().split("\n")
        for row in (1, 5, 6, 10):
            self.assertEqual(before[row][:3], empty[row][:3], row)

    def test_selection_follows_its_row_when_rows_reorder(self):
        pane = Pane("p", "p")
        state = PaneState()
        lines = [Line([("a", "")], key="a"), Line([("b", "")], key="b"),
                 Line([("c", "")], key="c")]
        state.cursor, state.mark, state.chosen = 1, (1, "b"), True
        reordered = [lines[1], lines[0], lines[2]][::-1]  # c a b
        keep_selection(pane, state, reordered)
        self.assertEqual(reordered[state.cursor].key, "b")
        state.cursor, state.mark = 2, (1, "b")  # moved by a key since: left alone
        keep_selection(pane, state, lines)
        self.assertEqual(state.cursor, 2)

    def test_selection_stays_on_top_while_rows_load(self):
        """Nothing picked yet (a new grouping, rows still loading): a reorder leaves the
        cursor on the first row instead of chasing whatever row sat there before."""
        pane = Pane("p", "p")
        state = PaneState()
        lines = [Line([(k, "")], key=k) for k in ("high", "normal", "low", "debug")]
        state.cursor, state.mark = 0, (0, "high")
        loaded = [lines[1], lines[2], lines[0], lines[3]]
        keep_selection(pane, state, loaded)
        self.assertEqual(state.cursor, 0)

    def test_mouse_states(self):
        from slurm_usage.ui import terminal
        import curses
        self.assertEqual(terminal.mouse_action(curses.BUTTON4_PRESSED), "up")
        self.assertEqual(terminal.mouse_action(getattr(curses, "BUTTON5_PRESSED", 0x200000)),
                         "down")
        self.assertEqual(terminal.mouse_action(curses.BUTTON1_PRESSED), "click")
        self.assertEqual(terminal.mouse_action(curses.BUTTON1_CLICKED), "click")
        self.assertIsNone(terminal.mouse_action(curses.BUTTON1_RELEASED))

    def test_ascii_cells_for_new_glyphs(self):
        for glyph in "▀▄█┄┈╌▾–●":
            self.assertNotEqual(textmod.ascii_cell(glyph), "?", glyph)


if __name__ == "__main__":
    unittest.main()
