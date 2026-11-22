"""The palette, its fallbacks by color depth, glyph sets and the small charts."""

import unittest

from slurm_usage.ui import charts, fmt, theme
from slurm_usage.ui import text as textmod
from slurm_usage.ui.canvas import Canvas, on
from slurm_usage.ui.widgets import Column, Table, fit_parts, number_parts
from tests.test_ui import every_view, make_app


class DepthTest(unittest.TestCase):
    def test_detect_depth(self):
        self.assertEqual(theme.detect_depth(1 << 24, True, {}), "truecolor")
        self.assertEqual(theme.detect_depth(256, True, {"COLORTERM": "truecolor"}), "256")
        self.assertEqual(theme.detect_depth(256, True, {}), "256")
        self.assertEqual(theme.detect_depth(16, True, {}), "16")
        self.assertEqual(theme.detect_depth(8, True, {}), "8")
        self.assertEqual(theme.detect_depth(0, True, {}), "mono")
        self.assertEqual(theme.detect_depth(256, False, {}), "mono")
        self.assertEqual(theme.detect_depth(256, True, {"NO_COLOR": ""}), "mono")

    def test_every_color_has_every_depth(self):
        colors = [role.fg for role in theme.ROLES.values() if role.fg is not None]
        colors += list(theme.BACKGROUNDS.values())
        for color in colors:
            number, _ = theme.color_number(color, "truecolor")
            self.assertEqual(number, (color.rgb[0] << 16) | (color.rgb[1] << 8) | color.rgb[2])
            self.assertTrue(16 <= theme.color_number(color, "256")[0] <= 255, color)
            self.assertTrue(0 <= theme.color_number(color, "16")[0] <= 15, color)
            self.assertTrue(0 <= theme.color_number(color, "8")[0] <= 7, color)

    def test_256_is_close_to_the_rgb(self):
        def xterm(index):
            if index >= 232:
                level = 8 + 10 * (index - 232)
                return level, level, level
            index -= 16
            steps = (0, 95, 135, 175, 215, 255)
            return steps[index // 36], steps[index // 6 % 6], steps[index % 6]
        for name, role in theme.ROLES.items():
            if role.fg is None:
                continue
            got = xterm(role.fg.x256)
            off = max(abs(a - b) for a, b in zip(got, role.fg.rgb))
            self.assertLess(off, 48, (name, got, role.fg.rgb))

    def test_gray_in_8_colors_is_dimmed_white(self):
        number, dim = theme.color_number(theme.MUTED, "8")
        self.assertEqual((number, dim), (7, True))


class ResolveTest(unittest.TestCase):
    def test_roles_and_modifiers(self):
        look = theme.resolve("good+b@sel")
        self.assertEqual((look.fg, look.bg, look.bold), (theme.GOOD, "sel", True))
        self.assertEqual(theme.resolve("").fg, None)
        self.assertEqual(theme.resolve("no-such-role").fg, None)
        self.assertEqual(theme.resolve("tab").bg, "pill")
        self.assertEqual(theme.resolve("selected").bg, "sel")
        self.assertTrue(theme.resolve("dim").dim)

    def test_overlay_keeps_the_color(self):
        canvas = Canvas(10, 1)
        canvas.put(0, 0, "abc", "user")
        canvas.overlay(0, 0, 2, "sel")
        self.assertEqual([s for _, s in canvas.cells[0][:3]], ["user@sel", "user@sel", "user"])
        self.assertEqual(on("good+b@band", "sel"), "good+b@sel")

    def test_backgrounds_only_with_256_or_more(self):
        self.assertTrue(theme.backgrounds_shown("256"))
        self.assertTrue(theme.backgrounds_shown("truecolor"))
        for depth in ("16", "8", "mono"):
            self.assertFalse(theme.backgrounds_shown(depth))


class GlyphTest(unittest.TestCase):
    def test_box_fallbacks(self):
        self.assertEqual(theme.box(False, {"TERM": "xterm-256color"}).tl, "╭")
        self.assertEqual(theme.box(False, {"TERM": "linux"}).tl, "┌")
        self.assertEqual(theme.box(True, {}).tl, "+")
        for glyph in theme.ROUNDED + theme.SQUARE:
            self.assertNotEqual(textmod.ascii_cell(glyph), "?", glyph)

    def test_every_drawn_glyph_has_an_ascii_cell(self):
        """ASCII mode maps cells after drawing: nothing a screen draws may become '?'."""
        app = make_app()
        seen = set()
        for _, app in every_view(app):
            for width, height in ((80, 24), (160, 40)):
                canvas = Canvas(width, height)
                app.render(canvas)
                seen.update(c for line in canvas.cells for c, _ in line if c and ord(c) > 127)
        braille = {c for c in seen if 0x2800 <= ord(c) <= 0x28FF}  # drawn only in Unicode mode
        missing = [c for c in seen - braille if textmod.ascii_cell(c) == "?"]
        self.assertEqual(missing, [])


class ChartTest(unittest.TestCase):
    def test_widths_are_exact(self):
        for size in (1, 5, 13, 40):
            for values in (None, [], [0, 0], [1, 5, 0, 3], list(range(100))):
                for ascii_only in (False, True):
                    self.assertEqual(charts.width(charts.spark(values, size, ascii_only)), size)
                    self.assertEqual(charts.width(charts.braille(values, size, ascii_only)), size)
            for fraction in (None, 0, 0.13, 0.5, 0.999, 1, 1.4):
                self.assertEqual(charts.width(charts.gauge(fraction, size)), size)
                self.assertEqual(charts.width(charts.gauge(fraction, size, True)), size)
            self.assertEqual(charts.width(charts.stacked([(0.3, "good"), (0.7, "warn")], size)),
                             size)

    def test_gauge_precision_and_ramp(self):
        parts = charts.gauge(0.5625, 8)  # 4.5 cells
        self.assertEqual(parts[0], ("████", "good"))
        self.assertEqual(parts[1], ("▌", "good"))
        self.assertEqual(charts.ramp_style(0.95), "bad")
        self.assertEqual(charts.ramp_style(0.8), "warn")
        self.assertEqual(charts.gauge(0.2, 4, True), [("#", "good"), ("...", "faint")])

    def test_no_color_charts_carry_meaning_in_glyphs(self):
        cells = charts.heat_cells([0.1, 0.5, 1.0], color=False)
        self.assertEqual("".join(t for t, _ in cells), "░▒█")
        self.assertTrue(all(style == "" for _, style in cells))

    def test_delta_arrows(self):
        self.assertEqual(charts.delta(150, 100), [("▲", "good"), ("50%", "")])
        self.assertEqual(charts.delta(50, 100), [("▼", "warn"), ("50%", "")])
        self.assertEqual(charts.delta(None, 100), [(fmt.NA, "faint")])

    def test_units_recede(self):
        self.assertEqual(number_parts("≥72.5k"), [("≥", "dim"), ("72.5", ""), ("k", "dim")])
        self.assertEqual(number_parts("3h12m"), [("3h12m", "")])
        self.assertEqual(charts.width(fit_parts([("abc", ""), ("def", "dim")], 4)), 4)

    def test_empty_efficiency_columns_are_left_out(self):
        columns = [Column("Name", 6), Column("GPU util*", 9, ">"), Column("GPU-h", 6, ">")]
        table = Table(columns, 40, prune=("GPU util*",))
        table.row(["a", ("?", "faint"), "12"])
        self.assertNotIn("GPU util*", table.header().text)
        table = Table(columns, 40, prune=("GPU util*",))
        line = table.row(["a", ("54%", "good"), "12"])
        self.assertIn("GPU util*", table.header().text)
        self.assertIn("54%", line.text)


if __name__ == "__main__":
    unittest.main()
