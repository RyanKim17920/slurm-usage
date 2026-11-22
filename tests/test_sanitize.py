import unittest

from slurm_usage.sanitize import REPLACEMENT, clean, clean_id, clean_multiline


class CleanTest(unittest.TestCase):
    def test_plain_text_unchanged(self):
        self.assertEqual(clean("train-v2 lr=3e-4"), "train-v2 lr=3e-4")

    def test_none(self):
        self.assertEqual(clean(None), "")
        self.assertIsNone(clean_id(None))

    def test_csi_and_sgr(self):
        self.assertEqual(clean("a\x1b[31mred\x1b[0m b\x1b[2J\x1b[H"), "ared b")

    def test_osc_title_and_hyperlink(self):
        self.assertEqual(clean("x\x1b]0;owned\x07y"), "xy")
        self.assertEqual(clean("\x1b]8;;http://evil\x1b\\click\x1b]8;;\x1b\\"), "click")

    def test_dcs_and_unterminated(self):
        self.assertEqual(clean("a\x1bPq#0;2;0;0;0\x1b\\b"), "ab")
        self.assertEqual(clean("safe\x1b]0;never ends"), "safe")
        self.assertEqual(clean("x\x1b"), "x")

    def test_c1_controls(self):
        self.assertEqual(clean("a\x9b31mb\x9d0;t\x9cc\x85d"), "abcd")

    def test_c0_controls_and_whitespace(self):
        self.assertEqual(clean("a\x00b\x07c\x08d\x7fe"), "abcde")
        self.assertEqual(clean("one\ntwo\rthree\tfour"), "one two three four")

    def test_bidi_and_invisible(self):
        self.assertEqual(clean("abc‮dcba⁦x⁩​﻿"), "abcdcbax")

    def test_line_separators(self):
        self.assertEqual(clean("a b c"), "a b c")

    def test_undecodable_bytes(self):
        raw = b"ok\xff\xfe".decode("utf-8", "surrogateescape")
        self.assertEqual(clean(raw), "ok" + REPLACEMENT * 2)

    def test_unicode_kept(self):
        self.assertEqual(clean("modèle-日本語-🙂"), "modèle-日本語-🙂")

    def test_truncation(self):
        self.assertEqual(clean("abcdef", max_len=4), "abc…")
        self.assertEqual(clean("abc", max_len=0), "")

    def test_multiline_keeps_newlines_only(self):
        self.assertEqual(clean_multiline("a\r\nb\x1b[1m\nc\rd"), "a\nb\nc d")

    def test_clean_id_marks_changes(self):
        self.assertEqual(clean_id("alice"), "alice")
        self.assertEqual(clean_id("ali​ce"), "alice" + REPLACEMENT)
        self.assertNotEqual(clean_id("ali​ce"), clean_id("alice"))

    def test_long_hostile_input_is_fast(self):
        import time
        text = "\x1b[" + "1;" * 100000 + "\x1b]" * 50000
        started = time.monotonic()
        clean(text)
        self.assertLess(time.monotonic() - started, 2.0)


if __name__ == "__main__":
    unittest.main()
