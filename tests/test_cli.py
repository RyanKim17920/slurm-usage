"""The command line, with the live collector and history replaced by invented data."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from slurm_usage import __version__, cli
from slurm_usage.ui import data as hd
from slurm_usage.ui.anonymize import AnonCollector, AnonHistory, Anonymizer
from tests import ui_fakes as uf


def fake_build(anonymize=False):
    def build(args):
        history, fake = uf.real_history(days=20)
        collector = uf.FakeCollector(uf.make_snapshot(fake.now))
        data = hd.HistoryData(history, clock=lambda: fake.now)
        if args.history == "off":
            data = hd.HistoryData(hd.NoHistory())
        anon = None
        if args.anonymize:
            anon = Anonymizer()
            collector = AnonCollector(collector, anon)
            data.history = AnonHistory(data.history, anon)
        return collector, data, anon
    return build


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(cli, "build", fake_build()), redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue()


class CliTest(unittest.TestCase):
    def test_version(self):
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            cli.main(["--version"])
        self.assertIn(__version__, out.getvalue())

    def test_args(self):
        args = cli.parse_args(["--range", "month", "--user", "u", "--account", "a",
                               "--history", "off", "--anonymize"])
        self.assertEqual((args.range, args.user, args.account, args.history, args.anonymize),
                         ("month", "u", "a", "off", True))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(["--range", "decade"])

    def test_once(self):
        code, out = run(["--once", "--range", "day"])
        self.assertEqual(code, 0)
        for needle in ("GPU", "Range", "Account", "physics"):
            self.assertIn(needle, out)

    def test_non_tty_falls_back_to_once(self):
        code, out = run([])  # stdout is a StringIO, not a terminal
        self.assertEqual(code, 0)
        self.assertIn("Range", out)

    def test_json_with_user_and_account(self):
        code, out = run(["--json", "--user", "alice", "--account", "physics"])
        data = json.loads(out)
        self.assertEqual([a["account"] for a in data["accounts"]], ["physics"])
        self.assertEqual(data["person"]["user"], "alice")
        self.assertIn("trend", data)

    def test_json_trend_sparklines_for_every_range_come_from_the_rollup(self):
        def no_spark(*args, **kwargs):
            raise AssertionError("a sparkline needed a full history pass")
        with mock.patch.object(hd.HistoryData, "spark", no_spark):
            code, out = run(["--json"])
        trends = {t["range"]: t for t in json.loads(out)["trend"]}
        self.assertEqual(set(trends), set(hd.RANGE_NAMES))
        for name in ("week", "month", "year", "all"):
            self.assertTrue(any(trends[name]["series"]), name)

    def test_once_on_a_non_utf8_stdout(self):
        for encoding in ("ascii", "latin-1"):
            raw = io.BytesIO()
            stream = io.TextIOWrapper(raw, encoding=encoding)
            with mock.patch.object(cli, "build", fake_build()), redirect_stdout(stream), \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["--once"]), 0)
                self.assertEqual(cli.main(["--json"]), 0)
            stream.flush()
            out = raw.getvalue().decode(encoding)
            self.assertIn("Range", out)
            self.assertNotIn("\u2581", out)

    def test_legacy_terminal_gets_ascii_and_json_is_ascii(self):
        with mock.patch.dict("os.environ", {"TERM": "vt100"}):
            code, out = run(["--once"])
        self.assertTrue(all(ord(c) < 128 for c in out), out)
        code, out = run(["--json"])
        self.assertTrue(all(ord(c) < 128 for c in out))
        json.loads(out)

    def test_partly_loaded_rows_are_marked_everywhere(self):
        from slurm_usage.ui import text
        row = {"range": "year", "gpu_hours_allocated": 10.0, "cpu_hours_allocated": 20.0,
               "gpu_utilization_of_live_capacity": 0.5,
               "cpu_utilization_of_live_capacity": None, "gpu_util_peak": None,
               "cpu_efficiency": None, "previous_gpu_hours_allocated": None,
               "previous_cpu_hours_allocated": None, "attempts": 3, "jobs": 2,
               "complete": False, "previous_complete": False, "series": []}
        out = text.render({"range": "year", "generated_at": 0, "trend": [row],
                           "person": {"user": "alice", "trend": [row]}})
        trend_line = [l for l in out.splitlines() if l.startswith("year")][0]
        self.assertIn("≥50%", trend_line)  # the util column is a lower bound too
        self.assertNotIn("≥n/a", trend_line)
        person_line = [l for l in out.splitlines() if l.strip().startswith("year")
                       and "jobs" in l][0]
        self.assertIn("≥10", person_line)
        self.assertIn("≥2 jobs", person_line)

    def test_no_history(self):
        code, out = run(["--json", "--history", "off"])
        data = json.loads(out)
        self.assertEqual(data["history"]["status"], "off")
        self.assertNotIn("trend", data)

    def test_anonymized_json_has_no_real_names(self):
        code, out = run(["--json", "--anonymize", "--user", "alice", "--range", "month"])
        data = json.loads(out)
        self.assertTrue(data["anonymized"])
        self.assertEqual(data["person"]["user"], "user-01")
        for real in uf.REAL_IDENTIFIERS:
            self.assertNotIn(real, out, real)


if __name__ == "__main__":
    unittest.main()
