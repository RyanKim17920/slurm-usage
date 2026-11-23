"""The invented demo cluster (scripts/demo) loads cleanly through the real code.

Runs the fake Slurm executables as subprocesses through the real runner, with
only them on PATH, so no real cluster is ever asked.
"""

import importlib.util
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

from slurm_usage.history import History, Window
from slurm_usage.live import LiveCollector
from slurm_usage.runner import Runner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "scripts", "demo")
DAY = 86400


def load(name: str):
    spec = importlib.util.spec_from_file_location("demo_" + name, os.path.join(DEMO, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DemoClusterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_demo = load("run_demo")
        cls.scratch = tempfile.mkdtemp(prefix="orion-test-")
        cls.env = cls.run_demo.demo_environment(cls.scratch)
        cls.runner = Runner(environ=cls.env)
        cls.as_alice = mock.patch("pwd.getpwuid", cls.run_demo.demo_getpwuid("alice"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.scratch, ignore_errors=True)

    def test_only_fakes_on_path(self):
        for program in ("squeue", "scontrol", "sacct", "sacctmgr", "sstat", "sinfo", "sprio",
                        "sshare", "sreport"):
            found = shutil.which(program, path=self.env["PATH"])
            self.assertEqual(os.path.dirname(found), os.path.join(DEMO, "bin"), program)

    def test_live_snapshot(self):
        collector = LiveCollector(self.runner, uid=os.getuid(), user="alice", euid=os.geteuid(),
                                  workers=1)
        snap = collector.snapshot()
        self.assertEqual(snap.cluster.name, "orion")
        self.assertEqual([w.code for w in snap.warnings], [])
        self.assertEqual(len(snap.nodes), 10)
        drained = [n for n in snap.nodes if n.reason]
        self.assertEqual([n.name for n in drained], ["gpu-a06"])
        running = [j for j in snap.jobs if j.state == "RUNNING"]
        pending = [j for j in snap.jobs if j.state == "PENDING"]
        self.assertTrue(running and pending)
        self.assertTrue(all(j.detail_status == "ok" for j in running),
                        {j.job_id: j.detail_status for j in running})
        reasons = {j.reason for j in pending}
        self.assertTrue({"Resources", "JobArrayTaskLimit", "Dependency"} <= reasons, reasons)
        allocated = sum(n.gpus_alloc or 0 for n in snap.nodes)
        self.assertEqual(allocated, sum(j.gpus_alloc or 0 for j in running))

    def test_history_week_and_detail(self):
        cache = tempfile.mkdtemp(dir=self.scratch)
        with self.as_alice:
            history = History(self.runner, cache_dir=cache)
            try:
                info = history.info()
                self.assertEqual(info.cluster, "orion")
                self.assertEqual(info.visibility, "all")
                self.assertEqual([n.code for n in info.notes], [])
                window = Window.last(7 * DAY)
                started = time.time()
                history.refresh(window)
                self.assertLess(time.time() - started, 60)
                report = history.query(window, group_by=["user", "account"])
                users = {key[0] for key in report.groups[("user",)]}
                accounts = {key[0] for key in report.groups[("account",)]}
                self.assertTrue(report.complete)
                self.assertEqual([n.code for n in report.notes], [])
                self.assertGreater(report.total.attempts, 300)
                self.assertIn("alice", users)
                self.assertIn("vision", accounts)

                # the requeued multi-node run: a node failure, then running again
                runs = [row for row in history.jobs(window).rows if row.name == "pretrain_moe_v2"]
                self.assertEqual(len({row.job_id for row in runs}), 1)
                detail = history.detail(runs[0].job_id)
                self.assertEqual(detail.status, "ok", detail.message)
                outcomes = [a.job.outcome for a in detail.attempts]
                self.assertEqual(outcomes, ["node_fail", "running"])
                self.assertTrue(detail.attempts[0].steps)
            finally:
                history.close()

    def test_history_never_exceeds_the_live_capacity(self):
        """No hour of the demo's history holds more GPUs than the demo cluster has (an
        invented cluster whose charts poke above capacity reads as a bug)."""
        fake = load("fakeslurm")
        capacity = sum(node[3] for node in fake.NODES)
        cache = tempfile.mkdtemp(dir=self.scratch)
        with self.as_alice:
            history = History(self.runner, cache_dir=cache)
            try:
                history.info()
                window = Window.last(30 * DAY)
                history.refresh(window)
                report = history.query(window)
                edges, values = report.edges, report.values["gpu"]
                held = [v / float(edges[i + 1] - edges[i]) for i, v in enumerate(values)]
                self.assertLessEqual(max(held), capacity + 1e-6)
                self.assertGreater(max(held), capacity * 0.8)  # and it does get busy
            finally:
                history.close()

    def test_a_kept_cache_is_one_per_demo_day(self):
        """The world is anchored to today's midnight: yesterday's cache must not mix in."""
        fake = load("fakeslurm")
        today = fake.world_id(1_800_000_000)
        self.assertEqual(today, fake.world_id(1_800_000_000 + 60))
        self.assertNotEqual(today, fake.world_id(1_800_000_000 + DAY))


if __name__ == "__main__":
    unittest.main()
