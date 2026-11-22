"""Cache, incremental refresh, requeues, cancellation, visibility."""

import os
import stat
import threading
import unittest

from tests.test_history_support import H, T0, FakeSlurm, job, make_history, temp_dir

from slurm_usage.history import Cancelled, HistoryError, Status, Window
from slurm_usage.history import store as store_module

DAY = 86400


class CacheTest(unittest.TestCase):
    def test_private_modes_and_reuse(self):
        directory = os.path.join(temp_dir(self), "cache")
        fake = FakeSlurm([job(1, start=T0, end=T0 + H)], live=[])
        h = make_history(fake, persist=True, cache_dir=directory)
        h.refresh()
        path = h.info().cache_path
        self.assertEqual(stat.S_IMODE(os.stat(directory).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        h.close()
        fake.calls.clear()
        again = make_history(fake, persist=True, cache_dir=directory)
        self.assertEqual(again.query().total.attempts, 1)  # from the cache, no sacct history
        self.assertFalse(fake.sacct_calls("-D"))

    def test_existing_shared_directory_keeps_its_mode(self):
        shared = os.path.join(temp_dir(self), "shared")
        os.mkdir(shared)
        os.chmod(shared, 0o2775)  # group-writable and setgid, as a team directory may be
        fake = FakeSlurm([job(1, start=T0, end=T0 + H)], live=[])
        h = make_history(fake, persist=True, cache_dir=shared)
        h.refresh()
        self.assertEqual(stat.S_IMODE(os.stat(shared).st_mode), 0o2775)  # untouched
        path = h.info().cache_path
        self.assertEqual(os.path.dirname(path), os.path.join(shared, "slurm-usage"))
        self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode), 0o700)
        h.close()

    def test_non_private_subdirectory_is_refused_not_changed(self):
        shared = os.path.join(temp_dir(self), "shared")
        os.makedirs(os.path.join(shared, "slurm-usage"))
        os.chmod(shared, 0o775)
        os.chmod(os.path.join(shared, "slurm-usage"), 0o755)
        fake = FakeSlurm([job(1, start=T0, end=T0 + H)], live=[])
        h = make_history(fake, persist=True, cache_dir=shared)
        h.refresh()
        self.assertIsNone(h.info().cache_path)
        self.assertTrue(any(n.code == "cache_memory" for n in h.info().notes))
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(shared, "slurm-usage")).st_mode),
                         0o755)

    def test_default_location_honours_xdg(self):
        old = os.environ.get("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = "/somewhere/cache"
        try:
            self.assertEqual(store_module.default_cache_dir(), "/somewhere/cache/slurm-usage")
            os.environ["XDG_CACHE_HOME"] = "relative"
            self.assertTrue(store_module.default_cache_dir().endswith(".cache/slurm-usage"))
        finally:
            if old is None:
                del os.environ["XDG_CACHE_HOME"]
            else:
                os.environ["XDG_CACHE_HOME"] = old

    def test_symlinked_cache_falls_back_to_memory(self):
        directory = temp_dir(self)
        target = os.path.join(directory, "elsewhere")
        open(target, "w").close()
        name = store_module.cache_name("x")
        os.symlink(target, os.path.join(directory, name))
        store = store_module.open_store(directory, name)
        self.assertIsNone(store.path)
        self.assertEqual(store.notes[0].code, "cache_memory")

    def test_corrupt_cache_is_rebuilt(self):
        directory = temp_dir(self)
        name = store_module.cache_name("y")
        path = os.path.join(directory, name)
        with open(path, "wb") as handle:
            handle.write(b"not a database" * 100)
        os.chmod(path, 0o600)
        store = store_module.open_store(directory, name)
        self.assertEqual(store.path, path)
        self.assertEqual(store.notes[0].code, "cache_rebuilt")

    def test_old_schema_versions_are_removed(self):
        directory = temp_dir(self)
        name = store_module.cache_name("z")
        stale = os.path.join(directory, "history-v0-" + name.split("-", 2)[-1])
        open(stale, "w").close()
        store_module.open_store(directory, name)
        self.assertFalse(os.path.exists(stale))

    def test_identity_separates_viewers(self):
        self.assertNotEqual(store_module.cache_name("c", "1", "all"),
                            store_module.cache_name("c", "2", "all"))

    def test_memory_store_gives_the_same_answers(self):
        jobs = [job(1, start=T0, end=T0 + 5 * H, gpus=2),
                job(2, start=T0 + H, end=None, state="RUNNING", gpus=1)]
        answers = []
        for factory in (store_module.MemoryStore, lambda: store_module.SqliteStore()):
            fake = FakeSlurm(jobs, now=T0 + 6 * H, live=[2])
            h = make_history(fake)
            h._open()
            h._store = factory()
            from slurm_usage.history.ingest import Ingestor
            h._ingest = Ingestor(fake, h._store, h._env)
            h.refresh()
            r = h.query(Window(T0, T0 + DAY), group_by=["user", "node"])
            answers.append((r.total.gpu_seconds, r.total.attempts, r.total.peak_gpus,
                            len(h.jobs().rows), h.build_rollup() > 0,
                            h.overview().ranges[0].total.gpu_seconds))
        self.assertEqual(answers[0], answers[1])


class RefreshTest(unittest.TestCase):
    def test_running_job_is_updated_when_it_ends(self):
        fake = FakeSlurm([job(1, start=T0, end=None, state="RUNNING", index=5)],
                         now=T0 + 2 * H, live=[1])
        h = make_history(fake)
        h.refresh()
        self.assertEqual(h.query(Window(T0, T0 + DAY)).total.gpu_seconds, 2 * H)
        fake.jobs = [job(1, start=T0, end=T0 + 3 * H, state="COMPLETED", index=5)]
        fake.now, fake.live = T0 + 10 * DAY, []
        h.refresh()
        r = h.query(Window(T0, T0 + DAY))
        self.assertEqual((r.total.gpu_seconds, r.total.running), (3 * H, 0))

    def test_long_running_job_across_the_cache_boundary(self):
        # started long before what the forward refresh re-reads
        fake = FakeSlurm([job(1, start=T0, end=None, state="RUNNING", index=5)],
                         now=T0 + 20 * DAY, live=[1])
        h = make_history(fake)
        h.refresh(Window.last(DAY, fake.now))
        fake.now += 5 * DAY
        fake.jobs = [job(1, start=T0, end=fake.now - DAY, state="TIMEOUT", index=5)]
        h.refresh(Window.last(DAY, fake.now))
        rows = h.attempts("1")
        self.assertEqual((len(rows), rows[0].state, rows[0].end), (1, "TIMEOUT", fake.now - DAY))

    def test_requeue_adds_an_attempt(self):
        first = job(1, start=T0, end=T0 + H, state="PREEMPTED", index=1, submit=T0)
        fake = FakeSlurm([first], now=T0 + 2 * H, live=[1])
        h = make_history(fake)
        h.refresh()
        fake.jobs = [first, job(1, start=T0 + 3 * H, end=None, state="RUNNING", index=2,
                                submit=T0 + H, restarts=1)]
        fake.now = T0 + 4 * H
        h.refresh()
        r = h.query(Window(T0, T0 + DAY))
        self.assertEqual((r.total.attempts, r.total.jobs, r.total.gpu_seconds), (2, 1, 2 * H))

    def test_vanished_open_row_is_dropped(self):
        fake = FakeSlurm([job(1, start=T0, end=None, state="RUNNING", index=9)],
                         now=T0 + H, live=[1])
        h = make_history(fake)
        h.refresh()
        fake.jobs, fake.now = [], T0 + 2 * H
        h.refresh()
        self.assertEqual(h.attempts("1"), [])

    def test_refresh_uses_epochs_and_bounded_pieces(self):
        fake = FakeSlurm([job(i, start=T0 + i * H, end=T0 + i * H + 60) for i in range(1, 50)],
                         now=T0 + 3 * DAY, live=[])
        h = make_history(fake)
        h.refresh()
        calls = fake.sacct_calls("-X", "-D", "-a")
        self.assertGreater(len(calls), 1)
        for call in calls:
            self.assertTrue(any(a.startswith("--delimiter=\x1f") for a in call))
        self.assertEqual(h.query().total.attempts, 49)

    def test_failed_piece_does_not_mark_coverage(self):
        fake = FakeSlurm([job(1, start=T0, end=T0 + H)], live=[])
        fake.fail["-X"] = Status.TIMEOUT
        h = make_history(fake)
        with self.assertRaises(HistoryError):
            h.refresh()
        with self.assertRaises(HistoryError):
            h.query()
        self.assertTrue(any(n.code == "refresh_failed" for n in h.info().notes))

    def test_cancel_and_progress(self):
        fake = FakeSlurm([job(i, start=T0 + i * 60, end=T0 + i * 60 + 30) for i in range(1, 300)],
                         now=T0 + 40 * DAY, live=[])
        fake.now = T0 + 6 * H
        h = make_history(fake)
        seen = []
        recent = Window(T0 + 4 * H, None)
        h.refresh(recent, progress=seen.append)
        before = h.query(recent).total.attempts
        self.assertTrue(seen and seen[-1].phase in ("history", "refresh"))
        stop = threading.Event()
        fake.on_line = lambda at: stop.set() if at > 10 else None
        with self.assertRaises(Cancelled):
            h.refresh(progress=None, cancel=stop)
        # what was committed stays usable and complete
        self.assertGreater(before, 0)
        self.assertEqual(h.query(recent).total.attempts, before)
        self.assertTrue(h.query(recent).complete)
        self.assertFalse(h.query().complete)

    def test_other_cluster_rows_are_ignored(self):
        fake = FakeSlurm([job(1, start=T0, end=T0 + H), job(2, start=T0, end=T0 + H,
                                                            Cluster="sibling")], live=[])
        h = make_history(fake)
        h.refresh()
        self.assertEqual(h.query().total.attempts, 1)


class EnvironmentTest(unittest.TestCase):
    def test_no_accounting_says_so(self):
        config = FakeSlurm().config.replace("accounting_storage/slurmdbd",
                                            "accounting_storage/none")
        h = make_history(FakeSlurm([], config=config))
        info = h.info()
        self.assertEqual(info.accounting, Status.UNAVAILABLE)
        self.assertTrue(any(n.code == "no_accounting" for n in info.notes))
        with self.assertRaises(HistoryError) as caught:
            h.refresh()
        self.assertEqual(caught.exception.status, Status.UNAVAILABLE)
        self.assertEqual(h.detail("1").status, Status.UNAVAILABLE)

    def test_restricted_visibility_is_reported(self):
        config = FakeSlurm().config.replace("PrivateData = (null)", "PrivateData = jobs,usage")
        own = os.geteuid()
        fake = FakeSlurm([job(1, start=T0, end=T0 + H, uid=own)], config=config, live=[])
        info = make_history(fake).info()
        self.assertEqual(info.visibility, "restricted")
        self.assertTrue(any(n.code == "restricted" for n in info.notes))

    def test_seeing_others_without_privilege_means_all(self):
        fake = FakeSlurm([job(1, start=T0 + 29 * DAY, end=None, state="RUNNING", uid=12345)],
                         live=[1])
        self.assertEqual(make_history(fake).info().visibility, "all")

    def test_visibility_probe_reads_one_hour_and_stops_at_another_user(self):
        now = T0 + 30 * DAY
        jobs = [job(i, start=now - 1800, end=None, state="RUNNING", uid=os.geteuid() + 1 + i)
                for i in range(50)]
        fake = FakeSlurm(jobs, now=now, live=list(range(50)))
        streams = []
        original = fake.stream

        def stream(argv, timeout=None, max_bytes=None):
            found = original(argv, timeout, max_bytes)
            if argv[-1] == "UID":
                streams.append((argv, found))
            return found
        fake.stream = stream
        self.assertEqual(make_history(fake).info().visibility, "all")
        (argv, probe), = streams
        self.assertEqual(argv[argv.index("-S") + 1], "now-1hours")
        self.assertTrue(probe.closed_early)  # stopped at the first other user

    def test_no_origin_stops_after_an_empty_year_not_ten(self):
        now = T0 + 30 * DAY
        fake = FakeSlurm([job(1, start=now - 2 * DAY, end=now - DAY)], now=now, live=[])
        h = make_history(fake)
        h.refresh()  # all of history; the fake records no accounting origin
        windows = [c for c in fake.sacct_calls("-S") if "-X" in c and "UID" not in c[-1]]
        self.assertLess(len(windows), 40)  # 10 years would be far more pieces
        self.assertEqual(h.info().covered_from, 0)
        self.assertEqual(h.query().total.attempts, 1)

    def test_background_pieces_are_paced(self):
        from slurm_usage.history import ingest
        now = T0 + 30 * DAY
        fake = FakeSlurm([job(i, start=now - i * DAY, end=now - i * DAY + H)
                          for i in range(1, 20)], now=now, live=[])
        h = make_history(fake)
        h.refresh(Window(now - DAY, now))
        sleeps = []
        old_pause, old_sleep = ingest.PIECE_PAUSE, ingest.time.sleep
        ingest.PIECE_PAUSE, ingest.time.sleep = 0.2, sleeps.append
        try:
            h.refresh(Window(now - 20 * DAY, now), update=False)  # background backfill
            paced = len(sleeps)
            sleeps.clear()
            h.refresh(Window(now - 25 * DAY, now), update=True)  # the range on screen
        finally:
            ingest.PIECE_PAUSE, ingest.time.sleep = old_pause, old_sleep
        self.assertGreater(paced, 0)
        self.assertEqual(sleeps, [])

    def test_frequent_refresh_rereads_overlap_not_a_whole_day(self):
        now = T0 + 30 * DAY
        fake = FakeSlurm([job(1, start=now - DAY, end=now - H)], now=now, live=[])
        h = make_history(fake)
        h.refresh(Window(now - DAY, now))
        fake.calls.clear()
        fake.now += 600
        h._clock = lambda: fake.now
        h.refresh(Window(now - DAY, now))
        starts = [c[c.index("-S") + 1] for c in fake.sacct_calls("-S") if "-X" in c
                  and c[-1] != "UID"]
        from tests.test_history_support import stamp_to_epoch
        self.assertTrue(starts)
        self.assertGreaterEqual(min(stamp_to_epoch(s, fake.now) for s in starts),
                                now - 5 * H)  # overlap only, not a day

    def test_unchanged_rows_are_not_rewritten(self):
        now = T0 + 30 * DAY
        fake = FakeSlurm([job(i, start=now - 20 * H + i * H, end=now - 19 * H + i * H)
                          for i in range(10)], now=now, live=[])
        h = make_history(fake, persist=True, cache_dir=temp_dir(self))
        h.refresh(Window(now - DAY, now))
        h.build_rollup()
        self.assertIsNone(h._store.get_meta("dirty"))
        h._store.set_meta("lookback_at", 0)  # force the full day to be read again
        h.refresh(Window(now - DAY, now))
        self.assertTrue(fake.sacct_calls("-S"))
        self.assertIsNone(h._store.get_meta("dirty"))  # nothing changed, nothing to redo
        fake.jobs[3]["End"] = fake.jobs[3]["End"] + 600  # one row really changed
        h._store.set_meta("lookback_at", 0)
        h.refresh(Window(now - DAY, now))
        self.assertIsNotNone(h._store.get_meta("dirty"))
        h.close()

    def test_admin_sees_all(self):
        fake = FakeSlurm([], admin="Administrator")
        self.assertEqual(make_history(fake).info().visibility, "all")

    def test_no_environment_queries_and_no_bulk_free_text(self):
        fake = FakeSlurm([job(1, start=T0, end=T0 + H)], live=[])
        h = make_history(fake)
        h.refresh()
        h.load_efficiency()
        h.detail("1")
        for call in fake.calls:
            joined = " ".join(call)
            self.assertNotIn("env-vars", joined)
            if "-j" not in call:
                self.assertNotIn("SubmitLine", joined)
                self.assertNotIn("WorkDir", joined)


if __name__ == "__main__":
    unittest.main()
