import unittest

from slurm_usage.live import applicable_limits, group_by, reconcile_gpus
from slurm_usage.model import Status
from tests import cluster_fixture as cf
from tests import fixtures as fx


class StreamingRunner(fx.FakeRunner):
    """A runner that also streams, padding every scontrol job record with a huge tail."""

    def __init__(self, table):
        super().__init__(table)
        self.streamed, self.caps = [], []

    def run(self, argv, timeout=None, max_bytes=None):
        self.caps.append(max_bytes)
        return super().run(argv, timeout, max_bytes)

    def stream(self, argv, timeout=None, max_bytes=None):
        result = super().run(argv, timeout, max_bytes)
        self.streamed.append(list(argv))
        lines = [line + " Comment=" + "x" * 200000 for line in result.lines()]

        class Lines:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def __iter__(self):
                yield from lines
                self.result = result
        return Lines()


class LiveMemoryTest(unittest.TestCase):
    def test_job_detail_is_streamed_and_trimmed_and_runs_are_capped(self):
        from slurm_usage.live import collector as collector_module
        runner = StreamingRunner(cf.responses())
        live = collector_module.LiveCollector(runner, uid=cf.ALICE, user="alice",
                                              euid=cf.ALICE, workers=1)
        snap = live.snapshot()
        self.assertEqual(runner.streamed, [collector_module.queries.JOB_DETAIL])
        self.assertEqual(snap.job("101").detail_status, "ok")  # still matched
        self.assertTrue(runner.caps)
        self.assertTrue(all(cap == collector_module.LIVE_MAX_BYTES for cap in runner.caps))
        self.assertLess(collector_module.LIVE_MAX_BYTES, 256 * 1024 * 1024)


class GpuClusterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner_table = cf.responses()
        cls.collector = cf.collector(cls.runner_table)
        cls.snap = cls.collector.snapshot()

    def test_bulk_call_budget(self):
        runner = self.collector.runner
        calls = len(runner.calls)
        self.collector.snapshot()
        self.assertEqual(len(runner.calls) - calls, 7)  # static queries are cached
        self.assertFalse(any(argv[0] in ("sstat", "sacct") for argv in runner.calls))

    def test_capabilities(self):
        for name in ("nodes", "jobs", "job_detail", "assoc_mgr", "qos", "associations"):
            self.assertEqual(self.snap.capabilities[name].status, Status.OK, name)

    def test_gpu_totals(self):
        gpus = self.snap.totals.gpus
        self.assertEqual((gpus.total, gpus.alloc, gpus.free, gpus.unavailable, gpus.reserved),
                         (8, 3, 1, 4, 0))
        self.assertEqual(self.snap.totals.gpu_types["a100"].total, 8)
        self.assertEqual(self.snap.totals.nodes, 3)
        self.assertEqual(self.snap.totals.nodes_schedulable, 2)

    def test_cpu_totals_and_untracked_memory(self):
        cpus = self.snap.totals.cpus
        self.assertEqual((cpus.total, cpus.alloc, cpus.unavailable), (192, 32, 64))
        self.assertEqual(cpus.free, 96)
        self.assertIsNone(self.snap.totals.mem.alloc)
        self.assertIsNone(self.snap.totals.mem.free)

    def test_reconcile(self):
        result = reconcile_gpus(self.snap)
        self.assertEqual((result.node_alloc, result.jobs_alloc, result.difference), (3, 3, 0))
        self.assertTrue(result.consistent)

    def test_jobs(self):
        job = self.snap.job("101")
        self.assertEqual((job.user, job.uid, job.account, job.qos, job.partition),
                         ("alice", cf.ALICE, "acct-a", "normal", "gpu"))
        self.assertEqual((job.cpus_alloc, job.gpus_alloc, job.gpu_types), (16, 2, {"a100": 2}))
        self.assertEqual(job.allocations[0].gpu_indices, [0, 1])
        self.assertIsNone(job.mem_alloc)  # CR_CORE: not tracked
        self.assertEqual(job.detail_status, "ok")

    def test_pending_array_is_one_record(self):
        sweep = self.snap.job("200")
        self.assertEqual((sweep.task_count, sweep.array_task, sweep.display_id),
                         (3, "0-2", "200_[0-2]"))
        self.assertEqual(sweep.gpus_req, 1)
        self.assertEqual(self.snap.job("201").partition, "gpu,all")

    def test_group_by(self):
        users = {usage.key: usage for usage in group_by(self.snap.jobs, "user")}
        self.assertEqual((users["alice"].gpus, users["alice"].running, users["alice"].pending),
                         (2, 1, 1))
        self.assertEqual((users["bob"].gpus, users["bob"].pending, users["bob"].pending_gpus),
                         (1, 3, 3))
        nodes = {usage.key: usage for usage in group_by(self.snap.jobs, "node")}
        self.assertEqual(nodes["gpu01"].gpus, 3)
        roles = group_by(self.snap.jobs, "role")
        self.assertIn(("acct-b", "normal", "cpu"), [usage.key for usage in roles])

    def test_mine(self):
        self.assertEqual(sorted(job.job_id for job in self.snap.my_jobs), ["101", "201", "90"])
        self.assertEqual([role.account for role in self.snap.my_roles], ["acct-a"])
        role = self.snap.my_roles[0]
        self.assertEqual(role.qos_allowed, ["normal", "idle"])
        self.assertTrue(role.is_default)

    def test_invoker(self):
        me = self.snap.invoker
        self.assertEqual((me.user, me.admin_level, me.default_account, me.kind),
                         ("alice", "None", "acct-a", "coordinator"))
        self.assertEqual(me.coordinator_accounts, ["acct-b"])
        self.assertFalse(me.is_slurm_user)

    def test_health(self):
        health = self.snap.health
        self.assertEqual([node.name for node in health.problem_nodes], ["gpu02"])
        drained = health.problem_nodes[0]
        self.assertEqual((drained.reason, drained.reason_user), ("bad gpu", "admin1"))
        self.assertEqual(dict(health.free_gpus_by_node), {"gpu01": 1, "gpu02": 0})
        self.assertEqual((health.largest_gpu_block.node, health.largest_gpu_block.gpus),
                         ("gpu01", 1))
        self.assertEqual(health.gpus_stranded_cpu, 0)
        self.assertIsNone(health.gpus_stranded_mem)
        reasons = {usage.key: usage.pending for usage in health.pending_by_reason}
        self.assertEqual(reasons, {"Priority": 3, "ReqNodeNotAvail": 1})
        self.assertEqual(health.reservations, [])
        self.assertTrue(health.scheduler[0].up)
        self.assertEqual([o.node for o in health.load_outliers], [])

    def test_capacity_mismatch_warning(self):
        self.assertEqual([(c.metric, c.live, c.accounting) for c in self.snap.capacity
                          if c.mismatch], [])
        table = cf.responses(**{"sacctmgr cluster": "testcluster|cpu=192,gres/gpu=16,node=3\n"})
        snap = cf.collector(table).snapshot()
        warning = [w for w in snap.warnings if w.code == "capacity_mismatch"][0]
        self.assertEqual(warning.values, {"metric": "gres/gpu", "live": 8, "accounting": 16})
        self.assertEqual(snap.totals.gpus.total, 8)

    def test_priority(self):
        self.assertTrue(self.snap.priority.fairshare_effective)
        table = cf.responses(**{"scontrol show config": fx.config(PriorityWeightFairshare="0")})
        self.assertFalse(cf.collector(table).snapshot().priority.fairshare_effective)

    def test_limits(self):
        found = applicable_limits(self.snap, "acct-a", "alice", "normal", "gpu")
        names = {(x.scope, x.name, x.metric) for x in found}
        self.assertIn(("assoc", "GrpTRES", "gres/gpu"), names)
        self.assertIn(("qos_user", "MaxTRESPU", "gres/gpu"), names)
        self.assertIn(("qos", "MaxWallPJ", None), names)  # from slurmdbd
        per_user = [x for x in found if x.name == "MaxTRESPU"]
        self.assertEqual(len(per_user), 1)  # the slurmdbd template is replaced by real usage
        self.assertEqual(per_user[0].used, 2)


class HostileTest(unittest.TestCase):
    def test_forged_fields_in_job_name(self):
        victim = fx.FakeJob(301, "victim", user="bob", uid=cf.BOB, account="acct-b",
                            nodes="gpu01", cpus=8, gpus=1,
                            per_node=["Nodes=gpu01 CPU_IDs=0-7 Mem=0 GRES=gpu:a100:1(IDX:0)"])
        forged_line = victim.scontrol().replace("gres/gpu=1", "gres/gpu=4").rstrip("\n")
        evil_name = ("x UserId=root(0) GroupId=root(0) Account=admin QOS=top "
                     "\x1b]0;pwn\x07|zzz\n" + forged_line)
        mallory = fx.FakeJob(302, evil_name, user="mallory", uid=cf.MALLORY, account="acct-c",
                             nodes="gpu01", cpus=4, gpus=1,
                             per_node=["Nodes=gpu01 CPU_IDs=8-11 Mem=0 "
                                       "GRES=gpu:a100:1(IDX:1)"])
        jobs = [victim, mallory]
        snap = cf.collector(cf.responses(jobs)).snapshot()
        evil = snap.job("302")
        self.assertEqual((evil.user, evil.uid, evil.account, evil.qos),
                         ("mallory", cf.MALLORY, "acct-c", "normal"))
        self.assertNotIn("\x1b", evil.name)
        self.assertNotIn("\n", evil.name)
        self.assertEqual(evil.gpus_alloc, 1)
        target = snap.job("301")
        self.assertEqual(target.user, "bob")
        self.assertIn(target.detail_status, ("ok", "ambiguous"))
        self.assertIn(target.gpus_alloc, (1, None))  # never the forged 4
        self.assertEqual(len(snap.jobs), 2)

    def test_mismatched_detail_is_ignored(self):
        job = fx.FakeJob(303, "real", nodes="gpu01", gpus=1, detail_uid=0,
                         per_node=["Nodes=gpu01 CPU_IDs=0-3 Mem=0 GRES=gpu:a100:1(IDX:0)"])
        snap = cf.collector(cf.responses([job])).snapshot()
        self.assertEqual(snap.job("303").detail_status, "missing")
        self.assertEqual(snap.job("303").uid, cf.ALICE)

    def test_key_like_account_rejects_detail(self):
        job = fx.FakeJob(304, "j", account="a AllocTRES=gres/gpu=64", nodes="cpu01", gpus=0)
        snap = cf.collector(cf.responses([job])).snapshot()
        self.assertEqual(snap.job("304").detail_status, "ambiguous")

    def test_hostile_identity_values_marked(self):
        job = fx.FakeJob(305, "j", user="ali‮ce", nodes="cpu01")
        snap = cf.collector(cf.responses([job])).snapshot()
        self.assertNotEqual(snap.job("305").user, "alice")
        self.assertNotIn("‮", snap.job("305").user)


class SiteShapeTest(unittest.TestCase):
    def test_private_data_jobs(self):
        everyone = cf.gpu_cluster_jobs()
        mine = [job for job in everyone if job.uid == cf.ALICE]
        table = cf.responses(mine, **{"scontrol show config": fx.config(PrivateData="jobs,usage")})
        snap = cf.collector(table).snapshot()
        self.assertIn("visibility_limited", [w.code for w in snap.warnings])
        self.assertIn("accounts_limited", [w.code for w in snap.warnings])
        result = reconcile_gpus(snap)
        self.assertEqual((result.node_alloc, result.jobs_alloc), (3, 2))
        self.assertTrue(result.explained)
        self.assertIn("other users' jobs are hidden (PrivateData)", result.causes)

    def test_no_accounting(self):
        table = cf.responses(**{
            "scontrol show config": fx.config(AccountingStorageType="accounting_storage/none",
                                              AccountingStorageTRES="cpu,mem,node"),
            "scontrol show assoc_mgr": fx.unavailable("Accounting storage is disabled"),
        })
        collector = cf.collector(table)
        snap = collector.snapshot()
        self.assertFalse(any(argv[0] == "sacctmgr" for argv in collector.runner.calls))
        self.assertIsNone(snap.roles)
        self.assertIsNone(snap.limits)
        self.assertEqual(snap.capabilities["history"].status, Status.UNAVAILABLE)
        self.assertEqual(snap.capabilities["assoc_mgr"].status, Status.UNAVAILABLE)
        self.assertEqual(snap.totals.gpus.total, 8)  # nodes still carry gres/gpu
        self.assertEqual(snap.capacity, [])

    def test_no_gpus(self):
        nodes = fx.node("c1", "MIXED", alloc_cpus=4) + fx.node("c2")
        jobs = [fx.FakeJob(1, partition="all", nodes="c1", cpus=4,
                           per_node=["Nodes=c1 CPU_IDs=0-3 Mem=0 GRES="]),
                fx.FakeJob(2, state="PENDING", partition="all", reason="Resources")]
        table = cf.responses(jobs, **{
            "scontrol show config": fx.config(GresTypes="(null)",
                                              AccountingStorageTRES="cpu,mem,node"),
            "scontrol show node": nodes,
            "scontrol show partition": fx.partition("all", "c[1-2]"),
            "sacctmgr cluster": "testcluster|cpu=128,node=2\n"})
        snap = cf.collector(table).snapshot()
        self.assertEqual((snap.totals.gpus.total, snap.totals.gpus.free), (0, 0))
        self.assertEqual(snap.job("1").gpus_alloc, 0)
        self.assertEqual(snap.job("2").gpus_req, 0)
        self.assertIsNone(snap.health.largest_gpu_block)
        self.assertTrue(reconcile_gpus(snap).consistent)

    def test_untyped_unaccounted_gpus(self):
        nodes = fx.node("g1", "MIXED", alloc_cpus=4, gres="gpu:2",
                        gres_used="gpu:1(IDX:0)")
        jobs = [fx.FakeJob(5, partition="all", nodes="g1", cpus=4, gpus=0,
                           gres_request="gres/gpu:1",
                           per_node=["Nodes=g1 CPU_IDs=0-3 Mem=0 GRES=gpu:1(IDX:0)"]),
                fx.FakeJob(6, state="PENDING", partition="all", gres_request="gres/gpu:2",
                           reason="Resources")]
        table = cf.responses(jobs, **{
            "scontrol show config": fx.config(AccountingStorageTRES="cpu,mem,node"),
            "scontrol show node": nodes,
            "scontrol show partition": fx.partition("all", "g1")})
        snap = cf.collector(table).snapshot()
        self.assertEqual((snap.totals.gpus.total, snap.totals.gpus.alloc), (2, 1))
        self.assertEqual(snap.job("5").gpus_alloc, 1)
        self.assertEqual(snap.job("6").gpus_req, 2)
        self.assertTrue(reconcile_gpus(snap).consistent)

    def test_memory_tracked(self):
        nodes = fx.node("c1", "MIXED", alloc_cpus=4, mem=1000, alloc_mem=600)
        jobs = [fx.FakeJob(1, partition="all", nodes="c1", cpus=4, mem="600M",
                           per_node=["Nodes=c1 CPU_IDs=0-3 Mem=600 GRES="])]
        table = cf.responses(jobs, **{
            "scontrol show config": fx.config(SelectTypeParameters="CR_CORE_MEMORY"),
            "scontrol show node": nodes, "scontrol show partition": fx.partition("all", "c1")})
        snap = cf.collector(table).snapshot()
        self.assertEqual(snap.totals.mem.alloc, 600 * 2**20)
        self.assertEqual(snap.totals.mem.free, 400 * 2**20)
        self.assertEqual(snap.job("1").mem_alloc, 600 * 2**20)

    def test_completing_job_still_holds(self):
        nodes = fx.node("gpu01", "MIXED+COMPLETING", alloc_cpus=16, gres="gpu:a100:4",
                        gres_used="gpu:a100:2(IDX:0-1)", gpus=4, alloc_gpus=2,
                        gpu_tres_type="a100", partitions="all")
        jobs = [fx.FakeJob(1, nodes="gpu01", cpus=16, gpus=2, partition="all",
                           per_node=["Nodes=gpu01 CPU_IDs=0-15 Mem=0 GRES=gpu:a100:2(IDX:0-1)"]),
                fx.FakeJob(2, state="COMPLETING", nodes="gpu01", cpus=8, gpus=1,
                           partition="all",
                           per_node=["Nodes=gpu01 CPU_IDs=16-23 Mem=0 GRES="])]
        table = cf.responses(jobs, **{"scontrol show node": nodes,
                                      "scontrol show partition": fx.partition("all", "gpu01")})
        snap = cf.collector(table).snapshot()
        node = snap.node("gpu01")
        self.assertEqual((node.gpus_completing, node.cpus_completing), (1, 8))
        self.assertEqual((node.gpus_free, node.cpus_free), (1, 40))
        gpus = snap.totals.gpus
        self.assertEqual((gpus.alloc, gpus.completing, gpus.free), (2, 1, 1))
        self.assertEqual(reconcile_gpus(snap).difference, 0)
        self.assertTrue(snap.job("2").holds_resources)

    def test_completing_already_counted_by_node(self):
        nodes = fx.node("gpu01", "MIXED+COMPLETING", alloc_cpus=8, gres="gpu:4", gpus=4,
                        alloc_gpus=1, partitions="all")
        jobs = [fx.FakeJob(2, state="COMPLETING", nodes="gpu01", cpus=8, gpus=1,
                           gpu_type=None, partition="all",
                           per_node=["Nodes=gpu01 CPU_IDs=0-7 Mem=0 GRES=gpu:1(IDX:0)"])]
        table = cf.responses(jobs, **{"scontrol show node": nodes,
                                      "scontrol show partition": fx.partition("all", "gpu01")})
        snap = cf.collector(table).snapshot()
        self.assertEqual((snap.node("gpu01").gpus_completing, snap.totals.gpus.free), (0, 3))

    def test_everything_refused(self):
        refused = {key: fx.denied() for key in cf.responses()}
        snap = cf.collector(refused).snapshot()
        self.assertIsNone(snap.nodes)
        self.assertIsNone(snap.jobs)
        self.assertEqual(snap.capabilities["jobs"].status, Status.DENIED)
        self.assertIsNone(snap.totals.gpus.total)
        self.assertIn("source_incomplete", [w.code for w in snap.warnings])

    def test_slurm_not_installed(self):
        programs = ("scontrol", "squeue", "sacctmgr", "sacct", "sstat")
        snap = cf.collector({}, missing=programs).snapshot()
        self.assertEqual(snap.capabilities["nodes"].status, Status.UNAVAILABLE)
        self.assertEqual(snap.capabilities["steps"].status, Status.UNAVAILABLE)

    def test_hidden_node_in_job(self):
        jobs = [fx.FakeJob(1, nodes="secret01", gpus=1, partition="all",
                           per_node=["Nodes=secret01 CPU_IDs=0-3 Mem=0 GRES=gpu:a100:1(IDX:0)"])]
        snap = cf.collector(cf.responses(jobs)).snapshot()
        result = reconcile_gpus(snap)
        self.assertIn("node is not visible to the invoker", result.causes)


if __name__ == "__main__":
    unittest.main()
