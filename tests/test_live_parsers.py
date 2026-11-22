import unittest

from slurm_usage.live import cluster as cluster_text
from slurm_usage.live import limits, nodes
from slurm_usage.model import Availability, ClusterInfo
from tests import fixtures as fx

CR_CORE = ClusterInfo(select_parameters="CR_CORE", mem_tracked=False, gpu_accounted=True)
CR_MEM = ClusterInfo(select_parameters="CR_CORE_MEMORY", mem_tracked=True, gpu_accounted=True)


def one(text, info=CR_CORE, partitions=None):
    found = nodes.parse_nodes(text, info, partitions)
    return found[0] if len(found) == 1 else found


class NodeTest(unittest.TestCase):
    def test_typed_gpu_node(self):
        node = one(fx.node("gpu01", "MIXED", alloc_cpus=16, gres="gpu:a100:4(S:0-1)",
                           gres_used="gpu:a100:2(IDX:0,2)", gpus=4, alloc_gpus=2,
                           gpu_tres_type="a100", partitions="all,gpu"))
        self.assertEqual((node.gpus_total, node.gpus_alloc, node.gpus_free), (4, 2, 2))
        self.assertEqual(node.gpu_types, {"a100": 4})
        self.assertEqual(node.gpu_indices_alloc, [0, 2])
        self.assertEqual(node.partitions, ["all", "gpu"])
        self.assertEqual(node.availability, Availability.SCHEDULABLE)
        self.assertEqual(node.cpus_free, 48)

    def test_duplicate_records_once(self):
        text = fx.node("cpu01") + fx.node("cpu01", "DOWN") + fx.node("cpu02")
        self.assertEqual([n.name for n in nodes.parse_nodes(text, CR_CORE)], ["cpu01", "cpu02"])

    def test_natural_sort(self):
        text = fx.node("n10") + fx.node("n9") + fx.node("n9a")
        self.assertEqual([n.name for n in nodes.parse_nodes(text, CR_CORE)], ["n9", "n9a", "n10"])

    def test_alloc_mem_untracked(self):
        self.assertIsNone(one(fx.node("cpu01", alloc_mem=0)).mem_alloc)
        self.assertIsNone(one(fx.node("cpu01", alloc_mem=0)).mem_free)
        tracked = one(fx.node("cpu01", mem=1000, alloc_mem=0), CR_MEM)
        self.assertEqual(tracked.mem_alloc, 0)
        self.assertEqual(tracked.mem_free, 1000 * 2**20)

    def test_partition_level_memory_tracking(self):
        parts = cluster_text.parse_partitions(fx.partition(
            "mem", "cpu01", select="CR_CORE_MEMORY"))
        node = one(fx.node("cpu01", partitions="mem", alloc_mem=100), CR_CORE, parts)
        self.assertEqual(node.mem_alloc, 100 * 2**20)

    def test_down_and_drain_not_free(self):
        for state in ("DOWN", "IDLE+DRAIN", "MIXED+DRAIN", "DOWN*", "IDLE*", "IDLE+MAINTENANCE",
                      "IDLE+FAIL", "IDLE+NOT_RESPONDING", "IDLE$"):
            node = one(fx.node("gpu01", state, gres="gpu:4", gpus=4, reason="bad gpu"))
            self.assertEqual(node.availability, Availability.UNAVAILABLE, state)
            self.assertEqual(node.gpus_free, 0, state)
            self.assertEqual(node.cpus_free, 0, state)

    def test_reserved(self):
        node = one(fx.node("cpu01", "IDLE+RESERVED"))
        self.assertEqual(node.availability, Availability.RESERVED)
        self.assertEqual(node.cpus_free, 0)

    def test_completing_and_planned_are_available(self):
        self.assertTrue(one(fx.node("a", "MIXED+COMPLETING")).schedulable)
        self.assertTrue(one(fx.node("a", "IDLE+PLANNED")).schedulable)

    def test_reason_with_who_and_when(self):
        node = one(fx.node("gpu01", "IDLE+DRAIN",
                           reason="ECC errors State=IDLE [admin1@%d]" % fx.NOW))
        self.assertEqual(node.reason, "ECC errors State=IDLE")
        self.assertEqual(node.reason_user, "admin1")
        self.assertEqual(node.reason_time, fx.NOW)
        self.assertEqual(node.state, "IDLE")  # a reason cannot override the real state

    def test_hostile_reason(self):
        node = one(fx.node("gpu01", "DOWN", reason="\x1b]0;x\x07fix CPUTot=1 [a@b]"))
        self.assertEqual(node.cpus_total, 64)
        self.assertNotIn("\x1b", node.reason)

    def test_untyped_gpus_without_gpu_tres(self):
        info = ClusterInfo(mem_tracked=False, gpu_accounted=False)
        node = one(fx.node("g1", "MIXED", gres="gpu:2", gres_used="gpu:1(IDX:1)"), info)
        self.assertEqual((node.gpus_total, node.gpus_alloc, node.gpu_types), (2, 1, {}))
        self.assertEqual(node.gpu_indices_alloc, [1])

    def test_no_gpus_anywhere(self):
        info = ClusterInfo(mem_tracked=False, gpu_accounted=False)
        node = one(fx.node("c1"), info)
        self.assertEqual((node.gpus_total, node.gpus_alloc, node.gpus_free), (0, 0, 0))

    def test_mig_types(self):
        node = one(fx.node("m1", "MIXED", gres="gpu:1g.10gb:7,gpu:3g.40gb:2", gpus=9,
                           alloc_gpus=3, gres_used="gpu:1g.10gb:3(IDX:0-2),gpu:3g.40gb:0(IDX:N/A)"))
        self.assertEqual(node.gpus_total, 9)
        self.assertEqual(node.gpu_types_alloc, {"1g.10gb": 3})
        self.assertEqual(node.gpus_free_by_type(), {"1g.10gb": 4, "3g.40gb": 2})

    def test_shard_makes_gpu_unavailable_whole(self):
        node = one(fx.node("s1", "MIXED", gres="gpu:2,shard:8", gpus=2,
                           gres_used="gpu:0(IDX:N/A),shard:3(3/4,0/4)"))
        self.assertEqual(node.gpus_shared, 1)
        self.assertEqual(node.gpus_free, 1)

    def test_future_node_and_unknown_state(self):
        node = one(fx.node("f1", "FUTURE"))
        self.assertTrue(nodes.is_future(node))
        self.assertEqual(one(fx.node("u1", "UNKNOWN")).availability, Availability.UNAVAILABLE)

    def test_no_up_partition(self):
        parts = cluster_text.parse_partitions(fx.partition("p", "c1", state="DOWN"))
        node = one(fx.node("c1", partitions="p"), CR_CORE, parts)
        self.assertEqual(node.unavailable_because, ["NO_UP_PARTITION"])

    def test_stranded_gpus(self):
        node = one(fx.node("g1", "MIXED", cpus=32, alloc_cpus=32, gres="gpu:4", gpus=4,
                           alloc_gpus=2, threads=2))
        self.assertEqual((node.gpus_free, node.gpus_stranded_cpu, node.gpus_startable), (2, 2, 0))


class ClusterTextTest(unittest.TestCase):
    def test_priority_weights(self):
        conf = cluster_text.parse_config(fx.config(PriorityWeightFairshare="0"))
        prio = cluster_text.priority_config(conf)
        self.assertEqual(prio.weights["fairshare"], 0)
        self.assertFalse(prio.fairshare_effective)
        self.assertEqual(prio.decay_half_life, 7 * 86400)
        conf = cluster_text.parse_config(fx.config())
        self.assertTrue(cluster_text.priority_config(conf).fairshare_effective)
        basic = cluster_text.parse_config(fx.config(PriorityType="priority/basic"))
        self.assertFalse(cluster_text.priority_config(basic).fairshare_effective)

    def test_cluster_info(self):
        info = cluster_text.cluster_info(cluster_text.parse_config(fx.config()))
        self.assertEqual((info.name, info.mem_tracked, info.gpu_accounted, info.has_accounting),
                         ("testcluster", False, True, True))
        none = cluster_text.cluster_info(cluster_text.parse_config(fx.config(
            AccountingStorageType="accounting_storage/none")))
        self.assertFalse(none.has_accounting)

    def test_partitions(self):
        parts = cluster_text.parse_partitions(
            fx.partition("gpu", "gpu[01-02]", default=True,
                         extra="JobDefaults=DefCpuPerGPU=8,DefMemPerGPU=1000 ")
            + fx.partition("gpu", "dup"))
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].nodes, ["gpu01", "gpu02"])
        self.assertEqual(parts[0].default_cpus_per_gpu, 8)
        self.assertEqual(parts[0].max_time, 172800)
        self.assertIsNone(parts[0].allow_accounts)

    def test_ping(self):
        hosts = cluster_text.parse_ping(
            "Slurmctld(primary) at ctl1 is UP\nSlurmctld(backup) at ctl2 is DOWN\n")
        self.assertEqual([(h.role, h.host, h.up) for h in hosts],
                         [("primary", "ctl1", True), ("backup", "ctl2", False)])
        old = cluster_text.parse_ping("Slurmctld(primary/backup) at ctl1/ctl2 are UP/DOWN")
        self.assertEqual([h.up for h in old], [True, False])

    def test_reservations(self):
        text = ("ReservationName=maint StartTime=%d EndTime=%d Duration=01:00:00 "
                "Nodes=gpu[01-02] NodeCnt=2 CoreCnt=64 Features=(null) PartitionName=(null) "
                "Flags=MAINT,SPEC_NODES TRES=cpu=128 Users=root Groups=(null) Accounts=(null) "
                "Licenses=(null) State=INACTIVE BurstBuffer=(null) MaxStartDelay=(null)\n"
                % (fx.NOW, fx.NOW + 3600))
        found = cluster_text.parse_reservations(text)
        self.assertEqual(found[0].nodes, ["gpu01", "gpu02"])
        self.assertEqual(found[0].flags, ["MAINT", "SPEC_NODES"])
        self.assertEqual(cluster_text.parse_reservations("No reservations in the system\n"), [])


class AssocMgrTest(unittest.TestCase):
    def text(self):
        users = ("UserName=alice(2001) DefAccount=acct-a DefWckey= AdminLevel=None\n"
                 "UserName=bob(2002) DefAccount=acct-b DefWckey= AdminLevel=Operator\n")
        assocs = (fx.assoc("acct-a", grp_tres="cpu=N(8),gres/gpu=8(3)")
                  + fx.assoc("acct-a", "alice", 2001, default=True,
                             run_mins="cpu=N(4294967228),gres/gpu=N(42)"))
        qos = ("QOS=normal(1)\n    UsageRaw=100.000000\n"
               "    GrpJobs=N(2) GrpJobsAccrue=N(0) GrpSubmitJobs=N(3) GrpWall=N(5.00)\n"
               "    GrpTRES=cpu=N(8),gres/gpu=16(3)\n"
               "    MaxWallPJ=1440\n    MaxTRESPJ=\n    Priority=20\n    PreemptMode=OFF\n"
               "    Account Limits\n      acct-a\n"
               "        MaxJobsPA=N(1) MaxSubmitJobsPA=N(1)\n"
               "        MaxTRESPA=cpu=N(8),gres/gpu=4(3)\n"
               "    User Limits\n      alice(2001)\n"
               "        MaxJobsPU=N(1)\n        MaxTRESPU=gres/gpu=2(3)\n"
               "QOS=idle(2)\n    UsageRaw=0.000000\n"
               "    Account Limits\n        No Accounts\n    User Limits\n        No Users\n")
        return fx.assoc_mgr(users, assocs, qos)

    def test_roles_and_limits(self):
        data = limits.parse_assoc_mgr(self.text(), {"cpu": 128, "gres/gpu": 8})
        self.assertEqual([(u.name, u.uid, u.admin_level) for u in data.users],
                         [("alice", 2001, "None"), ("bob", 2002, "Operator")])
        account, alice = data.roles
        self.assertIsNone(account.user)
        self.assertEqual(alice.user, "alice")
        self.assertTrue(alice.is_default)
        self.assertEqual(alice.lineage, ["acct-a"])
        self.assertEqual(alice.fairshare_factor, 0.5)
        gpu = [x for x in account.limits if x.name == "GrpTRES" and x.metric == "gres/gpu"][0]
        self.assertEqual((gpu.limit, gpu.used, gpu.fraction), (8, 3, 0.375))
        cpu = [x for x in account.limits if x.name == "GrpTRES" and x.metric == "cpu"][0]
        self.assertTrue(cpu.unlimited)
        self.assertEqual(cpu.used, 8)

    def test_nonsense_run_mins(self):
        data = limits.parse_assoc_mgr(self.text(), {"cpu": 128, "gres/gpu": 8})
        alice = data.roles[1]
        bad = [x for x in alice.limits if x.name == "GrpTRESRunMins" and x.metric == "cpu"][0]
        self.assertTrue(bad.suspect)
        self.assertIsNone(bad.used)
        good = [x for x in alice.limits if x.name == "GrpTRESRunMins"
                and x.metric == "gres/gpu"][0]
        self.assertEqual(good.used, 42 * 60)

    def test_qos_scopes(self):
        data = limits.parse_assoc_mgr(self.text())
        normal = data.qos[0]
        self.assertEqual((normal.name, normal.priority, normal.usage_raw), ("normal", 20, 100.0))
        wall = [x for x in normal.limits if x.name == "MaxWallPJ"][0]
        self.assertEqual(wall.limit, 1440 * 60)
        per_user = [x for x in normal.limits if x.scope == "qos_user"
                    and x.metric == "gres/gpu"][0]
        self.assertEqual((per_user.user, per_user.limit, per_user.used), ("alice", 2, 3))
        per_account = [x for x in normal.limits if x.scope == "qos_account"
                       and x.metric == "gres/gpu"][0]
        self.assertEqual(per_account.account, "acct-a")
        self.assertEqual(data.qos[1].name, "idle")
        self.assertEqual(data.qos[1].limits, [])


if __name__ == "__main__":
    unittest.main()
