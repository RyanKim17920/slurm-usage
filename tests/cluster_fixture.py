"""A whole synthetic cluster, served by FakeRunner."""

from typing import Dict, List, Optional

from slurm_usage.live import LiveCollector
from tests import fixtures as fx

ALICE, BOB, MALLORY = 2001, 2002, 2003

USERS = ("UserName=alice(2001) DefAccount=acct-a DefWckey= AdminLevel=None\n")


def gpu_cluster_nodes() -> str:
    return (fx.node("gpu01", "MIXED", alloc_cpus=24, gres="gpu:a100:4(S:0-1)",
                    gres_used="gpu:a100:3(IDX:0-2)", gpus=4, alloc_gpus=3,
                    gpu_tres_type="a100", partitions="all,gpu", load=20.0)
            + fx.node("gpu02", "IDLE+DRAIN", gres="gpu:a100:4(S:0-1)",
                      gres_used="gpu:a100:0(IDX:N/A)", gpus=4, gpu_tres_type="a100",
                      partitions="all,gpu", reason="bad gpu [admin1@%d]" % fx.NOW)
            + fx.node("cpu01", "MIXED", alloc_cpus=8, partitions="all,cpu", load=7.5))


def gpu_cluster_partitions() -> str:
    return (fx.partition("all", "gpu[01-02],cpu01", default=True)
            + fx.partition("gpu", "gpu[01-02]", extra="JobDefaults=DefCpuPerGPU=8 ")
            + fx.partition("cpu", "cpu01"))


def gpu_cluster_jobs() -> List[fx.FakeJob]:
    return [
        fx.FakeJob(101, "train", nodes="gpu01", cpus=16, gpus=2,
                   per_node=["JOB_GRES=gpu:a100:2 Nodes=gpu01 CPU_IDs=0-15 Mem=16384 "
                             "GRES=gpu:a100:2(IDX:0-1)"]),
        fx.FakeJob(102, "eval", user="bob", uid=BOB, account="acct-b", nodes="gpu01",
                   cpus=8, gpus=1,
                   per_node=["JOB_GRES=gpu:a100:1 Nodes=gpu01 CPU_IDs=16-23 Mem=8192 "
                             "GRES=gpu:a100:1(IDX:2)"]),
        fx.FakeJob(103, "prep", user="bob", uid=BOB, account="acct-b", partition="cpu",
                   nodes="cpu01", cpus=8, gpus=0,
                   per_node=["Nodes=cpu01 CPU_IDs=0-7 Mem=4096 GRES="]),
        fx.FakeJob(200, "sweep", user="bob", uid=BOB, account="acct-b", state="PENDING",
                   reason="Priority", gpus=1, array="0", array_job=200, detail_array="0-2"),
        fx.FakeJob(200, "sweep", user="bob", uid=BOB, account="acct-b", state="PENDING",
                   reason="Priority", gpus=1, array="1", array_job=200, in_detail=False),
        fx.FakeJob(200, "sweep", user="bob", uid=BOB, account="acct-b", state="PENDING",
                   reason="Priority", gpus=1, array="2", array_job=200, in_detail=False),
        fx.FakeJob(201, "big", state="PENDING", partition="gpu,all",
                   reason="ReqNodeNotAvail, UnavailableNodes:gpu02", gpus=4),
        fx.FakeJob(90, "old", state="COMPLETED", nodes="cpu01", cpus=2),
    ]


def assoc_text() -> str:
    return fx.assoc_mgr(
        USERS,
        fx.assoc("acct-a", grp_tres="cpu=N(16),mem=N(0),gres/gpu=4(2)")
        + fx.assoc("acct-a", "alice", ALICE, default=True)
        + fx.assoc("acct-b", "bob", BOB),
        "QOS=normal(1)\n    UsageRaw=1.000000\n    GrpTRES=cpu=N(24),gres/gpu=N(3)\n"
        "    Priority=10\n    Account Limits\n        No Accounts\n"
        "    User Limits\n      alice(2001)\n        MaxTRESPU=gres/gpu=4(2)\n")


def responses(jobs: Optional[List[fx.FakeJob]] = None, **overrides) -> Dict[str, object]:
    jobs = gpu_cluster_jobs() if jobs is None else jobs
    table = {
        "scontrol show config": fx.config(),
        "scontrol show node": gpu_cluster_nodes(),
        "scontrol show partition": gpu_cluster_partitions(),
        "scontrol show job": "".join(job.scontrol_or_nothing() for job in jobs),
        "scontrol show reservation": "No reservations in the system\n",
        "scontrol show assoc_mgr": assoc_text(),
        "scontrol ping": "Slurmctld(primary) at ctl1 is UP\n",
        "squeue": lambda argv: fx.squeue_output(argv, jobs),
        "sacctmgr qos": "normal|10|2-00:00:00|cpu=512|gres/gpu=4|||10|\nidle|0||||||| \n",
        "sacctmgr cluster": "testcluster|cpu=192,mem=1536000M,node=3,gres/gpu=8\n",
        "sacctmgr assoc": "acct-a|||normal,idle|normal\nacct-a|alice||normal,idle|normal\n"
                          "acct-b|bob||normal|normal\n",
        "sacctmgr user": "alice|None|acct-a\n",
        "sacctmgr account": "acct-a|\nacct-b|alice\n",
    }  # type: Dict[str, object]
    table.update(overrides)
    return table


def collector(table: Dict[str, object], uid: int = ALICE, user: str = "alice",
              missing=()) -> LiveCollector:
    return LiveCollector(fx.FakeRunner(table, missing), uid=uid, user=user, euid=uid,
                         workers=1)
