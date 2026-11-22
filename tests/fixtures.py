"""Synthetic Slurm text in the shapes real clusters print. Every name is invented.

Times are epoch seconds because the runner sets SLURM_TIME_FORMAT=%s.
"""

import re
from typing import Callable, Dict, List, Optional, Sequence, Union

from slurm_usage.model import Status
from slurm_usage.runner import Result, check

NOW = 1_800_000_000
OS_TEXT = "Linux 6.1.0-test #1 SMP PREEMPT_DYNAMIC Mon Jan 1 00:00:00 UTC 2024"


def config(**overrides: str) -> str:
    values = {
        "ClusterName": "testcluster",
        "SLURM_VERSION": "24.05.1",
        "SelectType": "select/cons_tres",
        "SelectTypeParameters": "CR_CORE",
        "AccountingStorageType": "accounting_storage/slurmdbd",
        "AccountingStorageTRES": "cpu,mem,energy,node,billing,gres/gpu",
        "GresTypes": "gpu",
        "MinJobAge": "300 sec",
        "PriorityType": "priority/multifactor",
        "PriorityWeightAge": "1000",
        "PriorityWeightFairshare": "10000",
        "PriorityWeightQOS": "500",
        "PriorityWeightTRES": "(null)",
        "PriorityDecayHalfLife": "7-00:00:00",
        "PrivateData": "(null)",
        "SlurmUser": "slurm(64030)",
    }
    values.update(overrides)
    return "Configuration data as of 2024-01-01T00:00:00\n" + "".join(
        "%-24s= %s\n" % (key, value) for key, value in values.items() if value is not None)


def node(name: str, state: str = "IDLE", cpus: int = 64, alloc_cpus: int = 0,
         mem: int = 512000, alloc_mem: int = 0, gres: str = "(null)",
         gres_used: Optional[str] = None, gpus: Optional[int] = None,
         alloc_gpus: Optional[int] = None, gpu_tres_type: Optional[str] = None,
         partitions: str = "all", load: float = 0.0, reason: Optional[str] = None,
         extra: str = "", threads: int = 2, effective: Optional[int] = None) -> str:
    cfg = "cpu=%d,mem=%dM,billing=%d" % (cpus, mem, cpus)
    alloc = "cpu=%d" % alloc_cpus if alloc_cpus else ""
    if gpus is not None:
        cfg += ",gres/gpu=%d" % gpus
        if gpu_tres_type:
            cfg += ",gres/gpu:%s=%d" % (gpu_tres_type, gpus)
    if alloc_gpus:
        alloc += ",gres/gpu=%d" % alloc_gpus
        if gpu_tres_type:
            alloc += ",gres/gpu:%s=%d" % (gpu_tres_type, alloc_gpus)
    used = "" if gres_used is None else " GresUsed=" + gres_used
    line = (
        "NodeName={n} Arch=x86_64 CoresPerSocket=16 CPUAlloc={ac} CPUEfctv={ce} CPUTot={c} "
        "CPULoad={load:.2f} AvailableFeatures=fast,ib ActiveFeatures=fast,ib Gres={g} "
        "GresDrain=N/A{used} NodeAddr={n} NodeHostName={n} Version=24.05.1 OS={os} "
        "RealMemory={m} AllocMem={am} FreeMem={fm} Sockets=2 Boards=1 State={s} "
        "ThreadsPerCore={t} TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A Partitions={p} "
        "BootTime={boot} SlurmdStartTime={boot} LastBusyTime={boot} ResumeAfterTime=None "
        "CfgTRES={cfg} AllocTRES={alloc} CurrentWatts=0 AveWatts=0{extra}"
    ).format(n=name, ac=alloc_cpus, c=cpus, ce=cpus if effective is None else effective,
             load=load, g=gres, used=used, os=OS_TEXT, m=mem, am=alloc_mem, fm=mem // 4,
             s=state, t=threads, p=partitions, boot=NOW - 86400, cfg=cfg, alloc=alloc,
             extra=extra)
    if reason is not None:
        line += " Reason=" + reason
    return line + "\n"


def partition(name: str, nodes: str, state: str = "UP", default: bool = False,
              extra: str = "", total_cpus: int = 0, select: str = "NONE") -> str:
    return (
        "PartitionName={n} AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL AllocNodes=ALL "
        "Default={d} QoS=N/A DefaultTime=NONE DisableRootJobs=NO ExclusiveUser=NO "
        "GraceTime=0 Hidden=NO MaxNodes=UNLIMITED MaxTime=2-00:00:00 MinNodes=0 LLN=NO "
        "MaxCPUsPerNode=UNLIMITED Nodes={nodes} PriorityJobFactor=1 PriorityTier=1 "
        "RootOnly=NO ReqResv=NO OverSubscribe=NO OverTimeLimit=NONE PreemptMode=OFF "
        "State={s} TotalCPUs={tc} TotalNodes=1 SelectTypeParameters={sel} {extra}"
        "DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED TRES=cpu={tc}\n"
    ).format(n=name, d="YES" if default else "NO", nodes=nodes, s=state, tc=total_cpus,
             extra=extra, sel=select)


class FakeJob:
    """One squeue row and one scontrol record, consistent with each other."""

    def __init__(self, job_id: int, name: str = "train", user: str = "alice",
                 uid: int = 2001, account: str = "acct-a", qos: str = "normal",
                 partition: str = "gpu", state: str = "RUNNING", reason: str = "None",
                 nodes: str = "", cpus: int = 4, gpus: int = 0, mem: str = "16G",
                 gpu_type: Optional[str] = "a100", per_node: Optional[List[str]] = None,
                 array: Optional[str] = None, array_job: Optional[int] = None,
                 submit: int = NOW - 3600, start: Optional[int] = None,
                 comment: str = "", workdir: str = "/home/alice/project",
                 stdout: str = "/home/alice/project/slurm-%j.out", stderr: str = "",
                 node_count: int = 1, restarts: int = 0, gres_request: str = "N/A",
                 detail_name: Optional[str] = None, detail_uid: Optional[int] = None,
                 tail: str = "", in_detail: bool = True,
                 detail_array: Optional[str] = None) -> None:
        self.__dict__.update(locals())
        del self.__dict__["self"]
        if start is None:
            self.start = submit + 60 if state != "PENDING" else NOW + 3600

    @property
    def holds(self) -> bool:
        return self.state in ("RUNNING", "COMPLETING")

    def tres(self) -> str:
        text = "cpu=%d,mem=%s,node=%d,billing=%d" % (self.cpus, self.mem, self.node_count,
                                                     self.cpus)
        if self.gpus:
            text += ",gres/gpu=%d" % self.gpus
            if self.gpu_type:
                text += ",gres/gpu:%s=%d" % (self.gpu_type, self.gpus)
        return text

    def scontrol_or_nothing(self) -> str:
        return self.scontrol() if self.in_detail else ""

    def squeue_values(self) -> Dict[str, str]:
        display = str(self.job_id)
        if self.array is not None:
            display = "%d_%s" % (self.array_job or self.job_id, self.array)
        return {
            "%i": display, "%A": str(self.job_id),
            "%F": str(self.array_job or self.job_id) if self.array is not None else "N/A",
            "%K": self.array if self.array is not None else "N/A",
            "%U": str(self.uid), "%u": self.user, "%a": self.account, "%q": self.qos,
            "%P": self.partition, "%T": self.state, "%Q": "100", "%y": "0",
            "%V": str(self.submit), "%S": str(self.start),
            "%e": str(self.start + 7200), "%l": "2:00:00",
            "%M": "0:00" if not self.holds else "1:00:00", "%C": str(self.cpus),
            "%D": str(self.node_count), "%N": self.nodes if self.holds else "",
            "%B": self.nodes.split(",")[0] if self.holds else "n/a",
            "%v": "(null)", "%b": self.gres_request, "%w": "(null)", "%E": "(null)",
            "%r": self.reason, "%j": self.name, "%k": self.comment or "(null)",
            "%Z": self.workdir, "%o": "/home/alice/project/run.sh",
        }

    def scontrol(self) -> str:
        array = ""
        if self.array is not None:
            array = "ArrayJobId=%d ArrayTaskId=%s " % (self.array_job or self.job_id,
                                                       self.detail_array or self.array)
        holds = self.holds
        detail = ""
        if holds:
            for line in self.per_node or []:
                detail += " " + line
        uid = self.uid if self.detail_uid is None else self.detail_uid
        name = self.name if self.detail_name is None else self.detail_name
        return (
            "JobId={id} {array}JobName={name} UserId={user}({uid}) GroupId={user}({uid}) "
            "MCS_label=N/A Priority=100 Nice=0 Account={acct} QOS={qos} JobState={state} "
            "Reason={reason} Dependency=(null) Requeue=1 Restarts={restarts} BatchFlag=1 "
            "Reboot=0 ExitCode=0:0 RunTime=01:00:00 TimeLimit=02:00:00 TimeMin=N/A "
            "SubmitTime={submit} EligibleTime={submit} AccrueTime={submit} StartTime={start} "
            "EndTime={end} Deadline=N/A SuspendTime=None SecsPreSuspend=0 "
            "LastSchedEval={submit} Scheduler=Main Partition={part} AllocNode:Sid=login1:1 "
            "ReqNodeList=(null) ExcNodeList=(null) NodeList={nodes} BatchHost={bh} "
            "NumNodes={nn} NumCPUs={cpus} NumTasks=1 CPUs/Task={cpus} ReqB:S:C:T=0:0:*:* "
            "ReqTRES={tres} AllocTRES={atres} Socks/Node=* NtasksPerN:B:S:C=0:0:*:* "
            "CoreSpec=*{detail} MinCPUsNode={cpus} MinMemoryNode={mem} MinTmpDiskNode=0 "
            "Features=(null) DelayBoot=00:00:00 OverSubscribe=NO Exclusive=NO Contiguous=0 "
            "Licenses=(null) Network=(null) Command=/home/{user}/project/run.sh "
            "SubmitLine=sbatch run.sh WorkDir={wd} Comment={comment} StdErr={err} "
            "StdIn=/dev/null StdOut={out}{tail}\n"
        ).format(id=self.job_id, array=array, name=name, user=self.user, uid=uid,
                 acct=self.account, qos=self.qos, state=self.state, reason=self.reason,
                 restarts=self.restarts, submit=self.submit, start=self.start,
                 end=self.start + 7200, part=self.partition,
                 nodes=self.nodes if holds else "", bh=self.nodes.split(",")[0] if holds else "",
                 nn=self.node_count, cpus=self.cpus, tres=self.tres(),
                 atres=self.tres() if holds else "(null)", detail=detail, mem=self.mem,
                 wd=self.workdir, comment=self.comment, err=self.stderr or self.stdout,
                 out=self.stdout, tail=self.tail)


def squeue_output(argv: Sequence[str], jobs: Sequence[FakeJob]) -> str:
    fmt = argv[argv.index("-o") + 1]
    token = re.match(r"(z[0-9a-f]{10})b", fmt).group(1)
    begin, sep, end = token + "b", token + "s", token + "e"
    codes = fmt[len(begin):-len(end)].split(sep)
    out = []
    for job in jobs:
        values = job.squeue_values()
        out.append(begin + sep.join(values[code] for code in codes) + end + "\n")
    return "".join(out)


def assoc_mgr(users: str = "", assocs: str = "", qos: str = "") -> str:
    return ("Current Association Manager state\n\nUser Records\n\n%s\n"
            "Association Records\n\n%s\nQOS Records\n\n%s" % (users, assocs, qos))


def assoc(account: str, user: Optional[str] = None, uid: int = 0, parent: str = "root",
          grp_tres: str = "cpu=N(0),mem=N(0),gres/gpu=N(0)",
          run_mins: str = "cpu=N(0),gres/gpu=N(0)", default: bool = False) -> str:
    who = "%s(%d)" % (user, uid) if user else ""
    lineage = "/%s/%s" % (parent, account) if parent != "root" else "/%s" % account
    if user:
        lineage += "/0-%s" % user
    return (
        "ClusterName=testcluster Account={a} UserName={who} Partition= Priority=0 ID=1\n"
        "    SharesRaw/Norm/Level/Factor=1/0.25/4/0.50\n"
        "    UsageRaw/Norm/Efctv=1000.00/0.10/0.10\n"
        "    ParentAccount={p}(1) Lineage={lin}/ DefAssoc={d}\n"
        "    GrpJobs=N(1) GrpJobsAccrue=N(0)\n"
        "    GrpSubmitJobs=N(1) GrpWall=N(60.00)\n"
        "    GrpTRES={g}\n"
        "    GrpTRESMins=cpu=N(600),gres/gpu=N(60)\n"
        "    GrpTRESRunMins={r}\n"
        "    MaxJobs= MaxJobsAccrue= MaxSubmitJobs= MaxWallPJ=\n"
        "    MaxTRESPJ=\n    MaxTRESPN=\n    MaxTRESMinsPJ=\n    MinPrioThresh=\n"
        "    Comment=(null)\n"
    ).format(a=account, who=who, p=parent if not user else account, lin=lineage,
             d="Yes" if default else "No", g=grp_tres, r=run_mins)


Response = Union[str, Result, Callable[[List[str]], Union[str, Result]]]


class FakeRunner:
    """Answers argv from a table; checks every argv against the read-only policy."""

    def __init__(self, responses: Dict[str, Response], missing: Sequence[str] = ()) -> None:
        self.responses = responses
        self.calls = []  # type: List[List[str]]
        self.missing = set(missing)

    @staticmethod
    def key(argv: Sequence[str]) -> str:
        words = [arg for arg in argv if not arg.startswith("-")]
        if argv[0] == "scontrol":
            return "scontrol " + " ".join(words[1:3])
        if argv[0] == "sacctmgr":
            return "sacctmgr " + words[2] if len(words) > 2 else "sacctmgr"
        if argv[0] == "squeue":
            return "squeue log" if "-O" in argv else "squeue"
        if argv[0] == "sacct":
            return "sacct script" if "--batch-script" in argv else "sacct submit"
        return argv[0]

    def has(self, program: str) -> bool:
        return program not in self.missing

    def run(self, argv: Sequence[str], timeout: Optional[float] = None,
            max_bytes: Optional[int] = None) -> Result:
        check(argv)
        argv = list(argv)
        self.calls.append(argv)
        if argv[0] in self.missing:
            return Result(argv, Status.UNAVAILABLE, reason=argv[0] + " is not installed")
        response = self.responses.get(self.key(argv))
        if callable(response):
            response = response(argv)
        if response is None:
            return Result(argv, Status.ERROR, returncode=1, reason="no fixture")
        if isinstance(response, Result):
            response.argv = argv
            return response
        return Result(argv, Status.OK if response else Status.EMPTY, response, "", 0)

    def batch_script(self, job_id: str) -> Result:
        return self.run(["sacct", "-j", str(job_id), "--batch-script"])


def denied(reason: str = "Permission denied") -> Result:
    return Result([], Status.DENIED, "", reason, 1, reason)


def unavailable(reason: str = "Slurm accounting storage is disabled") -> Result:
    return Result([], Status.UNAVAILABLE, "", reason, 1, reason)
