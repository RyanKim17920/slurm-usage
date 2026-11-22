"""An invented cluster for UI tests and screenshots: a fake snapshot, a fake
collector and a fake History that answers with real Report/Stats/JobRow objects.
Every name here is made up.
"""

import bisect
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from slurm_usage.history import (AttemptDetail, Filter, HistoryError, HistoryInfo, JobDetail,
                                 JobPage, JobRow, Lane, LogPath, OUTCOMES, Report, Stats, Text,
                                 Window)
from slurm_usage.history.timeline import hour_edges
from slurm_usage.live import Fetched, LogPaths, LogTail, StepReport
from slurm_usage.live import summary as live_summary
from slurm_usage.model import (Allocation, Availability, Capability, ClusterInfo, Invoker, Job,
                               Limit, Node, Partition, Qos, Role, Snapshot, Status, Step,
                               Warning_)

CLUSTER = "orion"
USERS = ("alice", "bob", "carol", "dmitri", "erin")
ACCOUNTS = {"alice": "physics", "bob": "physics", "carol": "biolab", "dmitri": "biolab",
            "erin": "visiting"}
HOSTILE_NAME = "pwn\x1b]0;owned\x07\x1b[31mred\x9b2J"
REAL_IDENTIFIERS = (CLUSTER, "alice", "bob", "carol", "dmitri", "erin", "physics", "biolab",
                    "visiting", "gpu-a01", "gpu-a02", "gpu-b01", "cpu-c01", "sweep_alice")
UID = {name: 5000 + i for i, name in enumerate(USERS)}


def make_nodes() -> List[Node]:
    nodes = []
    for name, state, gpus, gpus_alloc, cpus_alloc, flags, avail, reason in (
            ("gpu-a01", "MIXED", 8, 6, 48, [], Availability.SCHEDULABLE, None),
            ("gpu-a02", "ALLOCATED", 8, 8, 96, [], Availability.SCHEDULABLE, None),
            ("gpu-b01", "IDLE", 4, 0, 0, ["DRAIN"], Availability.UNAVAILABLE, "bad gpu"),
            ("cpu-c01", "MIXED", 0, 0, 40, [], Availability.SCHEDULABLE, None)):
        nodes.append(Node(name=name, state=state, state_flags=flags, availability=avail,
                          partitions=["gpu"] if gpus else ["cpu"], cpus_total=96,
                          cpus_effective=96, cpus_alloc=cpus_alloc, threads_per_core=2,
                          cpu_load=cpus_alloc * 0.8, mem_total=1024 << 30,
                          mem_alloc=(cpus_alloc * 8) << 30, gpus_total=gpus,
                          gpus_alloc=gpus_alloc, gpu_types={"h100": gpus} if gpus else {},
                          gpus_shared=0, reason=reason))
    return nodes


def make_jobs(now: float, hostile: bool = True) -> List[Job]:
    jobs = []
    specs = [
        (101, "alice", "train_lr3_0042", "RUNNING", 4, 32, "gpu-a01", 3600 * 5),
        (102, "alice", "train_lr3_0043", "RUNNING", 2, 16, "gpu-a01", 1800),
        (103, "bob", "eval_ckpt", "RUNNING", 8, 96, "gpu-a02", 7200),
        (104, "carol", "fold_7", "RUNNING", 0, 40, "cpu-c01", 600),
        (105, "dmitri", HOSTILE_NAME if hostile else "plain", "PENDING", 8, 64, None, None),
        (106, "erin", "sweep_alice", "PENDING", 1, 8, None, None),
    ]
    for job_id, user, name, state, gpus, cpus, node, elapsed in specs:
        running = state == "RUNNING"
        job = Job(job_id=str(job_id), display_id=str(job_id), name=name, user=user, uid=UID[user],
                  account=ACCOUNTS[user], qos="normal", partition="gpu" if gpus else "cpu",
                  state=state, reason=None if running else "Resources", priority=1000 + job_id,
                  submit=int(now - (elapsed or 900) - 300), start=int(now - elapsed) if running else None,
                  elapsed=elapsed, time_limit=86400, nodes=[node] if node else [],
                  batch_host=node, cpus_alloc=cpus if running else None,
                  gpus_alloc=gpus if running else None, mem_alloc=(cpus * 8) << 30 if running else None,
                  cpus_req=cpus, gpus_req=gpus, mem_req=(cpus * 8) << 30,
                  workdir="/home/%s/project" % user, raw_workdir="/home/%s/project" % user,
                  detail_status="ok")
        if running:
            job.allocations = [Allocation(node, cpus, None, job.mem_alloc, gpus)]
        jobs.append(job)
    array = Job(job_id="107", display_id="107_[0-9]", array_job_id="107", task_count=10,
                name="prep_shard", user="carol", uid=UID["carol"], account="biolab", qos="low",
                partition="cpu", state="PENDING", reason="JobArrayTaskLimit",
                submit=int(now - 4000), cpus_req=4, gpus_req=0, mem_req=8 << 30)
    jobs.append(array)
    return jobs


def make_snapshot(now: Optional[float] = None, hostile: bool = True) -> Snapshot:
    now = time.time() if now is None else now
    nodes = make_nodes()
    jobs = make_jobs(now, hostile)
    snap = Snapshot(taken_at=now - 3, duration=0.4)
    snap.cluster = ClusterInfo(name=CLUSTER, version="24.05.1", has_accounting=True,
                               gpu_accounted=True, mem_tracked=True)
    snap.invoker = Invoker(user="alice", uid=UID["alice"], euid=UID["alice"], admin_level="None",
                           default_account="physics")
    snap.nodes, snap.jobs = nodes, jobs
    snap.partitions = [Partition("gpu", "UP", nodes=["gpu-a01", "gpu-a02", "gpu-b01"]),
                       Partition("cpu", "UP", nodes=["cpu-c01"])]
    roles = []
    for user in USERS:
        roles.append(Role(account=ACCOUNTS[user], user=user, is_default=True,
                          lineage=["root", ACCOUNTS[user]]))
    for account in sorted(set(ACCOUNTS.values())):
        role = Role(account=account, lineage=["root", account])
        if account == "physics":
            role.limits = [Limit("assoc", "GrpTRES", "gres/gpu", 16, 12, account="physics",
                                 source="assoc_mgr")]
        roles.append(role)
    snap.roles = roles
    snap.qos = [Qos("normal", 10), Qos("low", 0)]
    snap.limits = [limit for role in roles for limit in role.limits]
    for name in ("nodes", "jobs", "job_detail", "assoc_mgr", "partitions"):
        snap.capabilities[name] = Capability(name, Status.OK)
    live_summary.mark_completing(nodes, jobs)
    snap.totals = live_summary.totals(nodes)
    snap.health = live_summary.health(nodes, jobs)
    snap.warnings = [Warning_("capacity_mismatch", "gres/gpu capacity differs: nodes have 20, "
                              "the accounting database records 24; live totals use the nodes")]
    return snap


class FakeCollector:
    def __init__(self, snap: Optional[Snapshot] = None) -> None:
        self.snap = snap
        self.calls = []  # type: List[str]

    def snapshot(self) -> Snapshot:
        self.calls.append("snapshot")
        return self.snap if self.snap is not None else make_snapshot()

    def refresh(self) -> None:
        self.calls.append("refresh")

    def steps(self, job: Job) -> StepReport:
        self.calls.append("steps")
        return StepReport(Status.OK, [Step(job.job_id, "0", name="python", state="RUNNING",
                                           elapsed=1200, ntasks=4, total_cpu=3000.0,
                                           max_rss=12 << 30, gpu_util_max=88.0,
                                           tres_alloc={"cpu": 8})])

    def log_paths(self, job: Job) -> LogPaths:
        self.calls.append("log_paths")
        return LogPaths(Status.OK, "/home/x/out-%s.log" % job.job_id,
                        raw_stdout="/home/x/out-%s.log" % job.job_id)

    def tail_log(self, job: Job, raw: Optional[str], max_bytes: int = 0) -> LogTail:
        self.calls.append("tail_log")
        return LogTail(Status.OK, ["step 1 loss 2.31", "step 2 loss \x1b[2J2.10"])

    def batch_script(self, job: Job) -> Fetched:
        self.calls.append("batch_script")
        return Fetched(Status.DENIED, detail="Slurm refused: not your job")

    def submit_line(self, job: Job) -> Fetched:
        self.calls.append("submit_line")
        return Fetched(Status.OK, "sbatch --gres=gpu:4 run.sh")


# -- history -----------------------------------------------------------------

def make_attempts(now: float, days: int = 400, seed: int = 7) -> List[JobRow]:
    rng = random.Random(seed)
    rows = []
    job_id = 1000
    families = {"alice": ["train_lr{}_{:04d}", "eval_{}"], "bob": ["bench_v{}", "eval_{}"],
                "carol": ["fold_{}", "prep_{}"], "dmitri": ["md_run{}"], "erin": ["sweep_alice"]}
    start = now - days * 86400
    t = start
    while t < now - 600:
        t += rng.expovariate(1 / 14400.0)
        user = rng.choice(USERS)
        pattern = rng.choice(families[user])
        name = pattern.format(rng.randint(1, 9), rng.randint(1, 9999))
        gpus = rng.choice([0, 1, 1, 2, 4, 8, 8, 16, 32])
        nodes = "gpu-a01" if gpus <= 8 else "gpu-a[01-02]" if gpus <= 16 else "gpu-a[01-02],gpu-b01"
        if gpus == 0:
            nodes = "cpu-c01"
        job_id += 1
        attempts = 1 + (rng.random() < 0.1) * rng.randint(1, 3)
        begin = int(t)
        for number in range(attempts):
            elapsed = int(rng.expovariate(1 / 10800.0)) + 60
            end = begin + elapsed
            outcome = rng.choices(["completed", "failed", "timeout", "cancelled", "node_fail"],
                                  [70, 12, 8, 8, 2])[0]
            if number < attempts - 1:
                outcome = "preempted"
            state = {"completed": "COMPLETED", "failed": "FAILED", "timeout": "TIMEOUT",
                     "cancelled": "CANCELLED", "node_fail": "NODE_FAIL",
                     "preempted": "PREEMPTED"}[outcome]
            running = end > now
            if running:
                end, outcome, state = None, "running", "RUNNING"
            wait = int(rng.expovariate(1 / 600.0))
            rows.append(JobRow(
                key="%d.%d" % (job_id, number), job_id=str(job_id), display_id=str(job_id),
                name=name, user=user, uid=UID[user], account=ACCOUNTS[user],
                qos=rng.choice(["normal", "normal", "low"]), partition="gpu" if gpus else "cpu",
                state=state, outcome=outcome, nodes=nodes, submit=begin - wait - 5,
                eligible=begin - wait, start=begin, end=end,
                elapsed=(end - begin) if end else int(now - begin), cpus=max(4, gpus * 8),
                gpus=gpus, mem=(max(4, gpus * 8) * 8) << 30, time_limit=86400,
                time_unlimited=False, restarts=number, problem=None,
                cpu_efficiency=rng.random() if rng.random() < 0.7 else None,
                mem_efficiency=rng.random(),
                gpu_peak_utilization=rng.random() * 1.1 if gpus and rng.random() < 0.6 else None))
            if running:
                break
            begin = end + 120
    return rows


class FakeHistory:
    def __init__(self, now: Optional[float] = None, rows: Optional[List[JobRow]] = None) -> None:
        self.now = int(time.time() if now is None else now)
        self.rows = make_attempts(self.now) if rows is None else rows
        self.refreshed = []  # type: List[Any]

    def info(self) -> HistoryInfo:
        return HistoryInfo("ok", "all", CLUSTER, Invoker(user="alice", uid=UID["alice"]), True,
                           covered_from=0, as_of=self.now, cache_path="/tmp/cache")

    def refresh(self, window: Any = None, progress: Any = None, cancel: Any = None,
                update: bool = True) -> HistoryInfo:
        self.refreshed.append(window)
        return self.info()

    def _match(self, row: JobRow, where: Optional[Filter]) -> bool:
        if where is None:
            return True
        for role, values in where.items():
            if getattr(row, role, None) not in values:
                return False
        return True

    def _clip(self, row: JobRow, start: int, end: int) -> Optional[tuple]:
        stop = row.end if row.end is not None else self.now
        p, q = max(row.start, start), min(stop, end)
        if q <= p:
            return None
        return p, q

    def _span(self, window: Optional[Window]) -> tuple:
        window = window or Window()
        end = self.now if window.end is None else min(self.now, int(window.end))
        start = window.start if window.start is not None else min(r.start for r in self.rows)
        return int(start), int(end)

    def query(self, window: Any = None, where: Any = None, group_by: Any = (),
              series_by: Optional[str] = None, series_metric: str = "gpu",
              lanes_by: Optional[str] = None, lane_resolution: Optional[int] = None,
              efficiency: bool = False, progress: Any = None, cancel: Any = None,
              lanes_metric: Optional[str] = None) -> Report:
        start, end = self._span(window)
        specs = [("outcome",)] + [tuple((g,) if isinstance(g, str) else g) for g in group_by]
        groups = {spec: {} for spec in specs}  # type: Dict[tuple, Dict[tuple, Stats]]
        total = Stats(True)
        edges = hour_edges(start, end)
        values = {m: [0] * (len(edges) - 1) for m in ("gpu", "cpu", "mem")}
        keyed = {}  # type: Dict[str, List[int]]
        lanes = {}  # type: Dict[str, List[List[int]]]
        waits = []
        counts = {}  # type: Dict[str, int]
        rows = [r for r in self.rows if self._match(r, where)]
        for row in rows:
            counts[row.job_id] = counts.get(row.job_id, 0) + 1
        resolution = lane_resolution or max(1, (end - start) // 1000)
        for row in sorted(rows, key=lambda r: r.start):
            clipped = self._clip(row, start, end)
            if clipped is None:
                continue
            p, q = clipped
            targets = [total]
            for spec in specs:
                key = tuple(getattr(row, role) for role in spec)
                targets.append(groups[spec].setdefault(key, Stats(True)))
            for stats in targets:
                stats.add(p, q, q - p, row.cpus, row.gpus, row.mem)
                if counts[row.job_id] > 1:
                    stats.note_repeat((int(row.job_id), row.uid))
                if row.end is None:
                    stats.running += 1
                if row.start >= start and row.wait is not None:
                    stats.wait_count += 1
                    stats.wait_total += row.wait
                    stats.wait_max = max(stats.wait_max or 0, row.wait)
                if row.outcome in ("completed", "timeout") and row.end and row.end <= end:
                    stats.limit_count += 1
                    stats.limit_used += row.elapsed
                    stats.limit_requested += row.time_limit
                if efficiency and row.end is not None:
                    if row.cpu_efficiency is not None:
                        stats.cpu_covered += row.cpus * (q - p)
                        stats.cpu_used += row.cpu_efficiency * row.cpus * (q - p)
                    if row.gpu_peak_utilization is not None and row.gpus:
                        stats.gpu_covered += row.gpus * (q - p)
                        stats.gpu_used += min(1.0, row.gpu_peak_utilization) * row.gpus * (q - p)
            if row.start >= start and row.wait is not None:
                waits.append(row.wait)
            first = max(0, bisect.bisect_right(edges, p) - 1)
            last = min(len(edges) - 2, bisect.bisect_left(edges, q))
            for i in range(first, last + 1):
                a, b = max(p, edges[i]), min(q, edges[i + 1])
                if b > a:
                    for metric, rate in (("gpu", row.gpus), ("cpu", row.cpus), ("mem", row.mem)):
                        values[metric][i] += (rate or 0) * (b - a)
                    if series_by:
                        series = keyed.setdefault(getattr(row, series_by), [0] * (len(edges) - 1))
                        series[i] += (getattr(row, series_metric + "s") or 0) * (b - a)
            if lanes_by and (lanes_metric is None or getattr(row, lanes_metric + "s")):
                lane = lanes.setdefault(getattr(row, lanes_by), [])
                if lane and p <= lane[-1][1] + resolution:
                    lane[-1][1] = max(lane[-1][1], q)
                else:
                    lane.append([p, q])
        total.finish()
        for members in groups.values():
            for stats in members.values():
                stats.finish()
        outcome_groups = groups[("outcome",)]
        return Report(window=window or Window(), start=start, end=end, as_of=self.now,
                      where=where or Filter(), total=total, groups=groups,
                      outcomes={o: outcome_groups.get((o,)) or Stats(True) for o in OUTCOMES},
                      edges=edges, values=values, series_by=series_by, series_metric=series_metric,
                      keyed_values=keyed,
                      lanes=[Lane(k, [(a, b) for a, b in v]) for k, v in sorted(lanes.items())],
                      waits=sorted(waits), complete=True,
                      efficiency_unchecked=0 if efficiency else None)

    def jobs(self, window: Any = None, where: Any = None, by: str = "job", sort: str = "gpu",
             descending: bool = True, limit: int = 200, offset: int = 0,
             cancel: Any = None, match: Optional[str] = None) -> JobPage:
        import dataclasses
        start, end = self._span(window)
        found = {}  # type: Dict[str, JobRow]
        needle = (match or "").lower()
        for row in self.rows:
            if not self._match(row, where):
                continue
            if needle and not (needle in row.job_id or needle in row.name.lower()
                               or needle in row.user.lower()):
                continue
            clipped = self._clip(row, start, end)
            if clipped is None:
                continue
            seconds = clipped[1] - clipped[0]
            copy = dataclasses.replace(row, gpu_seconds=(row.gpus or 0) * seconds,
                                       cpu_seconds=(row.cpus or 0) * seconds)
            key = row.job_id if by == "job" else row.key
            old = found.get(key)
            if old is not None:
                copy.gpu_seconds += old.gpu_seconds
                copy.cpu_seconds += old.cpu_seconds
                copy.attempts = old.attempts + 1
            found[key] = copy
        rows = list(found.values())

        def order(r: JobRow) -> Any:
            v = {"gpu": r.gpu_seconds, "cpu": r.cpu_seconds, "start": r.start,
                 "elapsed": r.elapsed, "wait": r.wait, "attempts": r.attempts}.get(sort)
            return (v is not None, v or 0)
        rows.sort(key=order, reverse=descending)
        return JobPage(rows[offset:offset + limit], len(rows), True)

    def attempts(self, job_id: Any) -> List[JobRow]:
        return [r for r in self.rows if r.job_id == str(job_id)]

    def detail(self, job_id: Any, cancel: Any = None) -> JobDetail:
        rows = self.attempts(job_id)
        if not rows:
            return JobDetail(str(job_id), Status.EMPTY, "no such job in accounting")
        out = JobDetail(str(job_id), Status.OK)
        for row in rows:
            out.attempts.append(AttemptDetail(
                row, exit_code=1 if row.outcome == "failed" else 0,
                workdir="/home/%s/project" % row.user,
                stdout=LogPath("/home/%s/logs/%s.out" % (row.user, row.job_id),
                               "/home/%s/logs/%s.out" % (row.user, row.job_id)),
                steps=[Step(row.job_id, "batch", name="batch", state=row.state, elapsed=row.elapsed,
                            ntasks=1, total_cpu=float(row.elapsed or 0), max_rss=2 << 30,
                            tres_alloc={"cpu": row.cpus or 1})]))
        return out

    def submit_line(self, job_id: Any, key: Any = None) -> Text:
        return Text(Status.OK, "sbatch --gres=gpu:8 /home/alice/run.sh")

    def batch_script(self, job_id: Any) -> Text:
        return Text(Status.EMPTY, message="Slurm returned no script")

    def load_efficiency(self, window: Any = None, where: Any = None, progress: Any = None,
                        cancel: Any = None) -> int:
        return 0


class FailingHistory(FakeHistory):
    def query(self, *args: Any, **kwargs: Any) -> Report:
        raise HistoryError(Status.DENIED, "accounting refused")

    jobs = attempts = query


# -- the real History over a fake sacct ------------------------------------------

def real_history(days: int = 45, seed: int = 7) -> Tuple[Any, Any]:
    """A real history.History reading invented sacct output (tests/test_history_support)."""
    from tests import test_history_support as hs
    now = int(time.time()) // 60 * 60
    rows = make_attempts(now, days=days, seed=seed)
    states = {"completed": "COMPLETED", "failed": "FAILED", "timeout": "TIMEOUT",
              "cancelled": "CANCELLED", "node_fail": "NODE_FAIL", "preempted": "PREEMPTED",
              "running": "RUNNING"}
    jobs, live = [], []
    for index, row in enumerate(rows):
        if row.outcome == "running":
            live.append(int(row.job_id))
        jobs.append(hs.job(int(row.job_id), start=row.start, end=row.end, state=states[row.outcome],
                           user=row.user, uid=row.uid, account=row.account, qos=row.qos,
                           partition=row.partition, gpus=row.gpus, cpus=row.cpus,
                           mem="%dG" % (row.mem >> 30), nodes=row.nodes, submit=row.submit,
                           eligible=row.eligible, limit=row.time_limit // 60, name=row.name,
                           index=index + 1, restarts=row.restarts))
    fake = hs.FakeSlurm(jobs, now=now, live=live, cluster_tres="cpu=384,gres/gpu=24")
    return hs.make_history(fake), fake
