"""A deterministic, invented Slurm cluster ("orion") for demos and screenshots.

Every name, number and path here is made up. The executables in ``bin/``
(squeue, scontrol, sacct, sacctmgr, sstat, sinfo, sprio, sshare, sreport) all
call ``main()``, which answers in the formats real Slurm prints with
``SLURM_TIME_FORMAT=%s`` (what slurm-usage sets for every command).

The world is a small scheduler simulation over the last ``DAYS`` days, run
with a fixed seed against today's local midnight, so it always looks current
and two runs on the same day agree to the second. Whatever is past ``now`` is
not visible yet: those jobs are still pending, running, or not submitted. The
simulation is cached in ``$SLURM_DEMO_STATE`` (default: a private directory
under the system temp dir) because every command reads it.

Highlights built in, for the tour: daily and weekly cycles, growth over the
last weeks, a 32-GPU run that preempted the low-priority queue, a requeued
multi-node run (node failure, then running again now), a failure burst on one
afternoon, an account that appears three weeks ago, a drained node with a
reason, and a queue with varied pending reasons.
"""

import hashlib
import math
import os
import pickle
import random
import re
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

SEED = 20260117
DAYS = 120
CLUSTER = "orion"
MIN, HOUR, DAY = 60, 3600, 86400
TICK = 5 * MIN
FIRST_JOB = 418000
FIRST_INDEX = 7300000
GPU_TYPE = "h100"

# name, cpus, mem (MB), gpus, gpu type, partitions
NODES = (
    [("gpu-a%02d" % i, 112, 1031000, 8, "h100", "gpu") for i in range(1, 7)]
    + [("gpu-b%02d" % i, 64, 515000, 4, "l40s", "gpu") for i in range(1, 3)]
    + [("cpu-c%02d" % i, 192, 773000, 0, None, "cpu") for i in range(1, 3)]
)
DRAINED = "gpu-a06"
DRAIN_REASON = "NVLink errors on 2 GPUs, RMA pending"

ACCOUNTS = ("physics", "biolab", "astro", "visiting", "vision")
ACCOUNT_LIMITS = {"physics": 32, "visiting": 8, "vision": 24}  # GrpTRES gres/gpu
QOS = {  # name: (priority, max wall, MaxTRESPU gres/gpu)
    "high": (1000, "3-00:00:00", None),
    "normal": (100, "2-00:00:00", 24),
    "low": (0, "1-00:00:00", None),
    "debug": (500, "01:00:00", 2),
}
USER_GPU_CAP = 24  # normal QOS, per user

# name, uid (None = whoever runs the demo), account, efficiency (cpu, gpu)
USERS = (
    ("alice", None, "physics", 0.71, 0.86),
    ("bob", 20402, "physics", 0.55, 0.62),
    ("carol", 20403, "biolab", 0.83, 0.74),
    ("dmitri", 20404, "biolab", 0.48, 0.41),
    ("erin", 20405, "visiting", 0.38, 0.27),
    ("farid", 20406, "astro", 0.91, 0.58),
    ("grace", 20407, "astro", 0.66, 0.69),
    ("hiro", 20408, "vision", 0.62, 0.81),
    ("ines", 20409, "vision", 0.57, 0.52),
    ("jonas", 20410, "physics", 0.30, 0.22),
)
NEW_ACCOUNT_DAY = -21  # vision starts here

# per user: (name, weight, gpus, nodes, cpus, mem GB, median hours, spread, limit h, qos,
#            partition, array size, failure chance)
WORK = {
    "alice": [("train_gnn", 3, 8, 1, 64, 480, 9, 0.6, 24, "normal", "gpu", 0, 0.04),
              ("eval_ckpt", 5, 1, 1, 8, 64, 0.7, 0.5, 4, "normal", "gpu", 0, 0.03),
              ("sweep_lr", 1, 1, 1, 8, 48, 1.6, 0.4, 6, "low", "gpu", 8, 0.05),
              ("jupyter", 1, 1, 1, 8, 64, 3, 0.4, 8, "normal", "gpu", 0, 0.0)],
    "bob": [("md_sim", 4, 0, 1, 48, 96, 6, 0.6, 24, "normal", "cpu", 0, 0.05),
            ("train_vit", 2, 4, 1, 32, 240, 8, 0.6, 24, "normal", "gpu", 0, 0.06),
            ("eval_vit", 2, 1, 1, 8, 32, 0.5, 0.5, 2, "normal", "gpu", 0, 0.04)],
    "carol": [("fold_protein", 6, 2, 1, 16, 120, 2.5, 0.5, 12, "normal", "gpu", 0, 0.03),
              ("align_reads", 3, 0, 1, 64, 256, 1.8, 0.5, 8, "normal", "cpu", 0, 0.02)],
    "dmitri": [("seg_cells", 4, 1, 1, 12, 64, 1.4, 0.6, 6, "normal", "gpu", 0, 0.08),
               ("dock_screen", 2, 0, 1, 16, 32, 0.8, 0.5, 4, "normal", "cpu", 10, 0.04)],
    "erin": [("finetune_llm", 4, 4, 1, 32, 300, 10, 0.6, 24, "low", "gpu", 0, 0.05),
             ("eval_llm", 2, 2, 1, 16, 120, 1, 0.5, 4, "low", "gpu", 0, 0.05)],
    "farid": [("sim_galaxy", 2, 0, 1, 96, 384, 16, 0.5, 48, "normal", "cpu", 0, 0.03),
              ("denoise", 3, 2, 1, 16, 96, 3, 0.5, 12, "low", "gpu", 6, 0.04)],
    "grace": [("render_sky", 3, 0, 1, 32, 64, 2, 0.5, 8, "normal", "cpu", 0, 0.02),
              ("train_unet", 3, 2, 1, 16, 120, 5, 0.6, 24, "normal", "gpu", 0, 0.05)],
    "hiro": [("train_detr", 1.2, 8, 1, 64, 600, 11, 0.5, 36, "normal", "gpu", 0, 0.04),
             ("prep_coco", 2, 0, 1, 32, 128, 1.2, 0.4, 4, "normal", "cpu", 0, 0.02)],
    "ines": [("eval_detr", 4, 1, 1, 8, 64, 0.8, 0.5, 4, "normal", "gpu", 6, 0.03),
             ("train_vlm", 2, 4, 1, 32, 300, 7, 0.6, 24, "normal", "gpu", 0, 0.05)],
    "jonas": [("debug", 6, 1, 1, 4, 16, 0.15, 0.6, 1, "debug", "gpu", 0, 0.25),
              ("test_mpi", 2, 0, 1, 24, 32, 0.3, 0.5, 1, "debug", "cpu", 0, 0.15)],
}
RATE = {"alice": 5, "bob": 3, "carol": 6, "dmitri": 4, "erin": 3.2, "farid": 2.2,
        "grace": 3, "hiro": 3, "ines": 4, "jonas": 4}
HOURS = [1, 1, 1, 1, 1, 1, 2, 3, 6, 9, 10, 10, 8, 9, 10, 10, 9, 7, 5, 4, 3, 3, 2, 1]


# --------------------------------------------------------------------------- time
def now() -> int:
    return int(time.time())


def anchor(when: int) -> int:
    """Local midnight of ``when``."""
    day = time.localtime(when)
    return int(time.mktime((day.tm_year, day.tm_mon, day.tm_mday, 0, 0, 0, 0, 0, -1)))


def duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    days, rest = divmod(seconds, DAY)
    text = "%02d:%02d:%02d" % (rest // HOUR, rest % HOUR // MIN, rest % MIN)
    return "%d-%s" % (days, text) if days else text


def hostlist(names: Sequence[str]) -> str:
    groups = {}  # type: Dict[str, List[str]]
    order = []
    for name in names:
        match = re.match(r"(.*?)(\d+)$", name)
        prefix, number = (match.group(1), match.group(2)) if match else (name, "")
        if prefix not in groups:
            order.append(prefix)
        groups.setdefault(prefix, []).append(number)
    parts = []
    for prefix in order:
        numbers = groups[prefix]
        if len(numbers) == 1:
            parts.append(prefix + numbers[0])
            continue
        width = len(numbers[0])
        values = sorted(int(n) for n in numbers)
        runs, first, last = [], values[0], values[0]
        for value in values[1:] + [None]:
            if value is not None and value == last + 1:
                last = value
                continue
            runs.append(("%0*d" % (width, first)) if first == last
                        else "%0*d-%0*d" % (width, first, width, last))
            if value is not None:
                first = last = value
        parts.append("%s[%s]" % (prefix, ",".join(runs)))
    return ",".join(parts)


def ranges(values: Sequence[int]) -> str:
    values = sorted(values)
    out, start = [], None
    for i, value in enumerate(values):
        if start is None:
            start = value
        if i + 1 == len(values) or values[i + 1] != value + 1:
            out.append(str(start) if start == value else "%d-%d" % (start, value))
            start = None
    return ",".join(out)


# --------------------------------------------------------------------------- world
class Attempt:
    """One allocation attempt (what ``sacct -D -X`` prints one row for)."""

    __slots__ = ("job", "label", "array", "name", "user", "uid", "account", "qos", "partition",
                 "gpus", "nnodes", "cpus", "mem", "limit", "submit", "eligible", "start", "end",
                 "state", "exit", "nodes", "gpu_idx", "cpu_idx", "restarts", "index", "work",
                 "fail", "kind", "pinned", "reason")

    def __init__(self, **values: object) -> None:
        for name in self.__slots__:
            setattr(self, name, values.get(name))


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def _submissions(rng: random.Random, base: int, me: int) -> List[Attempt]:
    jobs = []  # type: List[Attempt]
    uids = {name: (me if uid is None else uid) for name, uid, *_ in USERS}
    accounts = {name: account for name, _, account, *_ in USERS}
    total_weight = float(sum(HOURS))
    for day in range(-DAYS, 1):
        day_start = base + day * DAY
        weekday = time.localtime(day_start + 12 * HOUR).tm_wday
        weekend = 0.35 if weekday >= 5 else 1.0
        growth = 0.75 + 0.4 * (day + DAYS) / DAYS  # busier lately
        for user, _uid, account, *_rest in USERS:
            if accounts[user] == "vision" and day < NEW_ACCOUNT_DAY:
                continue
            rate = 1.9 * RATE[user] * weekend * growth
            if user in ("hiro", "ines") and day < NEW_ACCOUNT_DAY + 4:
                rate *= 0.5  # getting started
            if user == "erin" and -60 < day < -45:
                rate = 0  # away
            for _ in range(_poisson(rng, rate)):
                pick = rng.random() * total_weight
                hour = 0
                while pick > HOURS[hour]:
                    pick -= HOURS[hour]
                    hour += 1
                submit = day_start + hour * HOUR + rng.randrange(HOUR)
                kinds = WORK[user]
                choice = rng.random() * sum(k[1] for k in kinds)
                for kind in kinds:
                    choice -= kind[1]
                    if choice <= 0:
                        break
                (name, _w, gpus, nnodes, cpus, mem, median, spread, limit, qos, partition,
                 array, fail) = kind
                tasks = array or 1
                for task in range(tasks):
                    hours = median * math.exp(rng.gauss(0, spread))
                    jobs.append(Attempt(
                        name=name, user=user, uid=uids[user], account=account, qos=qos,
                        partition=partition, gpus=gpus, nnodes=nnodes, cpus=cpus,
                        mem=mem * 1024, limit=limit * 60, submit=submit + task,
                        work=max(90, int(hours * HOUR)), fail=fail, kind=name,
                        array=(task if array else None), restarts=0))
    # the tour's set pieces -------------------------------------------------------
    def special(name, user, gpus, nnodes, submit, hours, limit_h, qos="high", **extra):
        values = dict(name=name, user=user, uid=uids[user], account=accounts[user], qos=qos,
                      partition="gpu", gpus=gpus, nnodes=nnodes, cpus=96, mem=900 * 1024,
                      limit=limit_h * 60, submit=submit, work=int(hours * HOUR), fail=0.0,
                      kind=name, array=None, restarts=0)
        values.update(extra)
        jobs.append(Attempt(**values))

    # low-priority waves that the high-priority runs below then preempt
    for day, user, name, count, gpus in ((-45, "farid", "denoise_full", 14, 4),
                                         (-8, "farid", "denoise", 18, 2),
                                         (-26, "grace", "train_unet", 12, 4)):
        for i in range(count):
            special(name, user, gpus, 1, base + day * DAY + 7 * HOUR + 97 * i,
                    rng.uniform(14, 22), 24, qos="low", cpus=8 * gpus, mem=60 * gpus * 1024)
    special("pretrain_moe", "alice", 8, 4, base - 45 * DAY + 9 * HOUR + 1234, 58, 72)
    special("pretrain_moe_v2", "alice", 8, 2, base - 3 * DAY + 10 * HOUR + 321, 150, 168,
            fail="node_after@%d" % (7 * HOUR))
    for i, day in enumerate((-9, -9, -8, -8, -8, -7, -7, -6)):
        special("train_detr_xl" if i == 2 else "train_detr", "hiro", 8, 2 if i == 2 else 1,
                base + day * DAY + (9 + 2 * i % 7) * HOUR + 77 * i, 40 if i == 2 else 7,
                48 if i == 2 else 12)
    jobs.sort(key=lambda a: (a.submit, a.array or 0))
    return jobs


class Sim:
    def __init__(self, base: int, me: int) -> None:
        self.base, self.me = base, me
        self.rng = random.Random(SEED)
        self.free_gpu = {n[0]: set(range(n[3])) for n in NODES}
        self.free_cpu = {n[0]: set(range(n[1])) for n in NODES}
        self.free_mem = {n[0]: n[2] for n in NODES}
        self.part = {n[0]: n[5] for n in NODES}
        self.down = {}  # type: Dict[str, int]  # node -> back up at
        self.running = []  # type: List[Attempt]
        self.done = []  # type: List[Attempt]
        self.pending = []  # type: List[Attempt]
        self.user_gpus = {}  # type: Dict[str, int]
        self.acct_gpus = {}  # type: Dict[str, int]
        self.array_running = {}  # type: Dict[int, int]
        self.index = FIRST_INDEX
        self.drain_at = base - 30 * HOUR
        self.failures = []  # type: List[Tuple[int, int]]  # (when, job): its first node fails

    def _fits(self, node: str, job: Attempt, t: int) -> bool:
        if self.part[node] != job.partition or self.down.get(node, 0) > t:
            return False
        if node == DRAINED and t >= self.drain_at:
            return False
        return (len(self.free_gpu[node]) >= job.gpus and len(self.free_cpu[node]) >= job.cpus
                and self.free_mem[node] >= job.mem)

    def _limits_ok(self, job: Attempt) -> bool:
        need = job.gpus * job.nnodes
        if job.qos == "normal" and self.user_gpus.get(job.user, 0) + need > USER_GPU_CAP:
            return False
        if job.qos == "debug" and self.user_gpus.get(job.user, 0) + need > 2:
            return False
        cap = ACCOUNT_LIMITS.get(job.account)
        if cap is not None and job.qos != "high" and self.acct_gpus.get(job.account, 0) + need > cap:
            return False
        if job.array is not None and self.array_running.get(job.array, 0) >= 4:
            return False
        return True

    def _place(self, job: Attempt, t: int) -> Optional[List[str]]:
        fit = [n[0] for n in NODES if self._fits(n[0], job, t)]
        if len(fit) < job.nnodes:
            return None
        fit.sort(key=lambda n: (len(self.free_gpu[n]) - job.gpus, len(self.free_cpu[n])))
        return sorted(fit[:job.nnodes])

    def _start(self, job: Attempt, nodes: List[str], t: int) -> None:
        job.start = t + self.rng.randrange(1, 50)
        job.nodes = nodes
        job.gpu_idx, job.cpu_idx = {}, {}
        for node in nodes:
            gpus = sorted(self.free_gpu[node])[:job.gpus]
            cpus = sorted(self.free_cpu[node])[:job.cpus]
            self.free_gpu[node].difference_update(gpus)
            self.free_cpu[node].difference_update(cpus)
            self.free_mem[node] -= job.mem
            job.gpu_idx[node], job.cpu_idx[node] = gpus, cpus
        need = job.gpus * job.nnodes
        self.user_gpus[job.user] = self.user_gpus.get(job.user, 0) + need
        self.acct_gpus[job.account] = self.acct_gpus.get(job.account, 0) + need
        if job.array is not None:
            self.array_running[job.array] = self.array_running.get(job.array, 0) + 1
        if isinstance(job.fail, str) and job.fail.startswith("node_after@"):
            self.failures.append((job.start + int(job.fail.partition("@")[2]), job.job))
        # how it ends, decided now so the simulation stays causal
        limit = job.limit * 60
        rng = self.rng
        if job.fail and isinstance(job.fail, float) and rng.random() < job.fail:
            roll = rng.random()
            if roll < 0.6:
                job.state, job.exit = "FAILED", "1:0"
                job.end = job.start + int(job.work * rng.uniform(0.02, 0.5))
            elif roll < 0.8:
                job.state, job.exit = "OUT_OF_MEMORY", "0:125"
                job.end = job.start + int(job.work * rng.uniform(0.1, 0.6))
            else:
                job.state, job.exit = "CANCELLED by %d" % job.uid, "0:15"
                job.end = job.start + int(job.work * rng.uniform(0.05, 0.9))
        elif job.work > limit:
            job.state, job.exit, job.end = "TIMEOUT", "0:0", job.start + limit + rng.randrange(30, 90)
        else:
            job.state, job.exit, job.end = "COMPLETED", "0:0", job.start + job.work
        if job.name.startswith("fold_protein") and self._burst(job.submit):
            job.state, job.exit = "FAILED", "127:0"  # a broken module on one afternoon
            job.end = job.start + rng.randrange(20, 140)
        job.end = max(job.end, job.start + 15)
        self.running.append(job)

    def _burst(self, when: int) -> bool:
        start = self.base - 17 * DAY + 13 * HOUR
        return start <= when < start + 5 * HOUR

    def _release(self, job: Attempt) -> None:
        for node in job.nodes:
            self.free_gpu[node].update(job.gpu_idx[node])
            self.free_cpu[node].update(job.cpu_idx[node])
            self.free_mem[node] += job.mem
        need = job.gpus * job.nnodes
        self.user_gpus[job.user] -= need
        self.acct_gpus[job.account] -= need
        if job.array is not None:
            self.array_running[job.array] -= 1
        self.running.remove(job)
        self.done.append(job)

    def _requeue(self, old: Attempt, t: int, remaining: int) -> None:
        values = {name: getattr(old, name) for name in Attempt.__slots__}
        values.update(start=None, end=None, state=None, exit=None, nodes=None, gpu_idx=None,
                      cpu_idx=None, submit=t, eligible=t + 120, restarts=(old.restarts or 0) + 1,
                      work=max(300, remaining), index=self._next_index(),
                      fail=None if isinstance(old.fail, str) else old.fail)
        self.pending.append(Attempt(**values))

    def _next_index(self) -> int:
        self.index += 1
        return self.index

    def _stop(self, job: Attempt, t: int, state: str, exit_code: str) -> None:
        planned_end = job.end
        job.end, job.state, job.exit = t, state, exit_code
        self._release(job)
        self._requeue(job, t, planned_end - t)

    def _preempt_for(self, job: Attempt, t: int) -> bool:
        """Free enough nodes for a high-priority job by preempting low ones."""
        candidates = []
        for name, _c, _m, _g, _t, part in NODES:
            if part != job.partition or self.down.get(name, 0) > t:
                continue
            if name == DRAINED and t >= self.drain_at:
                continue
            victims = [r for r in self.running if r.qos == "low" and name in r.nodes]
            gpus = len(self.free_gpu[name]) + sum(v.gpus for v in victims)
            cpus = len(self.free_cpu[name]) + sum(v.cpus for v in victims)
            if gpus >= job.gpus and cpus >= job.cpus:
                candidates.append((len(victims), name, victims))
        if len(candidates) < job.nnodes:
            return False
        candidates.sort(key=lambda c: (c[0], c[1]))
        for _count, _name, victims in candidates[:job.nnodes]:
            for victim in victims:
                if victim in self.running:
                    self._stop(victim, t, "PREEMPTED", "0:15")
        return True

    def run(self, jobs: List[Attempt]) -> List[Attempt]:
        job_id = FIRST_JOB
        array_base = {}  # (submit second, user, name) -> base id
        for job in jobs:
            if job.array is not None:
                key = (job.submit - job.array, job.user, job.name)
                if key not in array_base:
                    array_base[key] = job_id
                job.job = job_id
                job.label = "%d_%d" % (array_base[key], job.array)
                job.array = array_base[key]
            else:
                job.job = job_id
                job.label = str(job_id)
            job_id += 1
            job.eligible = job.submit
            job.index = self._next_index()
        horizon = self.base + 2 * DAY
        t = self.base - DAYS * DAY
        incoming = list(jobs)
        incoming.reverse()
        while t < horizon:
            for job in [r for r in self.running if r.end <= t]:
                self._release(job)
            self.failures.sort()
            while self.failures and self.failures[0][0] <= t:
                when, which = self.failures.pop(0)
                for job in list(self.running):
                    if job.job == which:
                        bad = job.nodes[0]
                        self.down[bad] = when + 90 * MIN
                        for other in list(self.running):
                            if bad in other.nodes:
                                self._stop(other, when, "NODE_FAIL", "0:0")
            while incoming and incoming[-1].submit <= t:
                self.pending.append(incoming.pop())
            self.pending.sort(key=lambda j: (-QOS[j.qos][0], j.eligible, j.job))
            blocked_big = False
            for job in list(self.pending):
                if job.eligible > t:
                    continue
                if not self._limits_ok(job):
                    continue
                if blocked_big and job.nnodes > 1:
                    continue
                nodes = self._place(job, t)
                if nodes is None and job.qos == "high" and self._preempt_for(job, t):
                    nodes = self._place(job, t)
                if nodes is None:
                    if job.nnodes > 1:
                        blocked_big = True
                    continue
                self.pending.remove(job)
                self._start(job, nodes, t)
            t += TICK
        attempts = self.done + self.running + self.pending
        attempts.sort(key=lambda a: (a.job, a.index))
        return attempts


def _state_dir() -> str:
    path = os.environ.get("SLURM_DEMO_STATE") or os.path.join(
        tempfile.gettempdir(), "slurm-usage-demo-%d" % os.getuid())
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def world(at: Optional[int] = None) -> Tuple[int, List[Attempt]]:
    """(anchor, every attempt of the simulated cluster). Cached per day."""
    at = now() if at is None else at
    base = anchor(at)
    me = os.getuid()
    with open(os.path.abspath(__file__), "rb") as source:  # a changed world is a new cache
        version = hashlib.sha1(source.read()).hexdigest()[:12]
    path = os.path.join(_state_dir(), "world-%s-%d-%d.pickle" % (version, base, me))
    try:
        with open(path, "rb") as handle:
            return base, pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError):
        pass
    sim = Sim(base, me)
    attempts = sim.run(_submissions(random.Random(SEED + 1), base, me))
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "wb") as handle:
        pickle.dump(attempts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return base, attempts


# --------------------------------------------------------------------------- views at now
ALWAYS_PENDING = (
    # user, name, gpus, nodes, qos, reason, minutes ago, tasks
    ("erin", "sweep_lora", 1, 1, "low", "JobArrayTaskLimit", 35, 10),
    ("hiro", "train_detr", 8, 1, "normal", "ReqNodeNotAvail, UnavailableNodes:gpu-a06", 50, 1),
    ("carol", "fold_complex", 8, 2, "normal", "Resources", 22, 1),
    ("dmitri", "seg_cells", 1, 1, "normal", "Priority", 14, 3),
    ("bob", "eval_vit", 1, 1, "normal", "Dependency", 9, 1),
    ("alice", "eval_ckpt", 1, 1, "normal", "QOSMaxGRESPerUser", 6, 2),
)


class View:
    """The world as of ``t``: what squeue, scontrol and sacct would show."""

    def __init__(self, t: Optional[int] = None) -> None:
        self.t = now() if t is None else t
        self.base, attempts = world(self.t)
        self.attempts = [a for a in attempts if a.submit <= self.t]
        uids = {name: (os.getuid() if uid is None else uid) for name, uid, *_ in USERS}
        self.uids = uids
        accounts = {name: account for name, _, account, *_ in USERS}
        next_id = max(a.job for a in attempts) + 1
        next_index = max(a.index for a in attempts) + 1
        self.extra = []  # type: List[Attempt]
        for user, name, gpus, nnodes, qos, reason, ago, tasks in ALWAYS_PENDING:
            base_id = next_id
            for task in range(tasks):
                submit = self.t - ago * MIN + task
                self.extra.append(Attempt(
                    job=next_id, label=("%d_%d" % (base_id, task)) if tasks > 1 else str(next_id),
                    array=base_id if tasks > 1 else None, name=name, user=user,
                    uid=uids[user], account=accounts[user], qos=qos, partition="gpu",
                    gpus=gpus, nnodes=nnodes, cpus=8 * gpus, mem=64 * 1024 * gpus, limit=720,
                    submit=submit, eligible=submit, restarts=0, index=next_index,
                    reason=reason, kind=name))
                next_id += 1
                next_index += 1
        self.attempts += self.extra

    def status(self, a: Attempt) -> str:
        if a.start is None or a.start > self.t:
            return "PENDING"
        if a.end is None or a.end > self.t:
            return "RUNNING"
        return a.state

    def live(self) -> List[Attempt]:
        """Attempts squeue still shows: pending, running, and the last few minutes."""
        found = []
        latest = {}  # job -> its newest attempt
        for a in self.attempts:
            if a.job not in latest or a.index > latest[a.job].index:
                latest[a.job] = a
        for a in latest.values():
            state = self.status(a)
            if state in ("PENDING", "RUNNING") or (a.end and self.t - a.end < 300):
                found.append(a)
        return sorted(found, key=lambda a: a.job)

    def running(self) -> List[Attempt]:
        return [a for a in self.attempts if self.status(a) == "RUNNING"]

    def _usage(self) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, int]]:
        if not hasattr(self, "_used"):
            users, accounts, arrays = {}, {}, {}  # type: Dict[str, int], Dict[str, int], Dict[int, int]
            for a in self.running():
                need = a.gpus * a.nnodes
                if a.qos == "normal":
                    users[a.user] = users.get(a.user, 0) + need
                if a.qos != "high":
                    accounts[a.account] = accounts.get(a.account, 0) + need
                if a.array is not None:
                    arrays[a.array] = arrays.get(a.array, 0) + 1
            self._used = (users, accounts, arrays)
        return self._used

    def pending_reason(self, a: Attempt, queue_head: Optional[int]) -> str:
        if a.reason:
            return a.reason
        if a.eligible > self.t:
            return "BeginTime"
        users, accounts, arrays = self._usage()
        need = a.gpus * a.nnodes
        cap = ACCOUNT_LIMITS.get(a.account)
        if need and cap is not None and a.qos != "high" and accounts.get(a.account, 0) + need > cap:
            return "AssocGrpGRES"
        if need and a.qos == "normal" and users.get(a.user, 0) + need > USER_GPU_CAP:
            return "QOSMaxGRESPerUser"
        if a.array is not None and arrays.get(a.array, 0) >= 4:
            return "JobArrayTaskLimit"
        return "Resources" if a.job == queue_head else "Priority"


# --------------------------------------------------------------------------- formatting
def tres(a: Attempt, alloc: bool = True) -> str:
    cpus = a.cpus * a.nnodes if a.cpus else 2
    text = "billing=%d,cpu=%d" % (cpus, cpus)
    if a.gpus:
        text += ",gres/gpu=%d" % (a.gpus * a.nnodes)
        if alloc:
            text += ",gres/gpu:%s=%d" % (gpu_type(a), a.gpus * a.nnodes)
    return text + ",mem=%dM,node=%d" % (a.mem * a.nnodes, a.nnodes)


def gpu_type(a: Attempt) -> str:
    if a.nodes and a.nodes[0].startswith("gpu-b"):
        return "l40s"
    return GPU_TYPE


def _rand(a: Attempt, salt: str) -> random.Random:
    return random.Random("%d/%d/%s" % (a.job, a.index, salt))


def efficiency(a: Attempt) -> Tuple[float, float]:
    for name, _uid, _acct, cpu, gpu in USERS:
        if name == a.user:
            rng = _rand(a, "eff")
            return (max(0.05, min(0.98, cpu + rng.gauss(0, 0.08))),
                    max(0.03, min(0.99, gpu + rng.gauss(0, 0.08))))
    return 0.5, 0.5


def workdir(a: Attempt) -> str:
    """Where the job ran. run_demo.py points SLURM_DEMO_FILES at a scratch directory and
    writes the demo user's logs there, so the log keys have something to show."""
    root = os.environ.get("SLURM_DEMO_FILES") or "/scratch"
    return "%s/%s/%s" % (root, a.user, a.kind.split("_")[0])


def log_path(a: Attempt, stream: str = "out") -> str:
    return "%s/logs/%s-%d.%s" % (workdir(a), a.name, a.job, stream)


def log_text(view: "View", a: Attempt) -> str:
    """A plausible training log, up to now."""
    rng = _rand(a, "log")
    state = view.status(a)
    lines = ["[%s] %s on %s, %d GPU(s) per node, attempt %d" % (
        time.strftime("%H:%M:%S", time.localtime(a.start)), a.name, hostlist(a.nodes),
        a.gpus, (a.restarts or 0) + 1)]
    if a.restarts:
        lines.append("resuming from checkpoints/last.pt (requeued after a node failure)")
    stop = view.t if state == "RUNNING" else a.end
    steps = max(1, (stop - a.start) // 90)
    loss = rng.uniform(2.4, 3.2)
    first = max(0, steps - 16)
    for step in range(first, steps):
        loss *= rng.uniform(0.992, 1.0015)
        lines.append("step %6d | loss %.4f | lr %.2e | %5.0f tok/s/gpu | gpu mem %4.1fG" % (
            (step + 1) * 50, loss, 3e-4 * (0.5 + 0.5 * math.cos(step / 4000.0)),
            rng.uniform(2900, 3300), rng.uniform(61, 63)))
        if step % 12 == 11:
            lines.append("eval  %6d | val_loss %.4f | saved checkpoints/step_%d.pt" % (
                (step + 1) * 50, loss * 1.04, (step + 1) * 50))
    if state not in ("RUNNING", "PENDING"):
        lines.append("finished: %s" % state.split()[0].lower())
    return "\n".join(lines) + "\n"


SACCT_FIELDS = """Account AdminComment AllocCPUS AllocNodes AllocTRES AssocID AveCPU AveCPUFreq
AveDiskRead AveDiskWrite AvePages AveRSS AveVMSize BlockID Cluster Comment Constraints
ConsumedEnergy ConsumedEnergyRaw Container CPUTime CPUTimeRAW DBIndex DerivedExitCode Elapsed
ElapsedRaw Eligible End ExitCode Extra FailedNode Flags GID Group JobID JobIDRaw JobName Layout
MaxDiskRead MaxDiskReadNode MaxDiskReadTask MaxDiskWrite MaxDiskWriteNode MaxDiskWriteTask
MaxPages MaxPagesNode MaxPagesTask MaxRSS MaxRSSNode MaxRSSTask MaxVMSize MaxVMSizeNode
MaxVMSizeTask McsLabel MinCPU MinCPUNode MinCPUTask NCPUS NNodes NodeList NTasks Partition
Planned PlannedCPU PlannedCPURAW Priority QOS QOSRAW Reason ReqCPUFreq ReqCPUFreqGov
ReqCPUFreqMax ReqCPUFreqMin ReqCPUS ReqMem ReqNodes ReqTRES Reservation ReservationId
Restarts SLUID Start State StdErr StdIn StdOut Submit SubmitLine Suspended SystemComment
SystemCPU Timelimit TimelimitRaw TotalCPU TRESUsageInAve TRESUsageInMax TRESUsageInMaxNode
TRESUsageInMaxTask TRESUsageInMin TRESUsageInMinNode TRESUsageInMinTask TRESUsageInTot
TRESUsageOutAve TRESUsageOutMax TRESUsageOutMaxNode TRESUsageOutMaxTask TRESUsageOutMin
TRESUsageOutMinNode TRESUsageOutMinTask TRESUsageOutTot UID User UserCPU WCKey WCKeyID
WorkDir"""


def allocation_row(view: View, a: Attempt) -> Dict[str, str]:
    state = view.status(a)
    started = state != "PENDING"
    ended = started and state != "RUNNING"
    elapsed = 0
    if started:
        elapsed = (a.end if ended else view.t) - a.start
    row = {
        "DBIndex": str(a.index), "JobIDRaw": str(a.job), "Cluster": CLUSTER, "JobID": a.label,
        "Submit": str(a.submit), "Eligible": str(a.eligible),
        "Start": str(a.start) if started else "Unknown",
        "End": str(a.end) if ended else "Unknown", "Suspended": "00:00:00",
        "ElapsedRaw": str(elapsed), "State": state,
        "AllocTRES": tres(a) if started else "", "ReqTRES": tres(a, alloc=False),
        "ReqMem": "%dM" % (a.mem * a.nnodes), "NNodes": str(a.nnodes),
        "AllocCPUS": str(a.cpus * a.nnodes if started else 0),
        "TimelimitRaw": str(a.limit), "Restarts": str(a.restarts or 0),
        "ExitCode": (a.exit or "0:0") if ended else "0:0", "UID": str(a.uid), "User": a.user,
        "Account": a.account, "QOS": a.qos, "Partition": a.partition,
        "NodeList": hostlist(a.nodes) if started else "None assigned", "JobName": a.name,
        "Reason": "None" if started else view.pending_reason(a, None),
        "WorkDir": workdir(a), "StdOut": workdir(a) + "/logs/%x-%j.out",
        "StdErr": workdir(a) + "/logs/%x-%j.err", "NTasks": "",
        "SubmitLine": "sbatch --qos=%s %s.sbatch" % (a.qos, a.kind),
        "TotalCPU": "", "AveCPU": "", "MaxRSS": "", "AveRSS": "", "TRESUsageInTot": "",
        "TRESUsageInMax": "", "TRESUsageInAve": "",
    }
    if ended:
        cpu_eff, _ = efficiency(a)
        row["TotalCPU"] = duration(elapsed * a.cpus * a.nnodes * cpu_eff)
    return row


def step_rows(view: View, a: Attempt, live: bool = False) -> List[Dict[str, str]]:
    state = view.status(a)
    if state == "PENDING":
        return []
    base = allocation_row(view, a)
    end = a.end if state != "RUNNING" else None
    elapsed = int(base["ElapsedRaw"])
    cpu_eff, gpu_eff = efficiency(a)
    rng = _rand(a, "steps")
    rows = []
    main_cpus = a.cpus * a.nnodes
    tasks = max(1, a.gpus * a.nnodes) if a.gpus else max(1, a.nnodes)
    mem_peak = int(a.mem * rng.uniform(0.35, 0.8))  # MB per node
    util = int(gpu_eff * 100 * max(1, a.gpus))  # summed over GPUs, as Slurm does
    gpumem = int(a.gpus * 80 * 1024 * rng.uniform(0.4, 0.9)) if a.gpus else 0

    def usage(cpu_seconds: float, mem_mb: int, with_gpu: bool) -> str:
        text = "cpu=%s,energy=0,fs/disk=%d" % (duration(cpu_seconds), rng.randrange(10 ** 6, 10 ** 9))
        if with_gpu and a.gpus:
            text += ",gres/gpumem=%dM,gres/gpuutil=%d" % (gpumem, util)
        return text + ",mem=%dM,pages=0,vmem=%dM" % (mem_mb, mem_mb * 2)

    def step(suffix: str, name: str, ntasks: int, cpus: int, share: float, with_gpu: bool,
             start_off: int, mem_mb: int) -> Dict[str, str]:
        row = dict(base)
        start = a.start + start_off
        stop = end if end is not None else None
        step_elapsed = max(0, (stop if stop is not None else view.t) - start)
        total = step_elapsed * cpus * cpu_eff * share
        step_state = "RUNNING" if stop is None else (
            "COMPLETED" if a.state == "COMPLETED" else
            "CANCELLED" if a.state in ("PREEMPTED", "TIMEOUT", "NODE_FAIL") or
            a.state.startswith("CANCELLED") else "FAILED")
        row.update(JobIDRaw="%d.%s" % (a.job, suffix), JobID="%s.%s" % (a.label, suffix),
                   JobName=name, NTasks=str(ntasks), Start=str(start),
                   End=str(stop) if stop is not None else "Unknown",
                   ElapsedRaw=str(step_elapsed), State=step_state,
                   AllocTRES=tres(a), TotalCPU=duration(total),
                   AveCPU=duration(total / max(1, ntasks)),
                   MaxRSS="%dM" % mem_mb, AveRSS="%dM" % int(mem_mb * 0.8),
                   TRESUsageInTot=usage(total, mem_mb * ntasks, with_gpu),
                   TRESUsageInMax=usage(total / max(1, ntasks), mem_mb, with_gpu),
                   TRESUsageInAve=usage(total / max(1, ntasks), int(mem_mb * 0.8), with_gpu),
                   ExitCode=a.exit if stop is not None and a.exit else "0:0")
        if suffix in ("batch", "extern"):
            row["NodeList"] = a.nodes[0] if suffix == "batch" else hostlist(a.nodes)
        return row

    if not live:
        rows.append(step("batch", "batch", 1, 1, 0.02, False, 0, 400))
    rows.append(step("extern", "extern", a.nnodes, 1, 0.0, False, 0, 2))
    launcher = "torchrun" if a.nnodes > 1 else "python"
    if a.gpus == 0:
        launcher = a.kind.split("_")[0]
    rows.append(step("0", launcher, tasks, main_cpus, 0.97, True, rng.randrange(5, 40),
                     max(200, mem_peak // tasks)))
    if a.kind.startswith(("train", "pretrain")) and elapsed > 2 * HOUR:
        rows.append(step("1", "eval", tasks, main_cpus, 0.3, True, elapsed // 2,
                         max(200, mem_peak // (2 * tasks))))
    return rows


def _options(argv: Sequence[str], with_value: Sequence[str]) -> Tuple[Dict[str, str], set]:
    options, flags = {}, set()
    i = 1
    while i < len(argv):
        token = argv[i]
        if "=" in token and token.startswith("--"):
            key, _, value = token.partition("=")
            options[key] = value
        elif token in with_value and i + 1 < len(argv):
            options[token] = argv[i + 1]
            i += 1
        else:
            flags.add(token)
        i += 1
    return options, flags


def _when(text: str, current: int) -> Optional[int]:
    if text == "now":
        return current
    match = re.fullmatch(r"now-(\d+)(days|hours|minutes)", text)
    if match:
        unit = {"days": DAY, "hours": HOUR, "minutes": MIN}[match.group(2)]
        return current - int(match.group(1)) * unit
    if text.isdigit():
        return int(text)
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return int(time.mktime(time.strptime(text, fmt)))
        except ValueError:
            pass
    return None


# --------------------------------------------------------------------------- commands
def sacct(argv: Sequence[str], out) -> int:
    if "--helpformat" in argv or "-e" in argv:
        out.write(SACCT_FIELDS + "\n")
        return 0
    options, flags = _options(argv, ("-S", "-E", "-j", "-o", "-u", "-A", "--starttime",
                                     "--endtime", "--jobs", "--format"))
    view = View()
    if "--batch-script" in flags:
        job = options.get("-j", "")
        found = [a for a in view.attempts if str(a.job) == job]
        if found:
            a = found[-1]
            out.write("Batch Script for %s\n%s\n" % (job, "-" * 80))
            out.write(script(a))
        return 0
    fields = (options.get("-o") or options.get("--format") or
              "JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode").split(",")
    separator = options.get("--delimiter", "|")
    steps = "-X" not in flags
    low = _when(options.get("-S", ""), view.t) if "-S" in options else None
    high = _when(options.get("-E", ""), view.t) if "-E" in options else None
    if "-S" not in options and "-j" not in options:
        low = view.base  # sacct's default: since midnight
    wanted = set(options["-j"].split(",")) if "-j" in options else None
    uid = options.get("-u")
    account = options.get("-A")
    lines = []
    for a in view.attempts:
        if wanted is not None and str(a.job) not in wanted and a.label not in wanted:
            continue
        if uid is not None and str(a.uid) != uid and a.user != uid:
            continue
        if account is not None and a.account != account:
            continue
        state = view.status(a)
        begin = a.start if state != "PENDING" else a.eligible
        finish = a.end if state not in ("PENDING", "RUNNING") else None
        if high is not None and begin > high:
            continue
        if low is not None and finish is not None and finish < low:
            continue
        row = allocation_row(view, a)
        lines.append(separator.join(row.get(f, "") for f in fields))
        if steps:
            for step in step_rows(view, a):
                lines.append(separator.join(step.get(f, "") for f in fields))
    if lines:
        out.write("\n".join(lines) + "\n")
    return 0


def script(a: Attempt) -> str:
    gpus = "#SBATCH --gpus-per-node=%d\n" % a.gpus if a.gpus else ""
    return ("#!/bin/bash\n#SBATCH --job-name=%s\n#SBATCH --account=%s\n#SBATCH --qos=%s\n"
            "#SBATCH --nodes=%d\n%s#SBATCH --cpus-per-task=%d\n#SBATCH --time=%s\n"
            "#SBATCH --output=logs/%%x-%%j.out\n\nsource ~/envs/%s/bin/activate\n"
            "srun python %s.py --config configs/%s.yaml\n"
            % (a.name, a.account, a.qos, a.nnodes, gpus, a.cpus, duration(a.limit * 60),
               a.kind.split("_")[0], a.kind, a.kind))


def sstat(argv: Sequence[str], out) -> int:
    options, _ = _options(argv, ("-j", "-o"))
    view = View()
    job = options.get("-j", "")
    fields = options.get("-o", "JobID,NTasks,AveCPU,AveRSS,MaxRSS").split(",")
    for a in view.running():
        if str(a.job) != job and a.label != job:
            continue
        for row in step_rows(view, a, live=True):
            values = dict(row, Nodelist=row["NodeList"], JobID=row["JobIDRaw"])
            out.write("|".join(values.get(f, "") for f in fields) + "\n")
    return 0


def node_text(view: View, t: int) -> str:
    running = view.running()
    lines = []
    for name, cpus, mem, gpus, gtype, part in NODES:
        here = [a for a in running if name in a.nodes]
        alloc_cpu = sum(a.cpus for a in here)
        alloc_gpu = sum(a.gpus for a in here)
        alloc_mem = sum(a.mem for a in here)
        idx = sorted(i for a in here for i in a.gpu_idx[name])
        if alloc_cpu == 0:
            state = "IDLE"
        elif alloc_cpu >= cpus or (gpus and alloc_gpu >= gpus):
            state = "ALLOCATED"
        else:
            state = "MIXED"
        drained = name == DRAINED
        if drained:
            state += "+DRAIN"
        rng = random.Random("%s/%d" % (name, t // 60))
        load = alloc_cpu * rng.uniform(0.55, 0.95)
        cfg = "cpu=%d,mem=%dM,billing=%d" % (cpus, mem, cpus)
        alloc = "cpu=%d,mem=%dM" % (alloc_cpu, alloc_mem) if alloc_cpu else ""
        gres = "(null)"
        used = ""
        if gpus:
            cfg += ",gres/gpu=%d,gres/gpu:%s=%d" % (gpus, gtype, gpus)
            if alloc_gpu:
                alloc += ",gres/gpu=%d,gres/gpu:%s=%d" % (alloc_gpu, gtype, alloc_gpu)
            gres = "gpu:%s:%d(S:0-1)" % (gtype, gpus)
            used = " GresUsed=gpu:%s:%d(IDX:%s)" % (gtype, alloc_gpu, ranges(idx) if idx else "N/A")
        boot = view.base - 41 * DAY + 3 * HOUR
        line = (
            "NodeName={n} Arch=x86_64 CoresPerSocket={cps} CPUAlloc={ac} CPUEfctv={c} CPUTot={c} "
            "CPULoad={load:.2f} AvailableFeatures={feat} ActiveFeatures={feat} Gres={g} "
            "GresDrain=N/A{used} NodeAddr={n} NodeHostName={n} Version=24.11.3 "
            "OS=Linux 6.8.0-orion #1 SMP PREEMPT_DYNAMIC RealMemory={m} AllocMem={am} "
            "FreeMem={fm} Sockets=2 Boards=1 State={s} ThreadsPerCore=1 TmpDisk=0 Weight=1 "
            "Owner=N/A MCS_label=N/A Partitions={p} BootTime={boot} SlurmdStartTime={boot} "
            "LastBusyTime={busy} ResumeAfterTime=None CfgTRES={cfg} AllocTRES={alloc} "
            "CurrentWatts=0 AveWatts=0"
        ).format(n=name, cps=cpus // 2, ac=alloc_cpu, c=cpus, load=load,
                 feat="%s,ib" % (gtype or "epyc"), g=gres, used=used, m=mem, am=alloc_mem,
                 fm=max(0, mem - alloc_mem - 20000), s=state, p=part, boot=boot,
                 busy=t if here else t - 1800, cfg=cfg, alloc=alloc)
        if drained:
            line += " Reason=%s [opsbot@%d]" % (DRAIN_REASON, view.base - 30 * HOUR)
        lines.append(line)
    return "\n".join(lines) + "\n"


def job_text(view: View) -> str:
    out = []
    for a in view.live():
        state = view.status(a)
        holds = state == "RUNNING"
        array = ""
        if a.array is not None:
            array = "ArrayJobId=%d ArrayTaskId=%s " % (a.array, a.label.rpartition("_")[2])
        detail = ""
        if holds:
            for node in a.nodes:
                gtype = gpu_type(a)
                gres = (" GRES=gpu:%s:%d(IDX:%s)" % (gtype, a.gpus, ranges(a.gpu_idx[node]))
                        if a.gpus else " GRES=")
                job_gres = "JOB_GRES=gpu:%s:%d " % (gtype, a.gpus * a.nnodes) if a.gpus else ""
                detail += " %sNodes=%s CPU_IDs=%s Mem=%d%s" % (
                    job_gres, node, ranges(a.cpu_idx[node]), a.mem, gres)
        start = a.start if holds else view.t + 3600
        out.append(
            "JobId={id} {array}JobName={name} UserId={user}({uid}) GroupId={user}({uid}) "
            "MCS_label=N/A Priority={prio} Nice=0 Account={acct} QOS={qos} JobState={state} "
            "Reason={reason} Dependency=(null) Requeue=1 Restarts={restarts} BatchFlag=1 "
            "Reboot=0 ExitCode=0:0 RunTime={run} TimeLimit={limit} TimeMin=N/A "
            "SubmitTime={submit} EligibleTime={elig} AccrueTime={elig} StartTime={start} "
            "EndTime={end} Deadline=N/A SuspendTime=None SecsPreSuspend=0 "
            "LastSchedEval={t} Scheduler=Main Partition={part} AllocNode:Sid=login1:4242 "
            "ReqNodeList=(null) ExcNodeList=(null) NodeList={nodes} BatchHost={bh} "
            "NumNodes={nn} NumCPUs={cpus} NumTasks={nn} CPUs/Task={cpt} ReqB:S:C:T=0:0:*:* "
            "ReqTRES={req} AllocTRES={alloc} Socks/Node=* NtasksPerN:B:S:C=0:0:*:* "
            "CoreSpec=*{detail} MinCPUsNode={cpt} MinMemoryNode={mem}M MinTmpDiskNode=0 "
            "Features=(null) DelayBoot=00:00:00 OverSubscribe=OK Contiguous=0 "
            "Licenses=(null) Network=(null) Command={wd}/{kind}.sbatch "
            "WorkDir={wd} Comment=(null) StdErr={wd}/logs/{name}-{id}.err "
            "StdIn=/dev/null StdOut={wd}/logs/{name}-{id}.out\n".format(
                id=a.job, array=array, name=a.name, user=a.user, uid=a.uid,
                prio=priority(a), acct=a.account, qos=a.qos, state=state,
                reason="None" if holds else view.pending_reason(a, None),
                restarts=a.restarts or 0, run=duration(view.t - a.start) if holds else "00:00:00",
                limit=duration(a.limit * 60), submit=a.submit, elig=a.eligible, start=start,
                end=start + a.limit * 60, t=view.t - 20, part=a.partition,
                nodes=hostlist(a.nodes) if holds else "", bh=a.nodes[0] if holds else "",
                nn=a.nnodes, cpus=a.cpus * a.nnodes, cpt=a.cpus, req=tres(a, alloc=False),
                alloc=tres(a) if holds else "(null)", detail=detail, mem=a.mem,
                wd=workdir(a), kind=a.kind))
    return "".join(out)


def priority(a: Attempt) -> int:
    return QOS[a.qos][0] * 10 + 5000 - (a.job % 997)


def squeue(argv: Sequence[str], out) -> int:
    view = View()
    jobs = view.live()
    options, flags = _options(argv, ("-o", "-O", "-j", "-t", "-u", "--format", "--Format"))
    if "-j" in options:
        jobs = [a for a in jobs if str(a.job) in options["-j"].split(",")]
    if "-O" in options:  # log paths: JobID:<sep>,STDOUT:<sep>,STDERR:<end>
        parts = re.findall(r"(\w+):([^,]*)", options["-O"])
        for a in jobs:
            values = {"JobID": str(a.job), "STDOUT": log_path(a), "STDERR": log_path(a, "err")}
            out.write("".join(values.get(name.upper(), values.get(name, "")) + suffix
                              for name, suffix in parts) + "\n")
        return 0
    fmt = options.get("-o", "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R")
    if fmt == "%A":
        for a in jobs:
            out.write("%d\n" % a.job)
        return 0
    match = re.match(r"(z[0-9a-f]{10})b", fmt)
    if not match:  # a human asking: a short table
        out.write("%18s %9s %12s %8s %2s %10s %6s %s\n" % (
            "JOBID", "PARTITION", "NAME", "USER", "ST", "TIME", "NODES", "NODELIST(REASON)"))
        for a in jobs:
            state = view.status(a)
            out.write("%18s %9s %12s %8s %2s %10s %6d %s\n" % (
                a.label, a.partition, a.name[:12], a.user, "R" if state == "RUNNING" else "PD",
                duration(view.t - a.start) if state == "RUNNING" else "0:00", a.nnodes,
                hostlist(a.nodes) if state == "RUNNING" else "(%s)" % view.pending_reason(a, None)))
        return 0
    token = match.group(1)
    begin, sep, end = token + "b", token + "s", token + "e"
    codes = fmt[len(begin):-len(end)].split(sep)
    waiting = [a for a in jobs if view.status(a) == "PENDING" and not a.reason]
    waiting.sort(key=lambda a: (-priority(a), a.job))
    head = waiting[0].job if waiting else None
    for a in jobs:
        state = view.status(a)
        holds = state == "RUNNING"
        if state not in ("RUNNING", "PENDING"):
            holds = False
        values = {
            "%i": a.label, "%A": str(a.job),
            "%F": str(a.array) if a.array is not None else "N/A",
            "%K": a.label.rpartition("_")[2] if a.array is not None else "N/A",
            "%U": str(a.uid), "%u": a.user, "%a": a.account, "%q": a.qos, "%P": a.partition,
            "%T": state, "%Q": str(priority(a)), "%y": "0", "%V": str(a.submit),
            "%S": str(a.start if a.start and state != "PENDING" else view.t + 2 * HOUR),
            "%e": str((a.start or view.t) + a.limit * 60), "%l": duration(a.limit * 60),
            "%M": duration(view.t - a.start) if holds else "0:00",
            "%C": str(a.cpus * a.nnodes), "%D": str(a.nnodes),
            "%N": hostlist(a.nodes) if holds else "",
            "%B": a.nodes[0] if holds else "n/a", "%v": "(null)",
            "%b": "gres/gpu:%d" % a.gpus if a.gpus else "N/A", "%w": "(null)",
            "%E": "afterok:%d" % (a.job - 1) if a.reason == "Dependency" else "(null)",
            "%r": "None" if holds else (view.pending_reason(a, head) if state == "PENDING"
                                         else "None"),
            "%j": a.name, "%k": "(null)", "%Z": workdir(a),
            "%o": "%s/%s.sbatch" % (workdir(a), a.kind),
        }
        out.write(begin + sep.join(values.get(code, "") for code in codes) + end + "\n")
    return 0


def config_text(view: View) -> str:
    values = [
        ("ClusterName", CLUSTER), ("SLURM_VERSION", "24.11.3"),
        ("SelectType", "select/cons_tres"), ("SelectTypeParameters", "CR_CORE_MEMORY"),
        ("AccountingStorageType", "accounting_storage/slurmdbd"),
        ("AccountingStorageHost", "orion-db"),
        ("AccountingStorageTRES", "cpu,mem,energy,node,billing,fs/disk,vmem,pages,gres/gpu,"
                                  "gres/gpumem,gres/gpuutil"),
        ("AccountingStoreFlags", "job_script"),
        ("GresTypes", "gpu"), ("MinJobAge", "300 sec"), ("PreemptMode", "REQUEUE"),
        ("PreemptType", "preempt/qos"), ("PriorityType", "priority/multifactor"),
        ("PriorityWeightAge", "1000"), ("PriorityWeightFairshare", "10000"),
        ("PriorityWeightQOS", "5000"), ("PriorityWeightTRES", "(null)"),
        ("PriorityDecayHalfLife", "7-00:00:00"), ("PrivateData", "(null)"),
        ("SlurmUser", "slurm(64030)"), ("SlurmctldHost[0]", "orion-ctl"),
    ]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(view.t))
    return "Configuration data as of %s\n" % stamp + "".join(
        "%-24s= %s\n" % pair for pair in values)


def partition_text() -> str:
    def line(name, nodes, total_cpus, total_nodes, default, extra=""):
        return (
            "PartitionName={n} AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL AllocNodes=ALL "
            "Default={d} QoS=N/A DefaultTime=01:00:00 DisableRootJobs=NO ExclusiveUser=NO "
            "GraceTime=0 Hidden=NO MaxNodes=UNLIMITED MaxTime=3-00:00:00 MinNodes=0 LLN=NO "
            "MaxCPUsPerNode=UNLIMITED Nodes={nodes} PriorityJobFactor=1 PriorityTier=1 "
            "RootOnly=NO ReqResv=NO OverSubscribe=NO OverTimeLimit=NONE PreemptMode=REQUEUE "
            "State=UP TotalCPUs={tc} TotalNodes={tn} SelectTypeParameters=NONE {extra}"
            "DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED TRES=cpu={tc}\n"
        ).format(n=name, d="YES" if default else "NO", nodes=nodes, tc=total_cpus,
                 tn=total_nodes, extra=extra)
    return (line("gpu", "gpu-a[01-06],gpu-b[01-02]", 6 * 112 + 2 * 64, 8, True,
                 "JobDefaults=DefCpuPerGPU=8 ")
            + line("cpu", "cpu-c[01-02]", 384, 2, False))


def reservation_text(view: View) -> str:
    start = view.base + 4 * DAY + 6 * HOUR
    return ("ReservationName=maint_nvlink StartTime={s} EndTime={e} Duration=04:00:00 "
            "Nodes=gpu-a06 NodeCnt=1 CoreCnt=112 Features=(null) PartitionName=gpu "
            "Flags=MAINT,SPEC_NODES TRES=cpu=112 Users=root Groups=(null) Accounts=(null) "
            "Licenses=(null) State=INACTIVE BurstBuffer=(null) MaxStartDelay=(null)\n"
            ).format(s=start, e=start + 4 * HOUR)


def assoc_mgr_text(view: View) -> str:
    running = view.running()
    by_user, by_acct = {}, {}
    for a in running:
        by_user[a.user] = by_user.get(a.user, 0) + a.gpus * a.nnodes
        by_acct[a.account] = by_acct.get(a.account, 0) + a.gpus * a.nnodes
    users = "".join("UserName=%s(%d) DefAccount=%s DefWckey= AdminLevel=None\n"
                    % (name, view.uids[name], acct) for name, _u, acct, *_ in USERS)

    def record(account, user=None):
        who = "%s(%d)" % (user, view.uids[user]) if user else ""
        lineage = "/%s" % account + ("/0-%s" % user if user else "")
        cap = ACCOUNT_LIMITS.get(account)
        used = by_user.get(user, 0) if user else by_acct.get(account, 0)
        grp = "cpu=N(0),mem=N(0),gres/gpu=%s(%d)" % (cap if cap and not user else "N", used)
        return (
            "ClusterName=orion Account={a} UserName={who} Partition= Priority=0 ID=1\n"
            "    SharesRaw/Norm/Level/Factor=1/0.20/5/0.50\n"
            "    UsageRaw/Norm/Efctv=1000.00/0.10/0.10\n"
            "    ParentAccount={p}(1) Lineage={lin}/ DefAssoc={d}\n"
            "    GrpJobs=N(1) GrpJobsAccrue=N(0)\n"
            "    GrpSubmitJobs=N(1) GrpWall=N(60.00)\n"
            "    GrpTRES={g}\n"
            "    GrpTRESMins=cpu=N(600),gres/gpu=N(60)\n"
            "    GrpTRESRunMins=cpu=N(0),gres/gpu=N(0)\n"
            "    MaxJobs= MaxJobsAccrue= MaxSubmitJobs= MaxWallPJ=\n"
            "    MaxTRESPJ=\n    MaxTRESPN=\n    MaxTRESMinsPJ=\n    MinPrioThresh=\n"
            "    Comment=(null)\n"
        ).format(a=account, who=who, p=account if user else "root", lin=lineage,
                 d="Yes" if user else "No", g=grp)

    assocs = "".join(record(acct) for acct in ACCOUNTS)
    assocs += "".join(record(acct, name) for name, _u, acct, *_ in USERS)
    qos = ""
    for name, (prio, _wall, per_user) in QOS.items():
        qos += "QOS=%s(%d)\n    UsageRaw=1000.000000\n    GrpTRES=cpu=N(0),gres/gpu=N(0)\n" % (
            name, prio + 1)
        qos += "    Priority=%d\n    Account Limits\n        No Accounts\n    User Limits\n" % prio
        if per_user:
            for user, _u, *_ in USERS[:4]:
                qos += "      %s(%d)\n        MaxTRESPU=gres/gpu=%d(%d)\n" % (
                    user, view.uids[user], per_user, by_user.get(user, 0))
        else:
            qos += "        No Users\n"
    return ("Current Association Manager state\n\nUser Records\n\n%s\n"
            "Association Records\n\n%s\nQOS Records\n\n%s" % (users, assocs, qos))


def scontrol(argv: Sequence[str], out) -> int:
    words = [w for w in argv[1:] if not w.startswith("-")]
    view = View()
    if words[:1] == ["ping"]:
        out.write("Slurmctld(primary) at orion-ctl is UP\n")
        return 0
    if words[:1] != ["show"] or len(words) < 2:
        sys.stderr.write("scontrol: the demo cluster only answers show and ping\n")
        return 1
    what = words[1]
    if what == "config":
        out.write(config_text(view))
    elif what == "node":
        out.write(node_text(view, view.t))
    elif what == "job":
        out.write(job_text(view))
    elif what == "partition":
        out.write(partition_text())
    elif what == "reservation":
        out.write(reservation_text(view))
    elif what == "assoc_mgr":
        out.write(assoc_mgr_text(view))
    elif what == "hostnames" and len(words) > 2:
        out.write("\n".join(_expand(words[2])) + "\n")
    else:
        sys.stderr.write("scontrol: nothing to show for %s in the demo\n" % what)
        return 1
    return 0


def _expand(text: str) -> List[str]:
    match = re.fullmatch(r"(.*)\[(.*)\]", text)
    if not match:
        return [text]
    names = []
    for part in match.group(2).split(","):
        first, _, last = part.partition("-")
        for i in range(int(first), int(last or first) + 1):
            names.append("%s%0*d" % (match.group(1), len(first), i))
    return names


def sacctmgr(argv: Sequence[str], out) -> int:
    words = [w for w in argv[1:] if not w.startswith("-")]
    view = View()
    entity = words[1] if len(words) > 1 else ""
    fmt = next((w.split("=", 1)[1] for w in words if w.startswith("format=")), "")
    if entity in ("assoc", "association", "associations"):
        rows = []
        for account in ACCOUNTS:
            rows.append("%s|||high,normal,low,debug|normal" % account)
        for name, _u, account, *_ in USERS:
            rows.append("%s|%s||high,normal,low,debug|normal" % (account, name))
        out.write("\n".join(rows) + "\n")
    elif entity == "qos":
        for name, (prio, wall, per_user) in QOS.items():
            out.write("%s|%d|%s||%s||||\n" % (
                name, prio, wall, "gres/gpu=%d" % per_user if per_user else ""))
    elif entity == "account":
        coord = {"physics": "bob", "biolab": "carol", "vision": "hiro"}
        for account in ACCOUNTS:
            out.write("%s|%s\n" % (account, coord.get(account, "")))
    elif entity == "user":
        name = words[2] if len(words) > 2 and "=" not in words[2] else ""
        accounts = {n: acct for n, _u, acct, *_ in USERS}
        if name in accounts:
            if "coordinators" in fmt:
                coordinated = {"bob": "physics", "carol": "biolab", "hiro": "vision"}
                out.write("%s|None|%s\n" % (name, coordinated.get(name, "")))
            else:
                out.write("%s|None|%s\n" % (name, accounts[name]))
    elif entity == "cluster":
        cpus = sum(n[1] for n in NODES)
        gpus = sum(n[3] for n in NODES)
        mem = sum(n[2] for n in NODES)
        out.write("%s|billing=%d,cpu=%d,energy=0,gres/gpu=%d,mem=%dM,node=%d\n"
                  % (CLUSTER, cpus, cpus, gpus, mem, len(NODES)))
    elif entity in ("event", "events"):
        out.write("%d|\n" % (view.base - DAYS * DAY - 3 * HOUR))
    elif entity in ("transaction", "transactions"):
        out.write("%d|\n" % (view.base - DAYS * DAY - 5 * HOUR))
    return 0


def sinfo(argv: Sequence[str], out) -> int:
    view = View()
    out.write("PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST\n")
    states = {}
    for line in node_text(view, view.t).splitlines():
        name = re.search(r"NodeName=(\S+)", line).group(1)
        state = re.search(r" State=(\S+)", line).group(1)
        part = re.search(r" Partitions=(\S+)", line).group(1)
        short = {"IDLE": "idle", "MIXED": "mix", "ALLOCATED": "alloc"}.get(
            state.split("+")[0], "unk") + ("" if "+DRAIN" not in state else "")
        if "+DRAIN" in state:
            short = "drain"
        states.setdefault((part, short), []).append(name)
    for (part, short), names in sorted(states.items()):
        star = "*" if part == "gpu" else ""
        out.write("%-9s    up 3-00:00:00 %6d %6s %s\n" % (part + star, len(names), short,
                                                       hostlist(names)))
    return 0


def sprio(argv: Sequence[str], out) -> int:
    view = View()
    out.write("          JOBID PARTITION   PRIORITY       SITE        AGE  FAIRSHARE        QOS\n")
    for a in view.live():
        if view.status(a) == "PENDING":
            p = priority(a)
            out.write("%15s %9s %10d %10d %10d %10d %10d\n" % (
                a.label, a.partition, p, 0, p % 1000, 5000 - a.job % 997, QOS[a.qos][0] * 10))
    return 0


def sshare(argv: Sequence[str], out) -> int:
    out.write("Account                    User  RawShares  NormShares    RawUsage  "
              "EffectvUsage  FairShare\n")
    out.write("-------------------- ---------- ---------- ----------- ----------- "
              "------------- ----------\n")
    out.write("root                                          0.000000   123456789      1.000000\n")
    for i, account in enumerate(ACCOUNTS):
        out.write(" %-19s %10s %10d %11.6f %11d %13.6f %10s\n" % (
            account, "", 1, 0.2, 20000000 - i * 3000000, 0.2 - i * 0.03, ""))
    return 0


def sreport(argv: Sequence[str], out) -> int:
    view = View()
    hours = {}
    for a in view.attempts:
        if a.start and a.start < view.t:
            end = min(a.end or view.t, view.t)
            hours[a.account] = hours.get(a.account, 0) + (end - a.start) * a.gpus * a.nnodes // HOUR
    out.write("Cluster/Account/User Utilization (gres/gpu hours, invented demo data)\n")
    for account, value in sorted(hours.items(), key=lambda kv: -kv[1]):
        out.write("  %-10s %-10s %12d\n" % (CLUSTER, account, value))
    return 0


COMMANDS = {"sacct": sacct, "sstat": sstat, "squeue": squeue, "scontrol": scontrol,
            "sacctmgr": sacctmgr, "sinfo": sinfo, "sprio": sprio, "sshare": sshare,
            "sreport": sreport}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    program = os.path.basename(argv[0])
    command = COMMANDS.get(program)
    if command is None:
        sys.stderr.write("fakeslurm: unknown command %s\n" % program)
        return 2
    try:
        return command(argv, sys.stdout)
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
