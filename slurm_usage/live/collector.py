"""LiveCollector: capability detection and snapshots with few bulk calls."""

import os
import pwd
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import parse
from ..model import (Capability, ClusterInfo, Invoker, Job, Qos, Role, Snapshot, Status,
                     Warning_)
from ..runner import Runner, describe
from ..sanitize import clean, clean_id
from . import accounts, cluster, jobs, limits, logs, nodes, ondemand, queries, summary

STATIC_TTL = 600.0  # configuration and slurmdbd tables
RETRY_AFTER = 60.0  # a failed static or refused optional query
SLOW_CALL = 1.0     # optional calls slower than this are reused
SLOW_REUSE = 30.0
# Live answers are parsed whole; a bounded cap keeps one refresh's memory bounded.
LIVE_MAX_BYTES = 64 * 1024 * 1024
# scontrol show job: only the start of each one-line record is used (up to Features=);
# the free text after it (command, paths, comments) can be long and is dropped as it arrives.
DETAIL_LINE_CAP = 16 * 1024

Query = Tuple[str, List[str]]
Parser = Optional[Callable[[str], Any]]


@dataclass
class _Answer:
    capability: Capability
    value: Any = None  # parsed, or raw text when there is no parser


class LiveCollector:
    """What the cluster looks like now, as far as this invoker may see.

    ``snapshot()`` is seven bulk calls (nodes, queue, job detail, partitions,
    assoc_mgr, reservations, ping), whatever the number of jobs, plus up to seven
    configuration/slurmdbd calls cached for ten minutes. It does not raise for
    Slurm failures: see ``Snapshot.capabilities``.

    Call snapshot() from one thread at a time. The per-job methods may be called
    from another. All block for at most the runner's timeout.
    """

    def __init__(self, runner: Optional[Runner] = None, uid: Optional[int] = None,
                 user: Optional[str] = None, euid: Optional[int] = None,
                 static_ttl: float = STATIC_TTL, workers: int = 4) -> None:
        self.runner = runner if runner is not None else Runner()
        # uid/user/euid exist for tests; they only label output and change no permission
        self._uid = os.getuid() if uid is None else uid
        self._euid = os.geteuid() if euid is None else euid
        if user is None:
            try:
                user = pwd.getpwuid(self._uid).pw_name
            except KeyError:
                user = None
        self._user = user
        self._static_ttl = static_ttl
        self._workers = max(1, workers)
        self._cache = {}  # type: Dict[str, Tuple[float, _Answer]]
        self._refused_steps = {}  # type: Dict[str, ondemand.StepReport]

    def refresh(self) -> None:
        """Forget cached configuration and refusals."""
        self._cache.clear()
        self._refused_steps.clear()

    # -- fetching ----------------------------------------------------------

    def _ask(self, name: str, argv: List[str], parser: Parser) -> _Answer:
        result = self.runner.run(argv, max_bytes=LIVE_MAX_BYTES)
        answer = _Answer(Capability(name, result.status, result.reason, describe(argv),
                                    result.elapsed))
        if result.ok:
            answer.value = parser(result.stdout) if parser else result.stdout
        return answer

    def _ask_job_detail(self) -> _Answer:
        """scontrol show job, read line by line with each record cut to its useful start."""
        argv = queries.JOB_DETAIL
        stream = getattr(self.runner, "stream", None)
        if stream is None:
            return self._ask("job_detail", argv, None)
        parts = []  # type: List[str]
        with stream(argv, getattr(self.runner, "timeout", None), LIVE_MAX_BYTES) as lines:
            for line in lines:
                parts.append(line[:DETAIL_LINE_CAP])
        result = lines.result
        answer = _Answer(Capability("job_detail", result.status, result.reason,
                                    describe(argv), result.elapsed))
        if result.ok:
            answer.value = "\n".join(parts) + ("\n" if parts else "")
        return answer

    def _ask_static(self, name: str, argv: List[str], parser: Parser) -> _Answer:
        cached = self._cache.get(name)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        answer = self._ask(name, argv, parser)
        ttl = self._static_ttl if answer.capability.available else RETRY_AFTER
        self._cache[name] = (time.monotonic() + ttl, answer)
        return answer

    def _ask_optional(self, name: str, argv: List[str], parser: Parser) -> _Answer:
        cached = self._cache.get(name)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        answer = self._ask(name, argv, parser)
        if answer.capability.status in (Status.DENIED, Status.UNAVAILABLE):
            self._cache[name] = (time.monotonic() + RETRY_AFTER, answer)
        elif (answer.capability.elapsed or 0) > SLOW_CALL:
            self._cache[name] = (time.monotonic() + SLOW_REUSE, answer)
        return answer

    def _static(self, pool: ThreadPoolExecutor) -> Dict[str, _Answer]:
        got = {"config": self._ask_static("config", queries.CONFIG, cluster.parse_config)}
        config = got["config"].value or {}
        info = cluster.cluster_info(config)
        wanted = [("qos", queries.QOS, accounts.parse_qos),
                  ("accounting_capacity", queries.ACCOUNTING_CAPACITY,
                   lambda text: accounts.parse_capacity(text, info.name)),
                  ]  # type: List[Tuple[str, List[str], Parser]]
        if info.name:
            wanted.append(("associations", queries.associations(info.name),
                           accounts.parse_associations))
        user = self._user
        if user and not user.startswith("-"):
            wanted.append(("user", queries.user(user),
                           lambda text: accounts.parse_user(text, user)))
            wanted.append(("coordinators", queries.COORDINATORS,
                           lambda text: accounts.parse_coordinated(text, user)))
        if not info.name:
            got["associations"] = _Answer(Capability(
                "associations", Status.UNKNOWN, "needs the cluster's name"))
        if info.has_accounting is False:
            for name, argv, _ in wanted:
                got[name] = _Answer(Capability(
                    name, Status.UNAVAILABLE, "this cluster has no accounting database",
                    describe(argv)))
            return got
        futures = [(name, pool.submit(self._ask_static, name, argv, parser))
                   for name, argv, parser in wanted]
        for name, future in futures:
            got[name] = future.result()
        return got

    # -- snapshot ----------------------------------------------------------

    def snapshot(self) -> Snapshot:
        started = time.monotonic()
        snap = Snapshot(taken_at=time.time())
        frame = parse.Frame.new()
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            static = self._static(pool)
            # the three views that must agree go first, together, to minimise skew
            pending = [
                ("job_detail", pool.submit(self._ask_job_detail)),
                ("jobs", pool.submit(self._ask, "jobs", queries.queue(frame), None)),
                ("nodes", pool.submit(self._ask, "nodes", queries.NODES, None)),
                ("partitions", pool.submit(self._ask, "partitions", queries.PARTITIONS,
                                           cluster.parse_partitions)),
                ("assoc_mgr", pool.submit(self._ask_optional, "assoc_mgr", queries.ASSOC_MGR,
                                          None)),
                ("reservations", pool.submit(self._ask_optional, "reservations",
                                             queries.RESERVATIONS, cluster.parse_reservations)),
                ("scheduler", pool.submit(self._ask, "scheduler", queries.PING,
                                          cluster.parse_ping)),
            ]
            answers = dict(static)
            for name, future in pending:
                answers[name] = future.result()
        for name, answer in answers.items():
            seen = answer.capability
            # copies: capabilities are annotated below and answers may be cached
            snap.capabilities[name] = Capability(name, seen.status, seen.detail, seen.source,
                                                 seen.elapsed)

        config = answers["config"].value or {}
        snap.cluster = cluster.cluster_info(config)
        snap.priority = cluster.priority_config(config)
        snap.partitions = answers["partitions"].value
        self._set_nodes(snap, answers["nodes"])
        self._set_jobs(snap, answers, frame, config)
        summary.mark_completing(snap.nodes, snap.jobs)
        snap.totals = summary.totals(snap.nodes)
        self._set_accounts(snap, answers)
        self._set_invoker(snap, answers, config)
        snap.health = summary.health(snap.nodes, snap.jobs)
        snap.health.reservations = answers["reservations"].value
        snap.health.scheduler = answers["scheduler"].value or None
        recorded = answers.get("accounting_capacity")
        snap.capacity = summary.capacity_checks(snap.totals,
                                                recorded.value if recorded else None)
        self._detect_tools(snap)
        self._warn(snap)
        snap.duration = time.monotonic() - started
        return snap

    def _set_nodes(self, snap: Snapshot, answer: _Answer) -> None:
        if answer.value is None:
            return
        found = nodes.parse_nodes(answer.value, snap.cluster, snap.partitions)
        if found:
            snap.nodes = found
            if any("gres/gpu" in node.tres for node in found):
                snap.cluster.gpu_accounted = True
        else:
            capability = snap.capabilities["nodes"]
            capability.status = Status.EMPTY
            capability.detail = "no nodes are visible to you"

    def _set_jobs(self, snap: Snapshot, answers: Dict[str, _Answer], frame: parse.Frame,
                  config: Dict[str, str]) -> None:
        text = answers["jobs"].value
        if text is None:
            return  # without squeue there is no unforgeable identity to show
        rows, bad = jobs.parse_queue(text.split("\n"), frame)
        capability = snap.capabilities["jobs"]
        if not rows and (bad or frame.begin in text
                         or (text.strip() and "No jobs" not in text)):
            capability.status = Status.ERROR
            capability.detail = "squeue output was not understood"
            return
        if bad:
            capability.status = Status.PARTIAL
            capability.detail = "%d unreadable records" % bad

        has_gpus = None
        counts = [node.gpus_total for node in snap.nodes or ()]
        if any(counts):
            has_gpus = True
        elif counts and None not in counts:
            has_gpus = False
        elif not counts and "GresTypes" in config:
            has_gpus = None if "gpu" in config["GresTypes"].split(",") else False
        snap.jobs = jobs.build_jobs(rows, answers["job_detail"].value, snap.cluster, has_gpus)

    def _set_accounts(self, snap: Snapshot, answers: Dict[str, _Answer]) -> None:
        text = answers["assoc_mgr"].value
        data = None
        if text is not None and "Records" in text:
            capacity = {}  # type: Dict[str, float]
            if snap.totals.nodes:
                for key, value in (("node", snap.totals.nodes), ("cpu", snap.totals.cpus.total),
                                   ("mem", snap.totals.mem.total),
                                   ("gres/gpu", snap.totals.gpus.total)):
                    if value:
                        capacity[key] = value
                billing = [n.tres.get("billing") for n in snap.nodes or ()]
                if billing and None not in billing:
                    capacity["billing"] = sum(b for b in billing if b is not None)
                for kind, resource in snap.totals.gpu_types.items():
                    if resource.total:
                        capacity["gres/gpu:" + kind] = resource.total
            data = limits.parse_assoc_mgr(text, capacity)
        database_roles = answers["associations"].value if "associations" in answers else None
        database_qos = answers["qos"].value  # type: Optional[List[Qos]]
        manager_roles = data.roles if data else None  # type: Optional[List[Role]]
        snap.roles = accounts.merge_roles(manager_roles, database_roles)
        snap.qos = accounts.merge_qos(data.qos if data else None, database_qos)
        if data is not None or snap.qos is not None:
            snap.limits = [limit for role in manager_roles or () for limit in role.limits]
            snap.limits.extend(limit for qos in snap.qos or () for limit in qos.limits)
        for entry in data.users if data else ():
            if entry.uid == self._uid:
                snap.invoker.admin_level = entry.admin_level
                snap.invoker.default_account = entry.default_account

    def _set_invoker(self, snap: Snapshot, answers: Dict[str, _Answer],
                     config: Dict[str, str]) -> None:
        me = snap.invoker
        me.uid, me.euid, me.user = self._uid, self._euid, clean_id(self._user)
        if me.user is None:
            me.user = next((job.user for job in snap.jobs or () if job.uid == me.uid), None)
        row = answers["user"].value if "user" in answers else None
        if row:
            me.admin_level = me.admin_level or row[0]
            me.default_account = me.default_account or row[1]
        if "coordinators" in answers:
            me.coordinator_accounts = answers["coordinators"].value
        for role in snap.roles or ():
            if role.user and role.user == me.user and role.is_default is None and me.default_account:
                role.is_default = role.account == me.default_account
        if "SlurmUser" in config:
            name, uid = limits.named(config["SlurmUser"])
            if uid is not None:
                me.is_slurm_user = uid == me.uid
            elif name:
                me.is_slurm_user = name == me.user
        if "PrivateData" in config:
            private = parse.optional_text(config["PrivateData"]) or ""
            me.private_data = [clean(item).lower() for item in private.split(",") if item]

    def _detect_tools(self, snap: Snapshot) -> None:
        has = getattr(self.runner, "has", lambda program: True)
        for name, program in (("steps", "sstat"), ("history", "sacct"),
                              ("log_paths", "squeue"), ("batch_script", "sacct"),
                              ("submit_line", "sacct")):
            if not has(program):
                capability = Capability(name, Status.UNAVAILABLE, program + " is not installed")
            elif program == "sacct" and snap.cluster.has_accounting is False:
                capability = Capability(name, Status.UNAVAILABLE,
                                        "this cluster has no accounting database")
            else:
                capability = Capability(name, Status.UNKNOWN,
                                        "asked on demand; Slurm decides what you may see")
            capability.source = program
            snap.capabilities[name] = capability

    def _warn(self, snap: Snapshot) -> None:
        warn = snap.warnings.append
        for check in snap.capacity:
            if check.mismatch:
                warn(Warning_(
                    "capacity_mismatch",
                    "%s capacity differs: nodes have %d, the accounting database records %d; "
                    "live totals use the nodes" % (check.metric, check.live, check.accounting),
                    {"metric": check.metric, "live": check.live, "accounting": check.accounting}))
        hidden = set(snap.invoker.private_data or ()) & {"jobs", "nodes", "partitions"}
        if hidden and snap.invoker.privileged is not True:
            warn(Warning_(
                "visibility_limited",
                "PrivateData hides other users' %s: cluster-wide figures may cover only "
                "what you may see" % ", ".join(sorted(hidden)),
                {"hidden": sorted(hidden)}))
        private = set(snap.invoker.private_data or ()) & {"accounts", "usage", "users"}
        if private and snap.invoker.privileged is not True:
            warn(Warning_(
                "accounts_limited",
                "PrivateData limits %s: other people's roles, limits and history may be "
                "missing" % ", ".join(sorted(private)),
                {"hidden": sorted(private)}))
        unclear = sum(job.detail_status == "ambiguous" for job in snap.jobs or ())
        if unclear:
            warn(Warning_("detail_ambiguous",
                          "%d jobs' scheduler detail was ignored as ambiguous" % unclear,
                          {"jobs": unclear}))
        suspect = sum(limit.suspect for limit in snap.limits or ())
        if suspect:
            warn(Warning_("suspect_limit_usage",
                          "Slurm reports impossible usage for %d limits; shown as unknown"
                          % suspect, {"limits": suspect}))
        for name in ("nodes", "jobs", "job_detail"):
            capability = snap.capabilities[name]
            if capability.status == Status.PARTIAL or not capability.available:
                warn(Warning_("source_incomplete",
                              "%s: %s %s" % (name, capability.status, capability.detail),
                              {"source": name, "status": capability.status}))

    # -- on demand, for one job the viewer opened --------------------------

    def steps(self, job: Job) -> ondemand.StepReport:
        """sstat. A refusal is remembered, so an auto-refreshing view asks only once."""
        refused = self._refused_steps.get(job.job_id)
        if refused is not None:
            return refused
        report = ondemand.fetch_steps(self.runner, job)
        if report.status == Status.DENIED:
            self._refused_steps[job.job_id] = report
        return report

    def log_paths(self, job: Job) -> ondemand.LogPaths:
        return ondemand.fetch_log_paths(self.runner, job)

    def tail_log(self, job: Job, raw_path: Optional[str],
                 max_bytes: int = logs.MAX_BYTES) -> logs.LogTail:
        return logs.tail_file(raw_path, job.uid, max_bytes)

    def batch_script(self, job: Job) -> ondemand.Fetched:
        return ondemand.fetch_batch_script(self.runner, job.job_id)

    def submit_line(self, job: Job) -> ondemand.Fetched:
        return ondemand.fetch_submit_line(self.runner, job.job_id)
