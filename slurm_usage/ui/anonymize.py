"""``--anonymize``: stable pseudonyms for a screen recording or a shared report.

Users become user-01, accounts acct-01, nodes node-001, the cluster cluster-1;
job names become an opaque pseudonym per name family (job-01): none of their
text is shown, since words in a job name can identify a project or a person.
Known names are replaced in free text only as whole tokens, so a short name
cannot garble other words or reveal the mapping.
Paths, submit lines, scripts and log tails are not shown at all. The same real
value gets the same pseudonym for the life of one ``Anonymizer``.

It works on the data, not the screen: ``AnonCollector`` and ``AnonHistory``
wrap the live collector and the history object, so every view and every output
format sees only pseudonyms, and filters typed against pseudonyms are mapped
back before Slurm or the cache is asked.

QOS and partition names are scheduling vocabulary and are kept.
"""

import dataclasses
import re
from typing import Any, Dict, List, Optional

from ..model import Status
from ..sanitize import clean
from .data import name_family

HIDDEN = "hidden by --anonymize"
_PREFIX = {"user": ("user-", 2), "account": ("acct-", 2), "node": ("node-", 3),
           "cluster": ("cluster-", 1), "host": ("host-", 1), "reservation": ("resv-", 2),
           "job": ("job-", 2)}
_ROLE_KIND = {"user": "user", "account": "account", "node": "node"}
_NODE_REASON = re.compile(r"((?:Unavailable)?Nodes?|ReqNodeList|ExcNodeList)[:=]\S+", re.I)
_BRACKETED = re.compile(r"\[[^\]]*\]")
_SENSITIVE = re.compile(r"\S*[/@~]\S*")
# TRES names such as gres/gpu are vocabulary, not paths
_TRES_NAME = re.compile(r"(?:gres|license|fs|bb|ic)/[\w:.-]*[,.;:)]?", re.I)
_NONE_VALUES = ("", "(none)", "(null)")


class Anonymizer:
    def __init__(self) -> None:
        self.forward = {kind: {} for kind in _PREFIX}  # type: Dict[str, Dict[str, str]]
        self.backward = {kind: {} for kind in _PREFIX}  # type: Dict[str, Dict[str, str]]
        self._pattern = None  # type: Any
        self._known = 0

    def name(self, kind: str, real: Optional[str]) -> Optional[str]:
        if real is None or real in _NONE_VALUES:
            return real
        if real in self.backward[kind]:  # already a pseudonym: idempotent
            return real
        found = self.forward[kind].get(real)
        if found is None:
            prefix, digits = _PREFIX[kind]
            found = "%s%0*d" % (prefix, digits, len(self.forward[kind]) + 1)
            self.forward[kind][real] = found
            self.backward[kind][found] = real
        return found

    def real(self, kind: str, pseudo: str) -> str:
        return self.backward[kind].get(pseudo, pseudo)

    def names(self, kind: str, values: Optional[List[str]]) -> Optional[List[str]]:
        return None if values is None else [self.name(kind, v) for v in values]

    def scrub(self, text: Optional[str]) -> Optional[str]:
        """Free text: known identifiers replaced, node lists, paths and addresses masked."""
        if not text:
            return text
        text = _NODE_REASON.sub(lambda m: m.group(1) + ":…", text)
        text = _BRACKETED.sub("[…]", text)
        text = _SENSITIVE.sub(lambda m: m.group(0) if _TRES_NAME.fullmatch(m.group(0))
                              else "…", text)
        known = sum(len(m) for m in self.forward.values())
        if known != self._known:
            pairs = [(real, pseudo) for kind in ("user", "account", "node", "cluster", "host",
                                                 "reservation")
                     for real, pseudo in self.forward[kind].items() if real.strip()]
            pairs.sort(key=lambda p: -len(p[0]))
            self._lookup = dict(pairs)
            # whole tokens only: never inside a longer word, path or host name
            self._pattern = (re.compile(r"(?<![\w.-])(?:%s)(?![\w.-])" % "|".join(
                re.escape(r) for r, _ in pairs)) if pairs else None)
            self._known = known
        if self._pattern is not None:
            text = self._pattern.sub(lambda m: self._lookup[m.group(0)], text)
        return text

    def job_name(self, name: Optional[str]) -> Optional[str]:
        """An opaque pseudonym per name family: runs of one sweep share it."""
        if name is None:
            return None
        return self.name("job", name_family(name))

    # -- live ------------------------------------------------------------------

    def snapshot(self, snap: Any) -> Any:
        """Rewrite a fresh Snapshot in place and return it."""
        cluster = snap.cluster
        cluster.name = self.name("cluster", cluster.name)
        inv = snap.invoker
        inv.user = self.name("user", inv.user)
        inv.default_account = self.name("account", inv.default_account)
        inv.coordinator_accounts = self.names("account", inv.coordinator_accounts)
        for role in snap.roles or ():
            role.account = self.name("account", role.account)
            role.user = self.name("user", role.user)
            role.lineage = self.names("account", role.lineage) or []
            for limit in role.limits:
                self._limit(limit)
        for qos in snap.qos or ():
            for limit in qos.limits:
                self._limit(limit)
        for limit in snap.limits or ():
            self._limit(limit)
        for node in snap.nodes or ():
            self._node(node)
        for node in snap.health.problem_nodes:
            self._node(node)
        for part in snap.partitions or ():
            part.nodes = self.names("node", part.nodes) or []
            part.allow_accounts = self.names("account", part.allow_accounts)
            part.deny_accounts = self.names("account", part.deny_accounts)
            part.allow_groups = None
        for job in snap.jobs or ():
            job.user = self.name("user", job.user)
            job.account = self.name("account", job.account)
            job.name = self.job_name(job.name)
            job.raw_name = job.raw_workdir = job.workdir = job.command = None
            job.comment = job.wckey = None
            job.nodes = self.names("node", job.nodes) or []
            job.batch_host = self.name("node", job.batch_host)
            job.reservation = self.name("reservation", job.reservation)
            for alloc in job.allocations:
                alloc.node = self.name("node", alloc.node)
            for attempt in job.attempts:
                attempt.account = self.name("account", attempt.account)
                attempt.nodes = self.names("node", attempt.nodes) or []
                attempt.reason = self.scrub(attempt.reason)
        for job in snap.jobs or ():  # after every name is known
            job.reason = self.scrub(job.reason)
        health = snap.health
        health.free_gpus_by_node = [(self.name("node", n), c) for n, c in health.free_gpus_by_node]
        for block in [health.largest_gpu_block] + list(health.largest_gpu_block_by_type.values()):
            if block is not None:
                block.node = self.name("node", block.node)
        for outlier in health.load_outliers:
            outlier.node = self.name("node", outlier.node)
        for usage in health.pending_by_reason:
            if isinstance(usage.key, str):
                usage.key = self.scrub(usage.key)
        for resv in health.reservations or ():
            resv.name = self.name("reservation", resv.name)
            resv.users = self.names("user", resv.users) or []
            resv.accounts = self.names("account", resv.accounts) or []
            resv.nodes = self.names("node", resv.nodes) or []
        for host in health.scheduler or ():
            host.host = self.name("host", host.host)
        for warning in snap.warnings:
            warning.message = self.scrub(warning.message)
            warning.values = {}
        for capability in snap.capabilities.values():
            capability.detail = self.scrub(capability.detail) or ""
            capability.source = self.scrub(capability.source) or ""
        return snap

    def _limit(self, limit: Any) -> None:
        limit.account = self.name("account", limit.account)
        limit.user = self.name("user", limit.user)
        limit.raw = ""

    def _node(self, node: Any) -> None:
        node.name = self.name("node", node.name)
        node.reason = self.scrub(node.reason)
        node.reason_user = None

    # -- history ---------------------------------------------------------------

    def row(self, row: Any) -> Any:
        from ..parse import expand_hostlist
        nodes = row.nodes
        if nodes:
            names = expand_hostlist(nodes, 4096) or [nodes]
            nodes = ",".join(self.name("node", n) for n in names[:64])
        return dataclasses.replace(row, user=self.name("user", row.user),
                                   account=self.name("account", row.account),
                                   name=self.job_name(row.name) or "", nodes=nodes)


def _translate(anon: Anonymizer, where: Any) -> Any:
    if where is None:
        return None
    changes = {}
    for role, values in where.items():
        kind = _ROLE_KIND.get(role)
        if kind:
            changes[role] = frozenset(anon.real(kind, v) for v in values)
    return where.replace(**changes) if changes else where


class AnonHistory:
    """A history.History whose answers carry pseudonyms."""

    def __init__(self, history: Any, anon: Anonymizer) -> None:
        self._history, self._anon = history, anon

    def info(self) -> Any:
        info = self._history.info()
        info = dataclasses.replace(info, cluster=self._anon.name("cluster", info.cluster),
                                   cache_path=None)
        inv = dataclasses.replace(info.invoker)
        inv.user = self._anon.name("user", inv.user)
        inv.default_account = self._anon.name("account", inv.default_account)
        inv.coordinator_accounts = self._anon.names("account", inv.coordinator_accounts)
        info.invoker = inv
        info.notes = [type(n)(n.code, self._anon.scrub(n.message)) for n in info.notes]
        return info

    def refresh(self, *args: Any, **kwargs: Any) -> Any:
        self._history.refresh(*args, **kwargs)
        return self.info()

    def query(self, window: Any = None, where: Any = None, **kwargs: Any) -> Any:
        report = self._history.query(window, _translate(self._anon, where), **kwargs)
        anon = self._anon
        groups = {}
        for spec, members in report.groups.items():
            kinds = [_ROLE_KIND.get(role) for role in spec]
            groups[spec] = {tuple(anon.name(k, v) if k else v for k, v in zip(kinds, key)): stats
                            for key, stats in members.items()}
        report.groups = groups
        kind = _ROLE_KIND.get(report.series_by or "")
        if kind:
            report.keyed_values = {anon.name(kind, k): v for k, v in report.keyed_values.items()}
        lanes_kind = _ROLE_KIND.get(kwargs.get("lanes_by") or "")
        if lanes_kind:
            report.lanes = [type(l)(anon.name(lanes_kind, l.key), l.intervals) for l in report.lanes]
        report.where = where if where is not None else report.where
        report.notes = [type(n)(n.code, anon.scrub(n.message)) for n in report.notes]
        return report

    def jobs(self, window: Any = None, where: Any = None, *args: Any, **kwargs: Any) -> Any:
        run_type = kwargs.get("run_type")
        if run_type is not None:
            kwargs["run_type"] = (run_type[0], self._real_type(run_type[0], run_type[1]))
        page = self._history.jobs(window, _translate(self._anon, where), *args, **kwargs)
        page.rows = [self._anon.row(r) for r in page.rows]
        return page

    def attempts(self, job_id: Any) -> List[Any]:
        return [self._anon.row(r) for r in self._history.attempts(job_id)]

    def detail(self, job_id: Any, cancel: Any = None) -> Any:
        found = self._history.detail(job_id, cancel=cancel)
        for att in found.attempts:
            att.job = self._anon.row(att.job)
            att.workdir = att.raw_workdir = att.stdout = att.stderr = None
            att.reason = self._anon.scrub(att.reason)
            for step in att.steps:
                step.nodes = self._anon.names("node", step.nodes) or []
                step.name = self._anon.job_name(step.name)
        found.message = self._anon.scrub(found.message) or ""
        return found

    def submit_line(self, *args: Any, **kwargs: Any) -> Any:
        from ..history import Text
        return Text(Status.UNAVAILABLE, message=HIDDEN)

    batch_script = submit_line

    def load_efficiency(self, window: Any = None, where: Any = None, **kwargs: Any) -> int:
        return self._history.load_efficiency(window, _translate(self._anon, where), **kwargs)

    def capacity(self) -> Any:
        return self._history.capacity()

    def build_rollup(self, *args: Any, **kwargs: Any) -> Any:
        return self._history.build_rollup(*args, **kwargs)

    def overview(self, *args: Any, **kwargs: Any) -> Any:
        found = self._history.overview(*args, **kwargs)
        anon = self._anon
        for summary in found.ranges:
            accounts = {}
            for name, item in summary.accounts.items():
                item.account = anon.name("account", item.account)
                item.users = {anon.name("user", u): a for u, a in item.users.items()}
                accounts[item.account] = item
            summary.accounts = accounts
        found.notes = [type(n)(n.code, anon.scrub(n.message)) for n in found.notes]
        return found

    def _pseudo_type(self, by: str, key: str) -> str:
        if by == "account":
            return self._anon.name("account", key)
        if by == "name_family":
            return self._anon.name("job", key) or key
        return key

    def _real_type(self, by: str, key: str) -> str:
        if by == "account":
            return self._anon.real("account", key)
        if by == "name_family":
            return self._anon.real("job", key)
        return key

    def run_types(self, window: Any = None, where: Any = None, by: str = "gpu_count",
                  **kwargs: Any) -> Any:
        found = self._history.run_types(window, _translate(self._anon, where), by=by, **kwargs)
        for item in found:
            item.key = self._pseudo_type(by, item.key)
            item.example = ""
        return found

    def events(self, window: Any = None, where: Any = None, **kwargs: Any) -> Any:
        found = self._history.events(window, _translate(self._anon, where), **kwargs)
        anon = self._anon
        for event in found:  # every name an event carries is known before any is scrubbed
            for role, values in (event.where.items() if event.where is not None else ()):
                kind = _ROLE_KIND.get(role)
                for value in values if kind else ():
                    anon.name(kind, value)
            for key, value in event.values.items():
                if key in ("account", "user") and isinstance(value, str):
                    anon.name(key, value)
        for event in found:
            event.sentence = anon.scrub(event.sentence) or ""
            event.values = {}
            if event.where is not None:
                changes = {}
                for role, values in event.where.items():
                    kind = _ROLE_KIND.get(role)
                    if kind:
                        changes[role] = frozenset(anon.name(kind, v) for v in values)
                event.where = event.where.replace(**changes) if changes else event.where
        return found


class AnonCollector:
    """A live.LiveCollector whose snapshots carry pseudonyms and that shows no files."""

    def __init__(self, collector: Any, anon: Anonymizer) -> None:
        self._collector, self._anon = collector, anon

    def snapshot(self) -> Any:
        return self._anon.snapshot(self._collector.snapshot())

    def refresh(self) -> None:
        self._collector.refresh()

    def steps(self, job: Any) -> Any:
        report = self._collector.steps(job)
        for step in report.steps:
            step.nodes = self._anon.names("node", step.nodes) or []
        report.detail = self._anon.scrub(report.detail) or ""
        return report

    def log_paths(self, job: Any) -> Any:
        from ..live import LogPaths
        return LogPaths(Status.UNAVAILABLE, detail=HIDDEN)

    def tail_log(self, *args: Any, **kwargs: Any) -> Any:
        from ..live import LogTail
        return LogTail(Status.UNAVAILABLE, detail=HIDDEN)

    def batch_script(self, job: Any) -> Any:
        from ..live import Fetched
        return Fetched(Status.UNAVAILABLE, detail=HIDDEN)

    submit_line = batch_script


def scrub_all(anon: Optional[Anonymizer], text: str) -> str:
    return clean(anon.scrub(text) if anon else text)
