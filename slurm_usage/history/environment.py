"""What this cluster records and what this invoker may see of it.

slurmdbd's own PrivateData is not read (the runner does not allow dumping the
database configuration). Visibility comes from evidence: the invoker's admin
level, coordinated accounts, and whether accounting returns anyone else's uid.
"""

import os
import pwd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..model import CapacityCheck, Invoker, Status
from ..parse import (gpus_from_gres, gpus_from_tres, key_values, parse_gres, parse_int,
                     parse_time, parse_tres, split_outside)
from ..sanitize import clean
from .errors import Note
from .records import ALLOCATION_FIELDS, REQUIRED_FIELDS

ALL, RESTRICTED, VISIBILITY_UNKNOWN = "all", "restricted", "unknown"
_NO_ACCOUNTING = ("", "(null)", "accounting_storage/none")


@dataclass
class Environment:
    invoker: Invoker
    cluster: Optional[str] = None
    accounting: str = Status.UNKNOWN
    accounting_detail: str = ""
    visibility: str = VISIBILITY_UNKNOWN  # all, restricted or unknown
    live_shows_everyone: bool = False  # whether absence from squeue proves a job is gone
    gpu_tracked: Optional[bool] = None
    fields: Optional[frozenset] = None  # what this sacct can print; None = could not ask
    notes: List[Note] = field(default_factory=list)
    identity: Tuple[str, ...] = ()

    def supported(self, wanted: Tuple[str, ...]) -> Tuple[str, ...]:
        if self.fields is None:
            return wanted
        return tuple(name for name in wanted if name.lower() in self.fields)


def _pairs(lines: List[str]) -> Dict[str, str]:
    found = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            found.setdefault(key.strip(), value.strip())
    return found


def _private(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    if value.lower() in ("", "none", "(null)"):
        return []
    return [part.strip().lower() for part in value.split(",")]


def probe(runner: Any) -> Environment:
    euid = os.geteuid()
    try:
        user = pwd.getpwuid(euid).pw_name  # type: Optional[str]
    except KeyError:
        user = None
    invoker = Invoker(user=user, uid=euid, euid=euid)
    env = Environment(invoker)

    config_result = runner.run(["scontrol", "show", "config"])
    config = _pairs(config_result.lines()) if config_result.ok else {}
    env.cluster = clean(config.get("ClusterName")) or None
    storage = config.get("AccountingStorageType")
    tres = config.get("AccountingStorageTRES")
    if tres:
        env.gpu_tracked = any(name == "gres/gpu" or name.startswith("gres/gpu:")
                              for name in split_outside(tres.lower()))
    slurm_private = _private(config.get("PrivateData"))
    invoker.private_data = slurm_private

    formats = runner.run(["sacct", "--helpformat"])
    known = frozenset(formats.stdout.lower().split()) if formats.ok else frozenset()
    env.fields = known if "jobidraw" in known else None
    if formats.status == Status.UNAVAILABLE:
        env.accounting, env.accounting_detail = Status.UNAVAILABLE, formats.reason
    elif storage is not None and storage.lower() in _NO_ACCOUNTING:
        env.accounting = Status.UNAVAILABLE
        env.accounting_detail = "this cluster does not store job accounting"
    elif env.fields is not None and len(env.supported(REQUIRED_FIELDS)) < len(REQUIRED_FIELDS):
        env.accounting = Status.UNAVAILABLE
        env.accounting_detail = "this sacct is too old: it cannot print job ids and times"
    if env.accounting == Status.UNAVAILABLE:
        env.notes.append(Note("no_accounting", "no job history: %s" % env.accounting_detail))
        return env

    admin_line = ""
    if user and not user.startswith("-"):
        found = runner.run(["sacctmgr", "-n", "-P", "show", "user", user, "withcoord",
                            "format=user,adminlevel,coordinators"])
        for line in found.lines() if found.ok else ():
            name, _, rest = line.partition("|")
            if name == user:
                admin_line = rest
                level, _, coordinated = rest.partition("|")
                invoker.admin_level = clean(level)
                invoker.coordinator_accounts = [clean(a) for a in coordinated.split(",") if a]
                break

    slurm_user = config.get("SlurmUser", "")
    if slurm_user:
        name, _, number = slurm_user.partition("(")
        invoker.is_slurm_user = (number.rstrip(")") == str(euid) if number else name == user)

    others = None  # type: Optional[bool]
    if "uid" in (env.fields or ("uid",)):
        others = visible_others(runner, euid)

    hides_jobs = "jobs" in (slurm_private or ())
    if invoker.privileged:
        env.visibility = ALL
    elif others and invoker.coordinator_accounts == []:
        env.visibility = ALL  # only hiding jobs from everyone else could have prevented this
    elif hides_jobs:
        env.visibility = RESTRICTED
    env.live_shows_everyone = bool(invoker.privileged or (slurm_private is not None
                                                          and "jobs" not in slurm_private))

    if env.visibility == RESTRICTED:
        env.notes.append(Note("restricted", "Slurm limits job history to what you may see: "
                                            "totals are not cluster totals"))
    elif env.visibility == VISIBILITY_UNKNOWN:
        env.notes.append(Note("visibility_unknown", "cannot tell whether Slurm shows you every "
                                                    "job: totals may not be cluster totals"))
    if env.fields is not None:
        missing = {"DBIndex", "AllocTRES", "Suspended"} - set(env.supported(ALLOCATION_FIELDS))
        if missing:
            env.notes.append(Note("old_sacct", "this sacct lacks %s: history is less exact"
                                  % ", ".join(sorted(missing))))
    env.identity = (env.cluster or os.environ.get("SLURM_CONF", ""),
                    config.get("AccountingStorageHost", ""), str(euid), env.visibility,
                    str(others), ",".join(sorted(slurm_private or ())), admin_line)
    return env


def visible_others(runner: Any, euid: int) -> Optional[bool]:
    """Whether accounting shows this invoker anyone else's jobs. Asks about the last hour
    only, and stops at the first other user, so a busy cluster costs one short read."""
    argv = ["sacct", "-a", "-X", "-n", "-P", "-S", "now-1hours", "-E", "now", "-o", "UID"]
    with runner.stream(argv, timeout=30) as lines:
        for line in lines:
            line = line.strip()
            if line.isascii() and line.isdigit() and int(line) != euid:
                return True
    result = lines.result
    return False if result is not None and result.status in (Status.OK, Status.EMPTY) else None


def find_origin(runner: Any) -> Optional[int]:
    """When accounting began: the first cluster event or registration still on record."""
    found = []
    for argv in (["sacctmgr", "-n", "-P", "show", "event", "event=cluster",
                  "start=1970-01-02", "format=start"],
                 ["sacctmgr", "-n", "-P", "show", "transaction", "action=Add Clusters",
                  "format=time"]):
        result = runner.run(argv)
        for line in result.lines() if result.ok else ():
            when = parse_time(line.partition("|")[0])
            if when:
                found.append(when)
    return min(found) if found else None


def capacity(runner: Any, cluster: Optional[str]) -> List[CapacityCheck]:
    """Physical capacity now beside slurmdbd's belief, which is often wrong. Use ``live``."""
    believed = {}  # type: Dict[str, float]
    reported = runner.run(["sacctmgr", "-n", "-P", "show", "cluster", "format=cluster,tres"])
    if reported.ok:
        rows = [line.partition("|") for line in reported.lines() if "|" in line]
        for name, _, text in rows:
            if name == cluster or (cluster is None and len(rows) == 1):
                believed = parse_tres(text.rstrip("|"))

    cpus = gpus = None  # type: Optional[int]
    shown = runner.run(["scontrol", "-o", "show", "node"])
    if shown.status == Status.OK:
        nodes = {}
        for line in shown.lines():
            values = key_values(line.partition(" Reason=")[0])
            if values.get("NodeName"):
                nodes[values["NodeName"]] = values
        if nodes:
            cpu_counts, gpu_counts = [], []  # type: List[Optional[int]], List[Optional[int]]
            for values in nodes.values():
                cfg = parse_tres(values.get("CfgTRES"))
                cpu = cfg.get("cpu")
                cpu_counts.append(int(cpu) if cpu is not None
                                  else parse_int(values.get("CPUTot")))
                gpu, _ = gpus_from_tres(cfg)
                if gpu is None and "Gres" in values:
                    gpu, _ = gpus_from_gres(parse_gres(values["Gres"]))
                gpu_counts.append(gpu)
            cpus = None if None in cpu_counts else sum(cpu_counts)
            gpus = None if None in gpu_counts else sum(gpu_counts)

    believed_gpus, _ = gpus_from_tres(believed)
    believed_cpus = believed.get("cpu")
    return [CapacityCheck("gres/gpu", gpus, believed_gpus),
            CapacityCheck("cpu", cpus, int(believed_cpus) if believed_cpus is not None else None)]


PROBE_TTL = 300.0  # seconds a probe of this cluster is reused by later runs
_PROBE_FILE = "probe-v1.json"


def _probe_key() -> List[str]:
    import socket
    return [str(os.geteuid()), os.environ.get("SLURM_CONF", ""), socket.gethostname()]


def _to_json(env: Environment) -> Dict[str, Any]:
    import dataclasses
    out = dataclasses.asdict(env)
    out["fields"] = sorted(env.fields) if env.fields is not None else None
    out["notes"] = [list(n) for n in env.notes]
    out["identity"] = list(env.identity)
    return out


def _from_json(found: Dict[str, Any]) -> Environment:
    values = dict(found)
    values["invoker"] = Invoker(**values["invoker"])
    values["fields"] = frozenset(values["fields"]) if values["fields"] is not None else None
    values["notes"] = [Note(*n) for n in values["notes"]]
    values["identity"] = tuple(values["identity"])
    return Environment(**values)


def cached_probe(runner: Any, directory: Optional[str], clock: Any = None) -> Environment:
    """probe(), reused for PROBE_TTL seconds from an owner-only file in the private cache
    directory (``directory`` None: do not keep it). What it found is from Slurm and a
    few minutes old at most; accounting that was unavailable is asked again each time."""
    import json
    import time
    now = (clock or time.time)()
    path = None
    if directory is not None:
        try:
            from .store import private_dir
            path = os.path.join(private_dir(directory), _PROBE_FILE)
            handle = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                found = os.fstat(handle)
                if found.st_uid == os.geteuid() and not found.st_mode & 0o077:
                    with os.fdopen(os.dup(handle), "r", encoding="utf-8") as stream:
                        kept = json.load(stream)
                    if (kept.get("key") == _probe_key()
                            and 0 <= now - float(kept.get("at", 0)) < PROBE_TTL):
                        return _from_json(kept["env"])
            finally:
                os.close(handle)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    env = probe(runner)
    if path is not None and env.accounting != Status.UNAVAILABLE:
        try:
            temporary = path + ".%d" % os.getpid()
            handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                             0o600)
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump({"key": _probe_key(), "at": now, "env": _to_json(env)}, stream)
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError):
            pass
    return env
