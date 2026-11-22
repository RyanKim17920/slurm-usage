"""Cluster-level text: configuration, partitions, reservations, scheduler ping."""

import re
from typing import Dict, List, Optional

from .. import parse
from ..model import ClusterInfo, Partition, PriorityConfig, Reservation, SchedulerHost
from ..sanitize import clean, clean_id

_PING = re.compile(r"Slurmctld\(([\w/]+)\) at (\S+) (?:is|are) ([A-Z/]+)")


def parse_config(text: str) -> Dict[str, str]:
    config = {}  # type: Dict[str, str]
    for line in text.split("\n"):
        key, separator, value = line.partition("=")
        if separator and re.fullmatch(r"[A-Za-z][\w\[\]/]*", key.strip()):
            config.setdefault(key.strip(), value.strip())
    return config


def name_list(text: Optional[str]) -> Optional[List[str]]:
    """None for ALL or unset."""
    value = parse.optional_text(text)
    if value is None or value.upper() == "ALL":
        return None
    return [clean_id(item) or "" for item in value.split(",") if item]


def cluster_info(config: Dict[str, str]) -> ClusterInfo:
    info = ClusterInfo(name=clean_id(parse.optional_text(config.get("ClusterName"))),
                       version=clean(config.get("SLURM_VERSION")) or None)
    if "SelectTypeParameters" in config:
        info.select_parameters = clean(config["SelectTypeParameters"])
        if info.select_parameters:
            info.mem_tracked = "MEMORY" in info.select_parameters.upper()
    if "AccountingStorageTRES" in config:
        info.gpu_accounted = "gres/gpu" in config["AccountingStorageTRES"].split(",")
    if "AccountingStorageType" in config:
        storage = parse.optional_text(config["AccountingStorageType"])
        info.has_accounting = bool(storage) and not str(storage).endswith("/none")
    age = re.match(r"\d+", config.get("MinJobAge", ""))
    info.min_job_age = int(age.group()) if age else None
    return info


def priority_config(config: Dict[str, str]) -> PriorityConfig:
    result = PriorityConfig(plugin=clean(config.get("PriorityType")) or None)
    for key, value in config.items():
        if key.startswith("PriorityWeight") and key != "PriorityWeightTRES":
            weight = parse.parse_int(value)
            if weight is not None:
                result.weights[key[len("PriorityWeight"):].lower()] = weight
    result.tres_weights = clean(parse.optional_text(config.get("PriorityWeightTRES"))) or None
    half_life = parse.parse_duration(config.get("PriorityDecayHalfLife"))
    result.decay_half_life = int(half_life) if half_life is not None else None
    if result.plugin:
        if result.plugin.endswith("basic"):
            result.fairshare_effective = False
        elif "fairshare" in result.weights:
            result.fairshare_effective = result.weights["fairshare"] != 0
    return result


def parse_partitions(text: str) -> List[Partition]:
    found = {}  # type: Dict[str, Partition]
    for line in text.split("\n"):
        if not line.startswith("PartitionName="):
            continue
        kv = parse.key_values(line)
        name = clean_id(parse.optional_text(kv.get("PartitionName")))
        if not name or name in found:
            continue
        part = Partition(name=name, state=clean(kv.get("State")).upper() or None,
                         nodes=[clean_id(n) or "" for n in parse.expand_hostlist(kv.get("Nodes"))])
        part.is_default = kv["Default"] == "YES" if "Default" in kv else None
        part.hidden = kv["Hidden"] == "YES" if "Hidden" in kv else None
        part.max_time_unlimited = parse.is_unlimited(kv.get("MaxTime"))
        for attr, key in (("max_time", "MaxTime"), ("default_time", "DefaultTime")):
            seconds = parse.parse_duration(kv.get(key))
            setattr(part, attr, int(seconds) if seconds is not None else None)
        part.qos = clean_id(parse.optional_text(kv.get("QoS")))
        part.allow_accounts = name_list(kv.get("AllowAccounts"))
        part.deny_accounts = name_list(kv.get("DenyAccounts"))
        part.allow_qos = name_list(kv.get("AllowQos"))
        part.deny_qos = name_list(kv.get("DenyQos"))
        part.allow_groups = name_list(kv.get("AllowGroups"))
        part.preempt_mode = clean(kv.get("PreemptMode")) or None
        part.priority_tier = parse.parse_int(kv.get("PriorityTier"))
        per_gpu = re.search(r"(?:^|,)DefCpuPerGPU=(\d+)", kv.get("JobDefaults", ""))
        part.default_cpus_per_gpu = int(per_gpu.group(1)) if per_gpu else None
        part.select_parameters = clean(parse.optional_text(kv.get("SelectTypeParameters"))) or None
        part.tres = {clean(k): v for k, v in parse.parse_tres(kv.get("TRES")).items()}
        found[name] = part
    return list(found.values())


def parse_reservations(text: str) -> List[Reservation]:
    result = []
    for line in text.split("\n"):
        if not line.startswith("ReservationName="):
            continue
        kv = parse.key_values(line)
        name = clean_id(parse.optional_text(kv.get("ReservationName")))
        if not name:
            continue
        result.append(Reservation(
            name=name, state=clean(kv.get("State")) or None,
            start=parse.parse_time(kv.get("StartTime")), end=parse.parse_time(kv.get("EndTime")),
            nodes=[clean_id(n) or "" for n in parse.expand_hostlist(kv.get("Nodes"))],
            partition=clean_id(parse.optional_text(kv.get("PartitionName"))),
            users=name_list(kv.get("Users")) or [],
            accounts=name_list(kv.get("Accounts")) or [],
            flags=name_list(kv.get("Flags")) or [],
            tres={clean(k): v for k, v in parse.parse_tres(kv.get("TRES")).items()}))
    return result


def parse_ping(text: str) -> List[SchedulerHost]:
    hosts = []
    for roles, names, states in _PING.findall(text):
        # old versions put both controllers on one line: (primary/backup) at a/b are UP/DOWN
        for role, host, state in zip(roles.split("/"), names.split("/"), states.split("/")):
            up = True if state == "UP" else False if state == "DOWN" else None
            hosts.append(SchedulerHost(clean(role), clean(host), up))
    return hosts
