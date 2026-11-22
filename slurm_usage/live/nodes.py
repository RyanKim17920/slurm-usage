"""``scontrol -d -o show node``: each physical node once."""

import re
from typing import Dict, List, Optional, Tuple

from .. import parse
from ..model import Availability, ClusterInfo, Node, Partition
from ..sanitize import clean, clean_id

_SUFFIX = {"*": "NOT_RESPONDING", "~": "POWERED_DOWN", "#": "POWERING_UP",
           "!": "POWER_DOWN", "%": "POWERING_DOWN", "$": "MAINTENANCE",
           "@": "REBOOT_REQUESTED", "^": "REBOOT_ISSUED", "-": "PLANNED"}
_UNAVAILABLE = frozenset((
    "DOWN", "DRAIN", "DRAINED", "DRAINING", "FAIL", "FAILING", "ERROR", "UNKNOWN",
    "NOT_RESPONDING", "NO_RESPOND", "MAINT", "MAINTENANCE", "REBOOT_ISSUED", "POWERING_DOWN",
    "POWER_DOWN", "INVALID_REG", "INVAL", "INVALID", "BLOCKED", "FUTURE"))
_WHO_WHEN = re.compile(r"\[([^\[\]@\n]+)@([^\[\]@\n]+)\]")


def split_state(text: Optional[str]) -> Tuple[Optional[str], List[str]]:
    value = parse.optional_text(text)
    if value is None:
        # "UNKNOWN" is a real node state, not a missing value
        return ("UNKNOWN", []) if (text or "").strip().upper() == "UNKNOWN" else (None, [])
    words, flags = [], []  # type: List[str], List[str]
    for part in value.upper().split("+"):
        while part and part[-1] in _SUFFIX:
            flags.append(_SUFFIX[part[-1]])
            part = part[:-1]
        if part:
            words.append(clean(part))
    if not words:
        return None, flags
    return words[0], list(dict.fromkeys(words[1:] + flags))


def is_future(node: Node) -> bool:
    return node.state == "FUTURE" or "FUTURE" in node.state_flags


def _records(text: str) -> List[List[str]]:
    records = []  # type: List[List[str]]
    for line in text.split("\n"):
        if line.startswith("NodeName=") and " State=" in line:
            records.append([line])
        elif records and line:
            records[-1].append(line)  # continuation of a multi-line reason
    return records


def parse_nodes(text: str, cluster: ClusterInfo,
                partitions: Optional[List[Partition]] = None) -> List[Node]:
    found = {}  # type: Dict[str, Node]
    by_name = {part.name: part for part in partitions or ()}
    member_of = {}  # type: Dict[str, List[str]]
    for part in partitions or ():
        for host in part.nodes:
            member_of.setdefault(host, []).append(part.name)
    for record in _records(text):
        node = _node(record, cluster, by_name, member_of)
        if node is not None and node.name not in found:
            found[node.name] = node
    return sorted(found.values(), key=lambda node: natural_key(node.name))


def natural_key(text: str) -> list:
    # re.split alternates text and digits, so positions always compare like with like
    return [int(part) if index % 2 else part
            for index, part in enumerate(re.split(r"([0-9]+)", text))]


def _node(record: List[str], cluster: ClusterInfo, by_name: Dict[str, Partition],
          member_of: Dict[str, List[str]]) -> Optional[Node]:
    first, has_reason, reason_text = record[0].partition(" Reason=")
    kv = parse.key_values(first)  # nothing free-form before Reason decides a number
    name = clean_id(parse.optional_text(kv.get("NodeName")))
    if not name:
        return None
    node = Node(name=name)
    node.state, node.state_flags = split_state(kv.get("State"))
    listed = parse.optional_text(kv.get("Partitions"))
    node.partitions = list(dict.fromkeys(
        [clean_id(p) or "" for p in listed.split(",") if p] if listed else member_of.get(name, [])))

    node.cpus_total = parse.parse_int(kv.get("CPUTot"))
    node.cpus_effective = parse.parse_int(kv.get("CPUEfctv"))
    node.cpus_alloc = parse.parse_int(kv.get("CPUAlloc"))
    node.threads_per_core = parse.parse_int(kv.get("ThreadsPerCore"))
    node.cpu_load = parse.parse_float(kv.get("CPULoad"))
    node.boot_time = parse.parse_time(kv.get("BootTime"))
    node.last_busy = parse.parse_time(kv.get("LastBusyTime"))
    features = parse.optional_text(kv.get("AvailableFeatures"))
    node.features = [clean(item) for item in features.split(",")] if features else []

    _memory(node, kv, cluster, by_name)
    _gpus(node, kv, cluster)
    _availability(node, by_name)

    defaults = {by_name[p].default_cpus_per_gpu for p in node.partitions
                if p in by_name and by_name[p].default_cpus_per_gpu}
    if node.gpus_total and len(defaults) == 1:
        node.default_cpus_per_gpu = defaults.pop()

    if has_reason:
        reason = "\n".join([reason_text] + record[1:])
        stamp = None
        for stamp in _WHO_WHEN.finditer(reason):
            pass  # the last one is Slurm's own
        if stamp:
            node.reason_user = clean(stamp.group(1))
            node.reason_time = parse.parse_time(stamp.group(2))
            reason = reason[:stamp.start()]
        node.reason = clean(parse.optional_text(reason)) or None
    return node


def _memory(node: Node, kv: Dict[str, str], cluster: ClusterInfo,
            by_name: Dict[str, Partition]) -> None:
    node.mem_total = parse.parse_memory(kv.get("RealMemory"))
    node.mem_os_free = parse.parse_memory(kv.get("FreeMem"))
    reserved = parse.parse_memory(kv.get("MemSpecLimit"))
    if node.mem_total is not None:
        node.mem_schedulable = max(0, node.mem_total - (reserved or 0))
    tracked = cluster.mem_tracked
    for part in node.partitions:
        override = by_name[part].select_parameters if part in by_name else None
        if override and "MEMORY" in override.upper():
            tracked = True
    allocated = parse.parse_memory(kv.get("AllocMem"))
    # AllocMem=0 without CR_*_MEMORY means "not tracked", not "nothing allocated"
    if tracked or (tracked is None and allocated):
        node.mem_alloc = allocated


def _gpus(node: Node, kv: Dict[str, str], cluster: ClusterInfo) -> None:
    node.tres = {clean(k): v for k, v in parse.parse_tres(kv.get("CfgTRES")).items()}
    node.tres_alloc = {clean(k): v for k, v in parse.parse_tres(kv.get("AllocTRES")).items()}
    configured = parse.parse_gres(kv.get("Gres"))
    used = parse.parse_gres(kv.get("GresUsed"))

    total, types = parse.gpus_from_tres(node.tres)
    accounted = total is not None or bool(cluster.gpu_accounted)
    if "Gres" in kv:
        gres_total, gres_types = parse.gpus_from_gres(configured)
        types = gres_types or types
        if total is None:
            total = gres_total
    elif total is None and cluster.gpu_accounted and node.tres:
        total = 0
    node.gpus_total = total
    node.gpu_types = {clean(k): v for k, v in types.items()}
    if total == 0:
        node.gpus_alloc = node.gpus_shared = 0
        node.gpu_types_alloc = {}
        return

    alloc, alloc_types = parse.gpus_from_tres(node.tres_alloc)
    if alloc is None:
        if accounted and "AllocTRES" in kv:
            alloc = 0  # tracked and absent from AllocTRES: none allocated
        elif "GresUsed" in kv:
            alloc = parse.gpus_from_gres(used)[0]
    if "GresUsed" in kv:
        alloc_types = parse.gpus_from_gres(used)[1] or alloc_types
        node.gpu_indices_alloc = parse.gpu_indices(used)
    node.gpus_alloc = alloc
    if alloc == 0:
        node.gpu_types_alloc = {}
        node.gpu_indices_alloc = []
    elif alloc is not None and sum(alloc_types.values()) == alloc:
        node.gpu_types_alloc = {clean(k): v for k, v in alloc_types.items() if v}
    node.gpus_shared = _shared(kv, configured, used)


def _shared(kv: Dict[str, str], configured: list, used: list) -> Optional[int]:
    """GPUs unusable as whole GPUs: partly handed out as shard/MPS, or GRES-drained."""
    drained = parse.parse_gres(kv.get("GresDrain"))
    if any(g.name == "gpu" and g.count for g in drained):
        return None  # drained GPUs may overlap allocated ones, so free is unknown
    if not any(g.name in ("shard", "mps") for g in configured + used):
        return 0
    if "GresUsed" not in kv:
        return None
    busy = 0
    for gres in used:
        if gres.name in ("shard", "mps") and gres.count != 0:
            if not gres.shares:
                return None
            busy += sum(1 for in_use, _ in gres.shares if in_use)
    return busy


def _availability(node: Node, by_name: Dict[str, Partition]) -> None:
    if node.state is None:
        return
    words = [node.state] + node.state_flags
    node.unavailable_because = [word for word in words if word in _UNAVAILABLE]
    if by_name and not node.unavailable_because:
        states = [by_name[p].state for p in node.partitions if p in by_name]
        if not node.partitions or (states and all(s not in (None, "UP") for s in states)):
            node.unavailable_because = ["NO_UP_PARTITION"]
    if node.unavailable_because:
        node.availability = Availability.UNAVAILABLE
    elif "RESERVED" in words:
        node.availability = Availability.RESERVED
    else:
        node.availability = Availability.SCHEDULABLE
