"""TRES strings and GRES strings."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .hostlist import parse_ranges, split_outside
from .units import optional_text, parse_count, parse_duration, parse_float, parse_memory

_BYTE_KEYS = frozenset(("fs/disk", "fs/lustre", "gres/gpumem"))
_NOTE = re.compile(r"\(([^)]*)\)")
_SHARE = re.compile(r"(\d+)/(\d+)")
_GRES_FLAGS = frozenset(("no_consume",))


def split_tres(text: Optional[str]) -> List[Tuple[str, str]]:
    """Raw (name, value) pairs, in order."""
    value = optional_text(text)
    pairs = []
    for item in split_outside(value) if value else ():
        key, separator, amount = item.partition("=")
        if separator:
            pairs.append((key.strip(), amount.strip()))
    return pairs


def parse_tres(text: Optional[str], mem_unit: str = "M") -> Dict[str, float]:
    """{name: number}. Memory is bytes; unparsable values are omitted, not zero."""
    result = {}  # type: Dict[str, float]
    for key, raw in split_tres(text):
        if key in ("mem", "vmem"):
            number = parse_memory(raw, mem_unit)
        elif key in _BYTE_KEYS:
            number = parse_memory(raw, "")
        elif ":" in raw:
            number = parse_duration(raw)
        else:
            number = parse_count(raw)
        if number is not None:
            result[key] = number
    return result


def parse_usage_tres(text: Optional[str]) -> Dict[str, float]:
    """TRESUsage* from sstat/sacct: bare mem is bytes and cpu is seconds."""
    result = parse_tres(text, mem_unit="")
    for key, raw in split_tres(text):
        if key == "cpu":
            seconds = parse_duration(raw) if ":" in raw else parse_float(raw)
            if seconds is None:
                result.pop(key, None)
            else:
                result[key] = seconds
    return result


def gpus_from_tres(tres: Dict[str, float]) -> Tuple[Optional[int], Dict[str, int]]:
    """(total, by type). gres/gpu already includes gres/gpu:type, so never add both.

    total is None if no GPU key is present: only the caller knows whether that
    means zero (GPUs are a tracked TRES) or unknown (they are not).
    """
    types = {key[len("gres/gpu:"):]: int(value) for key, value in tres.items()
             if key.startswith("gres/gpu:")}
    if "gres/gpu" in tres:
        return int(tres["gres/gpu"]), types
    return (sum(types.values()) if types else None), types


@dataclass
class Gres:
    name: str
    count: Optional[int]
    type: Optional[str] = None
    indices: Optional[List[int]] = None
    shares: List[Tuple[int, int]] = field(default_factory=list)  # shard/MPS used/total per GPU
    raw: str = ""


def parse_gres(text: Optional[str]) -> List[Gres]:
    """Node Gres/GresUsed, job GRES=, TresPerNode.

    Accepts gpu, gpu:2, gpu:type:2, gres/gpu:type=2, gres:gpu:2, MIG types,
    (S:0-1) and (IDX:0,2-3) annotations, shard (used/total,...) annotations.
    """
    value = optional_text(text)
    result = []
    for raw in split_outside(value) if value else ():
        notes = _NOTE.findall(raw.replace("(null)", ""))
        body = _NOTE.sub("", raw.replace("(null)", "")).strip()
        if body.startswith("gres/") or body.startswith("gres:"):
            body = body[5:]
        body, has_count, amount = body.partition("=")
        parts = [part for part in body.split(":") if part and part.lower() not in _GRES_FLAGS]
        if not parts or parts[0].lower() in ("none", "n/a"):
            continue
        if has_count:
            count = parse_count(amount)
        elif len(parts) > 1 and parse_count(parts[-1]) is not None:
            count = parse_count(parts.pop())
        else:
            count = 1  # a bare request such as "gpu" or "gpu:type" means one
        entry = Gres(name=parts[0], count=count, type=":".join(parts[1:]) or None, raw=raw)
        for note in notes:
            if note.startswith("IDX:"):
                entry.indices = [] if entry.count == 0 else parse_ranges(note[4:])
            elif "/" in note:
                entry.shares = [(int(used), int(total)) for used, total in _SHARE.findall(note)]
        result.append(entry)
    return result


def gpus_from_gres(entries: List[Gres]) -> Tuple[Optional[int], Dict[str, int]]:
    """(total, by type) without counting a generic total on top of its typed parts.

    None means a GPU count was unreadable; no GPU entries at all gives 0, so
    only pass entries from a field that was actually present.
    """
    generic, types = 0, {}  # type: int, Dict[str, int]
    for entry in entries:
        if entry.name != "gpu":
            continue
        if entry.count is None:
            return None, {}
        if entry.type:
            types[entry.type] = types.get(entry.type, 0) + entry.count
        else:
            generic += entry.count
    typed = sum(types.values())
    # Slurm does not let a node mix typed and untyped GPUs, so both means total + breakdown.
    return max(generic, typed), types


def gpu_indices(entries: List[Gres]) -> Optional[List[int]]:
    """Union of IDX lists; None unless every allocated GPU entry has one."""
    found = []  # type: List[int]
    seen = False
    for entry in entries:
        if entry.name != "gpu" or entry.count == 0:
            continue
        if entry.indices is None:
            return None
        seen = True
        found.extend(entry.indices)
    return sorted(set(found)) if seen else None
