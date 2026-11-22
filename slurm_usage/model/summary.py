"""Derived values: totals, grouped usage, health, priority configuration."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .node import Node, Reservation


@dataclass
class Resource:
    """Capacity split into disjoint buckets.

    total = alloc + completing + shared + free + unavailable + reserved, when all known.
    alloc is the nodes' own figure; completing is what finishing jobs still hold
    beyond it. Jobs' allocations therefore add up to alloc + completing.
    """

    total: Optional[int] = None
    alloc: Optional[int] = None
    free: Optional[int] = None
    unavailable: Optional[int] = None  # unallocated, but not usable
    reserved: Optional[int] = None
    shared: Optional[int] = None  # GPUs held through shard/MPS
    completing: Optional[int] = None  # held by finishing jobs, see above


@dataclass
class Totals:
    cpus: Resource = field(default_factory=Resource)
    mem: Resource = field(default_factory=Resource)
    gpus: Resource = field(default_factory=Resource)
    gpu_types: Dict[str, Resource] = field(default_factory=dict)
    nodes: Optional[int] = None
    nodes_schedulable: Optional[int] = None


@dataclass
class Usage:
    """Jobs aggregated under one key. A sum with an unknown addend is None."""

    key: object
    running: int = 0  # records holding resources
    pending: int = 0  # tasks, counting each array element
    other: int = 0  # suspended or staging out
    cpus: Optional[int] = None
    gpus: Optional[int] = None
    mem: Optional[int] = None
    pending_cpus: Optional[int] = None
    pending_gpus: Optional[int] = None
    pending_mem: Optional[int] = None


@dataclass
class Warning_:
    code: str  # stable, e.g. capacity_mismatch
    message: str
    values: Dict[str, object] = field(default_factory=dict)


@dataclass
class CapacityCheck:
    metric: str
    live: Optional[int] = None  # deduplicated nodes: authoritative
    accounting: Optional[int] = None  # slurmdbd's belief, which can be wrong

    @property
    def mismatch(self) -> bool:
        return None not in (self.live, self.accounting) and self.live != self.accounting


@dataclass
class SchedulerHost:
    role: str
    host: str
    up: Optional[bool] = None


@dataclass
class LoadOutlier:
    node: str
    kind: str  # load_above_alloc, load_below_alloc
    load: float
    cpus_alloc: int


@dataclass
class GpuBlock:
    node: str
    gpus: int
    cpus_free: Optional[int] = None
    mem_free: Optional[int] = None


@dataclass
class Health:
    problem_nodes: List[Node] = field(default_factory=list)
    node_states: Dict[str, int] = field(default_factory=dict)
    reservations: Optional[List[Reservation]] = None  # None = could not ask
    scheduler: Optional[List[SchedulerHost]] = None
    pending_by_reason: List[Usage] = field(default_factory=list)
    free_gpus_by_node: List[Tuple[str, Optional[int]]] = field(default_factory=list)
    largest_gpu_block: Optional[GpuBlock] = None  # startable on one node
    largest_gpu_block_by_type: Dict[str, GpuBlock] = field(default_factory=dict)
    gpus_stranded_cpu: Optional[int] = None
    gpus_stranded_mem: Optional[int] = None
    load_outliers: List[LoadOutlier] = field(default_factory=list)


@dataclass
class PriorityConfig:
    plugin: Optional[str] = None
    weights: Dict[str, int] = field(default_factory=dict)  # fairshare, age, qos, ...
    tres_weights: Optional[str] = None
    decay_half_life: Optional[int] = None  # seconds
    fairshare_effective: Optional[bool] = None  # False = no effect on priority here
