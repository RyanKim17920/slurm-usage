"""Roles (associations), QOS and limits."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Limit:
    """One limit and/or its usage, in seconds, bytes or plain counts.

    ``limit is None`` with ``unlimited`` means no limit; without it, unknown.
    Metrics named cpu, gres/gpu etc. under a *Mins limit are TRES-seconds
    (mem: byte-seconds).
    """

    scope: str  # assoc, qos, qos_account, qos_user
    name: str  # Slurm's name, e.g. GrpTRES, MaxTRESPU, MaxWallPJ
    metric: Optional[str] = None  # TRES name for TRES limits
    limit: Optional[float] = None
    used: Optional[float] = None
    unlimited: bool = False
    suspect: bool = False  # impossible value reported by Slurm; used is None
    raw: str = ""
    account: Optional[str] = None
    user: Optional[str] = None
    qos: Optional[str] = None
    partition: Optional[str] = None
    source: str = ""

    @property
    def fraction(self) -> Optional[float]:
        if self.used is None or not self.limit:
            return None
        return self.used / self.limit

    @property
    def remaining(self) -> Optional[float]:
        if self.used is None or self.limit is None:
            return None
        return max(0, self.limit - self.used)


@dataclass
class Role:
    """An association: a user (None = the account itself) in an account."""

    account: str
    user: Optional[str] = None
    uid: Optional[int] = None
    partition: Optional[str] = None  # only for partition-specific associations
    qos_allowed: Optional[List[str]] = None
    default_qos: Optional[str] = None
    is_default: Optional[bool] = None
    lineage: List[str] = field(default_factory=list)  # accounts, outermost first
    shares_raw: Optional[int] = None
    shares_norm: Optional[float] = None
    usage_raw: Optional[float] = None  # decayed usage, not a historical total
    usage_effective: Optional[float] = None
    fairshare_factor: Optional[float] = None
    limits: List[Limit] = field(default_factory=list)


@dataclass
class Qos:
    name: str
    priority: Optional[int] = None
    preempt_mode: Optional[str] = None
    usage_raw: Optional[float] = None
    limits: List[Limit] = field(default_factory=list)
