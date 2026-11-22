"""Query status, per-invoker capabilities and who the invoker is."""

from dataclasses import dataclass, field
from typing import List, Optional


class Status:
    """String constants used by runner results and capabilities."""

    OK = "ok"                    # answered, with data
    EMPTY = "empty"              # answered, legitimately nothing
    PARTIAL = "partial"          # some data, some refused, failed or truncated
    DENIED = "denied"            # Slurm or the OS refused this invoker
    UNAVAILABLE = "unavailable"  # not installed, configured, reachable or supported
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"      # the caller stopped reading a stream
    UNKNOWN = "unknown"          # not attempted yet
    NOT_STORED = "not_stored"    # answered: Slurm keeps no copy of this (e.g. a batch script)

    USABLE = frozenset((OK, EMPTY, PARTIAL))


@dataclass
class Capability:
    """Whether one data source works for this invoker, and why not if it doesn't."""

    name: str
    status: str = Status.UNKNOWN
    detail: str = ""
    source: str = ""  # the command, for display
    elapsed: Optional[float] = None

    @property
    def available(self) -> bool:
        return self.status in Status.USABLE


@dataclass
class Invoker:
    """The person running the tool. Slurm, not this record, decides what is allowed."""

    user: Optional[str] = None
    uid: Optional[int] = None
    euid: Optional[int] = None
    admin_level: Optional[str] = None  # "None", "Operator", "Administrator"; None = unknown
    coordinator_accounts: Optional[List[str]] = None
    default_account: Optional[str] = None
    is_slurm_user: Optional[bool] = None
    private_data: Optional[List[str]] = None  # PrivateData of the cluster

    @property
    def privileged(self) -> Optional[bool]:
        """Root, SlurmUser, operator or admin. A display hint, never an access check."""
        if self.euid == 0 or self.is_slurm_user:
            return True
        if self.admin_level is None:
            return None
        return self.admin_level.lower() not in ("none", "")

    @property
    def kind(self) -> str:
        if self.euid == 0:
            return "root"
        if self.is_slurm_user:
            return "SlurmUser"
        if self.privileged:
            return str(self.admin_level).lower()
        if self.coordinator_accounts:
            return "coordinator"
        return "user" if self.admin_level is not None else "unknown"
