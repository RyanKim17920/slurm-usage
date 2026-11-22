"""Typed results, and turning exit codes plus stderr into a status."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..model import Status
from ..sanitize import clean

_NOT_AVAILABLE = re.compile(
    r"accounting storage is disabled|job accounting is disabled|unable to contact slurm"
    r"|connection refused|unable to connect|connection timed out|could not resolve"
    r"|socket timed out|invalid field|unrecognized option|invalid option"
    r"|invalid job format specification|invalid authentication credential|munge"
    r"|accounting_storage/none|not running a .*database|unable to register with slurm",
    re.I)
_DENIED = re.compile(
    r"permission denied|access denied|invalid user id|not authorized|unauthori[sz]ed"
    r"|you don't have|you do not have|operation not permitted|only admins|must be an? admin"
    r"|doesn't have privilege|insufficient privilege",
    re.I)
_NOTHING = re.compile(
    r"invalid job id|couldn't get steps|no steps? (?:found|running)|no jobs in the system", re.I)
_FAILED = re.compile(r"\berror:|\bfatal:", re.I)


@dataclass
class Result:
    """Outcome of one command.

    ``status`` is a model.Status. ``ok`` is True for ok, empty and partial.
    ``stdout`` is raw: sanitize anything taken from it before display.
    ``reason`` is already safe to display.
    """

    argv: List[str] = field(default_factory=list)
    status: str = Status.UNKNOWN
    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None
    reason: str = ""
    elapsed: float = 0.0
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.status in Status.USABLE

    def lines(self) -> List[str]:
        # not splitlines(): it also splits on \r, \x1c, U+2028, which users can inject
        lines = self.stdout.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return lines


def classify(returncode: Optional[int], has_output: bool, stderr: str) -> str:
    """Never inspects stdout's content: other users control parts of it."""
    failed = returncode != 0
    if has_output:
        bad = failed or _DENIED.search(stderr) or _FAILED.search(stderr)
        return Status.PARTIAL if bad else Status.OK
    if _NOT_AVAILABLE.search(stderr):
        return Status.UNAVAILABLE
    if _DENIED.search(stderr):
        return Status.DENIED
    if _NOTHING.search(stderr):
        return Status.EMPTY
    return Status.ERROR if failed or _FAILED.search(stderr) else Status.EMPTY


def explain(stderr: str, returncode: Optional[int], argv: Sequence[str]) -> str:
    lines = [line for line in stderr.split("\n") if line.strip()]
    if lines:
        # the line that decided the status is the most useful one
        for pattern in (_NOT_AVAILABLE, _DENIED):
            for line in lines:
                if pattern.search(line):
                    return clean(line.strip(), 300)
        return clean(lines[0].strip(), 300)
    if returncode:
        return "%s exited with status %d" % (argv[0], returncode)
    return ""
