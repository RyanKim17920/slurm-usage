"""The only place this program starts a process. Read-only by allowlist.

    runner = Runner()
    result = runner.run(["scontrol", "-o", "show", "node"])
    if result.ok: ...  # else result.status and result.reason say why

    with runner.stream(["sacct", "-n", "-P", ...]) as rows:
        for line in rows: ...
    rows.result.status

* argv lists only, no shell. Anything outside ``policy`` raises
  ReadOnlyViolation before a process exists: that is a programming error, not
  a runtime condition.
* Slurm decides what the invoker may see; a refusal is status ``denied`` and is
  never retried another way. There is no privilege change of any kind.
* Failed commands never raise. Statuses: ok, empty, partial, denied,
  unavailable, timeout, error, cancelled (model.Status).
* Children get LC_ALL=C, SLURM_TIME_FORMAT=%s (so timestamps are epoch seconds:
  read them with parse.parse_time) and SLURM_BITSTR_LEN=0 (array task lists are
  not truncated). Output-shaping variables such as SQUEUE_FORMAT are removed,
  as is SLURM_CLUSTERS: the local cluster is always the one described.
* stdout is decoded with surrogateescape and is not sanitized.

For tests, pass ``search_path`` pointing at fake executables, or replace the
Runner with any object that has ``run`` and ``stream``.
"""

import os
from typing import Dict, Mapping, Optional, Sequence

from ..model import Status
from .policy import ReadOnlyViolation, check, describe
from .process import Stream, child_environment, find_program
from .result import Result, classify

__all__ = ["ReadOnlyViolation", "Result", "Runner", "Status", "Stream", "check", "classify",
           "describe"]

RUN_MAX_BYTES = 256 * 1024 * 1024


class Runner:
    def __init__(self, timeout: float = 20.0, stream_timeout: float = 300.0,
                 search_path: Optional[str] = None,
                 environ: Optional[Mapping[str, str]] = None) -> None:
        base = os.environ if environ is None else environ
        self.timeout = timeout
        self.stream_timeout = stream_timeout
        self._path = base.get("PATH", os.defpath) if search_path is None else search_path
        self._env = child_environment(base)
        self._found = {}  # type: Dict[str, str]

    def _locate(self, program: str) -> Optional[str]:
        # cached: walking a PATH on a network filesystem costs more than the queries
        path = self._found.get(program)
        if path is None:
            path = find_program(program, self._path)
            if path is not None:
                self._found[program] = path
        return path

    def has(self, program: str) -> bool:
        return self._locate(program) is not None

    def stream(self, argv: Sequence[str], timeout: Optional[float] = None,
               max_bytes: Optional[int] = None) -> Stream:
        """For large output. The timeout includes the time the caller spends per line."""
        seconds = self.stream_timeout if timeout is None else timeout
        return Stream(argv, self._locate, self._env, seconds, max_bytes)

    def run(self, argv: Sequence[str], timeout: Optional[float] = None,
            max_bytes: int = RUN_MAX_BYTES) -> Result:
        seconds = self.timeout if timeout is None else timeout
        with self.stream(argv, seconds, max_bytes) as stream:
            lines = list(stream)
        result = stream.result
        assert result is not None
        if lines:
            result.stdout = "\n".join(lines) + "\n"
        return result

    def batch_script(self, job_id: str) -> Result:
        """One job's script, when the viewer asked for it. Slurm decides; do not cache."""
        return self.run(["sacct", "-j", str(job_id), "--batch-script"])
