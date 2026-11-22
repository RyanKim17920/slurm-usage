"""Starting and supervising the subprocesses."""

import os
import signal
import subprocess
import threading
import time
from typing import Callable, Iterator, List, Mapping, Optional, Sequence

from ..model import Status
from .policy import check
from .result import Result, classify, explain

_STDERR_KEEP = 16 * 1024
_SCRUB_PREFIXES = ("SQUEUE_", "SINFO_", "SACCT_", "SCONTROL_", "SSTAT_", "SPRIO_", "SSHARE_",
                   "SREPORT_", "SACCTMGR_")
_SCRUB_NAMES = ("SLURM_CLUSTERS", "SLURM_JSON", "SLURM_YAML", "SLURM_DEBUG_FLAGS")


def child_environment(base: Mapping[str, str]) -> dict:
    """The invoker's environment minus variables that silently change output shape.

    Identity and authentication are passed through untouched and never read.
    """
    env = {key: value for key, value in base.items()
           if not key.startswith(_SCRUB_PREFIXES) and key not in _SCRUB_NAMES}
    env.update(LC_ALL="C", LANG="C", SLURM_TIME_FORMAT="%s", SLURM_BITSTR_LEN="0")
    return env


def find_program(name: str, search_path: str) -> Optional[str]:
    """Like shutil.which, but ignores relative PATH entries (never runs ./squeue)."""
    for directory in search_path.split(os.pathsep):
        if os.path.isabs(directory):
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


class Stream:
    """Iterate over a command's stdout lines (without the newline).

    Use as a context manager. ``result`` is set once iteration ends or the
    stream is closed; its stdout is empty. Never raises for a failed command.
    """

    def __init__(self, argv: Sequence[str], locate: Callable[[str], Optional[str]], env: Mapping[str, str],
                 timeout: float, max_bytes: Optional[int]) -> None:
        check(argv)
        self.argv = list(argv)  # type: List[str]
        self.result = None  # type: Optional[Result]
        self._max_bytes = max_bytes
        self._started = time.monotonic()
        self._bytes = 0
        self._has_output = False
        self._eof = False
        self._truncated = False
        self._timed_out = threading.Event()
        self._stderr = bytearray()
        self._proc = None  # type: Optional[subprocess.Popen]

        path = locate(self.argv[0])
        if path is None:
            self._set(Status.UNAVAILABLE, "%s is not installed or not on PATH" % self.argv[0])
            return
        try:
            self._proc = subprocess.Popen(
                [path] + self.argv[1:], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=dict(env), shell=False, close_fds=True,
                start_new_session=True)
        except PermissionError:
            self._set(Status.DENIED, "not allowed to execute %s" % self.argv[0])
            return
        except OSError as error:
            self._set(Status.ERROR, "cannot start %s: %s" % (self.argv[0], error.strerror))
            return
        self._drain = threading.Thread(target=self._read_stderr, daemon=True)
        self._drain.start()
        self._timer = threading.Timer(timeout, self._expire)
        self._timer.daemon = True
        self._timer.start()

    def _set(self, status: str, reason: str, returncode: Optional[int] = None,
             stderr: str = "") -> None:
        self.result = Result(self.argv, status, "", stderr, returncode, reason,
                             time.monotonic() - self._started, self._truncated)

    def _read_stderr(self) -> None:
        assert self._proc is not None
        try:
            for chunk in iter(lambda: self._proc.stderr.read1(4096), b""):
                room = _STDERR_KEEP - len(self._stderr)
                if room > 0:
                    self._stderr.extend(chunk[:room])
        except (OSError, ValueError):
            pass

    def _kill(self) -> None:
        assert self._proc is not None
        try:
            os.killpg(self._proc.pid, signal.SIGKILL)
        except OSError:
            pass

    def _expire(self) -> None:
        self._timed_out.set()
        self._kill()

    def __iter__(self) -> Iterator[str]:
        if self._proc is None or self.result is not None:
            return
        out = self._proc.stdout
        try:
            while True:
                limit = -1 if self._max_bytes is None else self._max_bytes - self._bytes + 1
                line = out.readline(limit)
                if not line:
                    self._eof = True
                    break
                self._bytes += len(line)
                if self._max_bytes is not None and self._bytes > self._max_bytes:
                    self._truncated = True
                    break
                self._has_output = True
                if line.endswith(b"\n"):
                    line = line[:-1]
                yield line.decode("utf-8", "surrogateescape")
        finally:
            self.close()

    def close(self) -> None:
        if self._proc is None or self.result is not None:
            return
        proc = self._proc
        if not self._eof:
            self._kill()
        try:
            # at EOF the exit is imminent; the timer still bounds a child that hangs
            code = proc.wait()
        except BaseException:
            self._kill()
            raise
        finally:
            self._timer.cancel()
        self._drain.join(1.0)
        if self._drain.is_alive():  # a grandchild kept the pipe open
            self._kill()
            self._drain.join(1.0)
        proc.stdout.close()
        if not self._drain.is_alive():
            proc.stderr.close()
        stderr = bytes(self._stderr).decode("utf-8", "replace")
        if self._timed_out.is_set():
            self._set(Status.TIMEOUT, "%s timed out" % self.argv[0], code, stderr)
        elif self._truncated:
            self._set(Status.PARTIAL, "output limit reached", code, stderr)
        elif not self._eof:
            self._set(Status.CANCELLED, "stopped by the caller", code, stderr)
        else:
            self._set(classify(code, self._has_output, stderr),
                      explain(stderr, code, self.argv), code, stderr)

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
