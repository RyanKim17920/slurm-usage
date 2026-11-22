"""Bounded tail of a job's log, with nothing but the invoker's own permissions.

A job's owner chooses its log path, so it is hostile input:

* regular files only (checked before and after opening, opened non-blocking, so
  a FIFO or device cannot hang or be triggered);
* no /proc, /sys or /dev, and no well-known Slurm or munge credential files;
* as root, only a file owned by the job's owner;
* for someone else's job, never a file owned by the viewer (the owner must not
  be able to make the tool display the viewer's private files);
* bounded bytes, sanitized lines.
"""

import os
import stat
from dataclasses import dataclass, field
from typing import List, Optional

from ..model import Status
from ..sanitize import clean

MAX_BYTES = 64 * 1024
HARD_MAX_BYTES = 4 * 1024 * 1024
_PSEUDO = ("/proc", "/sys", "/dev")
_CREDENTIALS = frozenset(("munge.key", "slurm.key", "slurm.jwks", "jwt_hs256.key",
                          "slurmdbd.conf"))


@dataclass
class LogTail:
    status: str
    lines: List[str] = field(default_factory=list)  # sanitized
    detail: str = ""
    size: Optional[int] = None
    truncated: bool = False  # there is more above


def _refusal(resolved: str, info: os.stat_result, job_uid: Optional[int],
             uid: int, euid: int) -> Optional[str]:
    if not stat.S_ISREG(info.st_mode):
        return "not a regular file"
    if any(resolved == root or resolved.startswith(root + "/") for root in _PSEUDO):
        return "refusing a system pseudo-file"
    if os.path.basename(resolved) in _CREDENTIALS:
        return "refusing a credential file"
    if euid == 0 and (job_uid is None or info.st_uid != job_uid):
        return "as root, only files owned by the job's owner are read"
    if job_uid != uid and info.st_uid == uid:
        return "this path points at your own file, not the job owner's"
    return None


def tail_file(path: Optional[str], job_uid: Optional[int], max_bytes: int = MAX_BYTES,
              max_lines: int = 500) -> LogTail:
    if not path or "\0" in path or not os.path.isabs(path):
        return LogTail(Status.EMPTY, detail="no absolute log path")
    uid, euid = os.getuid(), os.geteuid()
    max_bytes = max(1, min(max_bytes, HARD_MAX_BYTES))
    fd = None
    try:
        resolved = os.path.realpath(path)
        why = _refusal(resolved, os.stat(resolved), job_uid, uid, euid)
        if why:
            return LogTail(Status.DENIED, detail=why)
        fd = os.open(resolved, os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY | os.O_CLOEXEC
                     | getattr(os, "O_NOFOLLOW", 0))
        info = os.fstat(fd)  # what was opened, in case the path changed meanwhile
        try:
            opened = os.readlink("/proc/self/fd/%d" % fd)
        except OSError:
            opened = resolved
        why = _refusal(opened, info, job_uid, uid, euid)
        if why:
            return LogTail(Status.DENIED, detail=why)
        offset = max(0, info.st_size - max_bytes)
        # one byte earlier tells whether the first line is cut
        peek = 1 if offset else 0
        os.lseek(fd, offset - peek, os.SEEK_SET)
        chunks, remaining = [], max_bytes + peek
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    except PermissionError:
        return LogTail(Status.DENIED, detail="your account may not read this file")
    except (FileNotFoundError, NotADirectoryError):
        return LogTail(Status.EMPTY, detail="file does not exist (from this host)")
    except (OSError, ValueError) as error:
        return LogTail(Status.ERROR, detail=clean(getattr(error, "strerror", None) or error))
    finally:
        if fd is not None:
            os.close(fd)

    data = b"".join(chunks)
    if peek:
        before, data = data[:1], data[1:]
        if before != b"\n" and b"\n" in data:
            data = data.split(b"\n", 1)[1]
    lines = data.decode("utf-8", "replace").split("\n")
    if lines[-1] == "":
        lines.pop()
    truncated = bool(peek) or len(lines) > max_lines
    lines = lines[max(0, len(lines) - max_lines):]
    # progress bars redraw with \r: only the last redraw is what a terminal shows
    shown = [clean(line.rstrip("\r").rsplit("\r", 1)[-1]) for line in lines]
    return LogTail(Status.OK if shown else Status.EMPTY, shown,
                   "" if shown else "file is empty", info.st_size, truncated)
