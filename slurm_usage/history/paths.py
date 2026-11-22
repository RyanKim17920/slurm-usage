"""sbatch filename patterns (%j, %A, %a, ...) in StdOut and StdErr."""

import os
import re
from typing import NamedTuple, Optional

from ..parse import expand_hostlist, optional_text
from ..sanitize import clean

_PATTERN = re.compile(r"%(%|(\d*)([AabJjNnstux]))")
_MAX_PAD = 10


class LogPath(NamedTuple):
    display: str
    raw: str  # for open() only, never for display
    merged: bool = False  # stderr that goes to stdout's file
    resolved: bool = True  # False if a pattern or the working directory was unknown

    def __repr__(self) -> str:
        return "LogPath(display=%r, merged=%r, resolved=%r)" % (self.display, self.merged,
                                                                self.resolved)


def expand_path(pattern: Optional[str], job_id: str, display_id: str, user: str, name: str,
                nodes: str, workdir: str) -> Optional[LogPath]:
    if optional_text(pattern) is None:
        return None
    assert pattern is not None
    resolved = True
    master, underscore, task = display_id.partition("_")
    has_task = bool(underscore) and task.isdigit()

    def replace(match: "re.Match") -> str:
        nonlocal resolved
        if match.group(1) == "%":
            return "%"
        width, letter = match.group(2), match.group(3)
        if letter == "A":
            value = master if has_task else job_id
        elif letter in ("a", "b"):
            if not has_task:
                resolved = False
                return match.group(0)
            value = task if letter == "a" else str(int(task) % 10)
        elif letter in ("j", "J"):
            value = job_id
        elif letter == "N":
            hosts = expand_hostlist(nodes)
            if not hosts:
                resolved = False
                return match.group(0)
            value = hosts[0]
        elif letter in ("n", "t"):
            value = "0"
        elif letter == "s":
            value = "batch"
        elif letter == "u":
            value = user
        else:
            value = name
        if width and value.isdigit():
            value = value.zfill(min(int(width), _MAX_PAD))
        return value

    if "\\" in pattern:  # a backslash switches pattern expansion off
        path = pattern.replace("\\", "")
    else:
        path = _PATTERN.sub(replace, pattern)
    if not os.path.isabs(path):
        if os.path.isabs(workdir):
            path = os.path.normpath(os.path.join(workdir, path))
        else:
            resolved = False
    return LogPath(clean(path), path, False, resolved)
