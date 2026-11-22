"""Record framing for Slurm's key=value and delimited text."""

import re
import secrets
from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence

# the class comes first so the scan is fast; the lookbehind then requires a word start
_KEY = re.compile(r"([A-Z](?<!\S.)[A-Za-z0-9_/:*.()\[\]-]*)=")


def key_values(text: str) -> Dict[str, str]:
    """Split "Key=value with spaces Other=x". The first occurrence of a key wins,
    so call this only on text that precedes any user-controlled field.
    """
    matches = list(_KEY.finditer(text))
    result = {}  # type: Dict[str, str]
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result.setdefault(match.group(1), text[match.end():stop].strip())
    return result


class Frame(NamedTuple):
    """An output format whose delimiters are unpredictable to whoever wrote the data.

    A value containing spaces, "|", newlines or a whole forged record cannot
    imitate a delimiter it has never seen. Make a new one for every command.
    """

    begin: str
    separator: str
    end: str

    @classmethod
    def new(cls) -> "Frame":
        token = "z" + secrets.token_hex(5)
        return cls(token + "b", token + "s", token + "e")

    def format(self, fields: Sequence[str]) -> str:
        return self.begin + self.separator.join(fields) + self.end

    def records(self, lines: Iterable[str], width: int) -> Iterator[Optional[List[str]]]:
        """Yield field lists, or None for a record of the wrong shape or cut short."""
        pending = None  # type: Optional[List[str]]
        for line in lines:
            if pending is None:
                start = line.find(self.begin)
                if start < 0:
                    continue
                pending = [line[start + len(self.begin):]]
            else:
                pending.append(line)
            if line.endswith(self.end):
                values = "\n".join(pending)[: -len(self.end)].split(self.separator)
                yield values if len(values) == width else None
                pending = None
        if pending is not None:
            yield None


SCRIPT_NOT_STORED = ("Slurm keeps no copy of this script: not a batch job (srun/salloc), or "
                     "this cluster does not store scripts (AccountingStoreFlags=job_script)")
SCRIPT_NOT_VISIBLE = ("Slurm returned no record: scripts are shown only to the job's owner "
                      "and Slurm operators, or the job is no longer in accounting")
_SCRIPT_HEADER = re.compile(r"Batch Script for \S+")


def batch_script_body(stdout: str) -> Optional[str]:
    """The script inside ``sacct -j ID --batch-script`` output.

    None: Slurm printed no record at all (the job was not returned to this
    invoker: not visible to you, or no longer in accounting).
    "": Slurm returned the record but no script ("NONE" under the header): not a
    batch job, or the cluster does not store scripts.
    """
    lines = stdout.split("\n")
    while lines and not lines[0].strip():
        del lines[0]
    header = bool(lines) and bool(_SCRIPT_HEADER.fullmatch(lines[0].strip()))
    if header:
        del lines[0]
        if lines and lines[0].strip() and set(lines[0].strip()) == {"-"}:
            del lines[0]
    while lines and not lines[-1].strip():
        lines.pop()
    body = "\n".join(lines)
    if not header and not body.strip():
        return None
    if body.strip() == "NONE":
        return ""
    return body
