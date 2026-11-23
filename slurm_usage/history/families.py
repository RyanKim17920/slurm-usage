"""Job-name families: reruns of one script grouped by their name with the numbers,
dates and hashes taken out."""

import re
from functools import lru_cache

NO_NAME = "(no name)"

_UUID = re.compile(r"[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?"
                   r"[0-9a-fA-F]{12}")
_DATE = re.compile(r"(?<![0-9])(?:19|20)[0-9]{2}[-_.]?[01][0-9][-_.]?[0-3][0-9]"
                   r"(?:[T_-]?[0-2][0-9][:_-]?[0-5][0-9](?:[:_-]?[0-5][0-9])?)?(?![0-9])")
_HEX = re.compile(r"(?<![0-9A-Za-z])(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{6,}"
                  r"(?![0-9A-Za-z])")
_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_REPEAT = re.compile(r"(<n>)(?:[-_.x,]?<n>)+")


@lru_cache(maxsize=65536)
def name_family(name: str) -> str:
    """"train_lr3e-4_seed2_20260101" -> "train_lr<n>e-<n>_seed<n>_<date>".

    Dates, UUIDs, hex hashes and numbers (array indices, seeds, steps) become
    placeholders, so reruns of one script group together.
    """
    text = name.strip()
    if not text:
        return NO_NAME
    text = _UUID.sub("<id>", text)
    text = _DATE.sub("<date>", text)
    text = _HEX.sub("<hex>", text)
    text = _NUMBER.sub("<n>", text)
    return _REPEAT.sub("<n>", text)
