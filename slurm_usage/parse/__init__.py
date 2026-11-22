"""Pure parsers for Slurm text. No I/O, nothing site-specific.

Scalars
    parse_int, parse_float, parse_count, parse_memory (bytes), memory_basis,
    parse_duration (seconds), is_unlimited, parse_time (epoch seconds),
    parse_exit_code, parse_state, optional_text
Lists
    expand_hostlist, parse_ranges, compress_ranges, split_outside
TRES and GRES
    split_tres, parse_tres, parse_usage_tres, gpus_from_tres, parse_gres,
    gpus_from_gres, gpu_indices, Gres
Records
    Frame (unforgeable delimiters for -o formats), key_values, batch_script_body

Parsers return None or omit a key for what they cannot read; they never
substitute 0. They do not sanitize: callers do that when storing display text.
"""

from .hostlist import compress_ranges, expand_hostlist, parse_ranges, split_outside
from .records import SCRIPT_NOT_STORED, SCRIPT_NOT_VISIBLE, Frame, batch_script_body, key_values
from .tres import (Gres, gpu_indices, gpus_from_gres, gpus_from_tres, parse_gres, parse_tres,
                   parse_usage_tres, split_tres)
from .units import (is_unlimited, memory_basis, optional_text, parse_count, parse_duration,
                    parse_exit_code, parse_float, parse_int, parse_memory, parse_state,
                    parse_time)

__all__ = [
    "SCRIPT_NOT_STORED", "SCRIPT_NOT_VISIBLE",
    "Frame", "Gres", "batch_script_body", "compress_ranges", "expand_hostlist", "gpu_indices", "gpus_from_gres",
    "gpus_from_tres", "is_unlimited", "key_values", "memory_basis", "optional_text",
    "parse_count", "parse_duration", "parse_exit_code", "parse_float", "parse_gres",
    "parse_int", "parse_memory", "parse_ranges", "parse_state", "parse_time", "parse_tres",
    "parse_usage_tres", "split_outside", "split_tres",
]
