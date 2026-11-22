
# Background history loading rests between pieces to spare slurmdbd; not in tests.
import slurm_usage.history.ingest as _ingest  # noqa: E402

_ingest.PIECE_PAUSE = 0.0
