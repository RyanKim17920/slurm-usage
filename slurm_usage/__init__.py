"""slurm_usage: a read-only terminal explorer for any Slurm cluster.

Layers, lowest first:

``sanitize``
    Makes text controlled by other users safe for a terminal.
``parse``
    Pure parsers for Slurm's text formats.
``model``
    Dataclasses shared by every layer.
``runner``
    The only place a subprocess is started; read-only by allowlist.
``live``
    What the cluster looks like now, as far as the invoker may see it.

A front end normally needs only ``live.LiveCollector`` and ``model``.
"""

__version__ = "0.1.0"
