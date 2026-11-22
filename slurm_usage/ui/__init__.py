"""The terminal front end.

    fmt        numbers, durations, placeholders, sparklines (pure)
    canvas     an in-memory screen; every string is sanitized on the way in
    widgets    panes, tables that fit the width, vertical layout
    loader     background work so drawing never waits on Slurm
    data       the only module that touches ``slurm_usage.history``
    anonymize  --anonymize: pseudonyms applied to the data, not the screen
    filestat   does a recorded path exist, and did it change after the job started
    screens    Overview, Me/Person, Nodes, Queue, People, Roles, Timeline, Job, Help
    app        navigation stack, keys, refresh policy (no curses)
    terminal   curses: colors, mouse, resize, draw loop
    text       --once and --json
"""
