# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.1.0] - 2026-11-22

First release.

### Added

- Full-screen terminal explorer with Overview, Me, Nodes, Queue, People, Roles
  and Timeline screens, drill-down to any job, filtering, sorting and mouse
  support.
- Live view from `scontrol`, `squeue` and `sacctmgr`: GPUs and CPUs allocated,
  free, unavailable and stranded; the queue with pending reasons; limits.
- Usage history from `sacct` for day, week, month, year and all time, each
  compared with the previous equal period, cached incrementally in a private
  SQLite file; an hourly rollup makes the overview and its trend sparklines
  instant for every range.
- Job detail with every attempt and step; the batch script, submit line and log
  tails fetched only on a keypress and never cached. A script Slurm does not keep
  is shown as `not stored`, one it does not return to you as `denied`.
- Efficiency (CPU, memory, sampled GPU utilization with coverage) on request.
- `--once` and `--json` summaries (ASCII fallback on non-UTF-8 output),
  `--anonymize` for demos, `--no-history`, `--no-full-history`.
- Paced background loading of all-time history; a private cache directory that
  never changes the mode of an existing directory.
- Read-only runner: every command is checked against an allowlist before it
  starts.
- Python 3.8+, standard library only.
