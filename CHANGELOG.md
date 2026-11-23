# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Range detail: any range of the cluster, an account or a person, opened from an
  overview tile, a Trend row or the overview's chart. GPUs in use over the range
  (hourly for a day, 3-hourly for a week, daily for a month, weekly for a year)
  against capacity, the average and the previous period, with the peak and now
  marked; total, average, utilization, peak, change and CPU-hours above it; who
  used it below, each row opening its own detail. `←` `→` `[` `]` or a click
  move between ranges in place; `g` switches GPUs and CPUs. The cluster's long
  ranges come from the hourly rollup (a busy year in well under a second).
- The overview leads with GPUs in use now in large figures, then a tile per
  range (GPU-hours, utilization, change, a small trend; `loading…` with progress
  instead of partial sums), and the chosen range's chart when there is room.
  An account's people page starts with the account's own Trend.
- The ranges in the header are clickable.

### Fixed

- Mouse: a click on the selected row opens it (no double click needed), the
  wheel scrolls the pane under the pointer, wheel-down works with ncurses
  builds without BUTTON5, and motion reports are no longer requested.
- A refresh that reorders a table keeps the same row selected.
- The overview no longer jumps when the live state arrives (the hero and the
  tiles keep their height while loading), and the accounts' `visible` column no
  longer appears and vanishes while history loads.
- Enter on the range tiles opens the range shown, even before the next frame.
- Where history begins inside a range (a young cluster's year), the average in
  use and the utilization are both over the covered span (average / capacity is
  the utilization), the tile and the detail say "from Jul 25", and the chart
  starts at the data with finer bins.

### Changed

- A redesigned terminal UI: panes in rounded boxes with the title and notes in
  the border and the focused one in the accent color; a header with tab pills,
  the range selector and a slim loading bar; key hints styled as keys.
- One palette by meaning (`ui/theme.py`): truecolor with a direct-color
  terminfo, a matched 256-color mapping, 16 and 8 colors, and no color.
- Overview opens on cards for GPUs, CPUs, nodes and the queue with gauges;
  sparklines are shaded on one color ramp, gauges have eighth-cell precision, timeline
  lanes and the activity heatmap use one color ramp with a legend.
- Less on screen at once: Me / person pages show four panes (events, jobs,
  activity and roles share one box with tabs), Job detail shows steps and
  attempts as tabs, efficiency columns appear once something is measured,
  secondary columns only when the width allows, and the explanatory notes that
  sat on screen moved to the `?` glossary.

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
