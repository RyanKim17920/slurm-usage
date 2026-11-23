# Architecture

One package, `slurm_usage`, standard library only. Layers, lowest first; each
imports only from layers listed before it (`history` does not use `live`).

| Package | Role |
| --- | --- |
| `sanitize` | makes text that other users control safe for a terminal |
| `parse` | pure parsers for Slurm's text formats: units, hostlists, TRES/GRES, record framing, batch-script output. No I/O |
| `model` | dataclasses shared by every layer: `Job`, `Node`, `Snapshot`, `Status`, ... |
| `runner` | the only place a subprocess is started. `policy.check(argv)` is the read-only allowlist; results carry a `Status` derived from exit code and stderr |
| `live` | what the cluster looks like now (`LiveCollector.snapshot()`), plus one-job queries made on request (`ondemand`) and bounded log tails (`logs`) |
| `history` | accounting history: incremental ingest into a disposable cache (`store`), queries over any window (`report`, `aggregate`, `series`), the hourly `rollup` behind the overview, run types, events, one job's detail |
| `ui` | the terminal front end: `data` adapts history for screens and loads ranges in the background, `screens` builds panes, `widgets` lays them out in boxes and fits tables to the width, `charts` draws sparklines, gauges and heat cells, `plot` the large figures and the range detail's chart, `breakdown` splits a list of attempts for the run-type breakdown and the Me page (GPU-seconds per bin, outcomes, runtime and wait quantiles, job-name families), `theme` holds the palette (truecolor, 256, 16, 8 colors or none) and glyph sets, `app` handles keys and the event loop, `text` renders `--once` / `--json`, `anonymize` wraps the collector and history with pseudonyms |
| `cli` | argument parsing (removed options stop with what replaced them), environment (`NO_COLOR`, `SLURM_USAGE_CACHE`), detecting ASCII terminals, and choosing between the TUI and `--once` |

## Status values

Every query result carries a `model.Status`: `ok`, `empty` (answered,
legitimately nothing), `partial`, `denied` (Slurm or the OS refused you),
`unavailable` (not installed, configured or supported), `timeout`, `error`,
`not_stored` (answered, but Slurm keeps no copy, e.g. a batch script). The UI
maps each to one placeholder (`n/a`, `—`, `?`, `denied`, `ERR`, `not stored`),
so a missing number is never shown as 0.

## Threads

The UI thread only draws. Slurm calls run on a small worker pool
(`ui/loader.py`, stale-while-revalidate), and history loads on one background
thread that loads the range on screen first. Switching to a shorter range stops
a longer load in progress (what it already committed stays in the cache) so the
requested range is ready first.

## History cache

`history/store.py` keeps attempts as numeric rows plus interned names and roles
in SQLite, or in memory. The file is owner-only, under `$SLURM_USAGE_CACHE`, else
`$XDG_CACHE_HOME/slurm-usage/` or `~/.cache/slurm-usage/`; when `XDG_CACHE_HOME`
is unset and `~/.cache` is on a network filesystem, it goes to node-local
`/var/tmp/slurm-usage-<uid>` (0700) if `/var/tmp` is local, root-owned and
sticky. `history/ingest.py` fetches newest first, one `sacct` at a time, in
windows sized to about 30,000 rows from the density of recent ones; a `sacct`
past 512 MB is stopped and its window split, and background windows pause for
half of slurmdbd's last answer time. Rows are written on a second thread while
the next window is fetched. It records which time span is covered; coverage only
grows, and pieces already committed stay valid if a load is cancelled.
`history/rollup.py` maintains hourly totals per role combination (user, account,
QOS, partition, state) for the hours whose data changed, and serves the
overview's per-range totals and trend series from them without touching Slurm.
The same pass keeps exact aggregates of finished allocations (`agg_hour`: usage,
allocation seconds, measured efficiency and first/last use per hour; `agg_start`:
per-attempt counts, waits and time-limit use by start hour). `aggregate.py`
answers a window from them for its whole hours and reads only the rows at its
edges, those running across its end, unfinished ones and odd ones (ending before
they start), so any range costs about the same as a day; the results equal a full
pass (`build_report(exact=True)`), and the few fields that do not add up over
hours (peak GPUs, active time, first/last use) come from one full pass the first
time one of them is read. Lanes use the aggregates once their resolution is an
hour or more. A cache from before the aggregates (or of another layout,
`AGG_VERSION`) rebuilds them from the stored rows on the next rollup, without
asking Slurm again. Events take their baselines (largest and longest before the
window, gaps, preemption chains) from SQLite and read only the rows inside the
window, and keep the coverage gaps and preemption chains of a selection until the
data changes (every range asks the same). The same aggregates are kept by
job-name family (`agg_fhour`, `agg_fstart`; `fams` interns the families and
`famnames` maps each stored name to its own), so run types by family come from
them like the others; their medians and example names, which only `--once` /
`--json` show, are read on first use. A list of one run type is narrowed in
SQLite (GPUs per allocation, roles, a family's names) and ranked there when it
can be; the newest attempts of a selection are read newest first and no further
than the page (down the start index when the selection is a large part of all
rows); a search is prefiltered with `instr()` and checked row by row. Step usage
for efficiency is read in pieces of about 3,000 attempts, four `sacct` at a time;
views redraw as pieces land. Per-kind sparklines (`History.kind_series`) are
summed into the chart's bins from the aggregates.

## Answers kept between runs

`ui/data.py`'s `ResultCache` keeps what history views showed last in
`views-v1.sqlite` beside the history cache (owner-only, pickled, never while
anonymized, only for the cache file and version they came from). The loader seeds a
view's entries from it, so a new run or a view opened again paints at once and then
refreshes in place; the overview, which depends on the cache alone, is reused
outright while the cache's data generation (bumped by every change, in `meta`) is
the same. `HistoryData` answers a repeated query once per data generation (a minute
at most); a load that changes nothing does not recompute what is on screen. The
first refresh of a run waits (up to 3 s) until the screen has been answered from the
cache, and switching screens drops work queued for the old one. What a screen draws
from a long list (the run-type breakdown, job-name trends) is worked out with the
answer, off the UI thread, not per frame. A job row that stays selected for half a
second has its detail (and its steps while it runs) fetched under the job screen's
keys, so Enter opens it filled in. The probe of Slurm's configuration is kept 5
minutes in `probe-v1.json`. `--once` takes the live snapshot while history loads,
and skips the refresh when the cache was refreshed in the last minute.

## Settings at run time

`ui/app.py` owns the switches the settings box (`O`) lists: anonymize (`A`),
colors (`C`), glyphs (`U`) and history full/recent/off (`H`). The app keeps the
real collector and history and one `Anonymizer` for the whole run; anonymizing
wraps them, drops every cached loader entry, derives the pseudonymous snapshot
from the real one (no new query), closes open file text and maps the
breadcrumb stack, typed filters and the selected row through the pseudonyms
(and back). ASCII mode maps every screen cell to one ASCII cell after drawing.
History off pauses the background loader (`HistoryData.pause`); what was
committed stays in the cache.

## Tests

`tests/` uses `unittest` only. `tests/test_demo_cluster.py` runs the invented
demo cluster in `scripts/demo/` (fake Slurm executables, the only ones on
`PATH`) through the real runner, collector and history. `tests/fixtures.py`, `tests/cluster_fixture.py`
and `tests/ui_fakes.py` build an invented cluster; the runner tests use small
fake `sacct`/`squeue` executables in a temporary directory. Two test modules run
read-only queries against a real cluster and skip themselves without one.
