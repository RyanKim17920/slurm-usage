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
| `ui` | the terminal front end: `data` adapts history for screens and loads ranges in the background, `screens` builds panes, `widgets` lays them out in boxes and fits tables to the width, `charts` draws sparklines, gauges and heat cells, `theme` holds the palette (truecolor, 256, 16, 8 colors or none) and glyph sets, `app` handles keys and the event loop, `text` renders `--once` / `--json`, `anonymize` wraps the collector and history with pseudonyms |
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
`/var/tmp/slurm-usage-<uid>` (0700) if `/var/tmp` is local, root-owned and sticky.
`history/ingest.py` fetches newest first, one `sacct` at a time, in windows sized
to about 30,000 rows from the density of recent ones; a `sacct` past 512 MB is
stopped and its window split, and background windows pause for half of
slurmdbd's last answer time. Rows are written on a second thread while the next
window is fetched. It records which time span is covered; coverage only grows,
and pieces already committed stay valid if a load is cancelled.
`history/rollup.py` maintains hourly totals per role combination (user, account,
QOS, partition, state) for the hours whose data changed, and serves the
overview's per-range totals and trend series from them without touching Slurm.

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
