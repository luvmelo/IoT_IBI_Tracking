# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Mac-side **offline analysis** of TI IWR1443 + DCA1000 mmWave radar captures. The
capture half of the pipeline (mmWave Studio + Lua + DCA1000 UDP listener) lives
on a Windows machine and is **not in this repo** — this codebase only consumes
the artifacts that flow out of it (`.lua` config + raw `.bin` or reshaped
`.npz`).

End goal of the broader project (see `revised_project_proposal.md`): contactless
interbeat-interval (IBI) and short-window HRV tracking from the radar. The code
currently in-repo is only the *ingest + parameter-extraction* layer; signal
processing for IBI lives downstream and is not yet implemented.

## Commands

Dependency management is `uv` (Python ≥ 3.11). `uv.lock` is committed.

```bash
uv sync                                                # install / refresh env
uv run python scripts/inspect_capture.py \             # parse Lua, load capture,
    --cfg data/your_capture.lua \                      #   print params + stats,
    --data data/your_capture.npz                       #   write range-FFT plot
                                                       #   to data/inspect_range_fft.png
uv run python scripts/_smoke_test.py                   # synthetic .bin round-trip
                                                       #   (reader + reshape sanity)
uv run python scripts/_demo_realistic.py               # generate two-target demo
                                                       #   .lua + .bin in data/
```

There is **no test framework wired up** (no pytest, no CI). The two `_*.py`
scripts in `scripts/` are the de-facto smoke tests — `_smoke_test.py` asserts
shape and tone-recovery; failures raise. Run them after touching
`radar_analysis/reader.py` or `radar_analysis/radar_config.py`.

## Architecture

```
Windows side (NOT in this repo)                Mac side (this repo)
─────────────────────────────────              ─────────────────────────
IWR1443 → DCA1000 → mmWave Studio   ──.lua──▶  radar_analysis.RadarConfig
                  ↓ UDP                          (regex-parses Lua,
       MakeyMakey record.py                       derives n_chirps, range_res, …)
                  ↓
                 .npz       ──.npz/.bin──▶     radar_analysis.load_capture
   (or mmWave Studio direct .bin)               → (n_frames, n_chirps,
                                                    n_samples, n_rx) complex64
```

Two files do all the work:

- **`radar_analysis/radar_config.py`** — `RadarConfig` is an `OrderedDict`
  populated by regex over the `.lua` file. We **do not execute Lua**; we just
  match `KEY = value` and `ar1.Func(args)` lines. `get_params()` then derives
  physical limits (range res, max range, velocity res, frame size in bytes).
  Adapted from `r-bt/MakeyMakey/src/xwr/radar_config.py`, kept stdlib-only so
  it runs without numba on the Mac.

- **`radar_analysis/reader.py`** — `load_capture(path, params)` dispatches on
  extension: `.npz` is unpacked directly (MakeyMakey's `record.py` already
  reshapes); `.bin` is run through `reshape_iwr1443_frame`, which assumes the
  IWR1443 BOOST LVDS layout from the notes PDF §24.6: groups of 8 int16 words
  = `[I_rx0..I_rx3, Q_rx0..Q_rx3]`. Output is always
  `(n_frames, n_chirps, n_samples, n_rx)` complex64.

Both scripts in `scripts/` insert the repo root onto `sys.path` rather than
relying on package install (the project is intentionally `package = false` in
`pyproject.toml`).

## Things to know before editing

- **The reshape is IWR1443-specific.** It assumes 4 RX, complex-1x ADC, and
  **single TX** (`END_CHIRP_TX = 0` in the Lua). If TDM-MIMO is ever enabled
  (TX2/TX3 firing), `reshape_iwr1443_frame` needs a deinterleave step before
  the `[I0..I3, Q0..Q3]` regroup. **Do not** copy the upstream
  `ConnectedSystemsLab/xwr_raw_ros` reshape — it targets AWR2243/cascade
  boards and will silently produce wrong arrays here.

- **`data/` is gitignored** except for `README.md` and `.gitkeep`. Captures
  are 10s–100s of MB (`.npz`) up to multiple GB (`.bin`). `*.bin`, `*.npz`,
  `_smoke.*`, and `*.png` are also ignored repo-wide as a defensive backstop.
  Don't commit captures even if they appear small.

- **The Lua parser is line-oriented and forgiving by design.** It coerces
  values to `int`/`float`/`str` based on whether they contain a `.`, strips
  trailing `--` comments, and ignores anything that doesn't match
  `KEY = value` or `something.Func(args)`. New Lua dialects can land without
  breaking it, but a typo in a key the analyzer reads (e.g. `ADC_SAMPLES`)
  will fail at `get_params()` rather than parse time.

- **Capture format provenance is in `docs/python_interface_setup.md`** —
  including the FPGA UDP control protocol, the static-IP requirement
  (`192.168.33.30`), and the gotchas list (Studio must be Admin, MATLAB
  toolbox doesn't work for IWR1443, etc.). When the user asks
  "why does the Windows side need X?" that doc is the source of truth.

- **Teammate handoff doc** (`docs/teammate_handoff.md`) is bilingual (中文 +
  English) and is what the Windows-side teammate is expected to read. If
  capture parameters change on their end, that's the doc to update.
