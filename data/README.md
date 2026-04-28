# `data/` — radar captures

This directory holds raw radar captures from the IWR1443 + DCA1000. **The
contents are gitignored** (see `.gitignore`); only this README and
`.gitkeep` are tracked.

## What goes here

| File | Source | Typical size |
|---|---|---|
| `*.lua` | mmWave Studio config script the teammate ran | a few KB |
| `*.npz` | MakeyMakey `record.py` output (already reshaped) | 10s–100s of MB |
| `adc_data.bin` | mmWave Studio `CaptureCardConfig_StartRecord` direct dump | 100s of MB – GB |
| `pktlogfile.txt` | DCA1000 packet log (debug only) | small |

## Why ignored

- Captures are too big for git
- They're easy to re-collect if the Lua config is preserved (the Lua file
  *is* committed when placed elsewhere, e.g. `scripts/configs/`)

## How to inspect a capture

```bash
uv run python scripts/inspect_capture.py \
    --cfg data/your_capture.lua \
    --data data/your_capture.npz
```

See `docs/teammate_handoff.md` for what to ask the teammate to send.
