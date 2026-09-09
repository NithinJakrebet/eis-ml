# Goal: EIS Data Visualization Dashboard

## Overview
Build a data visualization dashboard at `eis-ml/visualizer/` that lets lab members interactively explore raw EIS and capacity data for any cell, before any ML modeling is done. This is a diagnostic and exploration tool, not a model-results viewer.

---

## Architecture

### Backend — Python (`visualizer/api/`)
- Build a lightweight **Flask** (or FastAPI) server that:
  - Scans the `eis-ml/data/` directory at startup and returns available **folders** (e.g., `PEIS-HC-RT`) and **channels** (e.g., `A1`–`A8`, `B1`–`B6`) per folder
  - Loads and preprocesses a single channel's CSV file on demand using the existing pipeline conventions:
    - Parse using `pd.read_csv`, handle the Bio-Logic `.mpt`-style header (skip metadata rows until the column header row)
    - Add a `channel` column (matching filename convention used in `data_pipeline.loso`)
    - Apply the same frequency filter used in `state_vector.py`: `freq/Hz` in `(0.2, 20000]`
    - Do **not** build the full state vector — keep raw long-form data for visualization
  - Expose **3 JSON endpoints**:
    1. `GET /folders` → list of available data folder names
    2. `GET /channels?folder=<name>` → list of channel IDs found in that folder (derived from filenames)
    3. `GET /cell-data?folder=<name>&channel=<id>` → preprocessed data for that cell, returning:
       - `capacity_vs_cycle`: list of `{cycle, normalized_capacity}` — normalized by the channel's own initial capacity (cycle 1), consistent with pipeline leakage fix
       - `ns_states`: list of unique `Ns` values found in the file with their counts and the Ns states at which EIS is collected (e.g., `[{ns: 1, label: "Discharged", n_freqs: 37, n_cycles: 48}, ...]`)
       - `summary`: dict with `n_cycles`, `n_ns_states`, `freq_range_hz`, `cell_group` (A or B), `channel_id`, `folder`

### Frontend — React (`visualizer/client/`)
Keep it simple. Use **Recharts** for plots. No state management library needed — `useState` and `useEffect` are sufficient.

#### Layout
Single-page app with:
1. **Control panel** (top) — two dropdowns side by side:
   - Dropdown 1: Folder (populated from `/folders`)
   - Dropdown 2: Channel (populated from `/channels?folder=...`, re-fetches when folder changes)
   - A **"Load"** button that triggers `/cell-data` fetch
2. **Main panel** (below controls) — three sections arranged in a 2-column grid:
   - **Top-left**: Capacity vs. Cycle plot (line chart, x = cycle number, y = normalized capacity %)
   - **Top-right**: Ns States table — a simple HTML table listing each Ns state found, its label (Ns=1 → "Discharged", Ns=6 → "Charged"), frequency count, and cycle count
   - **Bottom (full width)**: Summary card — key metadata displayed as labeled stat boxes (channel, folder, cell group, n_cycles, freq range, Ns states present)

#### UX details
- Show a loading spinner while the API call is in flight
- Show a clear error message if the API returns an error or data is malformed
- Capacity axis: label as "Normalized Capacity (%)" and scale 0–110%
- Cycle axis: label as "Cycle Number"
- No x-axis padding needed; start from cycle 1

---

## File Structure
```
eis-ml/visualizer/
├── api/
│   ├── app.py            # Flask app — routes only, thin layer
│   ├── data_loader.py    # All data reading + preprocessing logic
│   └── requirements.txt  # flask, pandas, numpy
├── client/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Controls.jsx       # Folder + channel dropdowns + Load button
│   │   │   ├── CapacityPlot.jsx   # Recharts line chart
│   │   │   ├── NsStatesTable.jsx  # Ns state breakdown table
│   │   │   └── SummaryCard.jsx    # Metadata stat boxes
│   │   └── api.js                 # fetch wrapper for the 3 endpoints
│   ├── package.json
│   └── vite.config.js
└── README.md             # How to run (start Flask, start Vite, open localhost)
```

---

## Pipeline Invariants to Respect
These must be honored to stay consistent with the rest of `eis-ml`:

- **Channel identity**: each CSV file corresponds to exactly one channel. Never mix rows across channels.
- **Ns states**: the pipeline uses `Ns=1` (discharged, ~0.2–10 kHz, 37 freqs) and `Ns=6` (charged, ~1–10 kHz, 33 freqs). Display both in the Ns States table. Do not filter them out.
- **Capacity normalization**: normalize each channel's capacity by its own cycle-1 capacity (per-channel baseline). This matches the leakage-free normalization in the production pipeline.
- **Frequency filter**: only frequencies in `(0.2, 20000]` Hz — same as `state_vector.py`.
- **Do not build state vectors** here — this dashboard works with raw long-form data only.

---

## Known Data Quirks to Handle Gracefully
- Some channels may have missing cycles — handle gracefully (skip, don't crash)
- Some folders may have channels with different Ns state configurations (e.g., B cells may have GEIS instead of PEIS, or fewer Ns states) — just display whatever is present
- CSV files may have Bio-Logic `.mpt`-style metadata headers before the column row — the loader must skip those rows

---

## Non-Goals (explicitly out of scope)
- No model predictions or LOSO results in this dashboard
- No EIS Nyquist/Bode plots (raw impedance spectra) — capacity only
- No multi-channel overlay plots
- No file uploads
- No authentication

---

## README Requirements
The `visualizer/README.md` must include:
1. One-command start for the API (`cd api && flask run`)
2. One-command start for the client (`cd client && npm run dev`)
3. Expected data directory structure (`data/<folder>/<channel>.csv`)
4. A note that the data path is configured in `api/data_loader.py` (single line to change)
```

---

Key additions over your original:
- **Three explicit API endpoints** with exact response shapes, so there's no ambiguity for whoever implements the backend
- **Pipeline invariants section** — channel identity, normalization rule, frequency filter — so the implementer doesn't accidentally reintroduce the leakage bug or mix channels
- **Known data quirks** (Bio-Logic headers, missing cycles, GEIS vs PEIS B-cells) so edge cases are handled upfront rather than discovered during testing
- **Explicit file structure** so there's no debate about where things go
- **Non-goals** to prevent scope creep (especially no Nyquist plots, no multi-channel overlays)
- **README requirements** so the handoff is runnable by a new lab member