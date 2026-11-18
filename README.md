# GPS Trajectory Map-Matching & HMM Route Reconstruction

End-to-end machine learning and Hidden Markov Model (HMM) map-matching system with a **two-laptop LAN deployment** (FastAPI inference engine on Laptop 1 + Streamlit Evaluator Dashboard with interactive 4-layer Folium maps on Laptop 2).

---

## Repository Structure

```text
ml-hackathon/
├── data/                         # Raw dataset files: .track, .nodes, .arcs, .route
├── artifacts/                    # Serialized best_model.pkl, scalers, tuning configs
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Member 1: Parsers & NetworkX DiGraph builder
│   ├── candidate_gen.py          # Member 1: Spatial index (KDTree / STRtree) matcher
│   ├── features.py               # Member 2: Spatial, heading, speed & graph features
│   ├── train_models.py           # Member 2: 5-model GroupKFold benchmark
│   ├── bayesian_opt.py           # Member 2: Optuna tuning -> artifacts/best_model.pkl
│   ├── hmm_viterbi.py            # Member 3: HMM + Viterbi (`reconstruct_route`)
│   └── visualizer.py             # Member 4: 4-layer interactive Folium maps
├── api/
│   ├── __init__.py
│   ├── pipeline.py               # Member 4: End-to-end orchestrator & Contract 1-3 adapter
│   └── main.py                   # Member 4: FastAPI server (Laptop 1)
├── ui/
│   └── app.py                    # Member 4: Streamlit Evaluator UI (Laptop 2)
├── notebooks/                    # Member 1 & 3 EDA / HMM evaluation notebooks
├── report/
│   ├── sections/                 # Markdown sections (Sec 1-7)
│   ├── figures/                  # Report diagrams & benchmark plots
│   ├── build_report.py           # Compiles all sections -> Technical_Report.pdf
│   └── Technical_Report.pdf      # Compiled technical report
├── tests/
│   └── test_member4_integration.py
├── requirements.txt
└── README.md
```

---

## Quick Setup (Both Laptops)

```bash
git clone https://github.com/AtharvaGore/ml-hackathon.git
cd ml-hackathon

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Two-Laptop LAN Deployment Guide (Live Evaluator Viva)

Ensure **Laptop 1** and **Laptop 2** are connected to the same Wi-Fi network or mobile hotspot.

### Laptop 1 — FastAPI Inference Server

1. Verify dataset files (`.nodes`, `.arcs`, `.track`, `.route`) are in `data/` and trained model is in `artifacts/best_model.pkl`.  
   *(If teammates have not merged yet, the server automatically runs in `MOCK_DEMO` mode with a deterministic 14×16 directed urban road network and 10 evaluation trips so LAN connectivity and UI rendering can be verified immediately.)*
2. Start the server bound to all network interfaces (`0.0.0.0`):
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
3. Keep the terminal visible during the viva. On startup, the server prints its detected LAN URL, e.g.:
   ```text
   LAN Endpoint    : http://192.168.1.15:8000  (Enter this on Laptop 2 UI)
   ```
   *(If needed, find Laptop 1's IP manually via `ipconfig getifaddr en0` on macOS, `hostname -I` on Linux, or `ipconfig` on Windows.)*
4. Allow incoming TCP traffic on port `8000` if the OS firewall prompts.

### Laptop 2 — Streamlit Evaluator Dashboard

1. Launch the Evaluator UI:
   ```bash
   streamlit run ui/app.py
   ```
2. In the left sidebar:
   - Enter **Laptop 1's IP Address** (e.g., `192.168.1.15`) and port `8000`.
   - Click **Test Connection / Refresh** to confirm the green `● ONLINE` badge.
   - Select any of the **10 evaluation trips** from the dropdown and click **Run Map Matching**.
3. Inspect:
   - **KPI Cards**: Top-1 Accuracy (with HMM uplift vs. Classifier-only), Route Continuity, Server Processing Time & LAN RTT, Route F1, and Discontinuities.
   - **Interactive 4-Layer Folium Map**:
     - **Layer 1**: Raw GPS observations (blue circle markers with popups)
     - **Layer 2**: Road network arcs (light gray lines)
     - **Layer 3**: Ground-truth reference route (green line)
     - **Layer 4**: Model predicted reconstructed route (purple dashed overlay)
   - **10-Trip Batch Tab**: Evaluate all 10 trips in one click and export benchmark CSVs.

---

## API Contract Specification (Laptop 1)

| Method | Endpoint | Payload | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | — | Server status, pipeline mode (`full_real` / `partial_real` / `mock_demo`), graph stats, loaded modules |
| `GET` | `/trips` | — | Metadata for the 10 held-out evaluation trips (no ground-truth leakage) |
| `POST` | `/predict_trip` | `{"trip_id": "eval_trip_01"}` | Runs full pipeline (`Track -> Candidates -> Classifier -> HMM Viterbi -> Metrics`) |
| `POST` | `/predict_batch` | `{"trip_ids": null}` | Evaluates all 10 trips and returns per-trip + macro summary metrics |

Quick CLI verification from Laptop 2:
```bash
curl -X POST http://<LAPTOP_1_IP>:8000/predict_trip \
  -H "Content-Type: application/json" \
  -d '{"trip_id": "eval_trip_01"}'
```

---

## Plug-and-Play Team Integration (Contracts 1–3)

`api/pipeline.py` dynamically imports teammate modules without requiring edits or causing Git merge conflicts:

- **Member 1 (`feat/data-and-eda`)**: `src/data_loader.py` & `src/candidate_gen.py`
- **Member 2 (`feat/models-and-tuning`)**: `src/features.py` & `artifacts/best_model.pkl`
- **Member 3 (`feat/hmm-viterbi`)**: `src/hmm_viterbi.py` exposing:
  ```python
  def reconstruct_route(trip_id: str, gps_points: list, graph: nx.DiGraph) -> list[int]:
      ...
  ```
- **Member 4 (`feat/api-ui-deploy`)**: `api/main.py`, `api/pipeline.py`, `ui/app.py`, `src/visualizer.py`, `report/build_report.py`

---

## Compiling the Final Technical Report PDF

Members 1–3 add their text and plots into `report/sections/` and `report/figures/`. Then regenerate `report/Technical_Report.pdf` in one command:

```bash
python report/build_report.py
```

---

## Running Integration Tests

```bash
pytest -v tests/test_member4_integration.py
```
