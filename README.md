# PathSnap: Distributed Map-Matching & Route Reconstruction

PathSnap is an end-to-end distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy, sparse GPS observations to a directed physical road network.

---

## Repository Architecture

```
ml-hackathon/
├── data/                      # Raw trip directories (.track, .nodes, .arcs, .route)
│   └── 00000086/
├── artifacts/                 # Processed candidate datasets, figures, and interactive maps
│   ├── candidates_00000086.csv
│   ├── maps/
│   │   └── trip_00000086_map.html
│   └── figures/
├── src/                       # Core modular source code
│   ├── __init__.py
│   ├── data_loader.py         # [Member 1] Parsers, coordinate conversions & DiGraph builder
│   ├── candidate_gen.py       # [Member 1] STRtree/KDTree spatial candidate matcher
│   ├── visualizer.py          # [Member 1 & 4] Multi-layer interactive Folium maps
│   ├── features.py            # [Member 2] Feature extraction pipeline
│   ├── train_models.py        # [Member 2] 5 distinct classification models
│   ├── bayesian_opt.py        # [Member 2] Optuna Bayesian hyperparameter search
│   └── hmm_viterbi.py         # [Member 3] HMM transition model & Viterbi sequential decoder
├── api/                       # [Member 4] FastAPI backend service (Laptop 1)
│   └── main.py
├── ui/                        # [Member 4] Evaluator Streamlit dashboard (Laptop 2)
│   └── app.py
├── notebooks/                 # Exploratory data analysis & experiment notebooks
│   └── 01_eda.ipynb           # [Member 1] Phase 1 exploratory data analysis
├── report/                    # Consolidated technical report documentation
│   ├── section_2_2_eda.md     # [Member 1] Dataset understanding and EDA report
│   └── figures/               # High-resolution plots for report compilation
├── requirements.txt           # Project dependencies
└── README.md
```

---

## Member 1 Deliverables & Implementations

### 1. Robust Parsers & Road Graph (`src/data_loader.py`)
- High-performance parsers for `.track`, `.nodes`, `.arcs`, and `.route` files without assuming headers.
- Geodesic coordinate projection (WGS84 $\rightarrow$ local metric Cartesian coordinates via equirectangular projection centered on trip centroid).
- Haversine distance, forward bearing/azimuth, and point-to-segment perpendicular distance functions.
- Directed road network constructor producing `networkx.DiGraph` with node attributes `(lon, lat, x, y)` and edge attributes `(arc_id, length, bearing, weight)`.

### 2. Spatial Indexing & Candidate Generation (`src/candidate_gen.py`)
- Two-tier spatial indexing utilizing `shapely.STRtree` (R-tree with bounding box hierarchy) and `scipy.spatial.KDTree`.
- Queries top-$K$ candidate road segments within a search radius (e.g. $K=10$, $R=100\text{ m}$).
- Sub-second execution: processes all 4,143 GPS observations against 237,656 road arcs in $\approx 0.5\text{ s}$.
- Achieves **$100.00\%$ candidate recall** ($0.00\%$ miss rate) at $K=10, R=100\text{ m}$.
- Outputs standardized tabular format compliant with **Contract 1** (`candidates_df`).

### 3. Interactive Geographical Visualizer (`src/visualizer.py`)
- Folium multi-layer interactive visualization saved to `artifacts/maps/trip_00000086_map.html`:
  - **Layer 1:** Raw GPS observations (blue circle markers with popups + dashed trace).
  - **Layer 2:** Relevant road network corridor segments (light gray).
  - **Layer 3:** Ground-truth reference route (green polyline).
  - **Layer 4:** Predicted reconstructed route (purple polyline).
  - Integrated layer controls, popups, and start/finish flags.

### 4. Exploratory Data Analysis (`notebooks/01_eda.ipynb` & `report/section_2_2_eda.md`)
- Complete data quality and missing-value audit.
- Analysis of the nominal 1 Hz GPS sampling rate ($96.98\%$ exactly $1.0\text{ s}$) and dropout anomalies.
- Kinematic velocity profile ($65.41\text{ km}$ trajectory length, mean speed $56.88\text{ km/h}$).
- Network degree distribution (mean degree $2.07$, max $7$) and segment length distribution (median $64.2\text{ m}$, mean $95.8\text{ m}$).
- Geometric distance separation analysis between ground truth arcs (mean $4.97\text{ m}$) and distractor candidates (mean $47.82\text{ m}$).

---

## Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/AtharvaGore/ml-hackathon.git
cd ml-hackathon

# Install dependencies
pip install -r requirements.txt

# Run candidate generation & Folium map generation
python -c "
from src.data_loader import load_trip, build_road_graph
from src.candidate_gen import SpatialCandidateGenerator
from src.visualizer import create_folium_map, save_folium_map

trip = load_trip('00000086')
G, aug_arcs = build_road_graph(trip['nodes'], trip['arcs'])
gen = SpatialCandidateGenerator(aug_arcs, origin_lon=trip['nodes']['lon'].mean(), origin_lat=trip['nodes']['lat'].mean())
cands = gen.generate_candidates('00000086', trip['track'], trip['route'], k=10, max_distance=100.0)
print('Candidates generated successfully:', cands.shape)

m = create_folium_map(trip['track'], trip['nodes'], trip['arcs'], trip['route'], candidate_arc_ids=cands['arc_id'].unique())
save_folium_map(m, 'artifacts/maps/trip_00000086_map.html')
print('Map saved to artifacts/maps/trip_00000086_map.html')
"
```
