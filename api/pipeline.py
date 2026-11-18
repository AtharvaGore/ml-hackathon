"""End-to-end map-matching pipeline orchestrator & Contract 1-3 adapter (Member 4).

Design goals:
  1. Zero merge conflicts: lives strictly inside `api/`, never modifies P1/P2/P3 files.
  2. Automatic plug-and-play integration:
       - If P1 (`src.data_loader`, `src.candidate_gen`), P2 (`src.features`,
         `artifacts/best_model.pkl`), and P3 (`src.hmm_viterbi.reconstruct_route`)
         are present, calls them directly via Contracts 1, 2, and 3.
       - Before teammates merge (or for isolated Laptop 1 <-> Laptop 2 testing),
         runs a realistic deterministic urban road-network & 10-trip evaluation
         benchmark using real NetworkX directed graphs and real metrics.
  3. Strict server-side ground truth: ground-truth routes are loaded and evaluated
     exclusively on Laptop 1.
"""

from __future__ import annotations

import importlib
import inspect
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two WGS84 coordinates."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _point_to_segment_distance_and_frac(
    lat: float,
    lon: float,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> tuple[float, float]:
    """Local-metric perpendicular distance (m) and projection fraction t in [0, 1]."""
    cos_lat = math.cos(math.radians(lat))
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * cos_lat

    px = (lon - lon1) * m_per_deg_lon
    py = (lat - lat1) * m_per_deg_lat
    dx = (lon2 - lon1) * m_per_deg_lon
    dy = (lat2 - lat1) * m_per_deg_lat

    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return math.hypot(px, py), 0.0

    t = max(0.0, min(1.0, (px * dx + py * dy) / seg_len_sq))
    proj_x = t * dx
    proj_y = t * dy
    return math.hypot(px - proj_x, py - proj_y), t


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute forward compass bearing in degrees [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees [0, 180]."""
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _collapse_consecutive(seq: Sequence[int]) -> list[int]:
    """Remove consecutive duplicates while preserving traversal order."""
    collapsed: list[int] = []
    for item in seq:
        if not collapsed or collapsed[-1] != item:
            collapsed.append(int(item))
    return collapsed


def _safe_import(module_name: str) -> Any | None:
    """Import an optional teammate module without raising if absent."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


@dataclass
class ComponentStatus:
    """Tracks which teammate modules & artifacts are actively wired."""

    data_loader: bool = False
    candidate_gen: bool = False
    features: bool = False
    model_artifact: bool = False
    hmm_viterbi: bool = False
    real_dataset_files: bool = False

    @property
    def mode(self) -> str:
        if all(
            [
                self.data_loader,
                self.candidate_gen,
                self.features,
                self.model_artifact,
                self.hmm_viterbi,
            ]
        ):
            return "full_real"
        if any(
            [
                self.data_loader,
                self.candidate_gen,
                self.features,
                self.model_artifact,
                self.hmm_viterbi,
                self.real_dataset_files,
            ]
        ):
            return "partial_real"
        return "mock_demo"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "p1_data_loader": self.data_loader,
            "p1_candidate_gen": self.candidate_gen,
            "p2_features": self.features,
            "p2_model_artifact": self.model_artifact,
            "p3_hmm_viterbi": self.hmm_viterbi,
            "real_dataset_files": self.real_dataset_files,
        }


class MapMatchingPipeline:
    """Server-side orchestrator for GPS trajectory map-matching and evaluation."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        self.status = ComponentStatus()

        self.graph: nx.DiGraph = nx.DiGraph()
        self.arc_lookup: dict[int, dict[str, Any]] = {}
        self.eval_trips: dict[str, dict[str, Any]] = {}
        self.model: Any | None = None

        # Optional teammate modules
        self._p1_loader: Any | None = None
        self._p1_cand: Any | None = None
        self._p2_features: Any | None = None
        self._p3_hmm: Any | None = None
        self._shortest_path_cache: dict[tuple[int, int], float] = {}

        self._initialize()

    # ------------------------------------------------------------------
    # Startup discovery & loading
    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        """Load available teammate modules, model artifact, road graph, and eval trips."""
        self._p1_loader = _safe_import("src.data_loader")
        self._p1_cand = _safe_import("src.candidate_gen")
        self._p2_features = _safe_import("src.features")
        self._p3_hmm = _safe_import("src.hmm_viterbi")

        self.status.data_loader = self._p1_loader is not None
        self.status.candidate_gen = self._p1_cand is not None
        self.status.features = self._p2_features is not None
        self.status.hmm_viterbi = bool(
            self._p3_hmm is not None and hasattr(self._p3_hmm, "reconstruct_route")
        )

        # Load serialized best_model.pkl if present
        if self.model_path.exists():
            self.model = self._load_model_artifact(self.model_path)
            self.status.model_artifact = self.model is not None

        # Attempt to load real road network + evaluation trips if present
        loaded_real = self._try_load_real_data()
        if not loaded_real:
            self._build_synthetic_urban_benchmark()

        self._rebuild_arc_lookup()

    @staticmethod
    def _load_model_artifact(path: Path) -> Any | None:
        """Load pickled/joblib model artifact safely."""
        try:
            import joblib

            return joblib.load(path)
        except Exception:
            try:
                with path.open("rb") as fh:
                    return pickle.load(fh)
            except Exception:
                return None

    def _try_load_real_data(self) -> bool:
        """Attempt to load real road graph & evaluation trips via P1 or `data/` files."""
        if not self.data_dir.exists():
            return False

        has_files = any(
            list(self.data_dir.rglob(f"*{ext}"))
            for ext in (".nodes", ".arcs", ".track", ".route")
        )
        self.status.real_dataset_files = has_files
        if not has_files:
            return False

        # Prefer Member 1's data_loader if it exposes a standard loader function
        if self._p1_loader is not None:
            for fn_name in ("load_evaluation_bundle", "load_dataset", "load_road_network"):
                fn: Callable[..., Any] | None = getattr(self._p1_loader, fn_name, None)
                if callable(fn):
                    try:
                        res = fn(str(self.data_dir))
                        if isinstance(res, dict) and "graph" in res and "eval_trips" in res:
                            self.graph = res["graph"]
                            self.eval_trips = res["eval_trips"]
                            return len(self.graph.edges) > 0 and len(self.eval_trips) > 0
                    except Exception:
                        pass

        # Fallback: try generic parser if standard .nodes / .arcs / .track / .route exist
        try:
            return self._load_generic_dataset_files()
        except Exception:
            return False

    def _load_generic_dataset_files(self) -> bool:
        """Generic whitespace/CSV parser for standard .nodes/.arcs/.track/.route files."""
        nodes_files = sorted(self.data_dir.rglob("*.nodes"))
        arcs_files = sorted(self.data_dir.rglob("*.arcs"))
        track_files = sorted(self.data_dir.rglob("*.track"))
        route_files = sorted(self.data_dir.rglob("*.route"))
        if not (nodes_files and arcs_files and track_files):
            return False

        graph = nx.DiGraph()
        # Parse nodes: expect id, lon/lat or lat/lon
        with nodes_files[0].open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.replace(",", " ").replace("\t", " ").split()
                if len(parts) < 3:
                    continue
                try:
                    nid = int(parts[0])
                    c1, c2 = float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                # Heuristic: latitude is in [-90, 90] and for typical datasets |lat| < |lon|
                lat, lon = (c1, c2) if abs(c1) <= 90 and abs(c2) > 90 else (c2, c1)
                graph.add_node(nid, lat=lat, lon=lon)

        # Parse arcs: expect arc_id, from_node, to_node[, length] OR from_node, to_node
        arc_counter = 0
        with arcs_files[0].open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.replace(",", " ").replace("\t", " ").split()
                if len(parts) < 2:
                    continue
                try:
                    if len(parts) >= 3:
                        first, second, third = int(parts[0]), int(parts[1]), int(parts[2])
                        if second in graph.nodes and third in graph.nodes:
                            aid, u, v = first, second, third
                        elif first in graph.nodes and second in graph.nodes:
                            aid, u, v = arc_counter, first, second
                        else:
                            continue
                    else:
                        aid, u, v = arc_counter, int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                if u not in graph.nodes or v not in graph.nodes:
                    continue
                u_n, v_n = graph.nodes[u], graph.nodes[v]
                length = haversine_m(u_n["lat"], u_n["lon"], v_n["lat"], v_n["lon"])
                graph.add_edge(
                    u,
                    v,
                    arc_id=aid,
                    length=length,
                    weight=length,
                    coords=[[u_n["lat"], u_n["lon"]], [v_n["lat"], v_n["lon"]]],
                )
                arc_counter += 1

        if graph.number_of_edges() == 0:
            return False

        route_by_stem = {p.stem: p for p in route_files}
        # Select last 10 tracks as evaluation trips if > 10 exist
        selected_tracks = track_files[-10:] if len(track_files) > 10 else track_files
        eval_trips: dict[str, dict[str, Any]] = {}

        for tf in selected_tracks:
            trip_id = tf.stem
            gps_points: list[dict[str, Any]] = []
            with tf.open("r", encoding="utf-8", errors="ignore") as fh:
                for step_idx, line in enumerate(fh):
                    parts = line.replace(",", " ").replace("\t", " ").split()
                    if len(parts) < 2:
                        continue
                    try:
                        vals = [float(p) for p in parts[:3]]
                    except ValueError:
                        continue
                    if len(vals) == 3:
                        lon, lat, ts = vals[0], vals[1], vals[2]
                        if abs(lon) <= 90 and abs(lat) > 90:
                            lat, lon = lon, lat
                    else:
                        lon, lat = vals[0], vals[1]
                        if abs(lon) <= 90 and abs(lat) > 90:
                            lat, lon = lon, lat
                        ts = float(step_idx * 3.0)
                    gps_points.append(
                        {
                            "step_idx": len(gps_points),
                            "timestamp": ts,
                            "lat": lat,
                            "lon": lon,
                        }
                    )

            gt_arcs: list[int] = []
            rf = route_by_stem.get(trip_id)
            if rf and rf.exists():
                with rf.open("r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        parts = line.replace(",", " ").replace("\t", " ").split()
                        if not parts:
                            continue
                        try:
                            gt_arcs.append(int(parts[0]))
                        except ValueError:
                            continue

            if gps_points:
                eval_trips[trip_id] = {
                    "trip_id": trip_id,
                    "gps_points": gps_points,
                    "ground_truth_arc_ids": gt_arcs,
                    "description": f"Real track loaded from {tf.name}",
                }

        if not eval_trips:
            return False

        self.graph = graph
        self.eval_trips = eval_trips
        return True

    def _build_synthetic_urban_benchmark(self) -> None:
        """Build a realistic deterministic urban directed road graph & 10 evaluation trips.

        Used during parallel development (Contract 3 mock mode) so Laptop 1 and Laptop 2
        can be fully tested over LAN before P1/P2/P3 merge their branches.
        """
        rng = np.random.default_rng(seed=42)
        rows, cols = 14, 16
        base_lat, base_lon = 12.9640, 77.5840  # Central urban corridor (Bengaluru scale)
        step_lat = 0.00115  # ~128 m north-south blocks
        step_lon = 0.00125  # ~135 m east-west blocks

        graph = nx.DiGraph()
        for r in range(rows):
            for c in range(cols):
                nid = r * cols + c
                # Subtle realistic intersection jitter (~10m)
                j_lat = float(rng.normal(0.0, 0.00009))
                j_lon = float(rng.normal(0.0, 0.00009))
                lat = base_lat + r * step_lat + j_lat
                lon = base_lon + c * step_lon + j_lon
                graph.add_node(nid, lat=lat, lon=lon, row=r, col=c)

        arc_id = 1000

        def add_road_arc(u: int, v: int, road_type: str = "street") -> None:
            nonlocal arc_id
            u_n, v_n = graph.nodes[u], graph.nodes[v]
            length = haversine_m(u_n["lat"], u_n["lon"], v_n["lat"], v_n["lon"])
            graph.add_edge(
                u,
                v,
                arc_id=arc_id,
                length=round(length, 2),
                weight=round(length, 2),
                road_type=road_type,
                coords=[[u_n["lat"], u_n["lon"]], [v_n["lat"], v_n["lon"]]],
            )
            arc_id += 1

        # Build arterials (two-way) and side streets (mixed one-way & two-way)
        for r in range(rows):
            for c in range(cols):
                u = r * cols + c
                if c + 1 < cols:
                    v = r * cols + (c + 1)
                    if r % 3 == 0:
                        add_road_arc(u, v, "arterial")
                        add_road_arc(v, u, "arterial")
                    elif r % 2 == 0:
                        add_road_arc(u, v, "one_way_east")
                    else:
                        add_road_arc(v, u, "one_way_west")
                if r + 1 < rows:
                    v = (r + 1) * cols + c
                    if c % 3 == 0:
                        add_road_arc(u, v, "arterial")
                        add_road_arc(v, u, "arterial")
                    elif c % 2 == 0:
                        add_road_arc(u, v, "one_way_north")
                    else:
                        add_road_arc(v, u, "one_way_south")
                # Occasional diagonal connector avenues
                if r + 1 < rows and c + 1 < cols and (r + c) % 5 == 0:
                    v_diag = (r + 1) * cols + (c + 1)
                    add_road_arc(u, v_diag, "diagonal")
                    add_road_arc(v_diag, u, "diagonal")

        self.graph = graph

        # Generate 10 diverse evaluation trips (varying origins, lengths, noise profiles)
        trip_specs = [
            ("eval_trip_01", 0, 14, 7.5, 0.04, "Morning arterial commute (low noise)"),
            ("eval_trip_02", 18, 16, 10.0, 0.07, "Downtown one-way grid corridor"),
            ("eval_trip_03", 35, 19, 12.0, 0.10, "Urban canyon multipath stress test"),
            ("eval_trip_04", 5, 15, 8.5, 0.05, "Cross-town diagonal express route"),
            ("eval_trip_05", 48, 21, 11.5, 0.08, "Multi-turn residential & arterial mix"),
            ("eval_trip_06", 12, 13, 9.0, 0.06, "Short dense intersection loop"),
            ("eval_trip_07", 64, 18, 13.5, 0.11, "High-noise GPS dropout corridor"),
            ("eval_trip_08", 22, 17, 8.0, 0.05, "Steady sampling highway feeder"),
            ("eval_trip_09", 96, 15, 10.5, 0.08, "Reverse peak-hour avenue transit"),
            ("eval_trip_10", 3, 22, 11.0, 0.09, "Long cross-city benchmark trajectory"),
        ]

        eval_trips: dict[str, dict[str, Any]] = {}
        for idx, (trip_id, start_node, target_hops, noise_sigma_m, outlier_prob, desc) in enumerate(
            trip_specs, start=1
        ):
            trip_rng = np.random.default_rng(seed=1000 + idx * 37)
            path_edges = self._sample_directed_walk(graph, start_node, target_hops, trip_rng)

            gps_points: list[dict[str, Any]] = []
            gt_arc_ids: list[int] = []
            timestamp = 0.0

            for u, v in path_edges:
                edge = graph[u][v]
                aid = int(edge["arc_id"])
                u_lat, u_lon = graph.nodes[u]["lat"], graph.nodes[u]["lon"]
                v_lat, v_lon = graph.nodes[v]["lat"], graph.nodes[v]["lon"]

                # Sample 2 to 4 GPS observations along each road arc
                n_obs = int(trip_rng.integers(2, 5))
                fracs = np.linspace(0.15, 0.88, n_obs)
                for frac in fracs:
                    true_lat = u_lat + float(frac) * (v_lat - u_lat)
                    true_lon = u_lon + float(frac) * (v_lon - u_lon)

                    sigma = noise_sigma_m
                    if float(trip_rng.random()) < outlier_prob:
                        sigma = noise_sigma_m * 3.1  # Urban canyon multipath spike

                    d_north_m = float(trip_rng.normal(0.0, sigma))
                    d_east_m = float(trip_rng.normal(0.0, sigma))
                    noisy_lat = true_lat + d_north_m / 111_132.0
                    noisy_lon = true_lon + d_east_m / (
                        111_320.0 * math.cos(math.radians(true_lat))
                    )

                    gps_points.append(
                        {
                            "step_idx": len(gps_points),
                            "timestamp": round(timestamp, 1),
                            "lat": round(noisy_lat, 6),
                            "lon": round(noisy_lon, 6),
                        }
                    )
                    gt_arc_ids.append(aid)
                    timestamp += float(trip_rng.choice([2.0, 3.0, 4.0]))

            eval_trips[trip_id] = {
                "trip_id": trip_id,
                "gps_points": gps_points,
                "ground_truth_arc_ids": gt_arc_ids,
                "description": desc,
            }

        self.eval_trips = eval_trips

    @staticmethod
    def _sample_directed_walk(
        graph: nx.DiGraph,
        start_node: int,
        hops: int,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        """Sample a realistic non-backtracking directed path through the road network."""
        curr = start_node if start_node in graph.nodes else next(iter(graph.nodes))
        visited_nodes = {curr}
        prev_node: int | None = None
        edges: list[tuple[int, int]] = []

        for _ in range(hops):
            neighbors = list(graph.successors(curr))
            if not neighbors:
                break
            # Prefer forward, unvisited neighbors (avoid immediate U-turns)
            forward = [n for n in neighbors if n != prev_node and n not in visited_nodes]
            if not forward:
                forward = [n for n in neighbors if n != prev_node] or neighbors

            # Gently bias toward continuing in a consistent direction
            if prev_node is not None and len(forward) > 1:
                in_bearing = _bearing_deg(
                    graph.nodes[prev_node]["lat"],
                    graph.nodes[prev_node]["lon"],
                    graph.nodes[curr]["lat"],
                    graph.nodes[curr]["lon"],
                )
                weights = []
                for nxt in forward:
                    out_b = _bearing_deg(
                        graph.nodes[curr]["lat"],
                        graph.nodes[curr]["lon"],
                        graph.nodes[nxt]["lat"],
                        graph.nodes[nxt]["lon"],
                    )
                    turn = _angular_diff_deg(in_bearing, out_b)
                    weights.append(3.0 if turn < 35 else (1.5 if turn < 105 else 0.4))
                probs = np.array(weights, dtype=float)
                probs /= probs.sum()
                nxt = int(rng.choice(forward, p=probs))
            else:
                nxt = int(rng.choice(forward))

            edges.append((curr, nxt))
            prev_node = curr
            curr = nxt
            visited_nodes.add(curr)

        return edges

    def _rebuild_arc_lookup(self) -> None:
        """Index every directed edge by its arc_id for O(1) lookup."""
        self.arc_lookup.clear()
        self._shortest_path_cache.clear()
        for u, v, data in self.graph.edges(data=True):
            aid = int(data.get("arc_id", -1))
            if aid == -1:
                continue
            coords = data.get("coords")
            if not coords:
                u_n, v_n = self.graph.nodes[u], self.graph.nodes[v]
                coords = [[float(u_n["lat"]), float(u_n["lon"])], [float(v_n["lat"]), float(v_n["lon"])]]
            self.arc_lookup[aid] = {
                "arc_id": aid,
                "from_node": int(u) if isinstance(u, int) else u,
                "to_node": int(v) if isinstance(v, int) else v,
                "length": float(data.get("length", data.get("weight", 0.0))),
                "coords": coords,
            }

    # ------------------------------------------------------------------
    # Public inspection helpers
    # ------------------------------------------------------------------
    def list_eval_trips(self) -> list[dict[str, Any]]:
        """Return safe metadata for the evaluation trips (never exposes ground truth)."""
        summaries: list[dict[str, Any]] = []
        for trip_id in sorted(self.eval_trips.keys()):
            trip = self.eval_trips[trip_id]
            pts = trip["gps_points"]
            duration = (
                round(float(pts[-1]["timestamp"]) - float(pts[0]["timestamp"]), 1)
                if len(pts) >= 2
                else 0.0
            )
            summaries.append(
                {
                    "trip_id": trip_id,
                    "num_points": len(pts),
                    "duration_s": duration,
                    "description": trip.get("description", ""),
                }
            )
        return summaries

    def health_summary(self) -> dict[str, Any]:
        """Return health & diagnostic status for GET /health."""
        return {
            "status": "ok",
            "pipeline_mode": self.status.mode,
            "components": self.status.to_dict(),
            "graph_stats": {
                "num_nodes": self.graph.number_of_nodes(),
                "num_arcs": self.graph.number_of_edges(),
            },
            "num_eval_trips": len(self.eval_trips),
            "eval_trip_ids": sorted(self.eval_trips.keys()),
            "model_path": str(self.model_path),
            "model_loaded": self.model is not None,
        }

    # ------------------------------------------------------------------
    # Stage 2, 3, 4: Candidates -> Classifier -> HMM Viterbi
    # ------------------------------------------------------------------
    def _generate_candidates(
        self,
        gps_points: list[dict[str, Any]],
        k: int = 6,
        max_radius_m: float = 160.0,
    ) -> list[list[dict[str, Any]]]:
        """Generate top-K nearest candidate arcs for each GPS observation."""
        all_arcs = list(self.arc_lookup.values())
        candidates_per_step: list[list[dict[str, Any]]] = []

        for idx, pt in enumerate(gps_points):
            lat, lon = float(pt["lat"]), float(pt["lon"])

            # Compute GPS movement heading for feature enrichment
            if idx + 1 < len(gps_points):
                gps_heading = _bearing_deg(
                    lat, lon, float(gps_points[idx + 1]["lat"]), float(gps_points[idx + 1]["lon"])
                )
            elif idx > 0:
                gps_heading = _bearing_deg(
                    float(gps_points[idx - 1]["lat"]), float(gps_points[idx - 1]["lon"]), lat, lon
                )
            else:
                gps_heading = 0.0

            scored: list[dict[str, Any]] = []
            for arc in all_arcs:
                c0, c1 = arc["coords"][0], arc["coords"][-1]
                # Fast rough lat/lon bounding box reject (~250m)
                if (
                    abs(lat - 0.5 * (c0[0] + c1[0])) > 0.003
                    or abs(lon - 0.5 * (c0[1] + c1[1])) > 0.003
                ):
                    continue
                dist_m, proj_frac = _point_to_segment_distance_and_frac(
                    lat, lon, c0[0], c0[1], c1[0], c1[1]
                )
                if dist_m <= max_radius_m:
                    arc_bearing = _bearing_deg(c0[0], c0[1], c1[0], c1[1])
                    heading_delta = _angular_diff_deg(gps_heading, arc_bearing)
                    scored.append(
                        {
                            "arc_id": arc["arc_id"],
                            "from_node": arc["from_node"],
                            "to_node": arc["to_node"],
                            "length": arc["length"],
                            "distance_m": dist_m,
                            "proj_frac": proj_frac,
                            "heading_delta_deg": heading_delta,
                        }
                    )

            # Fallback if GPS point is far from every arc
            if not scored:
                for arc in all_arcs:
                    c0, c1 = arc["coords"][0], arc["coords"][-1]
                    dist_m, proj_frac = _point_to_segment_distance_and_frac(
                        lat, lon, c0[0], c0[1], c1[0], c1[1]
                    )
                    arc_bearing = _bearing_deg(c0[0], c0[1], c1[0], c1[1])
                    scored.append(
                        {
                            "arc_id": arc["arc_id"],
                            "from_node": arc["from_node"],
                            "to_node": arc["to_node"],
                            "length": arc["length"],
                            "distance_m": dist_m,
                            "proj_frac": proj_frac,
                            "heading_delta_deg": _angular_diff_deg(gps_heading, arc_bearing),
                        }
                    )

            scored.sort(key=lambda item: item["distance_m"])
            candidates_per_step.append(scored[:k])

        return candidates_per_step

    def _score_candidates(
        self,
        candidates_per_step: list[list[dict[str, Any]]],
    ) -> tuple[list[list[float]], list[int]]:
        """Score candidate arcs (emission probabilities) and return greedy Top-1 sequence."""
        # If Member 2's feature extractor + model are available, delegate via Contract 2
        if self.model is not None and self._p2_features is not None:
            for fn_name in ("extract_features", "compute_features", "build_features"):
                feat_fn = getattr(self._p2_features, fn_name, None)
                if callable(feat_fn):
                    try:
                        all_probs: list[list[float]] = []
                        greedy_ids: list[int] = []
                        for step_cands in candidates_per_step:
                            X = feat_fn(step_cands, self.graph)
                            probs = self.model.predict_proba(X)[:, 1].tolist()
                            all_probs.append([float(p) for p in probs])
                            best_idx = int(np.argmax(probs))
                            greedy_ids.append(int(step_cands[best_idx]["arc_id"]))
                        return all_probs, greedy_ids
                    except Exception:
                        break

        # Fallback: Gaussian distance + mild heading emission model (Contract 2 compatible)
        sigma_dist = 14.0
        sigma_head = 55.0
        all_probs = []
        greedy_ids = []

        for step_cands in candidates_per_step:
            raw_scores: list[float] = []
            for cand in step_cands:
                d = float(cand["distance_m"])
                hd = float(cand["heading_delta_deg"])
                # Emission likelihood combining perpendicular distance & heading alignment
                score = math.exp(-0.5 * (d / sigma_dist) ** 2) * math.exp(
                    -0.35 * (hd / sigma_head) ** 2
                )
                raw_scores.append(max(score, 1e-9))

            total = sum(raw_scores) or 1.0
            probs = [s / total for s in raw_scores]
            all_probs.append(probs)

            # Greedy classifier-only picks highest pointwise emission score (no graph transition)
            best_idx = int(np.argmax(probs))
            greedy_ids.append(int(step_cands[best_idx]["arc_id"]))

        return all_probs, greedy_ids

    def _network_distance_m(self, from_node: int, to_node: int, max_cutoff_m: float = 900.0) -> float:
        """Cached shortest directed road-network distance between two nodes."""
        if from_node == to_node:
            return 0.0
        key = (from_node, to_node)
        if key in self._shortest_path_cache:
            return self._shortest_path_cache[key]
        try:
            dist = float(
                nx.shortest_path_length(
                    self.graph, source=from_node, target=to_node, weight="length"
                )
            )
            if dist > max_cutoff_m:
                dist = math.inf
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            dist = math.inf
        self._shortest_path_cache[key] = dist
        return dist

    def _run_hmm_viterbi(
        self,
        trip_id: str,
        gps_points: list[dict[str, Any]],
        candidates_per_step: list[list[dict[str, Any]]],
        emission_probs: list[list[float]],
    ) -> list[int]:
        """Run Member 3's Contract 3 `reconstruct_route` if present, else fallback Viterbi."""
        if self._p3_hmm is not None and hasattr(self._p3_hmm, "reconstruct_route"):
            fn = self._p3_hmm.reconstruct_route
            sig = inspect.signature(fn)
            kwargs: dict[str, Any] = {}
            if "model" in sig.parameters and self.model is not None:
                kwargs["model"] = self.model
            if "candidates" in sig.parameters:
                kwargs["candidates"] = candidates_per_step
            if "emission_probs" in sig.parameters:
                kwargs["emission_probs"] = emission_probs
            try:
                result = fn(trip_id, gps_points, self.graph, **kwargs)
                if isinstance(result, (list, tuple)) and len(result) > 0:
                    return [int(a) for a in result]
            except Exception:
                pass

        # Built-in fallback log-space HMM Viterbi decoder (Newson & Krumm transition formulation)
        n_steps = len(gps_points)
        if n_steps == 0:
            return []

        beta = 22.0  # Transition exponential scale parameter (meters)
        dp: list[list[float]] = []
        backptr: list[list[int]] = []

        # Step 0 initialization
        dp.append([math.log(max(p, 1e-12)) for p in emission_probs[0]])
        backptr.append([-1] * len(candidates_per_step[0]))

        for t in range(1, n_steps):
            gps_disp = haversine_m(
                float(gps_points[t - 1]["lat"]),
                float(gps_points[t - 1]["lon"]),
                float(gps_points[t]["lat"]),
                float(gps_points[t]["lon"]),
            )
            prev_cands = candidates_per_step[t - 1]
            curr_cands = candidates_per_step[t]
            curr_dp: list[float] = []
            curr_bp: list[int] = []

            for j, c_curr in enumerate(curr_cands):
                log_emit = math.log(max(emission_probs[t][j], 1e-12))
                best_score = -math.inf
                best_prev = 0

                for i, c_prev in enumerate(prev_cands):
                    f_prev = float(c_prev.get("proj_frac", 0.5))
                    f_curr = float(c_curr.get("proj_frac", 0.5))
                    if c_prev["arc_id"] == c_curr["arc_id"]:
                        # Same arc: forward along-track distance; penalize backward jumps
                        along = (f_curr - f_prev) * c_prev["length"]
                        net_dist = abs(along)
                        back_pen = -3.0 if along < -15.0 else 0.0
                        trans_pen = -abs(net_dist - gps_disp) / beta + back_pen
                    elif c_prev["to_node"] == c_curr["from_node"]:
                        # Direct adjacent transition along directed road graph
                        net_dist = (1.0 - f_prev) * c_prev["length"] + f_curr * c_curr["length"]
                        trans_pen = -abs(net_dist - gps_disp) / beta
                    elif (
                        c_prev["from_node"] == c_curr["to_node"]
                        and c_prev["to_node"] == c_curr["from_node"]
                    ):
                        # Immediate U-turn penalty
                        trans_pen = -14.0
                    else:
                        sp = self._network_distance_m(c_prev["to_node"], c_curr["from_node"])
                        if math.isfinite(sp):
                            net_dist = (
                                (1.0 - f_prev) * c_prev["length"] + sp + f_curr * c_curr["length"]
                            )
                            trans_pen = -abs(net_dist - gps_disp) / beta - 0.5
                        else:
                            # Unreachable pair fallback penalty
                            trans_pen = -25.0

                    score = dp[t - 1][i] + trans_pen + log_emit
                    if score > best_score:
                        best_score = score
                        best_prev = i

                curr_dp.append(best_score)
                curr_bp.append(best_prev)

            dp.append(curr_dp)
            backptr.append(curr_bp)

        # Backtrack optimal state sequence
        best_last = int(np.argmax(dp[-1]))
        seq_indices = [best_last]
        for t in range(n_steps - 1, 0, -1):
            seq_indices.append(backptr[t][seq_indices[-1]])
        seq_indices.reverse()

        return [int(candidates_per_step[t][idx]["arc_id"]) for t, idx in enumerate(seq_indices)]

    # ------------------------------------------------------------------
    # Stage 5: Metrics & Bounding-Box Local Road Graph Extraction
    # ------------------------------------------------------------------
    def _compute_continuity(self, arc_sequence: Sequence[int]) -> dict[str, Any]:
        """Measure physical road-network continuity of a predicted arc sequence."""
        collapsed = _collapse_consecutive(arc_sequence)
        if len(collapsed) <= 1:
            return {
                "route_continuity": 1.0,
                "direct_continuity": 1.0,
                "transitions": 0,
                "valid_transitions": 0,
                "discontinuities": 0,
                "transition_flags": [],
            }

        valid_transitions = 0
        direct_transitions = 0
        transition_flags: list[bool] = []

        for a_prev, a_curr in zip(collapsed[:-1], collapsed[1:]):
            prev_rec = self.arc_lookup.get(a_prev)
            curr_rec = self.arc_lookup.get(a_curr)
            if not prev_rec or not curr_rec:
                transition_flags.append(False)
                continue

            if prev_rec["to_node"] == curr_rec["from_node"]:
                direct_transitions += 1
                valid_transitions += 1
                transition_flags.append(True)
            else:
                # Allow short connected bridge (<= 3 hops) when GPS skipped an intermediate short arc
                try:
                    hop_len = nx.shortest_path_length(
                        self.graph, source=prev_rec["to_node"], target=curr_rec["from_node"]
                    )
                    is_connected = hop_len <= 3
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    is_connected = False

                if is_connected:
                    valid_transitions += 1
                transition_flags.append(is_connected)

        total = len(collapsed) - 1
        return {
            "route_continuity": round(valid_transitions / total, 4),
            "direct_continuity": round(direct_transitions / total, 4),
            "transitions": total,
            "valid_transitions": valid_transitions,
            "discontinuities": total - valid_transitions,
            "transition_flags": transition_flags,
        }

    def _compute_metrics(
        self,
        gt_arc_ids: Sequence[int],
        pred_arc_ids: Sequence[int],
        greedy_arc_ids: Sequence[int],
    ) -> dict[str, Any]:
        """Compute Top-1 Accuracy, Route Continuity, F1, Jaccard, and HMM uplift."""
        pred_cont = self._compute_continuity(pred_arc_ids)
        greedy_cont = self._compute_continuity(greedy_arc_ids)

        gt_collapsed = _collapse_consecutive(gt_arc_ids)
        pred_collapsed = _collapse_consecutive(pred_arc_ids)

        # Pointwise Top-1 accuracy (when aligned per GPS step); otherwise arc-sequence overlap
        if len(gt_arc_ids) == len(pred_arc_ids) and len(gt_arc_ids) > 0:
            top1_acc = sum(1 for g, p in zip(gt_arc_ids, pred_arc_ids) if g == p) / len(gt_arc_ids)
            greedy_top1 = sum(1 for g, p in zip(gt_arc_ids, greedy_arc_ids) if g == p) / len(
                gt_arc_ids
            )
        elif gt_collapsed and pred_collapsed:
            gt_set = set(gt_collapsed)
            top1_acc = sum(1 for p in pred_arc_ids if p in gt_set) / max(1, len(pred_arc_ids))
            greedy_top1 = sum(1 for p in greedy_arc_ids if p in gt_set) / max(
                1, len(greedy_arc_ids)
            )
        else:
            top1_acc = 0.0
            greedy_top1 = 0.0

        gt_set = set(gt_collapsed)
        pred_set = set(pred_collapsed)
        tp = len(gt_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gt_set) if gt_set else 0.0
        route_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        jaccard = tp / len(gt_set | pred_set) if (gt_set | pred_set) else 0.0

        return {
            "top1_accuracy": round(top1_acc, 4),
            "route_continuity": pred_cont["route_continuity"],
            "direct_continuity": pred_cont["direct_continuity"],
            "discontinuities": pred_cont["discontinuities"],
            "route_precision": round(precision, 4),
            "route_recall": round(recall, 4),
            "route_f1": round(route_f1, 4),
            "sequence_jaccard": round(jaccard, 4),
            "classifier_only_top1_accuracy": round(greedy_top1, 4),
            "classifier_only_continuity": greedy_cont["route_continuity"],
            "classifier_only_discontinuities": greedy_cont["discontinuities"],
            "hmm_accuracy_gain": round(top1_acc - greedy_top1, 4),
            "hmm_continuity_gain": round(
                pred_cont["route_continuity"] - greedy_cont["route_continuity"], 4
            ),
            "num_gt_unique_arcs": len(gt_collapsed),
            "num_pred_unique_arcs": len(pred_collapsed),
        }

    def _extract_local_road_arcs(
        self,
        gps_points: Sequence[dict[str, Any]],
        pad_deg: float = 0.0022,
        max_arcs: int = 550,
    ) -> list[dict[str, Any]]:
        """Extract road network arcs within the trip bounding box for fast LAN transfer."""
        if not gps_points:
            return list(self.arc_lookup.values())[:max_arcs]

        lats = [float(p["lat"]) for p in gps_points]
        lons = [float(p["lon"]) for p in gps_points]
        min_lat, max_lat = min(lats) - pad_deg, max(lats) + pad_deg
        min_lon, max_lon = min(lons) - pad_deg, max(lons) + pad_deg

        local: list[dict[str, Any]] = []
        for arc in self.arc_lookup.values():
            if any(min_lat <= c[0] <= max_lat and min_lon <= c[1] <= max_lon for c in arc["coords"]):
                local.append(arc)
                if len(local) >= max_arcs:
                    break
        return local

    # ------------------------------------------------------------------
    # Main Orchestration Entrypoint
    # ------------------------------------------------------------------
    def predict_trip(self, trip_id: str) -> dict[str, Any]:
        """Run full pipeline for `trip_id` and return routes, geometries, and metrics."""
        if trip_id not in self.eval_trips:
            raise KeyError(f"Unknown trip_id '{trip_id}'. Available: {sorted(self.eval_trips)}")

        t_start = time.perf_counter()
        trip = self.eval_trips[trip_id]
        gps_points: list[dict[str, Any]] = trip["gps_points"]
        gt_arc_ids: list[int] = list(trip["ground_truth_arc_ids"])

        # Stage 2: Candidate generation
        t0 = time.perf_counter()
        candidates_per_step = self._generate_candidates(gps_points, k=6)
        candidate_ms = (time.perf_counter() - t0) * 1000.0
        total_candidates = sum(len(c) for c in candidates_per_step)

        # Stage 3: Emission probability scoring (Classifier)
        t1 = time.perf_counter()
        emission_probs, greedy_arc_ids = self._score_candidates(candidates_per_step)
        inference_ms = (time.perf_counter() - t1) * 1000.0

        # Stage 4: HMM Viterbi sequential decoding
        t2 = time.perf_counter()
        pred_arc_ids = self._run_hmm_viterbi(
            trip_id=trip_id,
            gps_points=gps_points,
            candidates_per_step=candidates_per_step,
            emission_probs=emission_probs,
        )
        viterbi_ms = (time.perf_counter() - t2) * 1000.0

        # Stage 5: Server-side ground-truth evaluation & geometry packing
        t3 = time.perf_counter()
        metrics = self._compute_metrics(gt_arc_ids, pred_arc_ids, greedy_arc_ids)
        gt_unique = _collapse_consecutive(gt_arc_ids)
        pred_unique = _collapse_consecutive(pred_arc_ids)

        gt_arcs = [self.arc_lookup[a] for a in gt_unique if a in self.arc_lookup]
        pred_arcs = [self.arc_lookup[a] for a in pred_unique if a in self.arc_lookup]
        local_road_arcs = self._extract_local_road_arcs(gps_points)

        step_details: list[dict[str, Any]] = []
        for idx, pt in enumerate(gps_points):
            gt_a = gt_arc_ids[idx] if idx < len(gt_arc_ids) else None
            pred_a = pred_arc_ids[idx] if idx < len(pred_arc_ids) else None
            greedy_a = greedy_arc_ids[idx] if idx < len(greedy_arc_ids) else None
            step_details.append(
                {
                    "step_idx": int(pt["step_idx"]),
                    "timestamp": float(pt["timestamp"]),
                    "lat": float(pt["lat"]),
                    "lon": float(pt["lon"]),
                    "ground_truth_arc_id": gt_a,
                    "classifier_only_arc_id": greedy_a,
                    "predicted_arc_id": pred_a,
                    "is_match": (gt_a == pred_a) if (gt_a is not None and pred_a is not None) else None,
                }
            )

        metrics_ms = (time.perf_counter() - t3) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0
        metrics["processing_time_ms"] = round(total_ms, 2)

        stage_timings_ms = {
            "candidate_gen_ms": round(candidate_ms, 2),
            "model_inference_ms": round(inference_ms, 2),
            "hmm_viterbi_ms": round(viterbi_ms, 2),
            "metrics_and_pack_ms": round(metrics_ms, 2),
            "total_ms": round(total_ms, 2),
        }

        return {
            "trip_id": trip_id,
            "description": trip.get("description", ""),
            "pipeline_mode": self.status.mode,
            "num_gps_points": len(gps_points),
            "num_candidate_pairs": total_candidates,
            "metrics": metrics,
            "stage_timings_ms": stage_timings_ms,
            "gps_points": gps_points,
            "road_network_arcs": local_road_arcs,
            "ground_truth_arc_ids": gt_unique,
            "predicted_arc_ids": pred_unique,
            "classifier_only_arc_ids": _collapse_consecutive(greedy_arc_ids),
            "ground_truth_arcs": gt_arcs,
            "predicted_arcs": pred_arcs,
            "step_details": step_details,
        }
