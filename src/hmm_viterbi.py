"""
PathSnap HMM & Viterbi Sequential Route Reconstruction Engine.

Implements the Hidden Markov Model (HMM) map-matching framework (Newson & Krumm 2009)
with extensions for classifier-probability fusion, GPS gap handling,
bounded graph search, and comprehensive evaluation metrics.
"""

import math
import time
import json
from typing import Any, Optional, Dict, List, Tuple
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from src.spatial_utils import project_to_meters, point_to_segment_project


class MapMatchingHMM:
    """
    Hidden Markov Model decoder for map-matching noisy GPS trajectories
    to directed road networks.
    """

    def __init__(
        self,
        sigma_z: float = 7.5,
        beta: float = 7.0,
        search_radius: float = 50.0,
        max_candidates: int = 10,
        disconnected_penalty: float = -35.0,
        max_speed_ms: float = 45.0,
        model_weight: float = 0.5,
    ):
        """
        Parameters:
            sigma_z: Standard deviation of GPS observation error (meters).
            beta: Exponential distribution scale parameter for transition error (meters).
            search_radius: Spatial radius (meters) for candidate segment queries.
            max_candidates: Maximum number of candidate road segments per observation.
            disconnected_penalty: Log probability penalty for graph disconnections.
            max_speed_ms: Maximum plausible vehicle speed (45 m/s = 162 km/h).
            model_weight: Weight alpha in [0, 1] for fusing classifier scores with Gaussian emission.
        """
        self.sigma_z = float(sigma_z)
        self.beta = float(beta)
        self.search_radius = float(search_radius)
        self.max_candidates = int(max_candidates)
        self.disconnected_penalty = float(disconnected_penalty)
        self.max_speed_ms = float(max_speed_ms)
        self.model_weight = float(model_weight)

    def compute_emission_log_prob(
        self, dist: float, classifier_score: float = None
    ) -> float:
        """
        Computes the log emission probability ln P(z_t | c_i).
        Combines Gaussian distance error with optional upstream classifier probability.
        """
        # Log of Gaussian error model: 1/(sqrt(2pi)*sigma) * exp(-dist^2 / (2*sigma^2))
        log_gaussian = -0.5 * (dist / self.sigma_z) ** 2 - math.log(
            math.sqrt(2.0 * math.pi) * self.sigma_z
        )

        if classifier_score is not None:
            # Clamp classifier probability to avoid log(0)
            p_clf = max(1e-6, min(1.0 - 1e-6, float(classifier_score)))
            log_clf = math.log(p_clf)
            # Log-linear fusion
            return (1.0 - self.model_weight) * log_gaussian + self.model_weight * log_clf

        return log_gaussian

    def generate_candidates(
        self,
        x_track: np.ndarray,
        y_track: np.ndarray,
        x_nodes: np.ndarray,
        y_nodes: np.ndarray,
        arcs: np.ndarray,
        arc_lengths: np.ndarray,
        classifier_probs: np.ndarray = None,
    ) -> list:
        """
        Spatial query using cKDTree on arc midpoints followed by exact orthogonal projection.
        Returns list of candidate dictionaries for each observation step t.
        """
        T = len(x_track)
        u_idx = arcs[:, 0]
        v_idx = arcs[:, 1]

        # Calculate arc midpoints
        arc_mid_x = 0.5 * (x_nodes[u_idx] + x_nodes[v_idx])
        arc_mid_y = 0.5 * (y_nodes[u_idx] + y_nodes[v_idx])

        # Build KD-Tree on midpoints
        tree = cKDTree(np.column_stack([arc_mid_x, arc_mid_y]))

        # Query nearest arc midpoints (query k=30 to ensure full coverage within radius)
        k_query = min(35, len(arcs))
        _, indices = tree.query(np.column_stack([x_track, y_track]), k=k_query)

        candidates_per_step = []
        for t in range(T):
            px, py = x_track[t], y_track[t]
            cands = []

            for arc_id in indices[t]:
                u, v = u_idx[arc_id], v_idx[arc_id]
                d, t_proj, proj_x, proj_y = point_to_segment_project(
                    px, py, x_nodes[u], y_nodes[u], x_nodes[v], y_nodes[v]
                )

                if d <= self.search_radius:
                    clf_score = (
                        float(classifier_probs[t, arc_id])
                        if classifier_probs is not None
                        else None
                    )
                    log_em = self.compute_emission_log_prob(d, clf_score)

                    cands.append(
                        {
                            "arc_id": int(arc_id),
                            "u": int(u),
                            "v": int(v),
                            "dist": float(d),
                            "t_proj": float(t_proj),
                            "length": float(arc_lengths[arc_id]),
                            "log_emission": float(log_em),
                            "proj_x": float(proj_x),
                            "proj_y": float(proj_y),
                        }
                    )

            # Sort by emission probability (highest log emission first)
            cands.sort(key=lambda c: c["log_emission"], reverse=True)

            # Robust fallback: if no arc within search_radius, take closest arc
            if not cands:
                top_arc = indices[t][0]
                u, v = u_idx[top_arc], v_idx[top_arc]
                d, t_proj, proj_x, proj_y = point_to_segment_project(
                    px, py, x_nodes[u], y_nodes[u], x_nodes[v], y_nodes[v]
                )
                cands.append(
                    {
                        "arc_id": int(top_arc),
                        "u": int(u),
                        "v": int(v),
                        "dist": float(d),
                        "t_proj": float(t_proj),
                        "length": float(arc_lengths[top_arc]),
                        "log_emission": self.compute_emission_log_prob(d, None),
                        "proj_x": float(proj_x),
                        "proj_y": float(proj_y),
                    }
                )

            candidates_per_step.append(cands[: self.max_candidates])

        return candidates_per_step

    def viterbi_decode(
        self,
        track: np.ndarray,
        nodes: np.ndarray,
        arcs: np.ndarray,
        G: nx.DiGraph,
        meta: dict,
        classifier_probs: np.ndarray = None,
    ) -> dict:
        """
        Executes the full Viterbi dynamic programming trellis.
        Returns predicted arc sequence, classifier-only baseline, and timing.
        """
        t_start = time.time()
        x_nodes, y_nodes = meta["x_nodes"], meta["y_nodes"]
        arc_lengths = meta["arc_lengths"]
        ref_lon, ref_lat = meta["ref_lon"], meta["ref_lat"]

        # Metric projection of GPS track
        x_track, y_track = project_to_meters(
            track[:, 0], track[:, 1], ref_lon, ref_lat
        )
        T = len(track)

        # 1. Candidate Generation
        t_cand_start = time.time()
        candidates_per_step = self.generate_candidates(
            x_track,
            y_track,
            x_nodes,
            y_nodes,
            arcs,
            arc_lengths,
            classifier_probs,
        )
        cand_time = time.time() - t_cand_start

        # Bounded Dijkstra cache
        sp_cache = {}

        def get_shortest_path(u: int, v: int, cutoff: float) -> float:
            if u == v:
                return 0.0
            # Round cutoff to reduce cache churn
            rounded_cutoff = math.ceil(cutoff / 50.0) * 50.0
            cache_key = (u, rounded_cutoff)
            if cache_key not in sp_cache:
                sp_cache[cache_key] = nx.single_source_dijkstra_path_length(
                    G, u, cutoff=rounded_cutoff, weight="weight"
                )
            return sp_cache[cache_key].get(v, None)

        # 2. Viterbi Forward Trellis in Log Space
        t_trellis_start = time.time()
        V = [{i: c["log_emission"] for i, c in enumerate(candidates_per_step[0])}]
        BP = [{}]

        for t in range(1, T):
            V_t = {}
            BP_t = {}
            prev_cands = candidates_per_step[t - 1]
            curr_cands = candidates_per_step[t]

            # GPS time and displacement
            dt = max(1e-3, track[t, 2] - track[t - 1, 2])
            d_gps = math.hypot(
                x_track[t] - x_track[t - 1], y_track[t] - y_track[t - 1]
            )

            # Adaptive beta for GPS gaps
            beta_eff = self.beta * (1.0 + max(0.0, (dt - 2.0) / 5.0))
            max_plausible_dist = max(150.0, self.max_speed_ms * dt + 50.0)

            for i, c_curr in enumerate(curr_cands):
                best_log_prob = -float("inf")
                best_prev_idx = 0

                for j, c_prev in enumerate(prev_cands):
                    # Compute network distance d_net
                    if c_prev["arc_id"] == c_curr["arc_id"]:
                        # Same road segment
                        delta_proj = c_curr["t_proj"] - c_prev["t_proj"]
                        if delta_proj >= 0.0:
                            d_net = delta_proj * c_curr["length"]
                        else:
                            # Slight backward jitter along arc
                            jitter_dist = abs(delta_proj) * c_curr["length"]
                            d_net = jitter_dist if jitter_dist < 10.0 else None
                    elif c_prev["v"] == c_curr["u"]:
                        # Directly contiguous arcs in graph
                        d_net = (
                            (1.0 - c_prev["t_proj"]) * c_prev["length"]
                            + c_curr["t_proj"] * c_curr["length"]
                        )
                    else:
                        # Non-adjacent arcs: find shortest network path
                        sp = get_shortest_path(
                            c_prev["v"], c_curr["u"], cutoff=max_plausible_dist
                        )
                        if sp is not None:
                            d_net = (
                                (1.0 - c_prev["t_proj"]) * c_prev["length"]
                                + sp
                                + c_curr["t_proj"] * c_curr["length"]
                            )
                        else:
                            d_net = None

                    # Calculate transition log probability
                    if d_net is not None:
                        diff = abs(d_net - d_gps)
                        log_trans = -math.log(beta_eff) - (diff / beta_eff)
                    else:
                        log_trans = self.disconnected_penalty

                    candidate_total = V[t - 1][j] + log_trans
                    if candidate_total > best_log_prob:
                        best_log_prob = candidate_total
                        best_prev_idx = j

                V_t[i] = best_log_prob + c_curr["log_emission"]
                BP_t[i] = best_prev_idx

            V.append(V_t)
            BP.append(BP_t)

        trellis_time = time.time() - t_trellis_start

        # 3. Backtracking to extract optimal sequence
        best_final_idx = max(V[-1].keys(), key=lambda i: V[-1][i])
        decoded_indices = [best_final_idx]
        for t in range(T - 1, 0, -1):
            best_final_idx = BP[t][decoded_indices[-1]]
            decoded_indices.append(best_final_idx)
        decoded_indices.reverse()

        viterbi_route = [
            candidates_per_step[t][idx]["arc_id"]
            for t, idx in enumerate(decoded_indices)
        ]
        classifier_route = [
            candidates_per_step[t][0]["arc_id"] for t in range(T)
        ]

        total_time = time.time() - t_start

        return {
            "viterbi_route": viterbi_route,
            "classifier_route": classifier_route,
            "candidates_per_step": candidates_per_step,
            "timings": {
                "candidate_gen_sec": cand_time,
                "viterbi_trellis_sec": trellis_time,
                "total_sec": total_time,
            },
        }

    @staticmethod
    def align_ground_truth(
        track: np.ndarray,
        route: np.ndarray,
        arcs: np.ndarray,
        x_nodes: np.ndarray,
        y_nodes: np.ndarray,
        ref_lon: float,
        ref_lat: float,
    ) -> np.ndarray:
        """
        Monotonically aligns GPS observations to the ground truth route sequence.
        Returns an array of length T where each element is the true arc_id for observation t.
        """
        if len(route) == 0:
            return np.array([], dtype=np.int64)

        x_track, y_track = project_to_meters(
            track[:, 0], track[:, 1], ref_lon, ref_lat
        )
        T = len(track)
        R = len(route)

        obs_gt = []
        curr_r = 0
        lookahead = 12

        for t in range(T):
            px, py = x_track[t], y_track[t]
            best_dist = float("inf")
            best_r = curr_r

            for r in range(curr_r, min(R, curr_r + lookahead)):
                arc_id = route[r]
                u, v = arcs[arc_id]
                d, _, _, _ = point_to_segment_project(
                    px, py, x_nodes[u], y_nodes[u], x_nodes[v], y_nodes[v]
                )
                if d < best_dist:
                    best_dist = d
                    best_r = r

            curr_r = best_r
            obs_gt.append(route[best_r])

        return np.array(obs_gt, dtype=np.int64)

    @staticmethod
    def compute_route_continuity(route_arcs: list, arcs: np.ndarray) -> float:
        """
        Calculates route continuity: percentage of consecutive arc transitions
        that are either identical or directly connected in the road network graph.
        """
        if len(route_arcs) <= 1:
            return 1.0

        transitions = len(route_arcs) - 1
        connected = 0
        for k in range(transitions):
            a1, a2 = route_arcs[k], route_arcs[k + 1]
            if a1 == a2 or arcs[a1, 1] == arcs[a2, 0]:
                connected += 1

        return float(connected / transitions)

    def evaluate_trip(
        self, trip_data: dict, classifier_probs: np.ndarray = None
    ) -> dict:
        """
        Evaluates a complete trip, comparing Classifier-Only vs Classifier+HMM/Viterbi.
        Accepts DataFrames (Member 1 pipeline) or raw NumPy arrays.
        """
        import pandas as pd
        from src.data_loader import build_road_graph, wgs84_to_local_metric

        track_raw = trip_data["track"]
        nodes_raw = trip_data["nodes"]
        arcs_raw = trip_data["arcs"]
        route_raw = trip_data.get("route")
        trip_id = str(trip_data.get("trip_id", "unknown"))

        # Normalize to numpy arrays
        if isinstance(track_raw, pd.DataFrame):
            track = track_raw[["lon", "lat", "timestamp"]].values
            track_df = track_raw
        else:
            track = np.asarray(track_raw)
            track_df = pd.DataFrame(track, columns=["lon", "lat", "timestamp"])

        if isinstance(nodes_raw, pd.DataFrame):
            nodes = nodes_raw[["lon", "lat"]].values
            nodes_df = nodes_raw
        else:
            nodes = np.asarray(nodes_raw)
            nodes_df = pd.DataFrame(nodes, columns=["lon", "lat"])

        if isinstance(arcs_raw, pd.DataFrame):
            arcs = arcs_raw[["from_node", "to_node"]].values
            arcs_df = arcs_raw
        else:
            arcs = np.asarray(arcs_raw)
            arcs_df = pd.DataFrame(arcs, columns=["from_node", "to_node"])
            if "arc_id" not in arcs_df.columns:
                arcs_df["arc_id"] = np.arange(len(arcs))

        if route_raw is not None:
            if isinstance(route_raw, pd.DataFrame):
                route = route_raw["arc_id"].values if "arc_id" in route_raw.columns else route_raw.values.flatten()
            else:
                route = np.asarray(route_raw).flatten()
        else:
            route = np.array([], dtype=np.int64)

        # Build Graph using Member 1's function
        G, aug_arcs = build_road_graph(nodes_df, arcs_df)

        ref_lon = float(nodes[:, 0].mean())
        ref_lat = float(nodes[:, 1].mean())
        x_nodes, y_nodes = wgs84_to_local_metric(nodes[:, 0], nodes[:, 1], ref_lon, ref_lat)
        arc_lengths = aug_arcs["arc_length"].values

        meta = {
            "ref_lon": ref_lon,
            "ref_lat": ref_lat,
            "x_nodes": x_nodes,
            "y_nodes": y_nodes,
            "arc_lengths": arc_lengths,
        }

        # Run Viterbi
        result = self.viterbi_decode(
            track, nodes, arcs, G, meta, classifier_probs
        )

        viterbi_arcs = result["viterbi_route"]
        classifier_arcs = result["classifier_route"]
        candidates = result["candidates_per_step"]
        T = len(track)

        # Continuity
        cont_clf = self.compute_route_continuity(classifier_arcs, arcs)
        cont_vit = self.compute_route_continuity(viterbi_arcs, arcs)

        # Ground truth alignment & metrics
        metrics = {
            "trip_id": trip_id,
            "observations": T,
            "continuity_classifier": cont_clf,
            "continuity_viterbi": cont_vit,
            "continuity_improvement": cont_vit - cont_clf,
            "timing_total_sec": result["timings"]["total_sec"],
        }

        if len(route) > 0:
            obs_gt = self.align_ground_truth(
                track,
                route,
                arcs,
                meta["x_nodes"],
                meta["y_nodes"],
                meta["ref_lon"],
                meta["ref_lat"],
            )

            # Top-1 Accuracy
            top1_clf = float(np.mean(np.array(classifier_arcs) == obs_gt))
            top1_vit = float(np.mean(np.array(viterbi_arcs) == obs_gt))

            # Candidate Miss Rate (proportion where ground truth arc is not in candidate set)
            miss_count = 0
            for t in range(T):
                c_arc_ids = {c["arc_id"] for c in candidates[t]}
                if obs_gt[t] not in c_arc_ids:
                    miss_count += 1
            miss_rate = float(miss_count / T)

            # Route set overlap
            gt_set = set(route)
            v_overlap = float(sum(1 for a in viterbi_arcs if a in gt_set) / T)
            c_overlap = float(sum(1 for a in classifier_arcs if a in gt_set) / T)

            metrics.update(
                {
                    "top1_accuracy_classifier": top1_clf,
                    "top1_accuracy_viterbi": top1_vit,
                    "accuracy_improvement": top1_vit - top1_clf,
                    "candidate_miss_rate": miss_rate,
                    "route_overlap_classifier": c_overlap,
                    "route_overlap_viterbi": v_overlap,
                }
            )

        return metrics, result


def reconstruct_route(
    trip_id: str,
    gps_points: list,
    graph: nx.DiGraph,
    candidates: list = None,
    emission_probs: list = None,
    model: Any = None,
    **kwargs,
) -> list[int]:
    """
    Contract 3 interface function for Member 4 API and pipeline integration.
    Takes trip_id, list of gps_points, and networkx road graph,
    and returns list of predicted arc_ids in chronological order.
    """
    if candidates is not None and len(candidates) > 0:
        n_steps = len(gps_points)
        if n_steps == 0:
            return []

        beta = 12.0
        disconnected_penalty = -35.0

        sp_cache: dict = {}

        def get_sp(u: int, v: int, cutoff: float) -> float:
            if u == v:
                return 0.0
            rounded = math.ceil(cutoff / 50.0) * 50.0
            k = (u, rounded)
            if k not in sp_cache:
                sp_cache[k] = nx.single_source_dijkstra_path_length(
                    graph, u, cutoff=rounded, weight="weight"
                )
            return sp_cache[k].get(v, None)

        V = []
        BP = []

        v0 = {}
        for i, c in enumerate(candidates[0]):
            p = (
                emission_probs[0][i]
                if emission_probs and len(emission_probs[0]) > i
                else 1.0 / len(candidates[0])
            )
            v0[i] = math.log(max(1e-9, p))
        V.append(v0)
        BP.append({})

        for t in range(1, n_steps):
            v_t = {}
            bp_t = {}
            prev_cands = candidates[t - 1]
            curr_cands = candidates[t]

            p_prev = gps_points[t - 1]
            p_curr = gps_points[t]

            lat1 = p_prev.get("lat", 0.0) if isinstance(p_prev, dict) else p_prev[1]
            lon1 = p_prev.get("lon", 0.0) if isinstance(p_prev, dict) else p_prev[0]
            lat2 = p_curr.get("lat", 0.0) if isinstance(p_curr, dict) else p_curr[1]
            lon2 = p_curr.get("lon", 0.0) if isinstance(p_curr, dict) else p_curr[0]
            ts1 = p_prev.get("timestamp", 0.0) if isinstance(p_prev, dict) else p_prev[2]
            ts2 = p_curr.get("timestamp", 0.0) if isinstance(p_curr, dict) else p_curr[2]

            dt = max(1e-3, ts2 - ts1)

            from src.spatial_utils import haversine_distance

            d_gps = haversine_distance(lon1, lat1, lon2, lat2)

            beta_eff = beta * (1.0 + max(0.0, (dt - 2.0) / 5.0))
            cutoff = max(150.0, 45.0 * dt + 50.0)

            for i, c_curr in enumerate(curr_cands):
                arc_curr = int(c_curr["arc_id"])
                u_curr = int(c_curr.get("from_node", c_curr.get("u", 0)))
                v_curr = int(c_curr.get("to_node", c_curr.get("v", 0)))
                l_curr = float(
                    c_curr.get(
                        "length_m", c_curr.get("arc_length", c_curr.get("length", 10.0))
                    )
                )
                frac_curr = float(c_curr.get("fraction", c_curr.get("t_proj", 0.5)))

                best_prob = -float("inf")
                best_j = 0

                for j, c_prev in enumerate(prev_cands):
                    arc_prev = int(c_prev["arc_id"])
                    u_prev = int(c_prev.get("from_node", c_prev.get("u", 0)))
                    v_prev = int(c_prev.get("to_node", c_prev.get("v", 0)))
                    l_prev = float(
                        c_prev.get(
                            "length_m",
                            c_prev.get("arc_length", c_prev.get("length", 10.0)),
                        )
                    )
                    frac_prev = float(
                        c_prev.get("fraction", c_prev.get("t_proj", 0.5))
                    )

                    if arc_prev == arc_curr:
                        delta = frac_curr - frac_prev
                        d_net = (
                            delta * l_curr
                            if delta >= 0.0
                            else abs(delta) * l_curr + 10.0
                        )
                    elif v_prev == u_curr:
                        d_net = (1.0 - frac_prev) * l_prev + frac_curr * l_curr
                    else:
                        sp = get_sp(v_prev, u_curr, cutoff)
                        d_net = (
                            (1.0 - frac_prev) * l_prev + sp + frac_curr * l_curr
                            if sp is not None
                            else None
                        )

                    if d_net is not None:
                        diff = abs(d_net - d_gps)
                        log_trans = -math.log(beta_eff) - (diff / beta_eff)
                    else:
                        log_trans = disconnected_penalty

                    tot = V[t - 1][j] + log_trans
                    if tot > best_prob:
                        best_prob = tot
                        best_j = j

                p_em = (
                    emission_probs[t][i]
                    if emission_probs and len(emission_probs[t]) > i
                    else 1.0 / len(curr_cands)
                )
                v_t[i] = best_prob + math.log(max(1e-9, p_em))
                bp_t[i] = best_j

            V.append(v_t)
            BP.append(bp_t)

        best_idx = max(V[-1].keys(), key=lambda k: V[-1][k])
        decoded = [best_idx]
        for t in range(n_steps - 1, 0, -1):
            best_idx = BP[t][decoded[-1]]
            decoded.append(best_idx)
        decoded.reverse()

        return [int(candidates[t][idx]["arc_id"]) for t, idx in enumerate(decoded)]

    from src.data_loader import load_trip

    trip_data = load_trip(str(trip_id))
    hmm = MapMatchingHMM()
    metrics, result = hmm.evaluate_trip(trip_data)
    return [int(a) for a in result["viterbi_route"]]
