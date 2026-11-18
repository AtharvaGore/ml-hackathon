"""
visualizer.py - Interactive Folium Map Visualizer for PathSnap.

Generates multi-layer interactive maps:
- Layer 1: Raw GPS observations (blue circle markers with popups + trajectory polyline).
- Layer 2: Road network arcs in trajectory corridor (light gray lines with arc metadata).
- Layer 3: Ground-truth reference route (green line).
- Layer 4: Model predicted route (purple line, when provided).
- Layer 5: Candidate arcs highlight (optional).
- Interactive LayerControl and informative tooltips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any

import folium
import numpy as np
import pandas as pd


def create_folium_map(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route_df: Optional[pd.DataFrame] = None,
    candidate_arc_ids: Optional[Union[List[int], Set[int], np.ndarray]] = None,
    predicted_arcs: Optional[List[int]] = None,
    title: str = "PathSnap Trajectory Visualization",
    gps_sample_step: int = 1,
) -> folium.Map:
    """
    Generate an interactive multi-layer Folium map for a trajectory.

    Args:
        track_df: DataFrame with ['lon', 'lat', 'timestamp', 'step_idx']
        nodes_df: DataFrame with ['lon', 'lat'] indexed by node_id
        arcs_df: DataFrame with ['from_node', 'to_node', 'arc_id']
        route_df: DataFrame with ground-truth ['arc_id'] (optional)
        candidate_arc_ids: Iterable of arc_ids in the spatial neighborhood (corridor)
        predicted_arcs: List of predicted arc_ids (optional)
        title: Map title
        gps_sample_step: Downsampling step for GPS circle markers to keep browser responsive

    Returns:
        folium.Map object
    """
    center_lat = float(track_df["lat"].mean())
    center_lon = float(track_df["lon"].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # -------------------------------------------------------------
    # Layer 2: Road Network Corridor (Light Gray)
    # If candidate_arc_ids provided, display those; otherwise display route + nearby arcs
    # -------------------------------------------------------------
    corridor_arcs: Set[int] = set()
    if candidate_arc_ids is not None:
        corridor_arcs.update(int(a) for a in candidate_arc_ids)
    if route_df is not None:
        corridor_arcs.update(int(a) for a in route_df["arc_id"].values)

    if corridor_arcs:
        network_group = folium.FeatureGroup(name="Road Network Corridor (Light Gray)", show=True)
        # Cap to a reasonable maximum for fast browser rendering
        selected_arc_ids = list(corridor_arcs)[:8000]
        subset_arcs = arcs_df.loc[selected_arc_ids]

        for row in subset_arcs.itertuples():
            u_node = nodes_df.iloc[row.from_node]
            v_node = nodes_df.iloc[row.to_node]
            folium.PolyLine(
                locations=[[u_node["lat"], u_node["lon"]], [v_node["lat"], v_node["lon"]]],
                color="#7f8c8d",
                weight=2.0,
                opacity=0.6,
                tooltip=f"Arc ID: {row.arc_id} ({row.from_node} -> {row.to_node})",
            ).add_to(network_group)
        network_group.add_to(m)

    # -------------------------------------------------------------
    # Layer 3: Ground-Truth Reference Route (Green)
    # -------------------------------------------------------------
    if route_df is not None:
        gt_group = folium.FeatureGroup(name="Ground-Truth Route (Green)", show=True)
        route_arc_ids = route_df["arc_id"].values
        route_lines = []
        for aid in route_arc_ids:
            if aid < len(arcs_df):
                arc_row = arcs_df.iloc[aid]
                u_node = nodes_df.iloc[arc_row["from_node"]]
                v_node = nodes_df.iloc[arc_row["to_node"]]
                route_lines.append([[u_node["lat"], u_node["lon"]], [v_node["lat"], v_node["lon"]]])

        for line_coords in route_lines:
            folium.PolyLine(
                locations=line_coords,
                color="#27ae60",
                weight=4.5,
                opacity=0.9,
            ).add_to(gt_group)
        gt_group.add_to(m)

    # -------------------------------------------------------------
    # Layer 4: Model Predicted Route (Purple) if provided
    # -------------------------------------------------------------
    if predicted_arcs is not None:
        pred_group = folium.FeatureGroup(name="Predicted Route (Purple)", show=True)
        for aid in predicted_arcs:
            if aid < len(arcs_df):
                arc_row = arcs_df.iloc[aid]
                u_node = nodes_df.iloc[arc_row["from_node"]]
                v_node = nodes_df.iloc[arc_row["to_node"]]
                folium.PolyLine(
                    locations=[[u_node["lat"], u_node["lon"]], [v_node["lat"], v_node["lon"]]],
                    color="#8e44ad",
                    weight=3.5,
                    opacity=0.85,
                ).add_to(pred_group)
        pred_group.add_to(m)

    # -------------------------------------------------------------
    # Layer 1: Raw GPS Observations (Blue Circle Markers & Trace)
    # -------------------------------------------------------------
    gps_group = folium.FeatureGroup(name="Raw GPS Observations (Blue)", show=True)

    # Connect raw GPS points with a subtle dotted trace
    gps_coords = track_df[["lat", "lon"]].values.tolist()
    folium.PolyLine(
        locations=gps_coords,
        color="#2980b9",
        weight=1.5,
        dash_array="4, 6",
        opacity=0.6,
    ).add_to(gps_group)

    # Downsample circle markers if track is large
    step = max(1, gps_sample_step)
    if len(track_df) > 1000 and gps_sample_step == 1:
        step = max(1, len(track_df) // 800)

    for idx, row in track_df.iloc[::step].iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3.5,
            color="#2980b9",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.8,
            popup=(
                f"<b>Observation</b>: {row.get('step_idx', idx)}<br>"
                f"<b>Time</b>: {row['timestamp']:.1f} s<br>"
                f"<b>Lat</b>: {row['lat']:.6f}<br>"
                f"<b>Lon</b>: {row['lon']:.6f}"
            ),
        ).add_to(gps_group)

    # Start and End markers
    start_pt = track_df.iloc[0]
    end_pt = track_df.iloc[-1]
    folium.Marker(
        location=[start_pt["lat"], start_pt["lon"]],
        popup=f"<b>Trip Start</b>: t={start_pt['timestamp']:.1f}s",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(gps_group)
    folium.Marker(
        location=[end_pt["lat"], end_pt["lon"]],
        popup=f"<b>Trip End</b>: t={end_pt['timestamp']:.1f}s",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
    ).add_to(gps_group)

    gps_group.add_to(m)

    # Add LayerControl
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def save_folium_map(
    m: folium.Map,
    output_path: Union[str, Path],
) -> Path:
    """Save folium map to HTML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    return output_path
