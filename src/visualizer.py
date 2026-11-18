"""Interactive Folium map generator for GPS map-matching visualization (Member 4).

Renders four core layers plus an optional diagnostics overlay — optimized for real
multi-thousand-point trajectories and city-scale road graphs:
  1. Raw GPS observations (blue circle markers)
  2. Road network arcs (light gray GeoJSON lines)
  3. Ground-truth reference route (green merged polylines)
  4. Model predicted reconstructed route (purple dashed merged polylines)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import folium
import networkx as nx


LAYER_COLORS = {
    "gps": "#2563EB",          # Blue circle markers
    "road_network": "#94A3B8", # Light gray road network lines
    "ground_truth": "#16A34A", # Green ground-truth route
    "predicted": "#9333EA",    # Purple predicted route (high contrast over green)
    "mismatch": "#DC2626",     # Red highlights for mismatched GPS points
}

SUPPORTED_TILES = {
    "OpenStreetMap": ("OpenStreetMap", None),
    "Light Minimal (Esri Gray)": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
    ),
    "Street Detail (Esri Street)": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "Tiles &copy; Esri",
    ),
}


def _merge_contiguous_polylines(arcs: Sequence[Mapping[str, Any]]) -> list[list[list[float]]]:
    """Merge chained road arcs into smooth continuous polylines for fast Leaflet rendering."""
    chains: list[list[list[float]]] = []
    current_chain: list[list[float]] = []

    for arc in arcs:
        coords = arc.get("coords", [])
        if len(coords) < 2:
            continue
        p0 = [float(coords[0][0]), float(coords[0][1])]
        p1 = [float(coords[-1][0]), float(coords[-1][1])]

        if not current_chain:
            current_chain = [p0, p1]
        else:
            last = current_chain[-1]
            # Connect seamlessly if endpoints match within ~1e-6 deg (~0.1m)
            if abs(last[0] - p0[0]) < 1e-6 and abs(last[1] - p0[1]) < 1e-6:
                current_chain.append(p1)
            else:
                chains.append(current_chain)
                current_chain = [p0, p1]

    if current_chain:
        chains.append(current_chain)
    return chains


def _build_legend_html(
    trip_id: str,
    metrics: Mapping[str, Any] | None = None,
    gps_note: str = "",
) -> str:
    """Create a clean floating HTML legend box for the Folium map."""
    metrics_block = ""
    if metrics:
        top1 = metrics.get("top1_accuracy")
        cont = metrics.get("route_continuity")
        ms = metrics.get("processing_time_ms")
        parts: list[str] = []
        if top1 is not None:
            parts.append(f"Top-1: <b>{float(top1) * 100:.1f}%</b>")
        if cont is not None:
            parts.append(f"Continuity: <b>{float(cont) * 100:.1f}%</b>")
        if ms is not None:
            parts.append(f"Time: <b>{float(ms):.1f} ms</b>")
        if parts:
            metrics_block = (
                "<div style='margin-top:6px;padding-top:6px;border-top:1px solid #E2E8F0;"
                "font-size:11px;color:#334155;'>"
                + " &middot; ".join(parts)
                + "</div>"
            )

    note_block = (
        f"<div style='margin-top:4px;font-size:10.5px;color:#64748B;'>{gps_note}</div>"
        if gps_note
        else ""
    )

    return f"""
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.14);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        color: #0F172A;
        min-width: 235px;
    ">
      <div style="font-weight: 700; font-size: 13px; margin-bottom: 6px;">
        Trip: {trip_id}
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                     background:{LAYER_COLORS['gps']};border:1.5px solid #1E3A8A;"></span>
        <span>Layer 1: Raw GPS Observations</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
        <span style="display:inline-block;width:18px;height:3px;
                     background:{LAYER_COLORS['road_network']};"></span>
        <span>Layer 2: Road Network Arcs</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
        <span style="display:inline-block;width:18px;height:4px;
                     background:{LAYER_COLORS['ground_truth']};border-radius:2px;"></span>
        <span>Layer 3: Ground-Truth Route</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
        <span style="display:inline-block;width:18px;height:0;
                     border-top:3.5px dashed {LAYER_COLORS['predicted']};"></span>
        <span>Layer 4: Predicted HMM Route</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                     background:{LAYER_COLORS['mismatch']};border:1.5px solid #7F1D1D;"></span>
        <span>Mismatched Observations</span>
      </div>
      {metrics_block}
      {note_block}
    </div>
    """


def _compute_center_and_bounds(
    gps_points: Sequence[Mapping[str, Any]],
    arcs: Sequence[Mapping[str, Any]],
) -> tuple[list[float], list[list[float]] | None]:
    """Calculate map center and bounding box from GPS points or arc geometries."""
    lats: list[float] = []
    lons: list[float] = []

    for pt in gps_points:
        lat = pt.get("lat", pt.get("gps_lat"))
        lon = pt.get("lon", pt.get("gps_lon"))
        if lat is not None and lon is not None:
            lats.append(float(lat))
            lons.append(float(lon))

    if not lats:
        for arc in arcs:
            for coord in arc.get("coords", []):
                if len(coord) >= 2:
                    lats.append(float(coord[0]))
                    lons.append(float(coord[1]))

    if not lats:
        return [12.9716, 77.5946], None

    center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    return center, bounds


def create_trip_map_from_payload(
    payload: Mapping[str, Any],
    tiles: str = "OpenStreetMap",
    show_layers: Mapping[str, bool] | None = None,
    max_gps_markers: int = 380,
) -> folium.Map:
    """Build an interactive 4-layer Folium map from a `/predict_trip` JSON payload."""
    trip_id = str(payload.get("trip_id", "unknown_trip"))
    gps_points: Sequence[Mapping[str, Any]] = payload.get("gps_points", [])
    road_arcs: Sequence[Mapping[str, Any]] = payload.get("road_network_arcs", [])
    gt_arcs: Sequence[Mapping[str, Any]] = payload.get("ground_truth_arcs", [])
    pred_arcs: Sequence[Mapping[str, Any]] = payload.get("predicted_arcs", [])
    step_details: Sequence[Mapping[str, Any]] = payload.get("step_details", [])
    metrics: Mapping[str, Any] | None = payload.get("metrics")

    visibility = {
        "gps": True,
        "road_network": True,
        "ground_truth": True,
        "predicted": True,
        "mismatches": True,
    }
    if show_layers:
        visibility.update(show_layers)

    tile_spec, tile_attr = SUPPORTED_TILES.get(tiles, ("OpenStreetMap", None))
    center, bounds = _compute_center_and_bounds(gps_points, gt_arcs or pred_arcs or road_arcs)

    map_kwargs: dict[str, Any] = {
        "location": center,
        "zoom_start": 14,
        "tiles": tile_spec,
        "control_scale": True,
        "prefer_canvas": True,
    }
    if tile_attr:
        map_kwargs["attr"] = tile_attr
    fmap = folium.Map(**map_kwargs)

    # Layer 2: Road Network Arcs rendered via fast GeoJSON FeatureCollection
    road_layer = folium.FeatureGroup(
        name="Layer 2: Road Network Arcs",
        show=visibility["road_network"],
    )
    if road_arcs:
        features = []
        for arc in road_arcs:
            coords = arc.get("coords", [])
            if len(coords) < 2:
                continue
            # GeoJSON uses [lon, lat] order
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[float(c[1]), float(c[0])] for c in coords],
                    },
                    "properties": {
                        "arc_id": arc.get("arc_id", -1),
                        "length": arc.get("length", 0.0),
                    },
                }
            )
        if features:
            folium.GeoJson(
                {"type": "FeatureCollection", "features": features},
                style_function=lambda _feat: {
                    "color": LAYER_COLORS["road_network"],
                    "weight": 1.8,
                    "opacity": 0.55,
                },
            ).add_to(road_layer)
    road_layer.add_to(fmap)

    # Layer 3: Ground-Truth Reference Route (Green merged polylines)
    gt_layer = folium.FeatureGroup(
        name="Layer 3: Ground-Truth Route (Green)",
        show=visibility["ground_truth"],
    )
    for chain_idx, chain in enumerate(_merge_contiguous_polylines(gt_arcs), start=1):
        folium.PolyLine(
            locations=chain,
            color=LAYER_COLORS["ground_truth"],
            weight=6.5,
            opacity=0.82,
            tooltip=f"Ground-Truth Route Segment #{chain_idx} ({len(gt_arcs)} total arcs)",
        ).add_to(gt_layer)
    gt_layer.add_to(fmap)

    # Layer 4: Model Predicted Reconstructed Route (Purple dashed merged polylines)
    pred_layer = folium.FeatureGroup(
        name="Layer 4: Predicted HMM Route (Purple)",
        show=visibility["predicted"],
    )
    for chain_idx, chain in enumerate(_merge_contiguous_polylines(pred_arcs), start=1):
        folium.PolyLine(
            locations=chain,
            color=LAYER_COLORS["predicted"],
            weight=3.8,
            opacity=0.95,
            dash_array="8, 6",
            tooltip=f"Predicted HMM Route Segment #{chain_idx} ({len(pred_arcs)} total arcs)",
        ).add_to(pred_layer)
    pred_layer.add_to(fmap)

    # Index per-step details by step_idx
    detail_by_step: dict[int, Mapping[str, Any]] = {
        int(item["step_idx"]): item for item in step_details if "step_idx" in item
    }

    # Layer 1: Raw GPS Observations (Blue circle markers; evenly strided for multi-thousand tracks)
    gps_layer = folium.FeatureGroup(
        name="Layer 1: Raw GPS Observations (Blue)",
        show=visibility["gps"],
    )
    mismatch_layer = folium.FeatureGroup(
        name="Diagnostics: Mismatches & Endpoints",
        show=visibility["mismatches"],
    )

    n_gps = len(gps_points)
    stride = max(1, n_gps // max_gps_markers) if n_gps > max_gps_markers else 1
    sampled_indices = set(range(0, n_gps, stride))
    if n_gps > 0:
        sampled_indices.add(0)
        sampled_indices.add(n_gps - 1)

    mismatch_count = 0
    for idx, pt in enumerate(gps_points):
        step_idx = int(pt.get("step_idx", idx))
        detail = detail_by_step.get(step_idx, {})
        is_match = detail.get("is_match")

        show_blue = idx in sampled_indices
        show_red = is_match is False and mismatch_count < 160
        if not show_blue and not show_red:
            continue

        lat = float(pt.get("lat", pt.get("gps_lat")))
        lon = float(pt.get("lon", pt.get("gps_lon")))
        ts = pt.get("timestamp", idx)
        gt_arc = detail.get("ground_truth_arc_id", "N/A")
        pred_arc = detail.get("predicted_arc_id", "N/A")
        status_label = "MATCH" if is_match is True else ("MISMATCH" if is_match is False else "N/A")

        popup_html = (
            f"<div style='font-family:sans-serif;font-size:12px;min-width:170px;'>"
            f"<b>GPS Observation #{step_idx}</b><br>"
            f"Timestamp: {ts}s<br>"
            f"Coords: ({lat:.5f}, {lon:.5f})<br>"
            f"GT Arc: <b>{gt_arc}</b><br>"
            f"Predicted Arc: <b>{pred_arc}</b><br>"
            f"Status: <b>{status_label}</b>"
            f"</div>"
        )

        if show_blue:
            folium.CircleMarker(
                location=[lat, lon],
                radius=4.0,
                color="#1E3A8A",
                weight=1.0,
                fill=True,
                fill_color=LAYER_COLORS["gps"],
                fill_opacity=0.85,
                tooltip=f"GPS #{step_idx} | GT={gt_arc} | Pred={pred_arc}",
                popup=folium.Popup(popup_html, max_width=240),
            ).add_to(gps_layer)

        if show_red:
            mismatch_count += 1
            folium.CircleMarker(
                location=[lat, lon],
                radius=7.0,
                color=LAYER_COLORS["mismatch"],
                weight=2.0,
                fill=True,
                fill_color=LAYER_COLORS["mismatch"],
                fill_opacity=0.35,
                tooltip=f"MISMATCH at GPS #{step_idx}: GT={gt_arc} vs Pred={pred_arc}",
                popup=folium.Popup(popup_html, max_width=240),
            ).add_to(mismatch_layer)

    gps_layer.add_to(fmap)

    if gps_points:
        first_pt, last_pt = gps_points[0], gps_points[-1]
        folium.Marker(
            location=[
                float(first_pt.get("lat", first_pt.get("gps_lat"))),
                float(first_pt.get("lon", first_pt.get("gps_lon"))),
            ],
            tooltip=f"Trip Start ({trip_id})",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(mismatch_layer)
        folium.Marker(
            location=[
                float(last_pt.get("lat", last_pt.get("gps_lat"))),
                float(last_pt.get("lon", last_pt.get("gps_lon"))),
            ],
            tooltip=f"Trip End ({trip_id})",
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
        ).add_to(mismatch_layer)

    mismatch_layer.add_to(fmap)

    if bounds is not None:
        fmap.fit_bounds(bounds, padding=(28, 28))

    gps_note = (
        f"Displaying {len(sampled_indices)} / {n_gps:,} GPS markers (1-in-{stride} stride)"
        if stride > 1
        else f"Displaying all {n_gps:,} GPS observations"
    )
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(_build_legend_html(trip_id, metrics, gps_note)))
    return fmap


def _extract_arc_record_from_graph(
    graph: nx.DiGraph,
    u: Any,
    v: Any,
    edge_data: Mapping[str, Any],
) -> dict[str, Any] | None:
    coords = edge_data.get("coords")
    if not coords:
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        u_lat = u_data.get("lat", u_data.get("y"))
        u_lon = u_data.get("lon", u_data.get("x"))
        v_lat = v_data.get("lat", v_data.get("y"))
        v_lon = v_data.get("lon", v_data.get("x"))
        if None in (u_lat, u_lon, v_lat, v_lon):
            return None
        coords = [[float(u_lat), float(u_lon)], [float(v_lat), float(v_lon)]]

    return {
        "arc_id": int(edge_data.get("arc_id", -1)),
        "from_node": int(u) if isinstance(u, int) else u,
        "to_node": int(v) if isinstance(v, int) else v,
        "length": float(edge_data.get("length", edge_data.get("weight", 0.0))),
        "coords": coords,
    }


def create_trip_map_from_graph(
    trip_id: str,
    gps_points: Sequence[Mapping[str, Any]],
    graph: nx.DiGraph,
    ground_truth_arc_ids: Sequence[int],
    predicted_arc_ids: Sequence[int],
    metrics: Mapping[str, Any] | None = None,
    tiles: str = "OpenStreetMap",
    bbox_pad_deg: float = 0.0025,
) -> folium.Map:
    """Build an interactive 4-layer Folium map directly from a NetworkX DiGraph."""
    arc_lookup: dict[int, dict[str, Any]] = {}
    all_arcs: list[dict[str, Any]] = []

    for u, v, data in graph.edges(data=True):
        rec = _extract_arc_record_from_graph(graph, u, v, data)
        if rec is None:
            continue
        all_arcs.append(rec)
        if rec["arc_id"] != -1:
            arc_lookup[rec["arc_id"]] = rec

    lats = [float(p.get("lat", p.get("gps_lat"))) for p in gps_points]
    lons = [float(p.get("lon", p.get("gps_lon"))) for p in gps_points]
    if lats and lons:
        min_lat, max_lat = min(lats) - bbox_pad_deg, max(lats) + bbox_pad_deg
        min_lon, max_lon = min(lons) - bbox_pad_deg, max(lons) + bbox_pad_deg
        local_road_arcs = [
            arc
            for arc in all_arcs
            if any(min_lat <= c[0] <= max_lat and min_lon <= c[1] <= max_lon for c in arc["coords"])
        ][:1100]
    else:
        local_road_arcs = all_arcs[:1100]

    gt_arcs = [arc_lookup[aid] for aid in ground_truth_arc_ids if aid in arc_lookup]
    pred_arcs = [arc_lookup[aid] for aid in predicted_arc_ids if aid in arc_lookup]

    payload = {
        "trip_id": trip_id,
        "gps_points": list(gps_points),
        "road_network_arcs": local_road_arcs,
        "ground_truth_arcs": gt_arcs,
        "predicted_arcs": pred_arcs,
        "step_details": [],
        "metrics": dict(metrics) if metrics else None,
    }
    return create_trip_map_from_payload(payload=payload, tiles=tiles)


def map_to_html_str(fmap: folium.Map) -> str:
    """Serialize a Folium map to a standalone HTML string."""
    return fmap.get_root().render()


def export_map_html(fmap: folium.Map, output_path: str | Path) -> Path:
    """Save an interactive Folium map to an HTML file on disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    return out
