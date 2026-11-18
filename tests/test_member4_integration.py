"""End-to-end integration tests for Member 4 deliverables.

Verifies:
  1. FastAPI endpoints (`/health`, `/trips`, `/predict_trip`, `/predict_batch`, 404 handling).
  2. Exact Contract 3 hook-in (`reconstruct_route(trip_id, gps_points, graph)`).
  3. 4-layer interactive Folium map generation (from API payload and from NetworkX DiGraph).
  4. Automated Technical_Report.pdf compilation & architecture diagram generation.
"""

from __future__ import annotations

import types
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from api.pipeline import MapMatchingPipeline
from report.build_report import build_technical_report
from src.visualizer import (
    create_trip_map_from_graph,
    create_trip_map_from_payload,
    export_map_html,
    map_to_html_str,
)


def test_health_and_trips_endpoints():
    with TestClient(app) as client:
        r_health = client.get("/health")
        assert r_health.status_code == 200
        health = r_health.json()
        assert health["status"] == "ok"
        assert health["num_eval_trips"] == 10
        assert health["graph_stats"]["num_nodes"] > 0
        assert health["graph_stats"]["num_arcs"] > 0

        r_trips = client.get("/trips")
        assert r_trips.status_code == 200
        trips_data = r_trips.json()
        assert trips_data["count"] == 10
        assert len(trips_data["trips"]) == 10
        # Ensure ground truth is NOT leaked in /trips
        for trip in trips_data["trips"]:
            assert "ground_truth_arc_ids" not in trip


def test_predict_trip_contract_and_404():
    with TestClient(app) as client:
        r_ok = client.post("/predict_trip", json={"trip_id": "eval_trip_01"})
        assert r_ok.status_code == 200
        data = r_ok.json()
        assert data["trip_id"] == "eval_trip_01"
        assert data["num_gps_points"] > 10
        assert len(data["gps_points"]) == data["num_gps_points"]
        assert len(data["road_network_arcs"]) > 0
        assert len(data["ground_truth_arcs"]) > 0
        assert len(data["predicted_arcs"]) > 0

        metrics = data["metrics"]
        assert 0.0 <= metrics["top1_accuracy"] <= 1.0
        assert 0.0 <= metrics["route_continuity"] <= 1.0
        assert metrics["processing_time_ms"] >= 0.0
        assert metrics["top1_accuracy"] >= 0.80
        assert metrics["route_continuity"] >= 0.85

        r_missing = client.post("/predict_trip", json={"trip_id": "nonexistent_trip"})
        assert r_missing.status_code == 404


def test_predict_batch_all_10_trips():
    with TestClient(app) as client:
        r = client.post("/predict_batch", json={})
        assert r.status_code == 200
        body = r.json()
        assert body["summary"]["num_trips"] == 10
        assert len(body["results"]) == 10
        assert body["summary"]["mean_top1_accuracy"] >= 0.85
        assert body["summary"]["mean_route_continuity"] >= 0.90


def test_contract3_reconstruct_route_hook():
    """Verify pipeline cleanly delegates to Member 3's Contract 3 `reconstruct_route`."""
    pipeline = MapMatchingPipeline()
    trip = pipeline.eval_trips["eval_trip_02"]
    gt = trip["ground_truth_arc_ids"]

    called = {"flag": False}

    def fake_reconstruct_route(trip_id: str, gps_points: list, graph):
        called["flag"] = True
        assert trip_id == "eval_trip_02"
        assert len(gps_points) == len(gt)
        assert graph.number_of_edges() > 0
        return list(gt)

    fake_mod = types.SimpleNamespace(reconstruct_route=fake_reconstruct_route)
    pipeline._p3_hmm = fake_mod

    result = pipeline.predict_trip("eval_trip_02")
    assert called["flag"] is True
    assert result["metrics"]["top1_accuracy"] == 1.0
    assert result["metrics"]["route_continuity"] == 1.0


def test_folium_4_layer_visualizer(tmp_path: Path):
    pipeline = MapMatchingPipeline()
    payload = pipeline.predict_trip("eval_trip_03")

    # Test building map from API JSON payload (Laptop 2 path)
    fmap = create_trip_map_from_payload(payload)
    html = map_to_html_str(fmap)
    assert "Layer 1: Raw GPS Observations (Blue)" in html
    assert "Layer 2: Road Network Arcs" in html
    assert "Layer 3: Ground-Truth Route (Green)" in html
    assert "Layer 4: Predicted HMM Route (Purple)" in html
    assert "eval_trip_03" in html

    out_html = export_map_html(fmap, tmp_path / "eval_trip_03_map.html")
    assert out_html.exists()
    assert out_html.stat().st_size > 2000

    # Test building map directly from NetworkX DiGraph
    fmap_g = create_trip_map_from_graph(
        trip_id="eval_trip_03",
        gps_points=payload["gps_points"],
        graph=pipeline.graph,
        ground_truth_arc_ids=payload["ground_truth_arc_ids"],
        predicted_arc_ids=payload["predicted_arc_ids"],
        metrics=payload["metrics"],
    )
    html_g = map_to_html_str(fmap_g)
    assert "Layer 4: Predicted HMM Route (Purple)" in html_g


def test_build_technical_report_pdf(tmp_path: Path):
    out_pdf = tmp_path / "Test_Report.pdf"
    built = build_technical_report(output_pdf=out_pdf)
    assert built.exists()
    assert built.stat().st_size > 10_000
    assert built.read_bytes()[:4] == b"%PDF"
