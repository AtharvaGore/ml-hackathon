"""
Benchmark script for Member 3: Evaluates HMM & Viterbi Route Reconstruction
across the 10 Student-Test trips and generates the mandatory Report Section 5 artifacts.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_trip
from src.hmm_viterbi import MapMatchingHMM

def run_benchmark():
    test_dir = "data/Student folder/student_test"
    trip_ids = sorted(os.listdir(test_dir))
    print(f"Starting HMM Benchmark across {len(trip_ids)} Student-Test trips...")
    
    hmm = MapMatchingHMM(
        sigma_z=7.5,
        beta=7.0,
        search_radius=50.0,
        max_candidates=10,
        disconnected_penalty=-35.0
    )
    
    results = []
    
    for i, trip_id in enumerate(trip_ids):
        print(f"[{i+1}/{len(trip_ids)}] Processing Trip {trip_id}...", end=" ", flush=True)
        trip_data = load_trip(trip_id)
        metrics, _ = hmm.evaluate_trip(trip_data)
        results.append(metrics)
        print(
            f"Done in {metrics['timing_total_sec']:.2f}s | "
            f"Top-1: {metrics['top1_accuracy_classifier']*100:.1f}% -> {metrics['top1_accuracy_viterbi']*100:.1f}% | "
            f"Continuity: {metrics['continuity_classifier']*100:.1f}% -> {metrics['continuity_viterbi']*100:.1f}%"
        )
        
    df = pd.DataFrame(results)
    
    # Save CSV
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    csv_path = "artifacts/hmm_student_test_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed metrics to {csv_path}")
    
    # Summary stats
    summary = {
        "mean_top1_classifier": float(df["top1_accuracy_classifier"].mean()),
        "mean_top1_viterbi": float(df["top1_accuracy_viterbi"].mean()),
        "mean_top1_improvement": float(df["accuracy_improvement"].mean()),
        "mean_continuity_classifier": float(df["continuity_classifier"].mean()),
        "mean_continuity_viterbi": float(df["continuity_viterbi"].mean()),
        "mean_continuity_improvement": float(df["continuity_improvement"].mean()),
        "mean_candidate_miss_rate": float(df["candidate_miss_rate"].mean()),
        "mean_inference_time_sec": float(df["timing_total_sec"].mean()),
    }
    
    summary_path = "artifacts/hmm_student_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    # Generate Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(df))
    width = 0.35
    
    # Top-1 Accuracy Plot
    axes[0].bar(x - width/2, df["top1_accuracy_classifier"] * 100, width, label="Classifier-Only", color="#3498db", alpha=0.85)
    axes[0].bar(x + width/2, df["top1_accuracy_viterbi"] * 100, width, label="Classifier + HMM Viterbi", color="#2ecc71", alpha=0.85)
    axes[0].set_title("Observation-level Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Trip ID")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([t[-4:] for t in df["trip_id"]], rotation=45)
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    
    # Route Continuity Plot
    axes[1].bar(x - width/2, df["continuity_classifier"] * 100, width, label="Classifier-Only", color="#e74c3c", alpha=0.85)
    axes[1].bar(x + width/2, df["continuity_viterbi"] * 100, width, label="Classifier + HMM Viterbi", color="#9b59b6", alpha=0.85)
    axes[1].set_title("Route Continuity (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Trip ID")
    axes[1].set_ylabel("Continuity (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([t[-4:] for t in df["trip_id"]], rotation=45)
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_path = "reports/hmm_vs_classifier_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Generated comparison plot: {plot_path}")
    
    print("\n" + "="*70)
    print("CONSOLIDATED HMM & VITERBI BENCHMARK SUMMARY (10 STUDENT-TEST TRIPS)")
    print("="*70)
    print(f"Classifier-Only Mean Top-1 Accuracy:     {summary['mean_top1_classifier']*100:.2f}%")
    print(f"HMM Viterbi Mean Top-1 Accuracy:         {summary['mean_top1_viterbi']*100:.2f}% (Delta: +{summary['mean_top1_improvement']*100:.2f}%)")
    print(f"Classifier-Only Mean Route Continuity:   {summary['mean_continuity_classifier']*100:.2f}%")
    print(f"HMM Viterbi Mean Route Continuity:       {summary['mean_continuity_viterbi']*100:.2f}% (Delta: +{summary['mean_continuity_improvement']*100:.2f}%)")
    print(f"Mean Candidate Miss Rate:                {summary['mean_candidate_miss_rate']*100:.2f}%")
    print(f"Average Inference Time per Trip:         {summary['mean_inference_time_sec']:.2f}s")
    print("="*70)

if __name__ == "__main__":
    run_benchmark()
