#!/usr/bin/env python3
"""
VQ-MAR — QUBO Landscape Analysis
================================
Computes condition numbers, eigenspectra, and energy landscape statistics
for both cost models (flat, real). Used to explain why real model p≥2
degrades in approximation ratio.

Outputs:
    georgia/qiskit-ready/qubo_matrices/landscape_analysis.json

Usage:
    python scripts/qaoa/analyze_qubo_landscape.py
"""

import os
import sys
import json
import datetime
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qiskit_qubo import load_qubo

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DEFAULT_BASE = os.environ.get("VQMAR_BASE", _REPO_ROOT)


def analyze_matrix(Q, label):
    """Compute spectral and landscape stats for a QUBO matrix."""
    n = Q.shape[0]
    eigs = np.linalg.eigvalsh(Q)  # sorted ascending, real (Q is symmetric)
    diag = Q.diagonal()
    off  = Q - np.diag(diag)

    # Condition number (ratio of largest to smallest absolute eigenvalue)
    abs_eigs = np.abs(eigs)
    kappa = float(abs_eigs.max() / abs_eigs[abs_eigs > 1e-10].min()) if abs_eigs[abs_eigs > 1e-10].size > 0 else float("inf")

    # Spectral gap: difference between two most negative eigenvalues
    neg_eigs = np.sort(eigs[eigs < 0])
    spectral_gap = float(neg_eigs[1] - neg_eigs[0]) if len(neg_eigs) >= 2 else None

    # Off-diagonal coupling stats
    triu_mask = np.triu(np.ones_like(off, dtype=bool), k=1)
    off_vals  = off[triu_mask]

    print(f"\n[{label}]")
    print(f"  Shape        : {Q.shape}")
    print(f"  κ(Q)         : {kappa:.2f}")
    print(f"  Eigenvalues  : min={eigs.min():.4f}  max={eigs.max():.4f}  "
          f"n_neg={int((eigs < 0).sum())}  n_pos={int((eigs > 0).sum())}")
    print(f"  Spectral gap : {spectral_gap:.4f}" if spectral_gap is not None else "  Spectral gap : N/A")
    print(f"  Diagonal     : min={diag.min():.4f}  max={diag.max():.4f}  "
          f"mean={diag.mean():.4f}  std={diag.std():.4f}")
    print(f"  Off-diagonal : min={off_vals.min():.4f}  max={off_vals.max():.4f}  "
          f"mean={off_vals.mean():.4f}  std={off_vals.std():.4f}")

    return {
        "n_qubits":         n,
        "kappa":            round(kappa, 4),
        "eigenvalues":      [round(float(e), 6) for e in eigs],
        "n_negative_eigs":  int((eigs < 0).sum()),
        "n_positive_eigs":  int((eigs > 0).sum()),
        "spectral_gap":     round(float(spectral_gap), 6) if spectral_gap is not None else None,
        "diagonal": {
            "min":  round(float(diag.min()), 6),
            "max":  round(float(diag.max()), 6),
            "mean": round(float(diag.mean()), 6),
            "std":  round(float(diag.std()), 6),
        },
        "off_diagonal": {
            "min":  round(float(off_vals.min()), 6),
            "max":  round(float(off_vals.max()), 6),
            "mean": round(float(off_vals.mean()), 6),
            "std":  round(float(off_vals.std()), 6),
        },
    }


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir  = os.environ.get("VQMAR_BASE", DEFAULT_BASE)

    print("=" * 60)
    print("  VQ-MAR — QUBO Landscape Analysis")
    print("=" * 60)

    results = {}
    for cost_model in ["flat", "real"]:
        data = load_qubo(base_dir, cost_model)
        Q    = data["Q"]
        meta = data["meta"]

        print(f"\n{'─'*60}")
        print(f"  Cost model: {cost_model}  (budget={meta['budget']:.4f})")
        stats = analyze_matrix(Q, cost_model.upper())
        stats["budget"]    = round(float(meta["budget"]), 6)
        stats["cost_range"] = {
            "min": round(float(min(meta["C"])), 6),
            "max": round(float(max(meta["C"])), 6),
        }
        results[cost_model] = stats

    # κ ratio — how much harder is real vs flat?
    kappa_ratio = results["real"]["kappa"] / results["flat"]["kappa"]
    print(f"\n{'─'*60}")
    print(f"  κ(real) / κ(flat) = {kappa_ratio:.1f}x harder")
    print(f"  Interpretation: real QUBO landscape is ~{kappa_ratio:.0f}× more ill-conditioned")
    print(f"  → COBYLA trust-region optimizer struggles at p≥2 (more parameters × rougher landscape)")

    results["run_timestamp"] = timestamp
    results["summary"] = {
        "kappa_ratio_real_vs_flat": round(kappa_ratio, 2),
        "interpretation": (
            f"Real QUBO is {kappa_ratio:.1f}x more ill-conditioned than flat. "
            "High κ → elongated energy valleys → COBYLA trust region collapses "
            "prematurely at p≥2 (4–6 parameters) without exploring sufficiently."
        ),
    }

    out_path = Path(base_dir) / "georgia" / "qiskit-ready" / "qubo_matrices" / f"landscape_analysis_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Export] {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
