#!/usr/bin/env python3
"""
VQ-MAR — QAOA Result Schema Converter
=====================================
Converts existing QAOA result JSONs to the immutable schema required
by extract_metrics.py.

Immutable schema keys (null for unavailable):
    run_id, variant, cost_model, n_sites, simulator, p_depth,
    energy, approximation_ratio, beta, gamma,
    top10_bitstrings, convergence_curve, seed, runtime_seconds

Usage:
    # Convert a single file:
    python scripts/qaoa/convert_results_to_schema.py \\
        --input georgia/qiskit-ready/qubo_matrices/flat/qaoa_native_diagonal_20_p1.json \\
        --variant baseline \\
        --output georgia/qiskit-ready/qubo_matrices/flat/top10_baseline.json

    # Batch-convert all native_diagonal results:
    python scripts/qaoa/convert_results_to_schema.py --batch

Output:
    Writes new JSON file(s) with immutable schema.
    Original files are NOT modified.
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DEFAULT_BASE = os.environ.get("VQMAR_BASE", _REPO_ROOT)

SIMULATOR_MAP = {
    "qaoa_native_diagonal": "native_diagonal",
    "qaoa_vqe_estimator":   "vqe_estimator",
    "qaoa_hardware":        "hardware",
    "qaoa_algorithms_sampler": "algorithms_sampler",
    "qaoa_mps":             "mps",
    "qaoa_noisy_aer":       "noisy_aer",
}


def infer_simulator(method_str):
    """Infer simulator from method field string."""
    if method_str is None:
        return None
    for key, val in SIMULATOR_MAP.items():
        if key in str(method_str):
            return val
    return str(method_str)


def extract_top10(src, n_sites):
    """Extract top10_bitstrings from various source formats."""
    # Already in new schema
    if "top10_bitstrings" in src:
        return src["top10_bitstrings"]

    # From top_samples (native_diagonal format)
    if "top_samples" in src:
        out = []
        for s in src["top_samples"][:10]:
            out.append({
                "bitstring":   s.get("bitstring", None),
                "energy":      s.get("energy", None),
                "sites":       None,   # not present in old format
                "benefit":     None,
                "cost":        None,
                "feasible":    None,
                "probability": s.get("probability", None),
            })
        return out

    # From result samples (algorithms_sampler format)
    if "result" in src and "samples" in src.get("result", {}):
        out = []
        for s in src["result"]["samples"][:10]:
            out.append({
                "bitstring":   s.get("bitstring", None),
                "energy":      s.get("energy", None),
                "sites":       s.get("sites", None),
                "benefit":     s.get("benefit", None),
                "cost":        s.get("cost", None),
                "feasible":    s.get("feasible", None),
                "probability": s.get("probability", None),
            })
        return out

    return []


def convert_single(src, variant, override_simulator=None):
    """
    Convert a single result dict to the immutable schema.
    Returns a new dict; original is not modified.
    """
    timestamp = src.get("run_timestamp", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    # ── Core fields ──────────────────────────────────────────────────────
    cost_model   = src.get("cost_model", None)
    n_sites      = src.get("n_sites", None)
    p_depth      = src.get("parameters", {}).get("reps", None)
    if p_depth is None:
        p_depth = src.get("p_depth", None)

    # Energy
    energy = None
    if "result" in src and "energy" in src["result"]:
        energy = src["result"]["energy"]
    elif "energy" in src:
        energy = src["energy"]

    # AR
    ar = None
    if "metrics" in src and "approximation_ratio" in src["metrics"]:
        ar = src["metrics"]["approximation_ratio"]
    elif "approximation_ratio" in src:
        ar = src["approximation_ratio"]

    # Beta / Gamma
    beta  = src.get("extra_metrics", {}).get("optimal_beta",  src.get("beta",  None))
    gamma = src.get("extra_metrics", {}).get("optimal_gamma", src.get("gamma", None))

    # Convergence curve
    conv = src.get("extra_metrics", {}).get("convergence_curve",
           src.get("convergence_curve", []))

    # Runtime
    runtime = src.get("runtime_seconds", None)

    # Seed
    seed = src.get("seed", None)
    if seed is None:
        seed = src.get("parameters", {}).get("seed", None)

    # Simulator
    simulator = override_simulator
    if simulator is None:
        simulator = infer_simulator(src.get("method", src.get("simulator", None)))

    # top10
    top10 = extract_top10(src, n_sites)

    # run_id
    run_id = src.get("run_id",
             f"{variant}_p{p_depth}_{cost_model}_{timestamp}")

    out = {
        "run_id":              run_id,
        "variant":             variant,
        "cost_model":          cost_model,
        "n_sites":             n_sites,
        "simulator":           simulator,
        "p_depth":             p_depth,
        "energy":              energy,
        "approximation_ratio": ar,
        "beta":                beta,
        "gamma":               gamma,
        "top10_bitstrings":    top10,
        "convergence_curve":   conv,
        "seed":                seed,
        "runtime_seconds":     runtime,
        "run_timestamp":       timestamp,
    }

    # Preserve original fields under "_original" for traceability
    out["_original"] = {k: v for k, v in src.items()
                        if k not in ("top_samples", "convergence_curve")}

    return out


def batch_convert(base_dir):
    """
    Convert all qaoa_native_diagonal result JSONs in both cost models
    to top10_baseline.json (for the baseline variant).
    """
    converted = []
    for cost_model in ["flat", "real"]:
        qubo_dir    = (Path(base_dir) / "georgia" / "qiskit-ready"
                       / "qubo_matrices" / cost_model)
        results_dir = qubo_dir / "n20" / "results"
        for p in [1, 2, 3]:
            src_path = results_dir / f"qaoa_native_diagonal_n20_{cost_model}_p{p}.json"
            if not src_path.exists():
                print(f"[Skip] Not found: {src_path}")
                continue

            with open(src_path) as f:
                src = json.load(f)

            out     = convert_single(src, variant="baseline")
            dst     = results_dir / f"top10_baseline_n20_{cost_model}_p{p}.json"
            with open(dst, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[Convert] {src_path.name} → {dst.name}  AR={out['approximation_ratio']}")
            converted.append(str(dst))

    print(f"\n[Batch] Converted {len(converted)} files.")
    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert result JSON to immutable schema")
    parser.add_argument("--input",  default=None, help="Path to source JSON")
    parser.add_argument("--output", default=None, help="Path for converted JSON")
    parser.add_argument("--variant", default="baseline",
                        choices=["baseline", "cvar", "choice_a", "choice_b"],
                        help="Variant label for this run")
    parser.add_argument("--simulator", default=None,
                        help="Override simulator field (native_diagonal, mps, hardware, etc.)")
    parser.add_argument("--batch", action="store_true",
                        help="Batch-convert all native_diagonal results to baseline schema")
    parser.add_argument("--base_dir", default=DEFAULT_BASE)
    args = parser.parse_args()

    if args.batch:
        batch_convert(args.base_dir)
        return

    if not args.input:
        parser.error("--input is required unless --batch is specified")

    with open(args.input) as f:
        src = json.load(f)

    out = convert_single(src, args.variant, override_simulator=args.simulator)

    if args.output:
        out_path = Path(args.output)
    else:
        # Default: same directory, top10_{variant}.json
        out_path = Path(args.input).parent / f"top10_{args.variant}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[Convert] {args.input}")
    print(f"          → {out_path}")
    print(f"  variant={out['variant']}, AR={out['approximation_ratio']}, "
          f"simulator={out['simulator']}, p={out['p_depth']}")


if __name__ == "__main__":
    main()
