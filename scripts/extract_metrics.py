"""
VQ-MAR — Metrics Extraction

Scans all solver output JSONs across both cost models (flat and real)
and produces a single unified CSV with one row per solver run.

Handles three JSON schema types:
  1. Classical solvers (greedy, SA, GA, brute_force): single result block
  2. QAOA runs (native_diagonal, hardware): single result + metrics block
  3. Top-10 variant files (baseline, cvar, spatial, hydraulic, noisy, MPS):
     top10_bitstrings array — extracts best (lowest energy) entry

Usage:
    python scripts/extract_metrics.py
    python scripts/extract_metrics.py --base_dir georgia/qiskit-ready/qubo_matrices
    python scripts/extract_metrics.py --output_dir results
"""

import json
import os
import csv
import glob
import argparse


# ============================================================
# Configuration
# ============================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_BASE = os.environ.get("VQMAR_BASE", _REPO_ROOT)

DEFAULT_BASE_DIR = os.path.join(DEFAULT_BASE, "georgia", "qiskit-ready", "qubo_matrices")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_BASE, "georgia", "qiskit-ready", "portfolio_analysis")

# Files to skip (metadata, not solver output)
SKIP_FILES = {
    "meta_20.json",
    "meta_50.json",
    "pairwise_20.json",
    "pairwise_50.json",
    "qubo_20.json",
    "qubo_50.json",
    "sanity_check_20.json",
    "sanity_check_50.json",
    "landscape_analysis.json",
}

# Also skip non-JSON files
SKIP_EXTENSIONS = {".npy", ".csv"}

# Column order for the output CSV
CSV_COLUMNS = [
    "cost_model",
    "method",
    "variant",
    "n_sites",
    "p_depth",
    "simulator",
    "optimizer",
    "energy",
    "bitstring",
    "n_selected",
    "selected_sites",
    "total_benefit",
    "total_cost",
    "budget",
    "budget_feasible",
    "excluded_used",
    "approximation_ratio",
    "p_success_optimal",
    "p_success_within_1pct",
    "runtime_seconds",
    "source_file",
]


# ============================================================
# Schema Detection and Extraction
# ============================================================

def detect_schema(data):
    """
    Detect which JSON schema a file uses.
    Returns: 'classical', 'qaoa_run', 'top10', or 'unknown'
    """
    if "top10_bitstrings" in data:
        return "top10"
    if "result" in data and "metrics" in data:
        return "qaoa_run"
    if "result" in data:
        return "classical"
    return "unknown"


def extract_classical(data, cost_model, filename):
    """Extract metrics from a classical solver JSON (greedy, SA, GA, brute_force)."""
    result = data.get("result", {})

    return {
        "cost_model": cost_model,
        "method": data.get("method", "unknown"),
        "variant": None,
        "n_sites": data.get("n_sites"),
        "p_depth": None,
        "simulator": None,
        "optimizer": None,
        "energy": result.get("energy"),
        "bitstring": result.get("bitstring"),
        "n_selected": result.get("n_selected"),
        "selected_sites": format_sites(result.get("selected_sites")),
        "total_benefit": result.get("total_benefit"),
        "total_cost": result.get("total_cost"),
        "budget": result.get("budget_B"),
        "budget_feasible": check_feasibility(result),
        "excluded_used": result.get("excluded_used"),
        "approximation_ratio": None,
        "p_success_optimal": None,
        "p_success_within_1pct": None,
        "runtime_seconds": data.get("runtime_seconds"),
        "source_file": filename,
    }


def extract_qaoa_run(data, cost_model, filename):
    """Extract metrics from a QAOA run JSON (native_diagonal, hardware)."""
    result = data.get("result", {})
    metrics = data.get("metrics", {})
    params = data.get("parameters", {})

    return {
        "cost_model": cost_model,
        "method": data.get("method", "unknown"),
        "variant": None,
        "n_sites": data.get("n_sites"),
        "p_depth": params.get("reps"),
        "simulator": params.get("sampler") or params.get("backend"),
        "optimizer": params.get("optimizer"),
        "energy": result.get("energy"),
        "bitstring": result.get("bitstring"),
        "n_selected": result.get("n_selected"),
        "selected_sites": format_sites(result.get("selected_sites")),
        "total_benefit": result.get("total_benefit"),
        "total_cost": result.get("total_cost"),
        "budget": result.get("budget_B") or metrics.get("budget_B"),
        "budget_feasible": metrics.get("budget_feasible"),
        "excluded_used": result.get("excluded_used"),
        "approximation_ratio": metrics.get("approximation_ratio"),
        "p_success_optimal": metrics.get("p_success_optimal"),
        "p_success_within_1pct": metrics.get("p_success_within_1pct"),
        "runtime_seconds": data.get("runtime_seconds"),
        "source_file": filename,
    }


def extract_top10(data, cost_model, filename):
    """
    Extract metrics from a top-10 bitstring JSON.
    Uses the best (lowest energy) entry from the top10_bitstrings array.
    """
    top10 = data.get("top10_bitstrings", [])

    # Filter to entries with valid energy
    valid = [e for e in top10 if e.get("energy") is not None]

    if valid:
        best = min(valid, key=lambda x: x["energy"])
    else:
        best = {}

    return {
        "cost_model": data.get("cost_model", cost_model),
        "method": "qaoa_top10",
        "variant": data.get("variant"),
        "n_sites": data.get("n_sites"),
        "p_depth": data.get("p_depth"),
        "simulator": data.get("simulator"),
        "optimizer": None,
        "energy": best.get("energy"),
        "bitstring": best.get("bitstring"),
        "n_selected": len(best.get("sites") or []),
        "selected_sites": format_sites(best.get("sites")),
        "total_benefit": best.get("benefit"),
        "total_cost": best.get("cost"),
        "budget": None,
        "budget_feasible": best.get("feasible"),
        "excluded_used": None,
        "approximation_ratio": data.get("approximation_ratio"),
        "p_success_optimal": None,
        "p_success_within_1pct": None,
        "runtime_seconds": data.get("runtime_seconds"),
        "source_file": filename,
    }


# ============================================================
# Helpers
# ============================================================

def format_sites(sites):
    """Convert a site list to a semicolon-separated string for CSV."""
    if not sites:
        return ""
    return ";".join(sites)


def check_feasibility(result):
    """Check budget feasibility from a classical result block."""
    cost = result.get("total_cost")
    budget = result.get("budget_B")
    if cost is not None and budget is not None:
        return cost <= budget
    return None


def infer_cost_model(dirpath):
    """Infer cost model from directory name."""
    dirname = os.path.basename(dirpath)
    if dirname == "flat":
        return "flat"
    elif dirname == "real":
        return "real"
    return "unknown"


# ============================================================
# Main Processing
# ============================================================

def scan_directory(dirpath, cost_model):
    """Scan a directory for solver JSON files and extract metrics."""
    rows = []
    # Recursive scan: picks up both the pre-restructure {cost}/*.json layout
    # and the post-April-23 {cost}/n{N}/results/*.json layout.
    json_files = sorted(glob.glob(os.path.join(dirpath, "**", "*.json"), recursive=True))

    for filepath in json_files:
        filename = os.path.basename(filepath)

        # Skip metadata files
        if filename in SKIP_FILES:
            continue

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARNING: Could not read {filename}: {e}")
            continue

        schema = detect_schema(data)

        if schema == "classical":
            row = extract_classical(data, cost_model, filename)
            rows.append(row)
            print(f"  {filename:<55} schema=classical   method={row['method']}")

        elif schema == "qaoa_run":
            row = extract_qaoa_run(data, cost_model, filename)
            rows.append(row)
            print(f"  {filename:<55} schema=qaoa_run    method={row['method']}")

        elif schema == "top10":
            row = extract_top10(data, cost_model, filename)
            rows.append(row)
            variant = row['variant'] or 'unknown'
            print(f"  {filename:<55} schema=top10       variant={variant}")

        else:
            print(f"  {filename:<55} schema=UNKNOWN     (skipped)")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="VQ-MAR Metrics Extraction — unified CSV across all solvers"
    )
    parser.add_argument(
        "--base_dir",
        default=DEFAULT_BASE_DIR,
        help=f"Base directory containing flat/ and real/ subdirs (default: {DEFAULT_BASE_DIR})"
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CSV (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("VQ-MAR — Metrics Extraction")
    print(f"Base directory: {args.base_dir}")
    print("=" * 70)

    all_rows = []

    # Scan flat/ and real/ directories
    for cost_model in ["flat", "real"]:
        dirpath = os.path.join(args.base_dir, cost_model)
        if not os.path.isdir(dirpath):
            print(f"\n  Directory not found: {dirpath} — skipping")
            continue

        print(f"\nScanning {cost_model}/ ...")
        rows = scan_directory(dirpath, cost_model)
        all_rows.extend(rows)
        print(f"  -> {len(rows)} runs extracted from {cost_model}/")

    # Also scan for any stray JSONs directly under base_dir (not in flat/ or real/).
    # Non-recursive: the recursive scans above already covered flat/... and real/...
    # subtrees; this only catches truly top-level stragglers.
    base_jsons = glob.glob(os.path.join(args.base_dir, "*.json"))
    if base_jsons:
        print(f"\nScanning {args.base_dir}/ (top-level only) ...")
        rows = []
        for filepath in sorted(base_jsons):
            filename = os.path.basename(filepath)
            if filename in SKIP_FILES:
                continue
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"  WARNING: Could not read {filename}: {e}")
                continue
            schema = detect_schema(data)
            if schema == "classical":
                row = extract_classical(data, "unknown", filename)
            elif schema == "qaoa_run":
                row = extract_qaoa_run(data, "unknown", filename)
            elif schema == "top10":
                row = extract_top10(data, "unknown", filename)
            else:
                row = None
            if row:
                print(f"  {filename:<55} schema={schema}")
                rows.append(row)
            else:
                print(f"  {filename:<55} schema={schema}     (skipped)")
        all_rows.extend(rows)
        print(f"  -> {len(rows)} runs extracted from top-level")

    if not all_rows:
        print("\nERROR: No solver output files found.")
        return

    # Print summary table
    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(all_rows)} total runs extracted")
    print("=" * 70)
    print(f"  {'Cost Model':<12} {'Method':<30} {'Variant':<12} {'Energy':>12} {'AR':>8} {'Feasible':>9}")
    print("  " + "-" * 85)
    for row in all_rows:
        energy = row['energy']
        energy_str = f"{energy:.4f}" if energy is not None else "N/A"
        ar = row['approximation_ratio']
        ar_str = f"{ar:.4f}" if isinstance(ar, (int, float)) else "N/A"
        feasible = row['budget_feasible']
        feas_str = "Yes" if feasible else ("No" if feasible is not None else "N/A")
        variant = row['variant'] or ""
        print(f"  {row['cost_model']:<12} {row['method']:<30} {variant:<12} {energy_str:>12} {ar_str:>8} {feas_str:>9}")

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "all_solver_metrics.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\nCSV saved to {csv_path}")

    # Also save as JSON for downstream tools
    json_path = os.path.join(args.output_dir, "all_solver_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(all_rows, f, indent=2, default=str)

    print(f"JSON saved to {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
