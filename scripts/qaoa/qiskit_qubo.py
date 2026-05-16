#!/usr/bin/env python3
"""
VQ-MAR — Qiskit QUBO Shared Loader
==================================
Shared data loader for all QAOA scripts. Supports N=20 and N=50 sites
(and any other size with the appropriate QUBO files on disk).

Loads the pre-computed QUBO matrix and constructs a Qiskit QuadraticProgram,
then verifies that re-evaluating H(x) on the optimal bitstring reproduces
the reference energy within tolerance (1e-6). PASS/FAIL printed to console.

For N≤28: uses brute_force_{n}.json as ground truth.
For N>28:  falls back to best_classical_{n}.json (best SA/GA result).

Inputs (parameterized by n_sites):
    georgia/qiskit-ready/qubo_matrices/{cost_model}/qubo_{n}.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/meta_{n}.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/brute_force_{n}.json  (N≤28)
    georgia/qiskit-ready/qubo_matrices/{cost_model}/best_classical_{n}.json (N>28)
    georgia/qiskit-ready/qubo_matrices/{cost_model}/greedy_{n}.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/pairwise_{n}.json
    georgia/qiskit-ready/site_metadata/{cost_model}/sites_{n}.csv

Outputs (when run standalone):
    Console: energy verification PASS/FAIL, QuadraticProgram summary

Usage:
    python scripts/qaoa/qiskit_qubo.py --cost_model flat
    python scripts/qaoa/qiskit_qubo.py --cost_model real
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from qiskit_optimization import QuadraticProgram

# ─── Configuration ────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DEFAULT_BASE = os.environ.get("VQMAR_BASE", _REPO_ROOT)

# Energy verification tolerance
ENERGY_TOL = 1e-6


def qubo_results_dir(base_dir, cost_model, n_sites):
    """Return the path where QAOA result JSONs are written for this run config."""
    return (Path(base_dir) / "georgia" / "qiskit-ready" / "qubo_matrices"
            / cost_model / f"n{n_sites}" / "results")


# ─── 1. DATA LOADING ─────────────────────────────────────────────────────────

def load_qubo(base_dir, cost_model="flat", n_sites=20):
    """
    Load all QUBO-related files for a given cost model and site count.
    Returns a dict with keys: Q, meta, site_ids, brute_force, greedy,
    pairwise, sites_df.

    For n_sites > 28 brute_force_{n}.json does not exist; falls back to
    best_classical_{n}.json (best SA/GA result) as the AR denominator.
    """
    n          = n_sites
    qiskit_dir = Path(base_dir) / "georgia" / "qiskit-ready"
    qubo_dir   = qiskit_dir / "qubo_matrices" / cost_model
    inputs_dir = qubo_dir / f"n{n}" / "inputs"
    meta_dir   = qiskit_dir / "site_metadata" / cost_model

    print(f"[Load] Reading QUBO files from {inputs_dir} (n_sites={n})")

    # Q matrix (numpy binary preferred; fall back to CSV)
    npy_path = inputs_dir / f"Q_{n}.npy"
    csv_path = inputs_dir / f"Q_{n}.csv"
    if npy_path.exists():
        Q = np.load(npy_path)
        print(f"  Q matrix:      {Q.shape} from Q_{n}.npy")
    elif csv_path.exists():
        Q = np.loadtxt(csv_path, delimiter=",")
        print(f"  Q matrix:      {Q.shape} from Q_{n}.csv")
    else:
        raise FileNotFoundError(f"Q_{n}.npy not found in {inputs_dir}")

    # Full QUBO spec (site_ids, wi, Ci, ei)
    qubo_path = inputs_dir / f"qubo_{n}.json"
    with open(qubo_path) as f:
        qubo_spec = json.load(f)
    site_ids = qubo_spec["site_ids"]
    print(f"  qubo_{n}.json: {len(site_ids)} sites, cost_model={cost_model}")

    # Metadata (w[], C[], budget, qubo_const)
    meta_path = inputs_dir / f"meta_{n}.json"
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"  meta_{n}.json: budget={meta['budget']:.4f}, "
          f"qubo_const={meta['qubo_const']:.4f}")

    # Ground truth: brute_force for N≤28, best_classical for N>28
    bf_path  = inputs_dir / f"brute_force_{n}.json"
    bc_path  = inputs_dir / f"best_classical_{n}.json"
    if bf_path.exists():
        with open(bf_path) as f:
            brute_force = json.load(f)
        print(f"  brute_force:   energy={brute_force['result']['energy']:.6f}")
    elif bc_path.exists():
        with open(bc_path) as f:
            brute_force = json.load(f)
        print(f"  best_classical (N={n}): energy={brute_force['result']['energy']:.6f}")
    else:
        raise FileNotFoundError(
            f"Neither brute_force_{n}.json nor best_classical_{n}.json found in {inputs_dir}. "
            f"For N>28, create best_classical_{n}.json from the best SA/GA result."
        )

    # Greedy baseline — used by downstream solvers for AR comparison
    greedy_path = inputs_dir / f"greedy_{n}.json"
    with open(greedy_path) as f:
        greedy = json.load(f)

    # Pairwise spatial interactions — needed for spatial_diversity variant
    pairwise_path = inputs_dir / f"pairwise_{n}.json"
    with open(pairwise_path) as f:
        pairwise = json.load(f)
    dist_km = np.array(pairwise["distance_matrix_km"])

    # Site metadata — Si scores used by downstream solvers for recharge potential
    sites_path = meta_dir / f"sites_{n}.csv"
    sites_df = pd.read_csv(sites_path, dtype={"site_id": str})
    print(f"  sites_{n}.csv: {len(sites_df)} sites, "
          f"{len(sites_df.columns)} columns")

    return {
        "Q":           Q,
        "meta":        meta,
        "qubo_spec":   qubo_spec,
        "site_ids":    site_ids,
        "brute_force": brute_force,
        "greedy":      greedy,
        "dist_km":     dist_km,
        "sites_df":    sites_df,
    }


# ─── 2. QUADRATIC PROGRAM CONSTRUCTION ───────────────────────────────────────

def build_quadratic_program(Q, site_ids):
    """
    Build a Qiskit QuadraticProgram from the 20×20 QUBO matrix Q.

    Q encodes the minimization problem H(x) = x^T Q x where x ∈ {0,1}^n.
    For a symmetric Q:
      - Diagonal Q[i,i]   → linear coefficient for x_i
      - Off-diagonal Q[i,j] → quadratic coefficient for x_i*x_j
        (sum upper + lower triangles = 2*Q[i,j] since Q is symmetric)

    Variable names: x0 ... x19 (index matches site_ids order).
    """
    n = len(site_ids)
    qp = QuadraticProgram("vqmar_qubo")

    for i in range(n):
        qp.binary_var(f"x{i}")

    # Linear terms from diagonal
    linear = {f"x{i}": float(Q[i, i]) for i in range(n)}

    # Quadratic terms from upper triangle (Q is symmetric)
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            val = float(Q[i, j] + Q[j, i])   # = 2 * Q[i,j]
            if val != 0.0:
                quadratic[(f"x{i}", f"x{j}")] = val

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


# ─── 3. ENERGY VERIFICATION ──────────────────────────────────────────────────

def verify_energy(Q, qubo_const, brute_force, n_sites=20, tol=ENERGY_TOL):
    """
    Energy verification:
    Re-evaluate H(x) = x^T Q x + qubo_const on the brute-force optimal
    bitstring and confirm it matches brute_force['result']['energy'].

    The stored energy in brute_force_{n}.json (N≤28) or best_classical_{n}.json
    (N>28) is the raw QUBO energy (x^T Q x), without the constant term, as 
    produced by unified_pipeline.py. We therefore compare against x^T Q x directly.

    Returns True (PASS) or False (FAIL).
    """
    bits = brute_force["result"]["bitstring"]
    x = np.array([int(b) for b in bits], dtype=float)
    expected = brute_force["result"]["energy"]

    computed = float(x @ Q @ x)

    diff = abs(computed - expected)
    passed = diff < tol

    status = "PASS" if passed else "FAIL"
    
    # Determine which file was actually loaded
    if n_sites <= 28:
        source_file = f"brute_force_{n_sites}.json"
    else:
        source_file = f"best_classical_{n_sites}.json (simulated annealing)"
    
    print(f"\nEnergy verification: {status}")
    print(f"     Bitstring : {bits}")
    print(f"     Expected  : {expected:.8f}  (from {source_file})")
    print(f"     Computed  : {computed:.8f}  (x^T Q x)")
    print(f"     Diff      : {diff:.2e}  (tol={tol:.0e})")

    if not passed:
        print("     WARNING: Energy mismatch — check Q matrix sign convention.")

    return passed


# ─── 4. MAIN (standalone verification) ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VQ-MAR — Qiskit QUBO pipeline and energy verification"
    )
    parser.add_argument("--cost_model", choices=["flat", "real"], default="flat",
                        help="Cost model: flat (Ci=0.5) or real (Eq. 15)")
    parser.add_argument("--n_sites", type=int, default=20,
                        help="Number of sites (20 or 50)")
    parser.add_argument("--base_dir", default=DEFAULT_BASE,
                        help="Project root directory (overrides $VQMAR_BASE)")
    args = parser.parse_args()

    print("=" * 65)
    print("  VQ-MAR — Qiskit QUBO Pipeline")
    print(f"  Cost model : {args.cost_model}")
    print(f"  N sites    : {args.n_sites}")
    print(f"  Base dir   : {args.base_dir}")
    print("=" * 65)
    print()

    # ── Load QUBO data ────────────────────────────────────────────────────
    data = load_qubo(args.base_dir, args.cost_model, n_sites=args.n_sites)
    Q        = data["Q"]
    meta     = data["meta"]
    site_ids = data["site_ids"]
    bf       = data["brute_force"]

    # ── Build QuadraticProgram ────────────────────────────────────────────
    print("\nBuilding QuadraticProgram...")
    qp = build_quadratic_program(Q, site_ids)

    n_vars   = qp.get_num_vars()
    n_linear = len([v for v in qp.objective.linear.to_dict().values() if v != 0])
    n_quad   = len(qp.objective.quadratic.to_dict())
    print(f"     Variables  : {n_vars} binary")
    print(f"     Linear     : {n_linear} non-zero terms")
    print(f"     Quadratic  : {n_quad} non-zero pairs")
    print(f"     Sparsity   : {1 - n_quad / (n_vars*(n_vars-1)/2):.1%} sparse")

    # ── Verify energy ─────────────────────────────────────────────────────
    passed = verify_energy(Q, meta["qubo_const"], bf)

    print()
    print("=" * 65)
    print(f"  Verification complete — {'PASS' if passed else 'FAIL'}")
    print("  QuadraticProgram ready for downstream QAOA solver scripts")
    print("=" * 65)


if __name__ == "__main__":
    main()
