#!/usr/bin/env python3
"""
VQ-MAR — QAOA Canonical (qiskit_algorithms.QAOA + StatevectorSampler)
=====================================================================

Method: Canonical IBM Qiskit QAOA pathway.
  - qiskit_algorithms.QAOA + StatevectorSampler + MinimumEigenOptimizer.
  - Maps directly to IBM Qiskit tutorials and documentation.
  - StatevectorSampler is the only primitive compatible with qiskit-algorithms
    0.4.0's DiagonalEstimator PUB format (AerSampler V1/V2 are not).

WARNING: SLOW. DiagonalEstimator recomputes 210-Pauli × 2^20 Ising diagonal
  on every COBYLA call (~60–90 s/call on M4 Pro). Expected total runtime:
  ~2–3 hr at maxiter=100, ~6–8 hr at maxiter=300. Use run_qaoa_native_diagonal.py
  for fast results. Run this script in background or overnight for benchmarking.

Inputs:
    georgia/qiskit-ready/qubo_matrices/{cost_model}/qubo_20.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/meta_20.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/brute_force_20.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/greedy_20.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/pairwise_20.json
    georgia/qiskit-ready/site_metadata/{cost_model}/sites_20.csv

Outputs:
    georgia/qiskit-ready/qubo_matrices/{cost_model}/qaoa_algorithms_sampler_20_p{reps}.json

Usage:
    # p=1 (baseline depth), run in background
    python scripts/qaoa/run_qaoa_algorithms_sampler.py --cost_model flat --reps 1

    # p=2, p=3 (deeper circuits)
    python scripts/qaoa/run_qaoa_algorithms_sampler.py --cost_model flat --reps 2 --maxiter 50
    python scripts/qaoa/run_qaoa_algorithms_sampler.py --cost_model flat --reps 3 --maxiter 50
"""

import os
import sys
import json
import time
import datetime
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path

from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Import shared QUBO helpers — loading and QuadraticProgram construction
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qiskit_qubo import load_qubo, build_quadratic_program, verify_energy, qubo_results_dir

# ─── Configuration ────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DEFAULT_BASE = os.environ.get("VQMAR_BASE", _REPO_ROOT)

# QAOA defaults
# Default maxiter=100 (not 300) to keep runtime ≤ 3 hours on M4 Pro.
# Use --maxiter 300 for the full benchmark (6–8 hours).
DEFAULT_REPS    = 1    # QAOA circuit depth p
DEFAULT_MAXITER = 100  # reduced from 300 — ~90s/call × 100 ≈ 2.5 hr
SEED            = 42   # reproducibility (matches SA, GA in unified_pipeline.py)

# Random portfolio baseline samples for recharge potential comparison
N_RANDOM_PORTFOLIOS = 1000

# Top-N samples to include in output JSON
TOP_N_SAMPLES = 20

# ─── Hardware optimization guide ──────────────────────────────────────────────
#
# Optimized for Apple M4 Pro (10 performance cores, 48 GB RAM).
# StatevectorSampler uses NumPy/BLAS via Apple Accelerate automatically.
#
# To adapt for other hardware:
#   1. DETECT CORE COUNT
#      n_cores = multiprocessing.cpu_count()
#      Set OMP_NUM_THREADS, VECLIB_MAXIMUM_THREADS, OPENBLAS_NUM_THREADS.
#
#   2. BOTTLENECK: DiagonalEstimator in qiskit-algorithms 0.4.0 allocates
#      ~1.68 GB per call (210 Pauli terms × 2^20 states). Cannot be disabled
#      without modifying the qiskit-algorithms source. If runtime is too slow,
#      use run_qaoa_native_diagonal.py instead.
#
#   3. GPU: not applicable here — DiagonalEstimator is CPU-only.


# ─── 1. QAOA RUN ─────────────────────────────────────────────────────────────

def run_qaoa(qp, reps, maxiter):
    """
    Run QAOA using qiskit_algorithms.QAOA + StatevectorSampler + MinimumEigenOptimizer.

    This is the canonical IBM Qiskit QAOA pathway. StatevectorSampler is the
    only primitive compatible with qiskit-algorithms 0.4.0's DiagonalEstimator
    PUB format — AerSampler V1 and V2 are both incompatible.

    Bottleneck: DiagonalEstimator rebuilds the full Ising diagonal per COBYLA
    call (~60–90 s on M4 Pro). At maxiter=100 total runtime ≈ 2.5 hours.

    Returns:
        opt_result  — OptimizationResult from MinimumEigenOptimizer
        qaoa_alg    — QAOA instance (holds result.optimal_parameters)
        betas       — list of optimal β angles at convergence
        gammas      — list of optimal γ angles at convergence
        conv_curve  — list of energy values per COBYLA call
        runtime_s   — wall-clock seconds
    """
    convergence_curve = []
    call_count = [0]
    t_start = time.time()

    def cobyla_callback(*args):
        # New scipy (pyprima in Python 3.14) calls callback(x) — just current params.
        # Old scipy called callback(nfev, x, fx, dx, accept). Accept both via *args.
        # Energy is not available here; only iteration count and timing are tracked.
        call_count[0] += 1
        nfev = call_count[0]
        if nfev == 1 or nfev % 10 == 0:
            elapsed  = time.time() - t_start
            avg_time = elapsed / max(nfev, 1)
            eta      = avg_time * maxiter
            print(
                f"nfev={nfev:4d} | elapsed={elapsed:.1f}s | "
                f"avg/iter={avg_time:.2f}s | ETA={eta:.1f}s",
                flush=True,
            )

    optimizer = COBYLA(maxiter=maxiter, callback=cobyla_callback)
    sampler   = StatevectorSampler(seed=SEED)
    qaoa_alg  = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)
    solver    = MinimumEigenOptimizer(qaoa_alg)

    print(f"[QAOA] Running qiskit_algorithms.QAOA p={reps}, "
          f"COBYLA maxiter={maxiter}...", flush=True)
    t0        = time.time()
    opt_result = solver.solve(qp)
    runtime_s  = time.time() - t0
    print(f"[QAOA] Done in {runtime_s:.1f}s | "
          f"nfev={len(convergence_curve)} | "
          f"energy={opt_result.fval:.6f}", flush=True)

    # Extract optimal β and γ from the QAOA result
    try:
        opt_params = qaoa_alg.result.optimal_parameters
        gammas = [float(v) for k, v in opt_params.items()
                  if "beta" not in str(k).lower()]
        betas  = [float(v) for k, v in opt_params.items()
                  if "beta" in str(k).lower()]
    except Exception:
        gammas, betas = [], []

    return opt_result, qaoa_alg, betas, gammas, convergence_curve, runtime_s


# ─── 2. RESULT DECODING ──────────────────────────────────────────────────────

def decode_result(opt_result, meta, site_ids):
    """
    Convert OptimizationResult to the standard VQ-MAR result dict,
    matching brute_force_20.json / greedy_20.json format.
    """
    x_vals   = opt_result.x
    bits     = "".join(str(int(v)) for v in x_vals)
    sel_idx  = [i for i, v in enumerate(x_vals) if v > 0.5]
    sel_sites = [site_ids[i] for i in sel_idx]

    w      = np.array(meta["w"])
    C      = np.array(meta["C"])
    budget = meta["budget"]

    return {
        "energy":          float(opt_result.fval),
        "bitstring":       bits,
        "n_selected":      len(sel_sites),
        "selected_sites":  sel_sites,
        "total_benefit":   float(np.sum(w[sel_idx])),
        "total_cost":      float(np.sum(C[sel_idx])),
        "excluded_used":   0,       # updated in compute_extra_metrics
        "budget_B":        budget,
    }


# ─── 3. CORE METRICS ─────────────────────────────────────────────────────────

def compute_metrics(opt_result, result_dict, meta, sites_df, brute_force,
                    greedy, runtime_s):
    """
    Compute the six core run metrics.
    Uses opt_result.samples (from MinimumEigenOptimizer) for the p_success metric.
    """
    n      = meta["n_sites"]
    budget = meta["budget"]
    C      = np.array(meta["C"])

    e_qaoa = result_dict["energy"]
    e_best = brute_force["result"]["energy"]
    ar     = e_qaoa / e_best if e_best != 0 else 0.0

    optimal_bits  = brute_force["result"]["bitstring"]
    e_1pct_thresh = e_best * 0.99

    p_optimal = 0.0
    p_1pct    = 0.0
    for s in opt_result.samples:
        sbits = "".join(str(int(v)) for v in s.x)
        if sbits == optimal_bits:
            p_optimal += s.probability
        if s.fval <= e_1pct_thresh:
            p_1pct += s.probability

    si_col = "Si" if "Si" in sites_df.columns else sites_df.columns[-1]
    si     = sites_df[si_col].values
    sel_idx = [i for i, b in enumerate(result_dict["bitstring"]) if b == "1"]

    sid_to_idx = {sid: i for i, sid in enumerate(sites_df["site_id"].tolist())}
    greedy_idx = [sid_to_idx[s] for s in greedy["result"]["selected_sites"]
                  if s in sid_to_idx]

    rp_qaoa   = float(np.sum(si[sel_idx])) if sel_idx else 0.0
    rp_greedy = float(np.sum(si[greedy_idx])) if greedy_idx else 0.0

    rng = np.random.default_rng(SEED)
    rp_rand = []
    for _ in range(N_RANDOM_PORTFOLIOS):
        perm = rng.permutation(n)
        chosen, cost_acc = [], 0.0
        for idx in perm:
            if cost_acc + C[idx] <= budget:
                chosen.append(idx)
                cost_acc += C[idx]
        rp_rand.append(float(np.sum(si[chosen])) if chosen else 0.0)
    rp_random_mean = float(np.mean(rp_rand))

    portfolio_cost  = result_dict["total_cost"]
    budget_feasible = portfolio_cost <= budget + 1e-9

    return {
        "approximation_ratio":            round(ar, 6),
        "p_success_optimal":              round(p_optimal, 6),
        "p_success_within_1pct":          round(p_1pct, 6),
        "n_qubits":                       n,
        "search_space":                   2 ** n,
        "recharge_potential_qaoa":        round(rp_qaoa, 6),
        "recharge_potential_greedy":      round(rp_greedy, 6),
        "recharge_potential_random_mean": round(rp_random_mean, 6),
        "portfolio_cost":                 round(portfolio_cost, 6),
        "budget_B":                       round(budget, 6),
        "budget_feasible":                budget_feasible,
    }


# ─── 4. EXTRA METRICS (supplementary, no additional runtime cost) ────────────

def compute_extra_metrics(opt_result, Q, result_dict, meta, sites_df, dist_km,
                          betas, gammas, convergence_curve, nfev):
    """
    Supplementary metrics derived from opt_result.samples — no extra circuit runs.

    Supplementary metric mapping:
        p_feasible, distribution_entropy, energy_gap → distribution diagnostics
        qubo_sparsity, qubo_condition_number         → landscape diagnostics
        optimal_beta, optimal_gamma                  → warm-start handoff
        optimizer_nfev, convergence_curve            → convergence diagnostics
        spatial_diversity_km                         → spatial coverage
        excluded_used                                → exclusion-constraint sanity
    """
    n      = meta["n_sites"]
    C      = np.array(meta["C"])
    ei     = np.array(meta.get("ei", [0] * n))
    budget = meta["budget"]

    # ── P(feasible) ───────────────────────────────────────────────────────
    p_feasible = 0.0
    for s in opt_result.samples:
        x_s    = np.array([int(v > 0.5) for v in s.x])
        cost_s = float(x_s @ C)
        excl_s = int(x_s @ ei)
        if cost_s <= budget + 1e-9 and excl_s == 0:
            p_feasible += s.probability

    # ── Distribution entropy (Shannon) ────────────────────────────────────
    probs = np.array([s.probability for s in opt_result.samples])
    probs = probs[probs > 0]
    dist_entropy = float(-np.sum(probs * np.log2(probs)))

    # ── Energy gap ────────────────────────────────────────────────────────
    energies   = sorted(set(round(s.fval, 8) for s in opt_result.samples))
    energy_gap = float(energies[1] - energies[0]) if len(energies) > 1 else 0.0

    # ── QUBO sparsity ─────────────────────────────────────────────────────
    n_offdiag    = n * (n - 1)
    n_nz_off     = int(np.count_nonzero(Q)) - int(np.count_nonzero(np.diag(Q)))
    qubo_sparsity = round(1.0 - n_nz_off / n_offdiag, 6) if n_offdiag > 0 else 1.0

    # ── QUBO condition number ─────────────────────────────────────────────
    try:
        qubo_cond = round(float(np.linalg.cond(Q)), 4)
    except Exception:
        qubo_cond = None

    # ── Spatial diversity ─────────────────────────────────────────────────
    sel_idx = [i for i, b in enumerate(result_dict["bitstring"]) if b == "1"]
    if len(sel_idx) >= 2:
        pairs = [(i, j) for ii, i in enumerate(sel_idx) for j in sel_idx[ii + 1:]]
        spatial_div = round(float(np.mean([dist_km[i][j] for i, j in pairs])), 4)
    else:
        spatial_div = 0.0

    # ── Excluded sites used ───────────────────────────────────────────────
    ei_col        = sites_df["ei"].values if "ei" in sites_df.columns else ei
    excluded_used = int(sum(ei_col[i] for i in sel_idx))

    return {
        "p_feasible":            round(float(p_feasible), 6),
        "distribution_entropy":  round(dist_entropy, 6),
        "energy_gap":            round(energy_gap, 6),
        "qubo_sparsity":         qubo_sparsity,
        "qubo_condition_number": qubo_cond,
        "optimal_beta":          [round(b, 8) for b in betas],
        "optimal_gamma":         [round(g, 8) for g in gammas],
        "optimizer_nfev":        nfev,
        "convergence_curve":     [round(e, 8) for e in convergence_curve],
        "spatial_diversity_km":  spatial_div,
        "excluded_used":         excluded_used,
    }


# ─── 5. EXPORT ───────────────────────────────────────────────────────────────

def export_result(out_dict, cost_model, reps, base_dir, timestamp=""):
    """
    Write qaoa_algorithms_sampler_n20_{cost_model}_p{reps}_{timestamp}.json.
    timestamp: YYYYMMDD_HHMMSS so each run is distinguishable.
    """
    n_sites = 20  # this script is N=20 only
    out_dir = qubo_results_dir(base_dir, cost_model, n_sites)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_part = f"_{timestamp}" if timestamp else ""
    out_path = out_dir / f"qaoa_algorithms_sampler_n{n_sites}_{cost_model}_p{reps}{ts_part}.json"
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)

    print(f"[Export] {out_path}")
    return out_path


# ─── 6. MAIN ─────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Mac/multi-core optimization ───────────────────────────────────────
    n_cores = multiprocessing.cpu_count()
    os.environ.setdefault("OMP_NUM_THREADS", str(n_cores))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n_cores))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_cores))

    parser = argparse.ArgumentParser(
        description="VQ-MAR — canonical qiskit_algorithms.QAOA (slow)"
    )
    parser.add_argument("--cost_model", choices=["flat", "real"], default="flat",
                        help="Cost model: flat (Ci=0.5) or real (Eq. 15)")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="QAOA circuit depth p")
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER,
                        help="COBYLA max function evaluations (default=100; "
                             "use 300 for the full benchmark)")
    parser.add_argument("--base_dir", default=DEFAULT_BASE,
                        help="Project root directory (overrides $VQMAR_BASE)")
    args = parser.parse_args()

    task = "qaoa_algorithms_sampler"

    print("=" * 65)
    print(f"  VQ-MAR — QAOA Canonical (qiskit_algorithms, p={args.reps})")
    print(f"  Cost model : {args.cost_model}")
    print(f"  QAOA depth : p={args.reps}")
    print(f"  Optimizer  : COBYLA (maxiter={args.maxiter})")
    print(f"  Sampler    : StatevectorSampler (seed={SEED})")
    print(f"  CPU cores  : {n_cores}")
    print("=" * 65)
    print()

    # ── Runtime warning ───────────────────────────────────────────────────
    est_min = args.maxiter * 90 / 60
    print(f"[WARNING] qiskit_algorithms.QAOA uses DiagonalEstimator.")
    print(f"          ~90 s per COBYLA call on Apple M4 Pro.")
    print(f"          Estimated total runtime: ~{est_min:.0f} min "
          f"at maxiter={args.maxiter}.")
    print(f"          For fast results: use run_qaoa_native_diagonal.py")
    print(f"          Ctrl+C cancels — partial results will NOT be saved.")
    print()

    # ── Load QUBO data ────────────────────────────────────────────────────
    data     = load_qubo(args.base_dir, args.cost_model)
    Q        = data["Q"]
    meta     = data["meta"]
    site_ids = data["site_ids"]
    n_sites  = len(site_ids)
    bf       = data["brute_force"]
    greedy   = data["greedy"]
    dist_km  = data["dist_km"]
    sites_df = data["sites_df"]

    # ── Build QuadraticProgram + verify energy ────────────────────────────
    print("\nBuilding QuadraticProgram...")
    qp = build_quadratic_program(Q, site_ids)
    verify_energy(Q, meta["qubo_const"], bf)

    # ── Run QAOA ─────────────────────────────────────────────────────────
    print()
    opt_result, qaoa_alg, betas, gammas, conv_curve, runtime_s = run_qaoa(
        qp, args.reps, args.maxiter
    )

    # ── Decode best result ────────────────────────────────────────────────
    result_dict = decode_result(opt_result, meta, site_ids)

    # ── Core metrics ─────────────────────────────────────────────────────
    print("\n[Metrics] Computing core metrics...")
    metrics = compute_metrics(
        opt_result, result_dict, meta, sites_df, bf, greedy, runtime_s
    )

    # ── Extra metrics (supplementary) ────────────────────────────────────
    print("[Metrics] Computing extra_metrics...")
    extra = compute_extra_metrics(
        opt_result, Q, result_dict, meta, sites_df, dist_km,
        betas, gammas, conv_curve, len(conv_curve)
    )
    result_dict["excluded_used"] = extra["excluded_used"]

    # ── Print summary ─────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  QAOA Canonical Results  "
          f"(cost_model={args.cost_model}, p={args.reps})")
    print("=" * 65)
    print(f"  Energy             : {result_dict['energy']:.6f}")
    print(f"  Ground truth       : {bf['result']['energy']:.6f}")
    print(f"  Approximation ratio: {metrics['approximation_ratio']:.4f}")
    print(f"  P(optimal)         : {metrics['p_success_optimal']:.4f}")
    print(f"  P(within 1%)       : {metrics['p_success_within_1pct']:.4f}")
    print(f"  Qubits / 2^n       : {metrics['n_qubits']} / "
          f"{metrics['search_space']:,}")
    print(f"  Runtime            : {runtime_s:.2f}s")
    print(f"  Recharge (QAOA)    : {metrics['recharge_potential_qaoa']:.4f}")
    print(f"  Recharge (greedy)  : {metrics['recharge_potential_greedy']:.4f}")
    print(f"  Portfolio cost     : {metrics['portfolio_cost']:.4f} / "
          f"{metrics['budget_B']:.4f}  "
          f"feasible={metrics['budget_feasible']}")
    print(f"  Selected sites     : {result_dict['selected_sites']}")
    print()
    print(f"  [extra] P(feasible)         : {extra['p_feasible']:.4f}")
    print(f"  [extra] Distribution entropy: {extra['distribution_entropy']:.4f}")
    print(f"  [extra] Energy gap          : {extra['energy_gap']:.6f}")
    print(f"  [extra] Spatial diversity   : {extra['spatial_diversity_km']:.2f} km")
    print(f"  [extra] QUBO sparsity       : {extra['qubo_sparsity']:.2%}")
    print(f"  [extra] COBYLA nfev         : {extra['optimizer_nfev']}")

    # ── Top samples from OptimizationResult ──────────────────────────────
    top_samples = [
        {
            "bitstring":   "".join(str(int(v)) for v in s.x),
            "energy":      round(float(s.fval), 8),
            "probability": round(float(s.probability), 8),
        }
        for s in sorted(opt_result.samples, key=lambda s: s.fval)[:TOP_N_SAMPLES]
    ]

    # ── Assemble output JSON ──────────────────────────────────────────────
    out = {
        "task":          task.split("(")[0],
        "method":        "qaoa_algorithms_sampler",
        "n_sites":       meta["n_sites"],
        "cost_model":    args.cost_model,
        "run_timestamp": timestamp,
        "parameters": {
            "reps":      args.reps,
            "optimizer": "COBYLA",
            "maxiter":   args.maxiter,
            "sampler":   "StatevectorSampler+qiskit_algorithms.QAOA",
            "seed":      SEED,
        },
        "runtime_seconds":     round(runtime_s, 4),
        "result":              result_dict,
        "metrics":             metrics,
        "extra_metrics":       extra,
        "top_samples":         top_samples,
        "ground_truth_energy": bf["result"]["energy"],
    }

    # ── Export ────────────────────────────────────────────────────────────
    print()
    export_result(out, args.cost_model, args.reps, args.base_dir, timestamp)

    print()
    print("=" * 65)
    print(f"  Run COMPLETE  [algorithms_sampler]")
    print(f"  AR = {metrics['approximation_ratio']:.4f}  |  "
          f"P(opt) = {metrics['p_success_optimal']:.4f}  |  "
          f"Runtime = {runtime_s:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
