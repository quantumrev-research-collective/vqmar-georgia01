#!/usr/bin/env python3
"""
VQ-MAR — CVaR-QAOA (Conditional Value at Risk)
==============================================
Method: CVaR-QAOA as in Barkoutsos et al. 2020.
  - Replaces the full expectation value ⟨ψ|H|ψ⟩ with the conditional
    mean of the α-worst energies in the shot distribution.
  - CVaR(α) = E[energy | energy ≤ α-quantile of distribution]
  - α=0.25 means we average over the best 25% of measurement outcomes.
  - Biases the optimizer toward low-energy solutions, improving AR vs
    standard expectation-value QAOA on combinatorial problems.

Usage:
    python scripts/qaoa/run_qaoa_cvar.py \\
        --cost_model flat --reps 1 --alpha 0.25 --maxiter 300 --n_restarts 3

Output:
    georgia/qiskit-ready/qubo_matrices/{cost_model}/top10_cvar_{timestamp}.json
"""

import os
import sys
import json
import time
import datetime
import argparse
import multiprocessing as _mp
import numpy as np
from pathlib import Path

from qiskit_aer import AerSimulator
from qiskit_optimization.translators import to_ising
from scipy.optimize import minimize as scipy_minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qiskit_qubo import load_qubo, build_quadratic_program, verify_energy, qubo_results_dir
from run_qaoa_native_diagonal import (
    build_qaoa_circuit,
    precompute_diagonal,
    decode_result,
    compute_metrics,
    compute_extra_metrics,
)

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT      = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DEFAULT_BASE    = os.environ.get("VQMAR_BASE", _REPO_ROOT)
DEFAULT_REPS    = 1
DEFAULT_MAXITER = 300
DEFAULT_ALPHA   = 0.25
SEED            = 42
TOP_N           = 10   # top-N bitstrings in output


# ─── CVaR objective ──────────────────────────────────────────────────────────

def make_cvar_objective(circuit, param_order, diagonal, sim, alpha,
                        convergence_curve, call_count, t_start, maxiter):
    """
    Return a COBYLA objective that uses CVaR instead of full expectation.

    CVaR(α) = mean(energies that fall within the α-worst quantile).
    For minimization (α=0.25): average over the 25% lowest-energy states
    weighted by their probability mass.
    """
    def objective(params):
        param_dict = dict(zip(param_order, params))
        bound      = circuit.assign_parameters(param_dict)
        sv_data    = sim.run(bound).result().get_statevector()
        probs      = np.abs(np.array(sv_data.data)) ** 2

        # Sort states by energy (ascending)
        order          = np.argsort(diagonal)
        sorted_diag    = diagonal[order]
        sorted_probs   = probs[order]

        # Cumulative probability — find cutoff at alpha
        cumprob = np.cumsum(sorted_probs)
        mask    = cumprob <= alpha

        if not np.any(mask):
            # Edge case: first state alone exceeds alpha
            cvar = float(sorted_diag[0])
        else:
            # Partial last bucket: include fractional contribution
            idx_cut = int(np.sum(mask))
            weight  = sorted_probs[:idx_cut].copy()
            if idx_cut < len(sorted_probs):
                remaining = alpha - float(cumprob[idx_cut - 1]) if idx_cut > 0 else alpha
                weight    = np.append(weight, remaining)
                energies  = sorted_diag[:idx_cut + 1]
            else:
                energies = sorted_diag[:idx_cut]
            total_weight = weight.sum()
            cvar = float(np.dot(weight, energies) / total_weight) if total_weight > 0 else float(sorted_diag[0])

        call_count[0] += 1
        convergence_curve.append(cvar)

        nc = call_count[0]
        if nc == 1 or nc % 10 == 0:
            elapsed = time.time() - t_start
            print(f"nfev={nc:4d} | CVaR(α={alpha})={cvar:.6f} | elapsed={elapsed:.1f}s",
                  flush=True)
        return cvar

    return objective


# ─── Main run ────────────────────────────────────────────────────────────────

def run_cvar(qp, Q, reps, maxiter, alpha, n_restarts):
    """Run CVaR-QAOA with optional random restarts."""
    ising_op, _ = to_ising(qp)
    circuit, gamma_params, beta_params = build_qaoa_circuit(ising_op, reps)
    param_order = list(beta_params) + list(gamma_params)
    n_params    = len(param_order)
    diagonal    = precompute_diagonal(Q)

    n_cores = _mp.cpu_count()
    sim     = AerSimulator(method="statevector", max_parallel_threads=n_cores,
                           statevector_parallel_threshold=12)

    # Warm-up
    _test = circuit.assign_parameters(dict(zip(param_order, np.zeros(n_params))))
    sim.run(_test).result().get_statevector()

    best_energy  = float("inf")
    best_run     = None
    total_rt     = 0.0
    total_nfev   = 0

    for restart in range(n_restarts):
        print(f"\n{'─'*55}")
        print(f"  CVaR restart {restart+1}/{n_restarts}"
              + (" (cold start)" if restart == 0 else " (random)"))
        print(f"{'─'*55}")

        if restart == 0:
            x0 = np.zeros(n_params)
            for i in range(reps):
                x0[i]        = np.pi / 8
                x0[reps + i] = np.pi / 4
        else:
            rng = np.random.default_rng(SEED + restart)
            x0  = rng.uniform(0, np.pi, n_params)

        conv_curve  = []
        call_count  = [0]
        t_start     = time.time()

        obj = make_cvar_objective(circuit, param_order, diagonal, sim, alpha,
                                  conv_curve, call_count, t_start, maxiter)

        result   = scipy_minimize(obj, x0=x0, method="COBYLA",
                                  options={"maxiter": maxiter, "rhobeg": 0.5})
        rt       = time.time() - t_start
        total_rt += rt
        total_nfev += len(conv_curve)

        # Decode best solution from final statevector
        param_dict  = dict(zip(param_order, result.x))
        sv_data     = sim.run(circuit.assign_parameters(param_dict)).result().get_statevector()
        probs_final = np.abs(np.array(sv_data.data)) ** 2

        run_energy = min(conv_curve) if conv_curve else float("inf")
        print(f"[CVaR] Restart {restart+1}: CVaR_final={result.fun:.6f} | "
              f"rt={rt:.1f}s | nfev={len(conv_curve)}")

        if run_energy < best_energy:
            best_energy = run_energy
            betas  = [float(result.x[i]) for i in range(reps)]
            gammas = [float(result.x[reps + i]) for i in range(reps)]
            best_run = (probs_final, diagonal, result.x, betas, gammas,
                        conv_curve, total_rt, total_nfev)

    return best_run + (total_rt, total_nfev)


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    n_cores = _mp.cpu_count()
    os.environ.setdefault("OMP_NUM_THREADS", str(n_cores))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n_cores))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_cores))

    parser = argparse.ArgumentParser(description="VQ-MAR CVaR-QAOA")
    parser.add_argument("--cost_model", choices=["flat", "real"], default="flat")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="CVaR confidence level α ∈ (0,1]. α=1 → standard expectation.")
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--n_restarts", type=int, default=1)
    parser.add_argument("--base_dir", default=DEFAULT_BASE)
    parser.add_argument("--n_sites", type=int, default=20,
                        help="Number of sites (default 20)")
    args = parser.parse_args()

    print("=" * 65)
    print("  VQ-MAR — CVaR-QAOA")
    print(f"  Cost model : {args.cost_model}")
    print(f"  QAOA depth : p={args.reps}")
    print(f"  α (CVaR)   : {args.alpha}")
    print(f"  Restarts   : {args.n_restarts}")
    print(f"  maxiter    : {args.maxiter}")
    print("=" * 65)
    print()

    # Load QUBO
    data     = load_qubo(args.base_dir, args.cost_model, n_sites=args.n_sites)
    Q        = data["Q"]
    meta     = data["meta"]
    site_ids = data["site_ids"]
    bf       = data["brute_force"]
    greedy   = data["greedy"]
    sites_df = data["sites_df"]
    dist_km  = data["dist_km"]

    qp = build_quadratic_program(Q, site_ids)
    verify_energy(Q, meta["qubo_const"], bf)

    # Run CVaR-QAOA
    result_tuple = run_cvar(qp, Q, args.reps, args.maxiter, args.alpha, args.n_restarts)
    probs, diagonal, opt_x, betas, gammas, conv_curve, _, _, total_rt, total_nfev = result_tuple

    # Decode & compute metrics
    result_dict = decode_result(probs, diagonal, meta, site_ids)
    metrics     = compute_metrics(probs, diagonal, result_dict, meta, sites_df,
                                  bf, greedy, total_rt)
    extra       = compute_extra_metrics(probs, diagonal, Q, result_dict, meta,
                                        sites_df, dist_km, betas, gammas,
                                        conv_curve, total_nfev)
    result_dict["excluded_used"] = extra["excluded_used"]

    ar = metrics["approximation_ratio"]
    e_best = bf["result"]["energy"]

    print()
    print("=" * 65)
    print(f"  CVaR-QAOA Results (p={args.reps}, α={args.alpha})")
    print("=" * 65)
    print(f"  Energy     : {result_dict['energy']:.6f}")
    print(f"  Ground truth: {e_best:.6f}")
    print(f"  AR          : {ar:.4f}")
    print(f"  Sites       : {result_dict['selected_sites']}")
    print(f"  Runtime     : {total_rt:.1f}s")

    # ── Build top-10 bitstrings list ──────────────────────────────────────
    C = np.array(meta["C"])
    w = np.array(meta["w"])
    budget = meta["budget"]
    n = meta["n_sites"]

    top_idx = np.argsort(diagonal)[:TOP_N * 5]   # candidate pool
    top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]  # sort by probability within pool
    # re-sort by energy for top-10
    top_by_energy = np.argsort(diagonal)[:TOP_N * 10]
    seen, top10 = set(), []
    for s in top_by_energy:
        if len(top10) >= TOP_N:
            break
        bits = format(int(s), f"0{n}b")
        if bits in seen:
            continue
        seen.add(bits)
        x   = np.array([int(b) for b in bits])
        sel = [site_ids[i] for i, v in enumerate(x) if v]
        top10.append({
            "bitstring": bits,
            "energy":    round(float(diagonal[s]), 8),
            "sites":     sel,
            "benefit":   round(float(np.dot(w, x)), 6),
            "cost":      round(float(np.dot(C, x)), 6),
            "feasible":  bool(np.dot(C, x) <= budget + 1e-9),
            "probability": round(float(probs[s]), 8),
        })

    # ── Immutable schema output ───────────────────────────────────────────
    out = {
        "run_id":              f"cvar_p{args.reps}_{args.cost_model}_{timestamp}",
        "variant":             "cvar",
        "cost_model":          args.cost_model,
        "n_sites":             n,
        "simulator":           "native_diagonal",
        "p_depth":             args.reps,
        "energy":              round(result_dict["energy"], 8),
        "approximation_ratio": round(ar, 6),
        "beta":                [round(b, 8) for b in betas],
        "gamma":               [round(g, 8) for g in gammas],
        "top10_bitstrings":    top10,
        "convergence_curve":   [round(e, 8) for e in conv_curve],
        "seed":                SEED,
        "runtime_seconds":     round(total_rt, 4),
        "run_timestamp":       timestamp,
        # ── Extended fields ───────────────────────────────────────────────
        "cvar_alpha":          args.alpha,
        "n_restarts":          args.n_restarts,
        "result":              result_dict,
        "metrics":             metrics,
        "extra_metrics":       extra,
        "ground_truth_energy": e_best,
    }

    # Export
    out_dir  = qubo_results_dir(args.base_dir, args.cost_model, args.n_sites)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"top10_cvar_n{args.n_sites}_{args.cost_model}_p{args.reps}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[Export] {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
