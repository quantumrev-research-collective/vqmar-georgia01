#!/usr/bin/env python3
"""
VQ-MAR — Spatial Diversity QAOA (Choice A)
==========================================
Method: Standard QAOA with a modified benefit vector that adds a
spatial diversity bonus to each site's weight wi.

QUBO modification (Choice A — Linear diversity):
    wi_A = wi_base + delta * diversity_bonus(i)

Diversity bonus: each site's bonus is proportional to its mean distance
to all other sites, normalized to [0,1]. Sites that are geographically
isolated get a higher bonus — selecting them increases portfolio spread.

    diversity_bonus(i) = mean_j(dist_km[i,j]) / max_k(mean_j(dist_km[k,j]))

delta (default 0.10) controls the diversity weight relative to the base
benefit. The resulting Q matrix differs from the baseline only in its
diagonal entries (linear terms in the QUBO).

Usage:
    python scripts/qaoa/run_qaoa_spatial_diversity.py \\
        --cost_model flat --reps 1 --delta 0.10 --maxiter 300 --n_restarts 3

Output:
    georgia/qiskit-ready/qubo_matrices/{cost_model}/top10_spatial_diversity_{timestamp}.json
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
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_ising
from scipy.optimize import minimize as scipy_minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qiskit_qubo import load_qubo, verify_energy, qubo_results_dir
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
DEFAULT_DELTA   = 0.10   # diversity bonus weight (fraction of wi range)
SEED            = 42
TOP_N           = 10


def build_choice_a_qubo(Q, meta, dist_km, delta):
    """
    Build a modified QUBO matrix for Choice A.

    Modifies diagonal entries: Q_A[i,i] = Q[i,i] - delta * diversity_bonus[i]
    (subtract because QUBO minimizes; higher bonus → more negative diagonal → preferred)

    Returns Q_A (modified matrix) and the diversity bonuses for logging.
    """
    n = Q.shape[0]
    mean_dist = np.array([np.mean([dist_km[i][j] for j in range(n) if j != i])
                          for i in range(n)])
    max_dist = mean_dist.max()
    diversity_bonus = mean_dist / max_dist if max_dist > 0 else np.zeros(n)

    Q_A = Q.copy()
    for i in range(n):
        Q_A[i, i] -= delta * diversity_bonus[i]

    return Q_A, diversity_bonus


def build_choice_a_qp(Q_A, site_ids):
    """Build QuadraticProgram from Choice A QUBO matrix."""
    n = Q_A.shape[0]
    qp = QuadraticProgram("vqmar_choice_a")
    for i in range(n):
        qp.binary_var(f"x{i}")
    linear    = {f"x{i}": float(Q_A[i, i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            val = float(Q_A[i, j] + Q_A[j, i])
            if val != 0.0:
                quadratic[(f"x{i}", f"x{j}")] = val
    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


def run_qaoa_variant(qp, Q, reps, maxiter, n_restarts):
    """Standard COBYLA QAOA with random restarts — identical to native diagonal."""
    ising_op, _ = to_ising(qp)
    circuit, gamma_params, beta_params = build_qaoa_circuit(ising_op, reps)
    param_order = list(beta_params) + list(gamma_params)
    n_params    = len(param_order)
    diagonal    = precompute_diagonal(Q)

    n_cores = _mp.cpu_count()
    sim     = AerSimulator(method="statevector", max_parallel_threads=n_cores,
                           statevector_parallel_threshold=12)

    _test = circuit.assign_parameters(dict(zip(param_order, np.zeros(n_params))))
    sim.run(_test).result().get_statevector()

    best_energy = float("inf")
    best_run    = None
    total_rt    = 0.0
    total_nfev  = 0

    for restart in range(n_restarts):
        print(f"\n{'─'*55}")
        print(f"  Restart {restart+1}/{n_restarts}"
              + (" (cold)" if restart == 0 else " (random)"))

        if restart == 0:
            x0 = np.zeros(n_params)
            for i in range(reps):
                x0[i]        = np.pi / 8
                x0[reps + i] = np.pi / 4
        else:
            rng = np.random.default_rng(SEED + restart)
            x0  = rng.uniform(0, np.pi, n_params)

        conv_curve = []
        call_count = [0]
        t_start    = time.time()

        def objective(params):
            param_dict = dict(zip(param_order, params))
            bound      = circuit.assign_parameters(param_dict)
            sv_data    = sim.run(bound).result().get_statevector()
            probs      = np.abs(np.array(sv_data.data)) ** 2
            energy     = float(np.dot(probs, diagonal))
            call_count[0] += 1
            conv_curve.append(energy)
            nc = call_count[0]
            if nc == 1 or nc % 10 == 0:
                print(f"nfev={nc:4d} | E={energy:.6f} | t={time.time()-t_start:.1f}s",
                      flush=True)
            return energy

        result   = scipy_minimize(objective, x0=x0, method="COBYLA",
                                  options={"maxiter": maxiter, "rhobeg": 0.5})
        rt       = time.time() - t_start
        total_rt += rt
        total_nfev += len(conv_curve)

        run_energy = min(conv_curve) if conv_curve else float("inf")
        print(f"[Choice A] Restart {restart+1}: E={result.fun:.6f} | rt={rt:.1f}s")

        if run_energy < best_energy:
            best_energy = run_energy
            betas  = [float(result.x[i]) for i in range(reps)]
            gammas = [float(result.x[reps + i]) for i in range(reps)]

            param_dict  = dict(zip(param_order, result.x))
            sv_data     = sim.run(circuit.assign_parameters(param_dict)).result().get_statevector()
            probs_final = np.abs(np.array(sv_data.data)) ** 2

            best_run = (probs_final, diagonal, result.x, betas, gammas,
                        conv_curve, total_rt, total_nfev)

    return best_run


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    n_cores = _mp.cpu_count()
    os.environ.setdefault("OMP_NUM_THREADS", str(n_cores))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n_cores))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_cores))

    parser = argparse.ArgumentParser(description="VQ-MAR Spatial Diversity QAOA — Linear Diversity QAOA")
    parser.add_argument("--cost_model", choices=["flat", "real"], default="flat")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA,
                        help="Diversity bonus weight (fraction of wi range, default 0.10)")
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--n_restarts", type=int, default=1)
    parser.add_argument("--base_dir", default=DEFAULT_BASE)
    parser.add_argument("--n_sites", type=int, default=20,
                        help="Number of sites (default 20)")
    args = parser.parse_args()

    print("=" * 65)
    print("  VQ-MAR — Spatial Diversity QAOA (Choice A)")
    print(f"  Cost model : {args.cost_model}")
    print(f"  QAOA depth : p={args.reps}")
    print(f"  delta      : {args.delta}")
    print(f"  Restarts   : {args.n_restarts}")
    print("=" * 65)
    print()

    data     = load_qubo(args.base_dir, args.cost_model, n_sites=args.n_sites)
    Q        = data["Q"]
    meta     = data["meta"]
    site_ids = data["site_ids"]
    bf       = data["brute_force"]
    greedy   = data["greedy"]
    sites_df = data["sites_df"]
    dist_km  = data["dist_km"]

    verify_energy(Q, meta["qubo_const"], bf)

    # Build Choice A QUBO
    Q_A, div_bonus = build_choice_a_qubo(Q, meta, dist_km, args.delta)
    qp_A           = build_choice_a_qp(Q_A, site_ids)

    print(f"\n[Choice A] Diversity bonuses (δ={args.delta}):")
    for i, (sid, b) in enumerate(zip(site_ids, div_bonus)):
        print(f"  {sid}: bonus={b:.4f}  Δdiag={-args.delta*b:.4f}")

    # Run QAOA on modified QUBO
    result_tuple = run_qaoa_variant(qp_A, Q_A, args.reps, args.maxiter, args.n_restarts)
    probs, diagonal, opt_x, betas, gammas, conv_curve, total_rt, total_nfev = result_tuple

    # Decode using ORIGINAL Q for proper energy comparison
    result_dict = decode_result(probs, diagonal, meta, site_ids)
    # Recompute energy using original Q for AR (fair comparison)
    n = meta["n_sites"]
    x_arr  = np.array([int(b) for b in result_dict["bitstring"]])
    true_energy = float(x_arr @ Q @ x_arr)
    result_dict["energy"] = true_energy

    metrics = compute_metrics(probs, diagonal, result_dict, meta, sites_df,
                               bf, greedy, total_rt)
    extra   = compute_extra_metrics(probs, diagonal, Q, result_dict, meta,
                                    sites_df, dist_km, betas, gammas,
                                    conv_curve, total_nfev)
    result_dict["excluded_used"] = extra["excluded_used"]

    # Recompute AR against original Q ground truth
    e_best = bf["result"]["energy"]
    ar     = true_energy / e_best if e_best != 0 else 0.0
    metrics["approximation_ratio"] = round(ar, 6)

    print()
    print("=" * 65)
    print(f"  Choice A Results (p={args.reps}, δ={args.delta})")
    print("=" * 65)
    print(f"  Energy (orig Q): {true_energy:.6f}  |  Ground truth: {e_best:.6f}")
    print(f"  AR             : {ar:.4f}")
    print(f"  Sites          : {result_dict['selected_sites']}")
    print(f"  Runtime        : {total_rt:.1f}s")

    # Build top-10
    C, w, budget = np.array(meta["C"]), np.array(meta["w"]), meta["budget"]
    top_by_energy = np.argsort(diagonal)[:TOP_N * 10]
    seen, top10 = set(), []
    for s in top_by_energy:
        if len(top10) >= TOP_N:
            break
        bits = format(int(s), f"0{n}b")
        if bits in seen:
            continue
        seen.add(bits)
        x_s = np.array([int(b) for b in bits])
        sel = [site_ids[i] for i, v in enumerate(x_s) if v]
        top10.append({
            "bitstring": bits,
            "energy":    round(float(x_s @ Q @ x_s), 8),  # original Q energy
            "sites":     sel,
            "benefit":   round(float(np.dot(w, x_s)), 6),
            "cost":      round(float(np.dot(C, x_s)), 6),
            "feasible":  bool(np.dot(C, x_s) <= budget + 1e-9),
            "probability": round(float(probs[s]), 8),
        })

    out = {
        "run_id":              f"choice_a_p{args.reps}_{args.cost_model}_{timestamp}",
        "variant":             "choice_a",
        "cost_model":          args.cost_model,
        "n_sites":             n,
        "simulator":           "native_diagonal",
        "p_depth":             args.reps,
        "energy":              round(true_energy, 8),
        "approximation_ratio": round(ar, 6),
        "beta":                [round(b, 8) for b in betas],
        "gamma":               [round(g, 8) for g in gammas],
        "top10_bitstrings":    top10,
        "convergence_curve":   [round(e, 8) for e in conv_curve],
        "seed":                SEED,
        "runtime_seconds":     round(total_rt, 4),
        "run_timestamp":       timestamp,
        "choice_a_delta":      args.delta,
        "diversity_bonuses":   [round(float(b), 6) for b in div_bonus],
        "result":              result_dict,
        "metrics":             metrics,
        "extra_metrics":       extra,
        "ground_truth_energy": e_best,
    }

    out_dir  = qubo_results_dir(args.base_dir, args.cost_model, args.n_sites)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"top10_spatial_diversity_n{args.n_sites}_{args.cost_model}_p{args.reps}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[Export] {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
