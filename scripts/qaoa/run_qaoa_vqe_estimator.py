#!/usr/bin/env python3
"""
VQ-MAR — QAOA via VQE + StatevectorEstimator
============================================

Method: Standard Qiskit VQE stack with QAOAAnsatz.
  - Uses VQE (qiskit_algorithms) instead of QAOA/SamplingVQE.
  - VQE routes through StatevectorEstimator — computes <ψ|H|ψ> via Pauli
    decomposition (O(terms × n)), NOT through DiagonalEstimator (O(2^n) per call).
  - Avoids the 1.68 GB per-call overhead of qiskit_algorithms.QAOA 0.4.0.
  - Final distribution obtained by running bound circuit through Statevector.

Expected runtime: ~5–15 min at p=1, maxiter=300 on Apple M4 Pro.

Inputs:
    georgia/qiskit-ready/qubo_matrices/{cost_model}/qubo_20.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/meta_20.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/brute_force_20.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/greedy_20.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/pairwise_20.json
    georgia/qiskit-ready/site_metadata/{cost_model}/sites_20.csv

Outputs:
    georgia/qiskit-ready/qubo_matrices/{cost_model}/qaoa_vqe_estimator_20_p{reps}.json

Usage:
    # p=1 (baseline depth)
    python scripts/qaoa/run_qaoa_vqe_estimator.py --cost_model flat --reps 1

    # p=2, p=3 (deeper circuits)
    python scripts/qaoa/run_qaoa_vqe_estimator.py --cost_model flat --reps 2
    python scripts/qaoa/run_qaoa_vqe_estimator.py --cost_model flat --reps 3
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

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorEstimator
from qiskit_optimization.translators import to_ising
from scipy.optimize import minimize as scipy_minimize
import multiprocessing as _mp

# Import shared QUBO helpers — loading and QuadraticProgram construction
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qiskit_qubo import load_qubo, build_quadratic_program, verify_energy, qubo_results_dir

# ─── Configuration ────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DEFAULT_BASE = os.environ.get("VQMAR_BASE", _REPO_ROOT)

# QAOA defaults
DEFAULT_REPS    = 1    # QAOA circuit depth p
DEFAULT_MAXITER = 300  # COBYLA max function evaluations
SEED            = 42   # reproducibility (matches SA, GA in unified_pipeline.py)

# Random portfolio baseline samples for recharge potential comparison
N_RANDOM_PORTFOLIOS = 1000

# Top-N samples to include in output JSON
TOP_N_SAMPLES = 20

# ─── Hardware optimization guide ──────────────────────────────────────────────
#
# Optimized for Apple M4 Pro (10 performance cores, 48 GB RAM).
# StatevectorEstimator uses BLAS via Apple Accelerate automatically.
#
# To adapt for other hardware:
#   1. DETECT CORE COUNT
#      n_cores = multiprocessing.cpu_count()
#      Set OMP_NUM_THREADS, VECLIB_MAXIMUM_THREADS, OPENBLAS_NUM_THREADS.
#
#   2. ESTIMATOR: StatevectorEstimator is the reference implementation (pure Python +
#      BLAS). For large n, consider AerEstimatorV2 (C++ backend).
#
#   3. GPU: not currently used by StatevectorEstimator. Build qiskit-aer with
#      CUDA/Metal for GPU statevector on NVIDIA/Apple Silicon.


# ─── 1. MANUAL QAOA CIRCUIT BUILDER ──────────────────────────────────────────

def build_qaoa_circuit(ising_op, reps):
    """
    Build QAOA circuit from basic gates (H, Rz, Rx, CX) — no PauliEvolutionGate.

    QAOAAnsatz and PauliEvolutionGate cause Statevector to hang indefinitely
    during decomposition of 190+ ZZ terms. This circuit is equivalent but uses
    only native gates that Statevector and StatevectorEstimator handle in <0.5s.

    Parameter convention: circuit.parameters sorted alphabetically:
      'beta' < 'gamma' → [beta[0],...,beta[reps-1], gamma[0],...,gamma[reps-1]]

    Cost layer: exp(-i*gamma*H_Ising) = product of Rz and CX-Rz-CX gates.
    Mixer layer: exp(-i*beta * sum X_i) = product of Rx(2*beta) gates.
    """
    n = ising_op.num_qubits
    gamma = ParameterVector('gamma', reps)
    beta  = ParameterVector('beta', reps)

    z_terms  = []
    zz_terms = []
    for label, coeff in zip(ising_op.paulis.to_labels(), ising_op.coeffs):
        positions = [i for i, p in enumerate(reversed(label)) if p == "Z"]
        if len(positions) == 1:
            z_terms.append((positions[0], float(coeff.real)))
        elif len(positions) == 2:
            zz_terms.append((positions[0], positions[1], float(coeff.real)))

    qc = QuantumCircuit(n)
    qc.h(range(n))

    for layer in range(reps):
        g = gamma[layer]
        b = beta[layer]
        for qi, h in z_terms:
            qc.rz(2.0 * g * h, qi)
        for qi, qj, J in zz_terms:
            qc.cx(qi, qj)
            qc.rz(2.0 * g * J, qj)
            qc.cx(qi, qj)
        for qi in range(n):
            qc.rx(2.0 * b, qi)

    # save_statevector() for AerSimulator post-optimization distribution read-out
    qc_with_sv = qc.copy()
    qc_with_sv.save_statevector()

    print(f"[VQE] Manual QAOA circuit: reps={reps}, "
          f"n_Z={len(z_terms)}, n_ZZ={len(zz_terms)}, "
          f"depth={qc.depth()}, n_params={qc.num_parameters}", flush=True)
    return qc, qc_with_sv, gamma, beta


# ─── 2. QAOA RUN (VQE + StatevectorEstimator) ────────────────────────────────

def run_qaoa(qp, reps, maxiter):
    """
    Run QAOA using StatevectorEstimator + scipy COBYLA (no qiskit_algorithms.VQE).

    Why scipy directly and not qiskit_algorithms.VQE:
      qiskit-algorithms COBYLA wraps scipy and passes the user callback straight
      through. New scipy (pyprima in Python 3.14) calls callback(x) — just params,
      no energy. The old 5-argument VQE callback signature crashes. By calling
      scipy.optimize.minimize directly we control the objective and convergence
      tracking without any callback compatibility issues.

      StatevectorEstimator computes <ψ|H|ψ> via Pauli decomposition — O(terms×n),
      not O(2^n). No DiagonalEstimator; no 1.68 GB allocation per call.

    Returns:
        probs      — float64 ndarray shape (2^n,), final measurement distribution
        ising_op   — SparsePauliOp Ising Hamiltonian (for metric computation)
        offset     — float, constant offset from QUBO→Ising conversion
        betas      — list of optimal β angles
        gammas     — list of optimal γ angles
        conv_curve — list of QUBO energy values per COBYLA call
        runtime_s  — wall-clock seconds
        nfev       — total COBYLA function evaluations
        ansatz     — QuantumCircuit ansatz (for post-hoc use)
        opt_params — dict mapping Parameter → float (optimal values)
    """
    # ── Convert QP → Ising SparsePauliOp ─────────────────────────────────
    ising_op, offset = to_ising(qp)
    print(f"[VQE] Ising Hamiltonian: {len(ising_op)} Pauli terms, "
          f"offset={offset:.6f}", flush=True)

    # ── Build manual QAOA circuit (basic gates only) ──────────────────────
    ansatz, ansatz_sv, gamma_params, beta_params = build_qaoa_circuit(ising_op, reps)

    # Sorted parameters: 'beta' < 'gamma' alphabetically
    # → layout [beta[0..reps-1], gamma[0..reps-1]]
    sorted_params = sorted(ansatz.parameters, key=lambda p: p.name)

    initial_point = np.zeros(2 * reps)
    initial_point[:reps] = np.pi / 8   # beta init
    initial_point[reps:] = np.pi / 4   # gamma init

    # ── Objective: <ψ|H_Ising|ψ> via StatevectorEstimator ────────────────
    estimator         = StatevectorEstimator()
    convergence_curve = []
    call_count        = [0]
    t_start           = time.time()

    def objective(params):
        param_dict = dict(zip(sorted_params, params))
        bound      = ansatz.assign_parameters(param_dict)
        job        = estimator.run([(bound, ising_op)])
        energy_ising = float(job.result()[0].data.evs)
        energy_qubo  = energy_ising + offset
        convergence_curve.append(energy_qubo)
        call_count[0] += 1
        nfev = call_count[0]
        if nfev == 1 or nfev % 10 == 0:
            elapsed  = time.time() - t_start
            avg_time = elapsed / nfev
            eta      = avg_time * maxiter
            print(
                f"nfev={nfev:4d} | energy={energy_qubo:.6f} | "
                f"elapsed={elapsed:.1f}s | avg/iter={avg_time:.2f}s | "
                f"ETA={eta:.1f}s",
                flush=True,
            )
        return energy_ising   # scipy minimizes this (Ising scale)

    print(f"\n[QAOA] Running StatevectorEstimator+COBYLA maxiter={maxiter}...",
          flush=True)
    t0     = time.time()
    result = scipy_minimize(
        objective, initial_point, method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 1.0},
    )
    runtime_s = time.time() - t0

    energy_ising = float(result.fun)
    energy_qubo  = energy_ising + offset
    print(f"[VQE] Done in {runtime_s:.1f}s | "
          f"nfev={result.nfev} | "
          f"eigenvalue={energy_ising:.6f} | "
          f"QUBO energy={energy_qubo:.6f}", flush=True)

    # ── Final statevector distribution (AerSimulator, ~0.07s) ────────────
    opt_params  = dict(zip(sorted_params, result.x))
    bound_final = ansatz_sv.assign_parameters(opt_params)
    n_cores     = _mp.cpu_count()
    sim         = AerSimulator(method="statevector", max_parallel_threads=n_cores,
                               statevector_parallel_threshold=12)
    sv_data     = sim.run(bound_final).result().get_statevector()
    probs       = np.abs(np.array(sv_data.data))**2

    # ── Extract betas and gammas — sorted params: [beta[0..], gamma[0..]] ─
    betas  = [float(result.x[i]) for i in range(reps)]
    gammas = [float(result.x[reps + i]) for i in range(reps)]

    return (probs, ising_op, offset, betas, gammas,
            convergence_curve, runtime_s, int(result.nfev),
            ansatz, opt_params)


# ─── 2. RESULT DECODING ──────────────────────────────────────────────────────

def decode_result(probs, Q, ising_op, offset, meta, site_ids):
    """
    Decode the optimal solution from the final statevector distribution.

    Uses the precomputed QUBO diagonal (x^T Q x) to assign energies to states
    rather than the Ising eigenvalue, so energy is directly comparable to
    brute_force_20.json and greedy_20.json.
    """
    n = meta["n_sites"]
    N = 2 ** n

    # Precompute diagonal for decoding (small n, fast)
    idx  = np.arange(N, dtype=np.uint32)
    bits = ((idx[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.float32)
    qb   = bits @ Q.astype(np.float32)
    diag = np.sum(bits * qb, axis=1).astype(np.float64)

    top_k      = np.argsort(probs)[-min(1000, N):]          # top-1000 by probability
    best_state = int(top_k[np.argmin(diag[top_k])])         # min-energy among them
    bitstring  = format(best_state, f"0{n}b")
    x_vals     = [int(b) for b in bitstring]

    w   = np.array(meta["w"])
    C   = np.array(meta["C"])
    sel = [i for i, v in enumerate(x_vals) if v]

    return diag, {
        "energy":         float(diag[best_state]),
        "bitstring":      bitstring,
        "n_selected":     len(sel),
        "selected_sites": [site_ids[i] for i in sel],
        "total_benefit":  float(np.sum(w[sel])),
        "total_cost":     float(np.sum(C[sel])),
        "excluded_used":  0,
        "budget_B":       meta["budget"],
    }


# ─── 3. CORE METRICS ─────────────────────────────────────────────────────────

def compute_metrics(probs, diagonal, result_dict, meta, sites_df,
                    brute_force, greedy, runtime_s):
    """
    Compute the six core run metrics.

    Same implementation as run_qaoa_native_diagonal.py — probs is a full
    statevector probability array, diagonal is the QUBO energy per state.
    """
    n      = meta["n_sites"]
    budget = meta["budget"]
    C      = np.array(meta["C"])

    e_qaoa = result_dict["energy"]
    e_best = brute_force["result"]["energy"]
    ar     = e_qaoa / e_best if e_best != 0 else 0.0

    e_1pct_thresh = e_best * 0.99
    p_optimal     = float(np.sum(probs[np.abs(diagonal - e_best) < 1e-6]))
    p_1pct        = float(np.sum(probs[diagonal <= e_1pct_thresh]))

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

def compute_extra_metrics(probs, diagonal, Q, result_dict, meta, sites_df,
                          dist_km, betas, gammas, convergence_curve, nfev):
    """
    Supplementary metrics derived from probs + diagonal — zero extra circuit runs.
    See run_qaoa_native_diagonal.py for field descriptions and metric mapping.
    """
    n      = meta["n_sites"]
    C      = np.array(meta["C"])
    ei     = np.array(meta.get("ei", [0] * n))
    budget = meta["budget"]
    N      = 2 ** n

    idx       = np.arange(N, dtype=np.uint32)
    bits      = ((idx[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.float32)
    costs     = (bits @ C.astype(np.float32)).astype(np.float64)
    excl_cnt  = (bits @ ei.astype(np.float32)).astype(np.float64)
    feasible  = (costs <= budget + 1e-9) & (excl_cnt < 0.5)
    p_feasible = float(np.sum(probs[feasible]))

    p_nz         = probs[probs > 0]
    dist_entropy = float(-np.sum(p_nz * np.log2(p_nz)))

    top_idx    = np.argsort(probs)[-min(10000, N):][::-1]
    energies   = sorted(set(round(float(diagonal[s]), 8) for s in top_idx))
    energy_gap = float(energies[1] - energies[0]) if len(energies) > 1 else 0.0

    n_offdiag    = n * (n - 1)
    n_nz_off     = int(np.count_nonzero(Q)) - int(np.count_nonzero(np.diag(Q)))
    qubo_sparsity = round(1.0 - n_nz_off / n_offdiag, 6) if n_offdiag > 0 else 1.0

    try:
        qubo_cond = round(float(np.linalg.cond(Q)), 4)
    except Exception:
        qubo_cond = None

    sel_idx = [i for i, b in enumerate(result_dict["bitstring"]) if b == "1"]
    if len(sel_idx) >= 2:
        pairs = [(i, j) for ii, i in enumerate(sel_idx) for j in sel_idx[ii + 1:]]
        spatial_div = round(float(np.mean([dist_km[i][j] for i, j in pairs])), 4)
    else:
        spatial_div = 0.0

    ei_col        = sites_df["ei"].values if "ei" in sites_df.columns else ei
    excluded_used = int(sum(ei_col[i] for i in sel_idx))

    return {
        "p_feasible":            round(p_feasible, 6),
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
    Write qaoa_vqe_estimator_n20_{cost_model}_p{reps}_{timestamp}.json.
    timestamp: YYYYMMDD_HHMMSS so each run is distinguishable.
    """
    n_sites = 20  # this script is N=20 only
    out_dir = qubo_results_dir(base_dir, cost_model, n_sites)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_part = f"_{timestamp}" if timestamp else ""
    out_path = out_dir / f"qaoa_vqe_estimator_n{n_sites}_{cost_model}_p{reps}{ts_part}.json"
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
        description="VQ-MAR — QAOA via VQE + StatevectorEstimator"
    )
    parser.add_argument("--cost_model", choices=["flat", "real"], default="flat",
                        help="Cost model: flat (Ci=0.5) or real (Eq. 15)")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="QAOA circuit depth p")
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER,
                        help="COBYLA max function evaluations")
    parser.add_argument("--base_dir", default=DEFAULT_BASE,
                        help="Project root directory (overrides $VQMAR_BASE)")
    args = parser.parse_args()

    task = "qaoa_vqe_estimator"

    print("=" * 65)
    print(f"  VQ-MAR — QAOA VQE + StatevectorEstimator (p={args.reps})")
    print(f"  Cost model : {args.cost_model}")
    print(f"  QAOA depth : p={args.reps}")
    print(f"  Optimizer  : COBYLA (maxiter={args.maxiter})")
    print(f"  Method     : VQE + StatevectorEstimator (no DiagonalEstimator)")
    print(f"  CPU cores  : {n_cores}")
    print("=" * 65)
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

    # ── Run QAOA via VQE ──────────────────────────────────────────────────
    print()
    (probs, ising_op, offset, betas, gammas,
     conv_curve, runtime_s, nfev,
     ansatz, opt_params) = run_qaoa(qp, args.reps, args.maxiter)

    # ── Decode best result (includes diagonal precompute) ─────────────────
    diagonal, result_dict = decode_result(probs, Q, ising_op, offset, meta, site_ids)

    # ── Core metrics ─────────────────────────────────────────────────────
    print("\n[Metrics] Computing core metrics...")
    metrics = compute_metrics(
        probs, diagonal, result_dict, meta, sites_df, bf, greedy, runtime_s
    )

    # ── Extra metrics (supplementary) ────────────────────────────────────
    print("[Metrics] Computing extra_metrics...")
    extra = compute_extra_metrics(
        probs, diagonal, Q, result_dict, meta, sites_df, dist_km,
        betas, gammas, conv_curve, nfev
    )
    result_dict["excluded_used"] = extra["excluded_used"]

    # ── Print summary ─────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  QAOA VQE Estimator Results  "
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

    # ── Build top_samples from full distribution ──────────────────────────
    n_sites = meta["n_sites"]
    top_idx = np.argsort(probs)[-TOP_N_SAMPLES:][::-1]
    top_samples = [
        {
            "bitstring":   format(int(s), f"0{n_sites}b"),
            "energy":      round(float(diagonal[s]), 8),
            "probability": round(float(probs[s]), 8),
        }
        for s in top_idx
    ]
    top_samples.sort(key=lambda x: x["energy"])

    # ── Assemble output JSON ──────────────────────────────────────────────
    out = {
        "task":          task.split("(")[0],
        "method":        "qaoa_vqe_estimator",
        "n_sites":       meta["n_sites"],
        "cost_model":    args.cost_model,
        "run_timestamp": timestamp,
        "parameters": {
            "reps":      args.reps,
            "optimizer": "COBYLA",
            "maxiter":   args.maxiter,
            "sampler":   "StatevectorEstimator+VQE",
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
    print(f"  Run COMPLETE  [vqe_estimator]")
    print(f"  AR = {metrics['approximation_ratio']:.4f}  |  "
          f"P(opt) = {metrics['p_success_optimal']:.4f}  |  "
          f"Runtime = {runtime_s:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
