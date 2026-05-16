#!/usr/bin/env python3
"""
VQ-MAR — QAOA Native Diagonal
=============================

Method: Native hand-crafted QAOA.
  - Builds QAOAAnsatz (qiskit.circuit.library) directly — no qiskit-algorithms.
  - Precomputes the full QUBO Hamiltonian diagonal (x^T Q x for all 2^n states)
    once before optimization (~3 s, ~80 MB).
  - Each COBYLA call: bind γ,β → Statevector.from_instruction() → np.dot(probs, diag).
  - Bypasses DiagonalEstimator entirely — no 1.68 GB per-call overhead.

Expected runtime: ~2–5 min at p=1, maxiter=300 on Apple M4 Pro.

Inputs:
    georgia/qiskit-ready/qubo_matrices/{cost_model}/qubo_20.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/meta_20.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/brute_force_20.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/greedy_20.json
    georgia/qiskit-ready/qubo_matrices/{cost_model}/pairwise_20.json
    georgia/qiskit-ready/site_metadata/{cost_model}/sites_20.csv

Outputs:
    georgia/qiskit-ready/qubo_matrices/{cost_model}/qaoa_native_diagonal_20_p{reps}.json

Usage:
    # p=1 (baseline depth)
    python scripts/qaoa/run_qaoa_native_diagonal.py --cost_model flat --reps 1

    # p=2, p=3 (deeper circuits)
    python scripts/qaoa/run_qaoa_native_diagonal.py --cost_model flat --reps 2
    python scripts/qaoa/run_qaoa_native_diagonal.py --cost_model flat --reps 3
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

# Random portfolio baseline samples for M5 recharge comparison
N_RANDOM_PORTFOLIOS = 1000

# Top-N samples to include in output JSON
TOP_N_SAMPLES = 20

# ─── Hardware optimization guide ──────────────────────────────────────────────
#
# Optimized for Apple M4 Pro (10 performance cores, 48 GB RAM).
# Statevector.from_instruction() uses Apple Accelerate (BLAS) automatically.
#
# To adapt for other hardware:
#   1. DETECT CORE COUNT
#      n_cores = multiprocessing.cpu_count()
#      Set OMP_NUM_THREADS, VECLIB_MAXIMUM_THREADS, OPENBLAS_NUM_THREADS.
#
#   2. MEMORY: precompute_diagonal allocates ~80 MB (float32 bit matrix).
#      For n>24 consider chunked computation or float16.
#
#   3. GPU: Statevector does not use GPU directly. For large n consider
#      qiskit-aer AerSimulator(method='statevector_gpu') instead.


# ─── 1. PRECOMPUTE DIAGONAL ──────────────────────────────────────────────────

def precompute_diagonal(Q):
    """
    Precompute the QUBO energy diagonal over all 2^n computational basis states.

    For each state s (bit string), the QUBO energy is x^T Q x where x is the
    bit vector. This is computed once and reused on every COBYLA call, replacing
    the per-call DiagonalEstimator overhead of qiskit_algorithms.QAOA.

    Memory: ~80 MB float32 for n=20 (2^20 × 20 bits).
    Runtime: ~3 s on M4 Pro for n=20.

    Returns:
        diagonal — float64 ndarray of shape (2^n,)
    """
    n = Q.shape[0]
    N = 2 ** n

    print(f"[Native] Precomputing QUBO diagonal for 2^{n}={N:,} states...",
          flush=True)
    t0 = time.time()

    idx  = np.arange(N, dtype=np.uint32)
    bits = ((idx[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.float32)

    # diagonal[s] = bits[s] @ Q @ bits[s] = sum_ij Q[i,j] * x_i * x_j
    Q_f32 = Q.astype(np.float32)
    qb    = bits @ Q_f32       # (N, n)  — Q times each bit vector
    diag  = np.sum(bits * qb, axis=1).astype(np.float64)   # (N,)

    print(f"[Native] Diagonal ready in {time.time() - t0:.1f}s  "
          f"| range [{diag.min():.4f}, {diag.max():.4f}]", flush=True)
    return diag


# ─── 2. QAOA RUN ─────────────────────────────────────────────────────────────

def build_qaoa_circuit(ising_op, reps):
    """
    Build a QAOA circuit from scratch using only basic gates: H, Rz, Rx, CX.

    Bypasses QAOAAnsatz / PauliEvolutionGate, which causes multi-minute hangs
    in Statevector.from_instruction() due to lazy decomposition of the cost
    operator. This circuit is functionally equivalent but simulates in <0.5s.

    Cost layer U_C(γ):  exp(-iγ * H_Ising) implemented gate-by-gate:
      - For each Z  term h_i  → Rz(2γh_i)  on qubit i
      - For each ZZ term J_ij → CX(i,j) · Rz(2γJ_ij, j) · CX(i,j)
    Mixer layer U_M(β): exp(-iβ * Σ X_i) → Rx(2β) on each qubit.

    Parameter convention: params = [γ_0, β_0, γ_1, β_1, ...]
    """
    n = ising_op.num_qubits
    # ParameterVector names use ASCII for stable sorted ordering:
    # 'beta' < 'gamma' alphabetically → circuit.parameters sorted as [beta[0],...,gamma[0],...]
    gamma = ParameterVector('gamma', reps)
    beta  = ParameterVector('beta', reps)

    # Parse Ising terms (all Z or ZZ, no X/Y in diagonal Hamiltonians)
    z_terms  = []   # (qubit_idx, coefficient)
    zz_terms = []   # (qubit_i, qubit_j, coefficient)
    for label, coeff in zip(ising_op.paulis.to_labels(), ising_op.coeffs):
        # Qiskit Pauli string: rightmost char = qubit 0
        positions = [i for i, p in enumerate(reversed(label)) if p == "Z"]
        if len(positions) == 1:
            z_terms.append((positions[0], float(coeff.real)))
        elif len(positions) == 2:
            zz_terms.append((positions[0], positions[1], float(coeff.real)))
        # len==0 is identity (constant) — no gate needed

    qc = QuantumCircuit(n)
    qc.h(range(n))           # initial |+>^n state

    for layer in range(reps):
        g = gamma[layer]
        b = beta[layer]

        # Cost unitary: exp(-i*g*H_Ising) gate-by-gate
        for qi, h in z_terms:
            qc.rz(2.0 * g * h, qi)
        for qi, qj, J in zz_terms:
            qc.cx(qi, qj)
            qc.rz(2.0 * g * J, qj)
            qc.cx(qi, qj)

        # Mixer unitary: exp(-i*b * sum X_i)
        for qi in range(n):
            qc.rx(2.0 * b, qi)

    qc.save_statevector()   # AerSimulator instruction to extract statevector

    print(f"[Native] Manual QAOA circuit: reps={reps}, "
          f"n_Z={len(z_terms)}, n_ZZ={len(zz_terms)}, "
          f"depth={qc.depth()}, n_params={qc.num_parameters}", flush=True)
    return qc, gamma, beta


def run_qaoa(qp, Q, reps, maxiter, warm_start_params=None,
             optimizer="cobyla", restart_seed=None):
    """
    Run QAOA using a manually built basic-gate circuit + precomputed diagonal + scipy COBYLA.

    Algorithm:
      1. Convert QuadraticProgram → Ising SparsePauliOp
      2. Build QAOA circuit from basic gates (H, Rz, Rx, CX) — no PauliEvolutionGate
      3. Precompute QUBO diagonal once (~0.1 s for n=20)
      4. COBYLA minimizes E(γ,β) = dot(|ψ(γ,β)|², diagonal)
         Each call: ~0.1–0.5 s (vs 60–90 s with DiagonalEstimator)
      5. Return full probability vector from final statevector

    Returns:
        probs      — float64 ndarray shape (2^n,), final measurement distribution
        diagonal   — float64 ndarray shape (2^n,), QUBO energies per state
        opt_params — 1D array [γ_0, β_0, γ_1, β_1, ...]
        betas      — list of optimal β angles
        gammas     — list of optimal γ angles
        conv_curve — list of energy values per COBYLA call
        runtime_s  — wall-clock seconds
        nfev       — total COBYLA function evaluations
    """
    # ── Convert QP → Ising SparsePauliOp ─────────────────────────────────
    ising_op, _offset = to_ising(qp)

    # ── Build circuit (basic gates only, no PauliEvolutionGate) ──────────
    circuit, gamma_params, beta_params = build_qaoa_circuit(ising_op, reps)
    # circuit.parameters sorted alphabetically: 'beta' < 'gamma'
    # → [beta[0],...,beta[reps-1], gamma[0],...,gamma[reps-1]]
    # We explicitly set param_order = betas first, gammas second to match.
    param_order = list(beta_params) + list(gamma_params)
    n_params    = len(param_order)

    # ── Precompute diagonal ───────────────────────────────────────────────
    diagonal = precompute_diagonal(Q)

    # ── AerSimulator (statevector, C++ backend ~0.07s/call vs 10s pure Python)
    n_cores = _mp.cpu_count()
    sim = AerSimulator(method="statevector", max_parallel_threads=n_cores,
                       statevector_parallel_threshold=12)

    # ── Warm-up: one call to trigger any JIT compilation ─────────────────
    print("[Native] Warming up AerSimulator...", flush=True)
    t_wu = time.time()
    _test = circuit.assign_parameters(dict(zip(param_order, np.zeros(n_params))))
    sim.run(_test).result().get_statevector()
    print(f"[Native] Warm-up done in {time.time()-t_wu:.2f}s", flush=True)

    # ── COBYLA objective ──────────────────────────────────────────────────
    convergence_curve = []
    call_count = [0]
    t_start = time.time()

    def objective(params):
        param_dict = dict(zip(param_order, params))
        bound      = circuit.assign_parameters(param_dict)
        sv_data    = sim.run(bound).result().get_statevector()
        probs_     = np.abs(np.array(sv_data.data))**2
        energy     = float(np.dot(probs_, diagonal))

        call_count[0] += 1
        convergence_curve.append(energy)

        nc = call_count[0]
        if nc == 1 or nc % 10 == 0:
            elapsed  = time.time() - t_start
            avg_time = elapsed / nc
            eta      = avg_time * maxiter
            print(
                f"nfev={nc:4d} | energy={energy:.6f} | "
                f"elapsed={elapsed:.1f}s | avg/iter={avg_time:.2f}s | "
                f"ETA={eta:.1f}s",
                flush=True,
            )
        return energy

    # ── Initial parameters ────────────────────────────────────────────────
    # param_order = [beta[0],...,beta[reps-1], gamma[0],...,gamma[reps-1]]
    # warm_start_params set in main() and passed in; None → cold start
    # restart_seed != None → random initial point (for n_restarts > 1)
    if warm_start_params is not None:
        initial_pt = warm_start_params.copy()
        assert len(initial_pt) == n_params, (
            f"warm_start param count mismatch: got {len(initial_pt)}, "
            f"expected {n_params} (reps={reps})"
        )
    elif restart_seed is not None:
        rng        = np.random.default_rng(restart_seed)
        initial_pt = rng.uniform(0, np.pi, n_params)
    else:
        initial_pt = np.zeros(n_params)
        for i in range(reps):
            initial_pt[i]        = np.pi / 8   # beta init
            initial_pt[reps + i] = np.pi / 4   # gamma init

    print(f"\n[QAOA] Running {optimizer.upper()} maxiter={maxiter}...", flush=True)
    t0 = time.time()

    if optimizer == "spsa":
        from qiskit_algorithms.optimizers import SPSA
        spsa_opt   = SPSA(maxiter=maxiter)
        opt_result = spsa_opt.minimize(objective, x0=initial_pt)
        opt_x      = np.array(opt_result.x)
        opt_fun    = float(opt_result.fun)
    else:
        scipy_result = scipy_minimize(
            objective,
            x0=initial_pt,
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": 0.5},
        )
        opt_x   = scipy_result.x
        opt_fun = scipy_result.fun

    runtime_s = time.time() - t0
    print(f"[QAOA] Done in {runtime_s:.1f}s | "
          f"nfev={len(convergence_curve)} | "
          f"energy={opt_fun:.6f}", flush=True)

    # ── Final statevector distribution ────────────────────────────────────
    param_dict  = dict(zip(param_order, opt_x))
    bound_final = circuit.assign_parameters(param_dict)
    sv_data     = sim.run(bound_final).result().get_statevector()
    probs       = np.abs(np.array(sv_data.data))**2

    # ── Split optimal params: [beta[0..reps-1], gamma[0..reps-1]] ────────
    betas  = [float(opt_x[i])        for i in range(reps)]
    gammas = [float(opt_x[reps + i]) for i in range(reps)]

    return (probs, diagonal, opt_x, betas, gammas,
            convergence_curve, runtime_s, len(convergence_curve))


# ─── 3. RESULT DECODING ──────────────────────────────────────────────────────

def decode_result(probs, diagonal, meta, site_ids):
    """
    Decode the optimal solution from the final statevector distribution.

    Uses the minimum-energy state among the top-1000 most probable states.
    This matches the MinimumEigenOptimizer strategy (pick best sample by energy),
    giving comparable results across all three QAOA scripts.

    Note: on real hardware we would take the most frequent measurement outcome.
    For statevector simulation, this hybrid (min-energy from top-probable) is
    equivalent and gives better solutions at equivalent p depth.
    """
    n          = meta["n_sites"]
    w          = np.array(meta["w"])
    C          = np.array(meta["C"])
    budget     = meta["budget"]

    N = len(probs)
    top_k   = np.argsort(probs)[-min(1000, N):]          # top-1000 by probability
    best_state = int(top_k[np.argmin(diagonal[top_k])])  # min-energy among them
    bitstring  = format(best_state, f"0{n}b")
    x_vals     = [int(b) for b in bitstring]

    sel_idx        = [i for i, v in enumerate(x_vals) if v]
    selected_sites = [site_ids[i] for i in sel_idx]

    return {
        "energy":          float(diagonal[best_state]),
        "bitstring":       bitstring,
        "n_selected":      len(sel_idx),
        "selected_sites":  selected_sites,
        "total_benefit":   float(np.sum(w[sel_idx])),
        "total_cost":      float(np.sum(C[sel_idx])),
        "excluded_used":   0,       # updated in compute_extra_metrics
        "budget_B":        budget,
    }


# ─── 4. CORE METRICS ─────────────────────────────────────────────────────────

def compute_metrics(probs, diagonal, result_dict, meta, sites_df,
                    brute_force, greedy, runtime_s):
    """
    Compute the six core run metrics.

    - approximation_ratio              — primary quality claim for the abstract
    - p_success_optimal,               — quantum signal (from full statevector)
      p_success_within_1pct
    - n_qubits / search_space          — resource metric
    - runtime_seconds                  — efficiency (same hardware as baselines)
    - recharge_potential_*             — domain impact (sum Si)
    - portfolio_cost / budget_feasible — budget feasibility
    """
    n      = meta["n_sites"]
    budget = meta["budget"]
    C      = np.array(meta["C"])

    # ── Approximation ratio ───────────────────────────────────────────────
    e_qaoa = result_dict["energy"]
    e_best = brute_force["result"]["energy"]
    ar     = e_qaoa / e_best if e_best != 0 else 0.0

    # ── P(success) — vectorized over full 2^n distribution ───────────────
    e_1pct_thresh = e_best * 0.99   # for negative energies: slightly less negative
    p_optimal     = float(np.sum(probs[np.abs(diagonal - e_best) < 1e-6]))
    p_1pct        = float(np.sum(probs[diagonal <= e_1pct_thresh]))

    # ── Resource ─────────────────────────────────────────────────────────
    n_qubits     = n
    search_space = 2 ** n

    # ── Recharge potential (sum Si) ──────────────────────────────────────
    si_col = "Si" if "Si" in sites_df.columns else sites_df.columns[-1]
    si     = sites_df[si_col].values

    sel_idx = [i for i, b in enumerate(result_dict["bitstring"]) if b == "1"]

    sid_to_idx = {sid: i for i, sid in enumerate(sites_df["site_id"].tolist())}
    greedy_idx = [sid_to_idx[s] for s in greedy["result"]["selected_sites"]
                  if s in sid_to_idx]

    rp_qaoa   = float(np.sum(si[sel_idx])) if sel_idx else 0.0
    rp_greedy = float(np.sum(si[greedy_idx])) if greedy_idx else 0.0

    # Random baseline: mean Si over N random budget-feasible portfolios
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

    # ── Portfolio cost and budget feasibility ─────────────────────────────
    portfolio_cost  = result_dict["total_cost"]
    budget_feasible = portfolio_cost <= budget + 1e-9

    return {
        "approximation_ratio":            round(ar, 6),
        "p_success_optimal":              round(p_optimal, 6),
        "p_success_within_1pct":          round(p_1pct, 6),
        "n_qubits":                       n_qubits,
        "search_space":                   search_space,
        "recharge_potential_qaoa":        round(rp_qaoa, 6),
        "recharge_potential_greedy":      round(rp_greedy, 6),
        "recharge_potential_random_mean": round(rp_random_mean, 6),
        "portfolio_cost":                 round(portfolio_cost, 6),
        "budget_B":                       round(budget, 6),
        "budget_feasible":                budget_feasible,
    }


# ─── 5. EXTRA METRICS (supplementary, no additional runtime cost) ────────────

def compute_extra_metrics(probs, diagonal, Q, result_dict, meta, sites_df,
                          dist_km, betas, gammas, convergence_curve, nfev):
    """
    Compute supplementary metrics at no additional runtime cost.
    All derived from probs (2^n vector) and diagonal — no extra circuit runs.

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
    N      = 2 ** n

    # ── Precompute bit matrix for feasibility checks ──────────────────────
    idx  = np.arange(N, dtype=np.uint32)
    bits = ((idx[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.float32)

    # ── P(feasible): budget-feasible AND no excluded sites ────────────────
    costs      = (bits @ C.astype(np.float32)).astype(np.float64)
    excl_count = (bits @ ei.astype(np.float32)).astype(np.float64)
    feasible   = (costs <= budget + 1e-9) & (excl_count < 0.5)
    p_feasible = float(np.sum(probs[feasible]))

    # ── Distribution entropy (Shannon H in bits) ──────────────────────────
    p_nz       = probs[probs > 0]
    dist_entropy = float(-np.sum(p_nz * np.log2(p_nz)))

    # ── Energy gap: 2nd distinct energy − best (landscape sharpness) ──────
    # Use top-probability states to find the gap efficiently
    top_idx  = np.argsort(probs)[-min(10000, N):][::-1]
    energies = sorted(set(round(float(diagonal[s]), 8) for s in top_idx))
    energy_gap = float(energies[1] - energies[0]) if len(energies) > 1 else 0.0

    # ── QUBO sparsity: fraction non-zero off-diagonal entries ─────────────
    n_offdiag         = n * (n - 1)
    n_nz_offdiag      = int(np.count_nonzero(Q)) - int(np.count_nonzero(np.diag(Q)))
    qubo_sparsity     = round(1.0 - n_nz_offdiag / n_offdiag, 6) if n_offdiag > 0 else 1.0

    # ── QUBO condition number ─────────────────────────────────────────────
    try:
        qubo_cond = round(float(np.linalg.cond(Q)), 4)
    except Exception:
        qubo_cond = None

    # ── Spatial diversity: mean pairwise km between selected sites ────────
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


# ─── 6. EXPORT ───────────────────────────────────────────────────────────────

def export_result(out_dict, cost_model, reps, base_dir,
                  optimizer_suffix="", timestamp=""):
    """
    Write qaoa_native_diagonal_n20_{cost_model}_p{reps}{optimizer_suffix}_{timestamp}.json.
    optimizer_suffix: e.g. "_spsa" for non-COBYLA runs.
    timestamp: YYYYMMDD_HHMMSS string so each run is distinguishable.
    """
    n_sites = 20  # this script is N=20 only
    out_dir = qubo_results_dir(base_dir, cost_model, n_sites)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_part = f"_{timestamp}" if timestamp else ""
    out_path = out_dir / f"qaoa_native_diagonal_n{n_sites}_{cost_model}_p{reps}{optimizer_suffix}{ts_part}.json"
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)

    print(f"[Export] {out_path}")
    return out_path


# ─── 7. MAIN ─────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Mac/multi-core optimization ───────────────────────────────────────
    n_cores = multiprocessing.cpu_count()
    os.environ.setdefault("OMP_NUM_THREADS", str(n_cores))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n_cores))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_cores))

    parser = argparse.ArgumentParser(
        description="VQ-MAR — QAOA native diagonal (fast)"
    )
    parser.add_argument("--cost_model", choices=["flat", "real"], default="flat",
                        help="Cost model: flat (Ci=0.5) or real (Eq. 15)")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="QAOA circuit depth p")
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER,
                        help="COBYLA max function evaluations")
    parser.add_argument("--base_dir", default=DEFAULT_BASE,
                        help="Project root directory (overrides $VQMAR_BASE)")
    parser.add_argument("--warm_start", default=None, metavar="PATH",
                        help="Path to a previous result JSON. Loads optimal_beta and "
                             "optimal_gamma and repeats them for all p layers "
                             "(layer-repetition warm-start). Recommended for real model p≥2.")
    parser.add_argument("--optimizer", choices=["cobyla", "spsa"], default="cobyla",
                        help="Optimizer: cobyla (default, gradient-free trust region) or "
                             "spsa (stochastic, handles rough/high-κ landscapes).")
    parser.add_argument("--n_restarts", type=int, default=1,
                        help="Number of independent random restarts. Restart 0 uses the "
                             "default cold start (π/8, π/4); restarts 1..N use random "
                             "initial points. Best result (lowest energy) is kept.")
    args = parser.parse_args()

    task = "qaoa_native_diagonal"

    # ── Resolve warm-start initial point ─────────────────────────────────
    warm_start_params = None
    if args.warm_start:
        with open(args.warm_start) as f:
            ws = json.load(f)
        b1 = ws["extra_metrics"]["optimal_beta"]   # list, length = p of source run
        g1 = ws["extra_metrics"]["optimal_gamma"]
        # Layer-repetition: repeat the p=1 angles for each layer
        warm_start_params = np.array(b1 * args.reps + g1 * args.reps)
        print(f"[Warm-start] Loaded from {args.warm_start}")
        print(f"             β₀={b1}  γ₀={g1}")
        print(f"             Initial point ({2*args.reps} params): {warm_start_params}")

    print("=" * 65)
    print(f"  VQ-MAR — QAOA Native Diagonal (p={args.reps})")
    print(f"  Cost model : {args.cost_model}")
    print(f"  QAOA depth : p={args.reps}")
    print(f"  Optimizer  : {args.optimizer.upper()} (maxiter={args.maxiter})")
    print(f"  Method     : Precomputed diagonal + Statevector (no Aer)")
    print(f"  Warm-start : {args.warm_start or 'cold (default π/8, π/4)'}")
    print(f"  Restarts   : {args.n_restarts}")
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

    # ── Run QAOA (with optional restarts) ────────────────────────────────
    print()
    best_energy = float("inf")
    best_run    = None
    total_runtime = 0.0

    for restart in range(args.n_restarts):
        if args.n_restarts > 1:
            print(f"\n{'─'*65}")
            label = "cold start" if restart == 0 else "random start"
            print(f"  Restart {restart + 1}/{args.n_restarts} ({label})", flush=True)

        # restart=0: use warm_start or default cold; restart>0: random seed
        seed = (SEED + restart) if restart > 0 else None
        run_result = run_qaoa(
            qp, Q, args.reps, args.maxiter,
            warm_start_params=warm_start_params if restart == 0 else None,
            optimizer=args.optimizer,
            restart_seed=seed,
        )
        _probs, _diag, _opt_x, _betas, _gammas, _curve, _rt, _nfev = run_result
        run_energy = min(_curve) if _curve else float("inf")
        total_runtime += _rt

        if run_energy < best_energy:
            best_energy = run_energy
            best_run    = run_result
            if args.n_restarts > 1:
                print(f"  [Restart {restart+1}] New best energy: {best_energy:.6f}", flush=True)

    if args.n_restarts > 1:
        print(f"\n{'─'*65}")
        print(f"  Best energy across {args.n_restarts} restarts: {best_energy:.6f}", flush=True)

    (probs, diagonal, opt_params, betas, gammas,
     conv_curve, runtime_s, nfev) = best_run
    runtime_s = total_runtime  # report total wall-clock across all restarts

    # ── Decode best result ────────────────────────────────────────────────
    result_dict = decode_result(probs, diagonal, meta, site_ids)

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
    print(f"  QAOA Native Diagonal Results  "
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
    top_idx = np.argsort(probs)[-TOP_N_SAMPLES:][::-1]   # highest prob first
    top_samples = [
        {
            "bitstring":   format(int(s), f"0{n_sites}b"),
            "energy":      round(float(diagonal[s]), 8),
            "probability": round(float(probs[s]), 8),
        }
        for s in top_idx
    ]
    # Sort by energy (ascending) for readability
    top_samples.sort(key=lambda x: x["energy"])

    # ── Assemble output JSON ──────────────────────────────────────────────
    out = {
        "task":       task.split("(")[0],
        "method":     "qaoa_native_diagonal",
        "n_sites":    meta["n_sites"],
        "cost_model": args.cost_model,
        "run_timestamp": timestamp,
        "parameters": {
            "reps":       args.reps,
            "optimizer":  args.optimizer.upper(),
            "maxiter":    args.maxiter,
            "n_restarts": args.n_restarts,
            "sampler":    "Statevector+precomputed_diagonal",
            "seed":       SEED,
        },
        "runtime_seconds":     round(runtime_s, 4),
        "result":              result_dict,
        "metrics":             metrics,
        "extra_metrics":       extra,
        "top_samples":         top_samples,
        "ground_truth_energy": bf["result"]["energy"],
    }

    # ── Export ────────────────────────────────────────────────────────────
    optimizer_suffix = f"_{args.optimizer}" if args.optimizer != "cobyla" else ""
    print()
    export_result(out, args.cost_model, args.reps, args.base_dir,
                  optimizer_suffix, timestamp)

    print()
    print("=" * 65)
    print(f"  Run COMPLETE  [native_diagonal]")
    print(f"  AR = {metrics['approximation_ratio']:.4f}  |  "
          f"P(opt) = {metrics['p_success_optimal']:.4f}  |  "
          f"Runtime = {runtime_s:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
