#!/usr/bin/env python3
"""
VQ-MAR — Matrix Product State (MPS) QAOA Simulator
==================================================

Performance optimizations:
  • Parallel restarts: Multiple random restarts now run in parallel across cores
  • Auto bond-dim fallback: Detects OOM and automatically downgrades χ
  • Auto-tuned parameters: bond_dim and maxiter auto-selected based on N
  • Reduced logging overhead: Log interval scaled to reduce I/O overhead
  • Per-restart simulators: Avoids thread contention by using single-threaded simulators

Why MPS? The native diagonal method precomputes 2^n diagonal values.
At N=50 this is 2^50 ≈ 10^15 values — intractable. MPS (tensor network)
represents the statevector approximately by keeping only the dominant
entanglement structure up to a bond dimension χ (mps_bond_dim).

Memory:
  N=20, χ=64 : ~20 × 64^2 × 8 bytes ≈ 6 MB    (exact for small χ)
  N=50, χ=64 : ~50 × 64^2 × 8 bytes ≈ 16 MB   (approximate, may OOM)
  N=50, χ=32 : ~50 × 32^2 × 8 bytes ≈ 4 MB    (recommended for N=50)

Trade-off: lower χ truncates entanglement → lower AR. For QAOA at p=1
the entanglement is limited, so χ=32 is usually sufficient for N=50.

Examples:

    # Full N=20 run (classic settings):
    python scripts/qaoa/run_qaoa_mps.py \\
        --cost_model flat --reps 1 --n_sites 20 --n_restarts 3

    # Full N=50 run (optimized settings, auto bond_dim + maxiter):
    python scripts/qaoa/run_qaoa_mps.py \\
        --cost_model flat --reps 1 --n_sites 50 --n_restarts 3
    
    # Manual override (if auto-selection fails):
    python scripts/qaoa/run_qaoa_mps.py \\
        --cost_model flat --reps 1 --n_sites 50 --mps_bond_dim 16 --maxiter 100 --n_restarts 3

Output:
    georgia/qiskit-ready/qubo_matrices/{cost_model}/top10_mps_{n_sites}q_bd{bond_dim}_{timestamp}.json
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
from concurrent.futures import ProcessPoolExecutor, as_completed

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_optimization.translators import to_ising
from scipy.optimize import minimize as scipy_minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qiskit_qubo import load_qubo, build_quadratic_program, verify_energy, qubo_results_dir
from run_qaoa_native_diagonal import build_qaoa_circuit

_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT       = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DEFAULT_BASE     = os.environ.get("VQMAR_BASE", _REPO_ROOT)
DEFAULT_REPS     = 1
DEFAULT_MAXITER  = 300
DEFAULT_BOND_DIM = 64
DEFAULT_N_SITES  = 20
SEED             = 42
TOP_N            = 10


# ─── MPS expectation value ────────────────────────────────────────────────────

def mps_objective(circuit, param_order, Q, sim, n_sites, convergence_curve,
                  call_count, t_start, maxiter):
    """
    Return COBYLA objective using MPS statevector simulation.

    Unlike native_diagonal (precomputes 2^n energies), here we compute
    the expectation value from shot sampling of the MPS statevector.
    Uses 'statevector' output mode from AerSim MPS — gives exact
    probabilities within the MPS approximation.
    """
    n = n_sites
    log_interval = max(1, maxiter // 10)  # Log ~10 times per run

    def objective(params):
        param_dict = dict(zip(param_order, params))
        bound      = circuit.assign_parameters(param_dict)

        if n <= 28:
            # Statevector: exact probabilities for N≤28
            result  = sim.run(bound).result()
            sv_data = result.get_statevector()
            probs   = np.abs(np.array(sv_data.data)) ** 2
            idx  = np.arange(2**n, dtype=np.uint32)
            bits = ((idx[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.float32)
            diag = np.sum(bits * (bits @ Q.astype(np.float32)), axis=1).astype(np.float64)
            energy = float(np.dot(probs, diag))
        else:
            # Shot-based: avoids materializing 2^n statevector for N>28
            meas = QuantumCircuit(bound.num_qubits, bound.num_qubits)
            for inst in bound.data:
                if inst.operation.name != 'save_statevector':
                    meas.append(inst)
            meas.measure_all(add_bits=False)
            counts = sim.run(meas, shots=2048).result().get_counts()
            total  = sum(counts.values())
            energy = 0.0
            for bs, cnt in counts.items():
                x = np.array([int(b) for b in bs[::-1][:n]], dtype=float)
                energy += cnt * float(x @ Q @ x)
            energy /= total if total > 0 else 1.0

        call_count[0] += 1
        convergence_curve.append(energy)
        nc = call_count[0]
        # Reduced logging for faster execution
        if nc == 1 or nc % log_interval == 0:
            elapsed = time.time() - t_start
            print(f"  nfev={nc:3d} | E={energy:.6f} | t={elapsed:.1f}s", flush=True)
        return energy

    return objective


def _run_single_restart(restart_idx, circuit, param_order, Q, reps, n_sites, maxiter, 
                        bond_dim, n_cores):
    """Run a single QAOA optimization restart (for parallelization)."""
    # Each restart gets its own simulator to avoid thread contention
    try:
        sim = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=bond_dim,
            max_parallel_threads=1,  # Avoid thread oversubscription
        )
    except Exception as e:
        print(f"[Restart {restart_idx+1}] OOM with bond_dim={bond_dim}, falling back to {bond_dim//2}", 
              flush=True)
        sim = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=bond_dim//2,
            max_parallel_threads=1,
        )
    
    n_params = len(param_order)
    if restart_idx == 0:
        x0 = np.zeros(n_params)
        for i in range(reps):
            x0[i]        = np.pi / 8
            x0[reps + i] = np.pi / 4
    else:
        rng = np.random.default_rng(SEED + restart_idx)
        x0  = rng.uniform(0, np.pi, n_params)

    conv_curve = []
    call_count = [0]
    t_start    = time.time()

    obj = mps_objective(circuit, param_order, Q, sim, n_sites,
                        conv_curve, call_count, t_start, maxiter)

    # Adaptive rhobeg based on problem size
    rhobeg = max(0.3, min(0.5, 2.0 / np.sqrt(n_sites)))
    
    result = scipy_minimize(obj, x0=x0, method="COBYLA",
                            options={"maxiter": maxiter, "rhobeg": rhobeg})
    
    rt = time.time() - t_start
    run_energy = min(conv_curve) if conv_curve else float("inf")
    
    return {
        "restart_idx": restart_idx,
        "energy": run_energy,
        "result": result,
        "conv_curve": conv_curve,
        "runtime": rt,
        "nfev": len(conv_curve),
        "reps": reps,
    }


def run_qaoa_mps(qp, Q, reps, maxiter, bond_dim, n_restarts, n_sites):
    """Run QAOA using MPS simulator with parallel random restarts."""
    ising_op, _ = to_ising(qp)
    circuit, gamma_params, beta_params = build_qaoa_circuit(ising_op, reps)
    # Remove save_statevector (added by build_qaoa_circuit for AerSim)
    # and add it back in a way compatible with MPS
    from qiskit import QuantumCircuit as QC
    stripped = QC(circuit.num_qubits)
    for inst in circuit.data:
        if inst.operation.name != 'save_statevector':
            stripped.append(inst)
    stripped.save_statevector()
    circuit = stripped

    param_order = list(beta_params) + list(gamma_params)
    n_params    = len(param_order)

    n_cores = _mp.cpu_count()
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # Prevent over-subscription
    
    print(f"[MPS] Simulator: method=matrix_product_state, bond_dim={bond_dim}")
    print(f"[MPS] Running {n_restarts} restart(s) in parallel across {n_cores} cores")

    # Warm-up on first simulator
    try:
        sim_wu = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=bond_dim,
            max_parallel_threads=1,
        )
        _test = circuit.assign_parameters(dict(zip(param_order, np.zeros(n_params))))
        t_wu  = time.time()
        if n_sites <= 28:
            sim_wu.run(_test).result().get_statevector()
        else:
            # Shot-based warm-up: avoids materializing 2^n statevector
            meas_wu = QuantumCircuit(_test.num_qubits, _test.num_qubits)
            for inst in _test.data:
                if inst.operation.name != 'save_statevector':
                    meas_wu.append(inst)
            meas_wu.measure_all(add_bits=False)
            sim_wu.run(meas_wu, shots=128).result().get_counts()
        print(f"[MPS] Warm-up done in {time.time()-t_wu:.2f}s", flush=True)
    except Exception as e:
        print(f"[MPS] Warm-up OOM with bond_dim={bond_dim}: {e}")
        print(f"[MPS] Auto-fallback to bond_dim={bond_dim//2}")
        bond_dim = bond_dim // 2

    best_energy = float("inf")
    best_run    = None
    total_rt    = 0.0
    total_nfev  = 0
    
    # Parallel restarts using ProcessPoolExecutor
    if n_restarts == 1:
        # Single restart: run sequentially
        result_dict = _run_single_restart(0, circuit, param_order, Q, reps, n_sites, 
                                         maxiter, bond_dim, n_cores)
        results = [result_dict]
    else:
        # Multiple restarts: parallelize across available cores
        n_workers = min(n_restarts, max(1, n_cores // 2))  # Leave room for other processes
        print(f"[MPS] Using {n_workers} worker(s)")
        
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_run_single_restart, i, circuit, param_order, Q, reps, 
                              n_sites, maxiter, bond_dim, n_cores)
                for i in range(n_restarts)
            ]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"[MPS] Restart failed: {e}", flush=True)

    # Select best result
    for result_dict in results:
        restart_idx = result_dict["restart_idx"]
        run_energy = result_dict["energy"]
        result = result_dict["result"]
        conv_curve = result_dict["conv_curve"]
        rt = result_dict["runtime"]
        nfev = result_dict["nfev"]
        
        print(f"\n[MPS] Restart {restart_idx+1}: E={result.fun:.6f} | rt={rt:.1f}s")
        
        total_rt += rt
        total_nfev += nfev
        
        if run_energy < best_energy:
            best_energy = run_energy
            betas  = [float(result.x[i]) for i in range(reps)]
            gammas = [float(result.x[reps + i]) for i in range(reps)]

            # Final state from best parameters — statevector for N≤28, shots for N>28
            try:
                sim_final = AerSimulator(
                    method="matrix_product_state",
                    matrix_product_state_max_bond_dimension=bond_dim,
                    max_parallel_threads=1,
                )
                param_dict  = dict(zip(param_order, result.x))
                circ_final  = circuit.assign_parameters(param_dict)
                if n_sites <= 28:
                    sv_data     = sim_final.run(circ_final).result().get_statevector()
                    probs_final = np.abs(np.array(sv_data.data)) ** 2
                else:
                    meas_f = QuantumCircuit(circ_final.num_qubits, circ_final.num_qubits)
                    for inst in circ_final.data:
                        if inst.operation.name != 'save_statevector':
                            meas_f.append(inst)
                    meas_f.measure_all(add_bits=False)
                    probs_final = sim_final.run(meas_f, shots=8192).result().get_counts()
            except Exception as e:
                print(f"[MPS] Failed to compute final state: {e}", flush=True)
                probs_final = {} if n_sites > 28 else np.array([])

            best_run = (probs_final, betas, gammas, conv_curve, total_rt, total_nfev)

    return best_run + (total_rt, total_nfev)


def decode_mps_result(probs, Q, meta, site_ids, n_sites):
    """Decode best solution from MPS result.

    probs: numpy array (N≤28, statevector probabilities) or
           dict of {bitstring: count} (N>28, shot-based counts).
    """
    n      = n_sites
    N      = 2 ** n
    w      = np.array(meta["w"])
    C      = np.array(meta["C"])
    budget = meta["budget"]

    if N <= 2 ** 28:
        # probs is a numpy probability array
        idx   = np.arange(N, dtype=np.uint32)
        bits  = ((idx[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.float32)
        diag  = np.sum(bits * (bits @ Q.astype(np.float32)), axis=1).astype(np.float64)
        top_k = np.argsort(probs)[-min(1000, N):]
        best  = int(top_k[np.argmin(diag[top_k])])
        bits_best = [int(b) for b in format(best, f"0{n}b")]
        energy    = float(diag[best])
    else:
        # probs is a counts dict from shot-based sampling — find min-energy shot
        counts    = probs if isinstance(probs, dict) else {}
        best_e    = float("inf")
        best_bits = [0] * n
        for bs, cnt in counts.items():
            bits_s = [int(b) for b in bs[::-1][:n]]
            x_s    = np.array(bits_s, dtype=float)
            e      = float(x_s @ Q @ x_s)
            if e < best_e:
                best_e    = e
                best_bits = bits_s
        bits_best = best_bits
        energy    = best_e

    bitstring = "".join(str(b) for b in bits_best)
    x_arr     = np.array(bits_best, dtype=float)
    sel_idx   = [i for i, v in enumerate(bits_best) if v]
    selected  = [site_ids[i] for i in sel_idx]

    return {
        "energy":         energy,
        "bitstring":      bitstring,
        "n_selected":     len(selected),
        "selected_sites": selected,
        "total_benefit":  float(np.dot(w[:len(x_arr)], x_arr)),
        "total_cost":     float(np.dot(C[:len(x_arr)], x_arr)),
        "excluded_used":  0,
        "budget_B":       budget,
    }


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    n_cores = _mp.cpu_count()
    os.environ.setdefault("OMP_NUM_THREADS", str(n_cores))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n_cores))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_cores))

    parser = argparse.ArgumentParser(description="VQ-MAR MPS QAOA Simulator")
    parser.add_argument("--cost_model", choices=["flat", "real"], default="flat")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--mps_bond_dim", type=int, default=None,
                        help="MPS max bond dimension χ. Auto-select if not provided: "
                             "64 for N≤30, 32 for N=50, 16 for N>50.")
    parser.add_argument("--n_sites", type=int, default=DEFAULT_N_SITES,
                        help="Number of sites. N=20 (default) or N=50 (needs qubo_50 files).")
    parser.add_argument("--variant", default="baseline",
                        choices=["baseline", "cvar", "choice_a", "choice_b"],
                        help="Variant label for output JSON")
    parser.add_argument("--maxiter", type=int, default=None,
                        help="Max iterations for COBYLA. Auto-select if not provided: "
                             "300 for N≤30, 200 for N=50, 100 for N>50.")
    parser.add_argument("--n_restarts", type=int, default=3,
                        help="Number of random restarts (default 3, runs in parallel)")
    parser.add_argument("--fixed_params", default=None, metavar="PATH",
                        help="Path to result JSON to load β/γ from (e.g., N=20 statevector "
                             "result). Runs a single fixed-params shot-sampled circuit instead "
                             "of COBYLA optimization. Recommended for N>28.")
    parser.add_argument("--shots", type=int, default=8192,
                        help="Shots for fixed-params mode (default 8192). Ignored in optimize mode.")
    parser.add_argument("--base_dir", default=DEFAULT_BASE)
    args = parser.parse_args()
    
    # Auto-select bond_dim and maxiter based on n_sites
    if args.mps_bond_dim is None:
        if args.n_sites <= 30:
            args.mps_bond_dim = 64
        elif args.n_sites <= 50:
            args.mps_bond_dim = 32
        else:
            args.mps_bond_dim = 16
        print(f"[Auto] bond_dim selected: {args.mps_bond_dim} for N={args.n_sites}")
    
    if args.maxiter is None:
        if args.n_sites <= 30:
            args.maxiter = 300
        elif args.n_sites <= 50:
            args.maxiter = 200
        else:
            args.maxiter = 100
        print(f"[Auto] maxiter selected: {args.maxiter} for N={args.n_sites}")

    print("=" * 65)
    print("  VQ-MAR — MPS QAOA Simulator")
    print(f"  Cost model : {args.cost_model}")
    print(f"  n_sites    : {args.n_sites}")
    print(f"  QAOA depth : p={args.reps}")
    print(f"  bond_dim   : χ={args.mps_bond_dim}")
    print(f"  Variant    : {args.variant}")
    if args.fixed_params:
        print(f"  Mode       : fixed-params ({args.shots} shots)")
        print(f"  Params src : {Path(args.fixed_params).name}")
    else:
        print(f"  Mode       : optimize (COBYLA, {args.n_restarts} restart(s))")
    print("=" * 65)
    print()

    data     = load_qubo(args.base_dir, args.cost_model, n_sites=args.n_sites)
    Q        = data["Q"]
    meta     = data["meta"]
    site_ids = data["site_ids"]
    bf       = data["brute_force"]

    # N=20 uses existing QUBO; N=50 requires separate files (generated by unified_pipeline.py)
    if args.n_sites != Q.shape[0]:
        raise ValueError(
            f"--n_sites {args.n_sites} but loaded Q is {Q.shape[0]}×{Q.shape[0]}. "
            f"Run unified_pipeline.py --n_sites {args.n_sites} first to generate "
            f"the {args.n_sites}-site QUBO files."
        )

    qp = build_quadratic_program(Q, site_ids)
    verify_energy(Q, meta["qubo_const"], bf, n_sites=args.n_sites)

    if args.fixed_params:
        # Fixed-params mode: load β/γ from JSON, run once, sample shots (no COBYLA)
        with open(args.fixed_params) as f:
            src = json.load(f)
        em     = src.get("extra_metrics", {})
        betas  = em.get("optimal_beta")  or src.get("beta",  [np.pi / 8])
        gammas = em.get("optimal_gamma") or src.get("gamma", [np.pi / 4])
        reps   = len(betas)

        print(f"[Fixed] β={[round(b,4) for b in betas]}, γ={[round(g,4) for g in gammas]}")

        from qiskit_optimization.translators import to_ising
        ising_op, _ = to_ising(qp)
        circuit, gamma_params, beta_params = build_qaoa_circuit(ising_op, reps)
        param_order  = list(beta_params) + list(gamma_params)
        param_values = list(betas) + list(gammas)
        bound        = circuit.assign_parameters(dict(zip(param_order, param_values)))

        meas = QuantumCircuit(bound.num_qubits, bound.num_qubits)
        for inst in bound.data:
            if inst.operation.name != 'save_statevector':
                meas.append(inst)
        meas.measure_all(add_bits=False)

        sim = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=args.mps_bond_dim,
            max_parallel_threads=n_cores,
        )
        print(f"[Fixed] Running {args.shots} shots...", flush=True)
        t0      = time.time()
        probs   = sim.run(meas, shots=args.shots).result().get_counts()
        total_rt   = time.time() - t0
        total_nfev = 1
        conv_curve = []
        print(f"[Fixed] Done in {total_rt:.1f}s | {len(probs)} unique bitstrings", flush=True)

    else:
        # COBYLA optimization mode
        result_tuple = run_qaoa_mps(qp, Q, args.reps, args.maxiter,
                                    args.mps_bond_dim, args.n_restarts, args.n_sites)
        probs, betas, gammas, conv_curve, _, _, total_rt, total_nfev = result_tuple

    result_dict = decode_mps_result(probs, Q, meta, site_ids, args.n_sites)

    e_best = bf["result"]["energy"]
    ar     = result_dict["energy"] / e_best if e_best != 0 else 0.0

    print()
    print("=" * 65)
    print(f"  MPS Results (N={args.n_sites}, χ={args.mps_bond_dim}, p={args.reps})")
    print("=" * 65)
    print(f"  Energy     : {result_dict['energy']:.6f}")
    print(f"  Ground truth: {e_best:.6f}")
    print(f"  AR          : {ar:.4f}")
    print(f"  Sites       : {result_dict['selected_sites']}")
    print(f"  Runtime     : {total_rt:.1f}s")

    if args.mps_bond_dim <= 32:
        print("\n  NOTE: bond_dim ≤ 32 — entanglement truncation may lower AR.")
        print("        Record in paper Methodology/Discussion.")

    # Build top-10: statevector for N≤28, shot counts for N>28
    n = args.n_sites
    C, w, budget = np.array(meta["C"]), np.array(meta["w"]), meta["budget"]
    top10 = []
    if n <= 28:
        N = 2 ** n
        idx   = np.arange(N, dtype=np.uint32)
        bits  = ((idx[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.float32)
        diag  = np.sum(bits * (bits @ Q.astype(np.float32)), axis=1).astype(np.float64)
        top_by_energy = np.argsort(diag)[:TOP_N * 10]
        seen = set()
        for s in top_by_energy:
            if len(top10) >= TOP_N:
                break
            bs = format(int(s), f"0{n}b")
            if bs in seen:
                continue
            seen.add(bs)
            x_s = np.array([int(b) for b in bs])
            sel = [site_ids[i] for i, v in enumerate(x_s) if v]
            top10.append({
                "bitstring": bs,
                "energy":    round(float(diag[s]), 8),
                "sites":     sel,
                "benefit":   round(float(np.dot(w, x_s)), 6),
                "cost":      round(float(np.dot(C, x_s)), 6),
                "feasible":  bool(np.dot(C, x_s) <= budget + 1e-9),
                "probability": round(float(probs[s]), 8),
            })
    elif isinstance(probs, dict) and probs:
        # N>28: build top10 from shot counts, ranked by energy
        total_shots = sum(probs.values())
        shot_entries = []
        for bs, cnt in probs.items():
            bits_s = [int(b) for b in bs[::-1][:n]]
            x_s    = np.array(bits_s, dtype=float)
            e      = float(x_s @ Q @ x_s)
            shot_entries.append((e, bits_s, cnt))
        shot_entries.sort(key=lambda t: t[0])
        seen = set()
        for e, bits_s, cnt in shot_entries:
            if len(top10) >= TOP_N:
                break
            bitkey = "".join(str(b) for b in bits_s)
            if bitkey in seen:
                continue
            seen.add(bitkey)
            sel = [site_ids[i] for i, v in enumerate(bits_s) if v]
            top10.append({
                "bitstring":   bitkey,
                "energy":      round(e, 8),
                "sites":       sel,
                "benefit":     round(float(np.dot(w[:n], bits_s)), 6),
                "cost":        round(float(np.dot(C[:n], bits_s)), 6),
                "feasible":    bool(np.dot(C[:n], bits_s) <= budget + 1e-9),
                "probability": round(cnt / total_shots, 8),
            })

    mode_tag = "fixed_params" if args.fixed_params else "optimize"
    out = {
        "run_id":              f"mps_{args.variant}_n{n}_{args.cost_model}_bd{args.mps_bond_dim}_{mode_tag}_{timestamp}",
        "variant":             args.variant,
        "cost_model":          args.cost_model,
        "n_sites":             n,
        "simulator":           "mps",
        "p_depth":             args.reps,
        "mode":                mode_tag,
        "energy":              round(result_dict["energy"], 8),
        "approximation_ratio": round(ar, 6),
        "beta":                [round(b, 8) for b in betas],
        "gamma":               [round(g, 8) for g in gammas],
        "top10_bitstrings":    top10,
        "convergence_curve":   [round(e, 8) for e in conv_curve],
        "seed":                SEED,
        "runtime_seconds":     round(total_rt, 4),
        "run_timestamp":       timestamp,
        "mps_bond_dim":        args.mps_bond_dim,
        "shots":               args.shots if args.fixed_params else None,
        "fixed_params_source": str(args.fixed_params) if args.fixed_params else None,
        "result":              result_dict,
        "ground_truth_energy": e_best,
        "bond_dim_note":       (
            "bond_dim ≤ 32: entanglement truncation may reduce AR — "
            "acknowledge in paper Methodology." if args.mps_bond_dim <= 32 else None
        ),
    }

    out_dir  = qubo_results_dir(args.base_dir, args.cost_model, args.n_sites)
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_suffix = {"choice_a": "_spatial_diversity_choiceA", "choice_b": "_hydraulic_diversity_choiceB"}.get(args.variant, "")
    fname    = f"top10_mps_n{n}_{args.cost_model}_bd{args.mps_bond_dim}_p{args.reps}_{timestamp}{variant_suffix}.json"
    out_path = out_dir / fname

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[Export] {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
