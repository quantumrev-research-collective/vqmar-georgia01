#!/usr/bin/env python3
"""
VQ-MAR — Noisy Aer QAOA Simulation
==================================
Simulates QAOA with a realistic hardware noise model derived from
FakeBrisbane (127-qubit Eagle r1 calibration data — same device family
as ibm_kingston used in the real hardware run).

This provides a noise-aware baseline sitting between:
  - Statevector simulator (no noise, AR=1.0)
  - Real hardware (ibm_kingston, AR=0.9939)

The noise model includes:
  - Depolarizing gate errors (1-qubit: ~0.1%, 2-qubit CX: ~0.5–1%)
  - Thermal relaxation (T1/T2)
  - Readout errors

We fix the optimal β/γ from the statevector run and measure the
shot distribution under noise — same methodology as the hardware run.

Usage:
    # Fixed-params mode (uses simulator β/γ):
    python scripts/qaoa/run_qaoa_noisy_aer.py \\
        --cost_model flat --reps 1 --shots 8192 \\
        --fixed_params georgia/qiskit-ready/qubo_matrices/flat/qaoa_native_diagonal_20_p1.json

    # Full optimization with noise (slow — noise model makes each call 10x slower):
    python scripts/qaoa/run_qaoa_noisy_aer.py \\
        --cost_model flat --reps 1 --shots 4096 --maxiter 100 --optimize

Output:
    georgia/qiskit-ready/qubo_matrices/{cost_model}/top10_noisy_aer_{timestamp}.json
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

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_optimization.translators import to_ising
from scipy.optimize import minimize as scipy_minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qiskit_qubo import load_qubo, build_quadratic_program, verify_energy, qubo_results_dir
from run_qaoa_native_diagonal import build_qaoa_circuit

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT      = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DEFAULT_BASE    = os.environ.get("VQMAR_BASE", _REPO_ROOT)
DEFAULT_REPS    = 1
DEFAULT_SHOTS   = 8192
DEFAULT_MAXITER = 100
SEED            = 42
TOP_N           = 10


def build_noisy_simulator(shots, n_sites=20):
    """
    Build an AerSimulator with a noise model.

    For n_sites ≤ 28: uses full FakeBrisbane noise model (statevector method).
    For n_sites > 28: uses MPS method with calibrated depolarizing noise
    (FakeBrisbane error rates: ~0.1% 1q, ~1% 2q). FakeBrisbane requires
    statevector internally, which is infeasible for N>28 (~17 PB).
    """
    n_cores = _mp.cpu_count()

    if n_sites > 28:
        from qiskit_aer.noise import depolarizing_error, NoiseModel as NM
        noise_model  = NM()
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ['rz', 'sx', 'x'])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01,  2), ['cx', 'ecr'])
        basis_gates  = ['cx', 'rz', 'sx', 'x']
        coupling_map = None
        print(f"[Noisy] N={n_sites}>28: using MPS method + depolarizing noise")
        print(f"        (FakeBrisbane error rates: 1q~0.1%, 2q~1%)")
        sim = AerSimulator(
            method="matrix_product_state",
            noise_model=noise_model,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            shots=shots,
            max_parallel_threads=1,
            matrix_product_state_max_bond_dimension=32,
        )
        return sim, noise_model

    try:
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        fake_backend = FakeBrisbane()
        noise_model  = NoiseModel.from_backend(fake_backend)
        basis_gates  = fake_backend.operation_names
        coupling_map = fake_backend.coupling_map
        print(f"[Noisy] Noise model from FakeBrisbane (Eagle r1 family)")
        print(f"        Basis gates: {[g for g in basis_gates if g in ['cx','ecr','rz','x','sx','measure']]}")
    except Exception as e:
        print(f"[Noisy] FakeBrisbane unavailable ({e}); using simple depolarizing noise")
        from qiskit_aer.noise import depolarizing_error, NoiseModel as NM
        noise_model  = NM()
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ['rz', 'sx', 'x'])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2),  ['cx'])
        basis_gates  = ['cx', 'rz', 'sx', 'x', 'measure']
        coupling_map = None

    sim = AerSimulator(
        noise_model=noise_model,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        shots=shots,
        max_parallel_threads=n_cores,
    )
    return sim, noise_model


def run_fixed_params_noisy(ising_op, betas, gammas, sim, shots, Q, meta, site_ids):
    """
    Build QAOA circuit, transpile for noise basis gates, run with noise model.
    Mirrors run_qaoa_hardware.py fixed-params approach but uses Aer noise model.
    """
    reps = len(betas)
    circuit, gamma_params, beta_params = build_qaoa_circuit(ising_op, reps)

    param_order  = list(beta_params) + list(gamma_params)
    param_values = list(betas) + list(gammas)
    bound        = circuit.assign_parameters(dict(zip(param_order, param_values)))

    # Strip save_statevector (AerSim instruction not compatible with noise sim)
    meas_circuit = QuantumCircuit(bound.num_qubits, bound.num_qubits)
    for inst in bound.data:
        if inst.operation.name != 'save_statevector':
            meas_circuit.append(inst)
    meas_circuit.measure_all(add_bits=False)

    print(f"[Noisy] Running with β={[round(b,4) for b in betas]}, "
          f"γ={[round(g,4) for g in gammas]}, shots={shots}")

    t0     = time.time()
    job    = sim.run(meas_circuit, shots=shots)
    result = job.result()
    rt     = time.time() - t0

    # Decode counts
    counts = result.get_counts()
    print(f"[Noisy] Done in {rt:.1f}s | {len(counts)} unique bitstrings")

    n = Q.shape[0]
    w      = np.array(meta["w"])
    C      = np.array(meta["C"])
    budget = meta["budget"]

    best_energy = float("inf")
    best_bits   = None
    samples     = []

    for bitstring, count in counts.items():
        # Qiskit returns MSB-first; reverse to qubit0=index0
        bits_rev = bitstring[::-1]
        x        = np.array([int(b) for b in bits_rev[:n]], dtype=float)
        energy   = float(x @ Q @ x)
        prob     = count / shots
        samples.append((energy, bits_rev, prob, count))
        if energy < best_energy:
            best_energy = energy
            best_bits   = bits_rev

    samples.sort(key=lambda t: t[0])

    x_best   = np.array([int(b) for b in best_bits[:n]], dtype=float)
    sel_idx  = [i for i, v in enumerate(x_best) if v > 0.5]
    selected = [site_ids[i] for i in sel_idx]

    result_dict = {
        "energy":         best_energy,
        "bitstring":      best_bits[:n],
        "n_selected":     len(selected),
        "selected_sites": selected,
        "total_benefit":  float(np.sum(w[sel_idx])),
        "total_cost":     float(np.sum(C[sel_idx])),
        "excluded_used":  0,
        "budget_B":       budget,
    }

    top10 = []
    for e, bits, prob, count in samples[:TOP_N]:
        x_s = np.array([int(b) for b in bits[:n]])
        sel = [site_ids[i] for i, v in enumerate(x_s) if v]
        top10.append({
            "bitstring":   bits[:n],
            "energy":      round(float(e), 8),
            "sites":       sel,
            "benefit":     round(float(np.dot(w[:n], x_s)), 6),
            "cost":        round(float(np.dot(C[:n], x_s)), 6),
            "feasible":    bool(np.dot(C[:n], x_s) <= budget + 1e-9),
            "probability": round(prob, 8),
            "count":       count,
        })

    return result_dict, top10, rt


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    n_cores = _mp.cpu_count()
    os.environ.setdefault("OMP_NUM_THREADS", str(n_cores))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n_cores))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_cores))

    parser = argparse.ArgumentParser(description="VQ-MAR Noisy Aer QAOA")
    parser.add_argument("--cost_model", choices=["flat", "real"], default="flat")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--fixed_params", default=None, metavar="PATH",
                        help="Path to simulator result JSON to load β/γ from. "
                             "Recommended: use statevector AR=1.0 result.")
    parser.add_argument("--optimize", action="store_true",
                        help="Run COBYLA optimization loop with noise (slow). "
                             "Default: fixed-params mode (one circuit).")
    parser.add_argument("--base_dir", default=DEFAULT_BASE)
    parser.add_argument("--n_sites", type=int, default=20,
                        help="Number of sites (20 or 50). Selects Q_{n}.npy and meta_{n}.json.")
    args = parser.parse_args()

    if not args.fixed_params and not args.optimize:
        # Default to using statevector optimal params
        default_src = (Path(args.base_dir) / "georgia" / "qiskit-ready"
                       / "qubo_matrices" / args.cost_model
                       / "n20" / "results" / f"qaoa_native_diagonal_n20_{args.cost_model}_p1.json")
        if default_src.exists():
            args.fixed_params = str(default_src)
            print(f"[Noisy] No --fixed_params specified; defaulting to {default_src.name}")
        else:
            parser.error("Provide --fixed_params PATH or --optimize flag.")

    print("=" * 65)
    print("  VQ-MAR — Noisy Aer Simulation")
    print(f"  Cost model  : {args.cost_model}")
    print(f"  QAOA depth  : p={args.reps}")
    print(f"  Shots       : {args.shots}")
    print(f"  Mode        : {'fixed-params' if args.fixed_params else 'optimize'}")
    noise_label = ("MPS + depolarizing (1q~0.1%, 2q~1%)" if args.n_sites > 28
                   else "FakeBrisbane (Eagle r1 family)")
    print(f"  Noise model : {noise_label}")
    print("=" * 65)
    print()

    data     = load_qubo(args.base_dir, args.cost_model, n_sites=args.n_sites)
    Q        = data["Q"]
    meta     = data["meta"]
    site_ids = data["site_ids"]
    bf       = data["brute_force"]

    qp = build_quadratic_program(Q, site_ids)
    verify_energy(Q, meta["qubo_const"], bf, n_sites=args.n_sites)

    sim, noise_model = build_noisy_simulator(args.shots, n_sites=args.n_sites)
    ising_op, _      = to_ising(qp)

    if args.fixed_params:
        with open(args.fixed_params) as f:
            src = json.load(f)
        betas  = src.get("extra_metrics", src).get("optimal_beta",  src.get("beta",  [np.pi/8]))
        gammas = src.get("extra_metrics", src).get("optimal_gamma", src.get("gamma", [np.pi/4]))
        reps   = len(betas)

        print(f"[Noisy] Fixed params: β={[round(b,4) for b in betas]}, "
              f"γ={[round(g,4) for g in gammas]}")

        result_dict, top10, runtime_s = run_fixed_params_noisy(
            ising_op, betas, gammas, sim, args.shots, Q, meta, site_ids
        )
        conv_curve = []

    else:
        # Optimize under noise (slow)
        circuit, gamma_params, beta_params = build_qaoa_circuit(ising_op, args.reps)
        param_order = list(beta_params) + list(gamma_params)
        n_params    = len(param_order)

        conv_curve = []
        call_count = [0]
        t_start    = time.time()

        def objective(params):
            param_dict = dict(zip(param_order, params))
            bound      = circuit.assign_parameters(param_dict)
            meas       = QuantumCircuit(bound.num_qubits, bound.num_qubits)
            for inst in bound.data:
                if inst.operation.name != 'save_statevector':
                    meas.append(inst)
            meas.measure_all(add_bits=False)
            counts  = sim.run(meas, shots=args.shots).result().get_counts()
            energies = []
            for bs, cnt in counts.items():
                x = np.array([int(b) for b in bs[::-1][:Q.shape[0]]], dtype=float)
                energies.extend([float(x @ Q @ x)] * cnt)
            energy = np.mean(energies)
            call_count[0] += 1
            conv_curve.append(energy)
            nc = call_count[0]
            if nc % 5 == 0:
                print(f"nfev={nc} | E={energy:.4f} | t={time.time()-t_start:.1f}s", flush=True)
            return energy

        x0 = np.array([np.pi/8] * args.reps + [np.pi/4] * args.reps)
        opt_result = scipy_minimize(objective, x0=x0, method="COBYLA",
                                    options={"maxiter": args.maxiter, "rhobeg": 0.5})
        betas  = [float(opt_result.x[i]) for i in range(args.reps)]
        gammas = [float(opt_result.x[args.reps + i]) for i in range(args.reps)]
        runtime_s = time.time() - t_start

        result_dict, top10, _ = run_fixed_params_noisy(
            ising_op, betas, gammas, sim, args.shots, Q, meta, site_ids
        )

    e_best = bf["result"]["energy"]
    ar     = result_dict["energy"] / e_best if e_best != 0 else 0.0

    print()
    print("=" * 65)
    print(f"  Noisy Aer Results (p={args.reps}, shots={args.shots})")
    print("=" * 65)
    print(f"  Energy (best shot) : {result_dict['energy']:.6f}")
    print(f"  Ground truth       : {e_best:.6f}")
    print(f"  Approximation ratio: {ar:.4f}")
    print(f"  Selected sites     : {result_dict['selected_sites']}")
    print(f"  Runtime            : {runtime_s:.1f}s")
    print()
    print("  Statevector (no noise): AR=1.000")
    print(f"  Noisy Aer            : AR={ar:.4f}")
    print(f"  Real hardware (K1)   : AR=0.9939")

    out = {
        "run_id":              f"noisy_aer_p{args.reps}_{args.cost_model}_{timestamp}",
        "variant":             "baseline",
        "cost_model":          args.cost_model,
        "n_sites":             meta["n_sites"],
        "simulator":           "noisy_aer",
        "p_depth":             args.reps,
        "energy":              round(result_dict["energy"], 8),
        "approximation_ratio": round(ar, 6),
        "beta":                [round(b, 8) for b in betas],
        "gamma":               [round(g, 8) for g in gammas],
        "top10_bitstrings":    top10,
        "convergence_curve":   [round(e, 8) for e in conv_curve],
        "seed":                SEED,
        "runtime_seconds":     round(runtime_s, 4),
        "run_timestamp":       timestamp,
        "shots":               args.shots,
        "noise_backend":       "FakeBrisbane (Eagle r1 family, ibm_kingston equivalent)",
        "fixed_params_source": str(args.fixed_params) if args.fixed_params else None,
        "result":              result_dict,
        "ground_truth_energy": e_best,
        "hardware_comparison": {
            "statevector_ar": 1.000,
            "noisy_aer_ar":   round(ar, 6),
            "hardware_ar":    0.9939,
            "hardware_job_id": "d79jmo1q1efs73d2tta0",
        },
    }

    out_dir  = qubo_results_dir(args.base_dir, args.cost_model, args.n_sites)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"top10_noisy_aer_n{args.n_sites}_{args.cost_model}_p{args.reps}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[Export] {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
