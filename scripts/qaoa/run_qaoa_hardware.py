#!/usr/bin/env python3
"""
VQ-MAR — QAOA on IBM Quantum Hardware
=====================================

Two execution modes:

1. --fixed_params PATH  (recommended — works on FREE IBM Quantum plan)
   Loads optimal β/γ from a completed simulator result JSON, builds the
   QAOA circuit with those fixed parameters, submits ONE hardware job
   (no Session required). Works on IBM Quantum open/free accounts.

   Example:
       python scripts/qaoa/run_qaoa_hardware.py --cost_model flat \
           --fixed_params georgia/qiskit-ready/qubo_matrices/flat/qaoa_native_diagonal_20_p1.json \
           --backend ibm_brisbane --shots 1024

2. Session mode (requires IBM Quantum Standard/Premium plan)
   Runs full COBYLA optimization loop inside a Session, batching all
   circuit evaluations into one queue reservation.

   Example:
       python scripts/qaoa/run_qaoa_hardware.py --cost_model flat \
           --backend ibm_brisbane --shots 1024 --maxiter 100

Requires IBM Quantum credentials saved via:
    QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='YOUR_TOKEN')
Or set IBM_QUANTUM_TOKEN in .env file at repo root.

Outputs:
    georgia/qiskit-ready/qubo_matrices/{cost_model}/qaoa_hardware_20_p{reps}.json

Notes:
    - 20-qubit QAOA requires a QPU with ≥ 20 qubits. Eagle (127q) or
      Heron r2 (ibm_torino, 133q) are preferred for circuit fidelity.
    - Hardware results have shot noise; P(success) will be lower than
      the statevector simulation. Report results as exploratory rather
      than as a central claim.
"""

import os
import sys
import json
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from qiskit_optimization.translators import to_ising

# Import shared QUBO helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qiskit_qubo import load_qubo, build_quadratic_program, verify_energy, qubo_results_dir
from run_qaoa_native_diagonal import build_qaoa_circuit


# ─── 0. AUTHENTICATION SETUP (Self-contained) ────────────────────────────────

def setup_ibm_credentials():
    """
    Load IBM Quantum credentials from .env file and save to QiskitRuntimeService.

    Supports two IBM Quantum platforms:
      - ibm_cloud (open/free plan):  token = IBM Cloud API key, instance = CRN
      - ibm_quantum_platform (legacy): token = IBM Quantum token, no instance needed

    Channel is auto-detected: if IBM_QUANTUM_INSTANCE is a CRN (starts with 'crn:'),
    uses 'ibm_cloud'; otherwise uses 'ibm_quantum_platform'.

    .env keys read (repo root .env — never committed):
        IBM_QUANTUM_TOKEN    — IBM Cloud API key (required)
        IBM_QUANTUM_INSTANCE — CRN for open/free plan, or hub/group/project
                               for legacy ibm_quantum_platform (required)

    Channel is auto-detected:
        instance starts with 'crn:'  →  ibm_cloud       (open/free plan)
        otherwise                    →  ibm_quantum_platform (legacy)
    """
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    token    = None
    instance = None

    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('IBM_QUANTUM_TOKEN='):
                        token = line.split('=', 1)[1].strip('"').strip("'")
                    elif line.startswith('IBM_QUANTUM_INSTANCE='):
                        instance = line.split('=', 1)[1].strip('"').strip("'")
        except Exception as e:
            print(f"[Info] Could not read .env file: {e}")
    else:
        print("[Warning] No .env file found at repo root.")

    if not token:
        print("[Warning] IBM_QUANTUM_TOKEN not set in .env. Using existing saved credentials.")
    if not instance:
        print("[Warning] IBM_QUANTUM_INSTANCE not set in .env. Connection may fail.")
        print("          Add IBM_QUANTUM_INSTANCE=<crn or hub/group/project> to .env")

    if not token:
        return  # rely on previously saved credentials

    # Detect channel from instance format
    channel = 'ibm_cloud' if (instance or '').startswith('crn:') else 'ibm_quantum_platform'

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        save_kwargs = dict(channel=channel, token=token, set_as_default=True, overwrite=True)
        if channel == 'ibm_cloud' and instance:
            save_kwargs['instance'] = instance
        QiskitRuntimeService.save_account(**save_kwargs)
        print(f"[Auth] Credentials configured  channel={channel}")
        if instance:
            print(f"       Instance : {instance[:70]}")
    except Exception as e:
        print(f"[Warning] Failed to save credentials: {e}")


# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_BACKEND  = "ibm_kingston"   # IBM Eagle r1 (27q); open/free plan backend
DEFAULT_SHOTS    = 1024             # shots per circuit evaluation
DEFAULT_REPS     = 1                # QAOA depth p
DEFAULT_MAXITER  = 100              # fewer iterations than sim — hardware is slow

TOP_N_SAMPLES    = 20


# ─── 1. AUTHENTICATION ───────────────────────────────────────────────────────

def get_backend(backend_name, instance=None):
    """
    Load IBM Quantum credentials and return the target backend.
    Falls back to DEFAULT_INSTANCE (open-plan CRN) if no instance provided.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("[Auth] ERROR: qiskit-ibm-runtime not installed.")
        print("       Run: pip install qiskit-ibm-runtime")
        raise SystemExit(1)

    # Read instance from .env if not passed via CLI
    if not instance:
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.strip().startswith('IBM_QUANTUM_INSTANCE='):
                        instance = line.strip().split('=', 1)[1].strip('"').strip("'")
                        break

    try:
        kwargs = {"instance": instance} if instance else {}
        service = QiskitRuntimeService(**kwargs)
        backend = service.backend(backend_name)
        print(f"[Auth] Connected to IBM Quantum backend: {backend.name}")
        print(f"       Qubits    : {backend.num_qubits}")
        print(f"       Status    : {backend.status().status_msg}")
        return service, backend
    except Exception as exc:
        print(f"\n[Auth] ERROR: Could not connect to IBM Quantum ({exc})")
        print()
        print("  To set up credentials:")
        print("  1. Get a free IBM Quantum account at https://quantum.ibm.com")
        print("  2. Copy your API token from the account page")
        print("  3. Run once in Python:")
        print("       from qiskit_ibm_runtime import QiskitRuntimeService")
        print("       QiskitRuntimeService.save_account(")
        print("           channel='ibm_quantum_platform', token='YOUR_TOKEN_HERE')")
        print()
        print("  Then re-run this script.")
        raise SystemExit(1)


# ─── 2a. FIXED-PARAMS MODE (single job, no Session, free plan) ───────────────

def run_fixed_params_job(ising_op, betas, gammas, backend, shots, timeout=600):
    """
    Build a QAOA circuit with fixed β/γ from a simulator result and submit
    ONE hardware job via SamplerV2 without a Session.

    This is the recommended path for IBM Quantum free/open plan accounts
    which do not support Sessions.

    Args:
        timeout  — seconds to wait for result; -1 = wait forever.
                   On timeout, returns (None, runtime_s, backend_name, job_id).

    Returns:
        counts       — dict {bitstring: count}, or None on timeout
        runtime_s    — wall-clock seconds (includes queue wait)
        backend_name — name of the QPU used
        job_id       — IBM Quantum job ID for provenance
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    reps = len(betas)
    circuit, gamma_params, beta_params = build_qaoa_circuit(ising_op, reps)

    # Bind parameters: param_order = [beta[0],...,beta[reps-1], gamma[0],...,gamma[reps-1]]
    param_order = list(beta_params) + list(gamma_params)
    param_values = np.array(list(betas) + list(gammas))
    bound = circuit.assign_parameters(dict(zip(param_order, param_values)))

    # Remove save_statevector (AerSim-only instruction) and add measurement
    from qiskit import QuantumCircuit
    meas_circuit = QuantumCircuit(bound.num_qubits, bound.num_qubits)
    # Copy gates except save_statevector
    for instruction in bound.data:
        if instruction.operation.name != 'save_statevector':
            meas_circuit.append(instruction)
    meas_circuit.measure_all(add_bits=False)

    print(f"[Hardware] Transpiling circuit for {backend.name}...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled = pm.run(meas_circuit)
    print(f"[Hardware] Transpiled circuit: depth={transpiled.depth()}, "
          f"gates={transpiled.size()}")

    print(f"[Hardware] Submitting job (shots={shots})...")
    print("           (includes queue wait time — may take minutes to hours)")
    t0 = time.time()

    sampler = SamplerV2(mode=backend)
    job = sampler.run([(transpiled,)], shots=shots)

    print(f"[Hardware] Job submitted. Job ID: {job.job_id()}")
    print("[Hardware] Waiting for result...")
    if timeout > 0:
        print(f"           (timeout={timeout}s; pass --timeout -1 to wait indefinitely)")

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(job.result)
        try:
            wait = timeout if timeout > 0 else None
            result = future.result(timeout=wait)
        except FuturesTimeout:
            runtime_s = time.time() - t0
            print(f"\n[Hardware] TIMEOUT after {runtime_s:.0f}s — job still in queue.")
            print(f"           Job ID : {job.job_id()}")
            print(f"           Retrieve later with:")
            print(f"               from qiskit_ibm_runtime import QiskitRuntimeService")
            print(f"               QiskitRuntimeService().job('{job.job_id()}').result()")
            return None, runtime_s, backend.name, job.job_id()

    runtime_s = time.time() - t0
    print(f"[Hardware] Result received in {runtime_s:.1f}s")

    # The classical register name depends on how the circuit was built ('meas', 'c', etc.)
    # Dynamically find the first classical register in the result DataBin.
    data_bin = result[0].data
    creg_name = next(iter(vars(data_bin)))
    counts = getattr(data_bin, creg_name).get_counts()
    return counts, runtime_s, backend.name, job.job_id()


def decode_counts_result(counts, Q, meta, site_ids, shots):
    """
    Convert hardware counts dict to the standard VQ-MAR result dict.

    Finds the bitstring with the lowest QUBO energy among all observed samples,
    computes probability distribution, and assembles output metrics.

    Returns:
        result_dict  — standard site-selection result
        top_samples  — list of top-N samples by energy
        probs_dict   — {bitstring: probability} for all observed bitstrings
    """
    n = Q.shape[0]
    w = np.array(meta["w"])
    C = np.array(meta["C"])
    budget = meta["budget"]

    best_energy  = float("inf")
    best_bits    = None
    sample_list  = []

    for bitstring, count in counts.items():
        # Qiskit returns bitstrings in MSB-first order (qubit n-1 is leftmost)
        # Reverse to get qubit 0 = index 0 (same convention as native diagonal)
        bits_rev = bitstring[::-1]
        x = np.array([int(b) for b in bits_rev[:n]], dtype=float)
        energy = float(x @ Q @ x)
        prob = count / shots
        sample_list.append((energy, bits_rev, prob))
        if energy < best_energy:
            best_energy = energy
            best_bits   = bits_rev

    sample_list.sort(key=lambda t: t[0])

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

    top_samples = [
        {
            "bitstring":   e[1][:n],
            "energy":      round(e[0], 8),
            "probability": round(e[2], 8),
        }
        for e in sample_list[:TOP_N_SAMPLES]
    ]

    probs_dict = {s[1][:n]: round(s[2], 8) for s in sample_list}

    return result_dict, top_samples, probs_dict


# ─── 2b. SESSION MODE (COBYLA loop on hardware, Standard/Premium plan) ────────

def run_qaoa_hardware_session(qp, backend, shots, reps, maxiter):
    """
    Run full COBYLA optimization loop inside a Session.

    Session batches all COBYLA circuit evaluations into a single
    queue reservation, reducing turnaround time and avoiding
    re-queuing overhead between iterations.

    Requires IBM Quantum Standard or Premium plan.

    Returns:
        result      — OptimizationResult
        betas       — list of optimal β angles
        gammas      — list of optimal γ angles
        conv_curve  — list of energy values per iteration
        runtime_s   — wall-clock seconds (includes queue wait)
        backend_name — name of the QPU used
    """
    from qiskit_ibm_runtime import Session, SamplerV2
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    convergence_curve = []

    def cobyla_callback(*args):
        convergence_curve.append(None)  # placeholder — energy not directly accessible

    optimizer = COBYLA(maxiter=maxiter, callback=cobyla_callback)

    print(f"[QAOA] Opening Session on {backend.name}...")
    print(f"       shots={shots}, reps={reps}, maxiter={maxiter}")
    print("       (includes queue wait time)")

    from qiskit_ibm_runtime import Sampler
    
    t0 = time.time()
    with Session(backend=backend) as session:
        sampler = Sampler(session=session)
        qaoa_alg = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=reps,
        )
        solver = MinimumEigenOptimizer(qaoa_alg)
        result = solver.solve(qp)

    runtime_s = time.time() - t0
    print(f"[QAOA] Session closed. Total time: {runtime_s:.1f}s")
    print(f"       nfev={len(convergence_curve)}, energy={result.fval:.6f}")

    try:
        opt_params = qaoa_alg.result.optimal_parameters
        gammas = [float(v) for k, v in opt_params.items()
                  if "beta" not in str(k).lower()]
        betas  = [float(v) for k, v in opt_params.items()
                  if "beta" in str(k).lower()]
    except Exception:
        gammas, betas = [], []

    return result, betas, gammas, convergence_curve, runtime_s, backend.name


# ─── 3. MAIN ─────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="VQ-MAR — QAOA on IBM Quantum hardware"
    )
    parser.add_argument("--cost_model", choices=["flat", "real"], default="flat",
                        help="Cost model: flat (Ci=0.5) or real (Eq. 15)")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="QAOA circuit depth p")
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS,
                        help="Shots per circuit evaluation")
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER,
                        help="COBYLA max function evaluations (Session mode only)")
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help="IBM Quantum backend name")
    parser.add_argument("--instance", default=None,
                        help="IBM Quantum instance CRN or hub/group/project. "
                             "Defaults to open-plan CRN (DEFAULT_INSTANCE).")
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
    parser.add_argument("--base_dir",
                        default=os.environ.get("VQMAR_BASE", _REPO_ROOT),
                        help="Project root directory (overrides $VQMAR_BASE)")
    parser.add_argument("--n_sites", type=int, default=20,
                        help="Number of sites (20 or 50). Selects Q_{n}.npy and meta_{n}.json.")
    parser.add_argument("--fixed_params", default=None, metavar="PATH",
                        help="Path to a completed simulator result JSON. "
                             "Loads optimal beta/gamma, builds circuit with those "
                             "fixed parameters, submits ONE hardware job (no Session). "
                             "Recommended for IBM Quantum free/open plan accounts.")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Seconds to wait for hardware job result. On timeout, "
                             "saves partial JSON with job_id for later retrieval. "
                             "-1 = wait indefinitely. (fixed_params mode only)")
    args = parser.parse_args()

    mode = "fixed_params" if args.fixed_params else "session"

    print("=" * 65)
    print("  VQ-MAR — QAOA on IBM Quantum Hardware")
    print(f"  Cost model : {args.cost_model}")
    print(f"  Backend    : {args.backend}")
    print(f"  QAOA depth : p={args.reps}")
    print(f"  Shots      : {args.shots}")
    print(f"  Mode       : {mode}")
    if mode == "session":
        print(f"  Optimizer  : COBYLA (maxiter={args.maxiter})")
    else:
        print(f"  Fixed params: {args.fixed_params}")
        timeout_str = f"{args.timeout}s" if args.timeout > 0 else "unlimited"
        print(f"  Timeout    : {timeout_str}")
    print("=" * 65)
    print()

    # ── Setup IBM credentials (self-contained) ────────────────────────────
    setup_ibm_credentials()

    # ── Load QUBO data ────────────────────────────────────────────────────
    data = load_qubo(args.base_dir, args.cost_model, n_sites=args.n_sites)
    Q        = data["Q"]
    meta     = data["meta"]
    site_ids = data["site_ids"]
    bf       = data["brute_force"]

    # ── Build QuadraticProgram + verify energy ────────────────────────────
    print("\nBuilding QuadraticProgram...")
    qp = build_quadratic_program(Q, site_ids)
    verify_energy(Q, meta["qubo_const"], bf)

    # ── Connect to IBM Quantum ────────────────────────────────────────────
    print()
    service, backend = get_backend(args.backend, args.instance)
    e_best = bf["result"]["energy"]

    # ─────────────────────────────────────────────────────────────────────
    # MODE A: Fixed-params (single job, no Session, free plan)
    # ─────────────────────────────────────────────────────────────────────
    if mode == "fixed_params":
        with open(args.fixed_params) as f:
            sim_result = json.load(f)

        em     = sim_result.get("extra_metrics", {})
        betas  = em.get("optimal_beta")  or sim_result.get("beta",  [np.pi/8])
        gammas = em.get("optimal_gamma") or sim_result.get("gamma", [np.pi/4])
        reps   = len(betas)

        # Allow --reps to override if specified; check consistency
        if args.reps != DEFAULT_REPS and args.reps != reps:
            print(f"[Warning] --reps={args.reps} but fixed_params has reps={reps}. "
                  f"Using reps={reps} from fixed_params.")
        reps = len(betas)

        print(f"\n[Fixed-params] Loaded β={[round(b,4) for b in betas]}, "
              f"γ={[round(g,4) for g in gammas]}  (from simulator, reps={reps})")

        ising_op, _ = to_ising(qp)

        counts, runtime_s, backend_name, job_id = run_fixed_params_job(
            ising_op, betas, gammas, backend, args.shots, timeout=args.timeout
        )

        out_dir = qubo_results_dir(args.base_dir, args.cost_model, args.n_sites)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Timeout path: export partial JSON with job_id for later retrieval ──
        if counts is None:
            partial_out = {
                "status":        "TIMEOUT",
                "job_id":        job_id,
                "run_timestamp": timestamp,
                "note": (
                    f"Job submitted but result not received within {args.timeout}s. "
                    f"Retrieve with: QiskitRuntimeService().job('{job_id}').result()"
                ),
                "parameters": {
                    "reps":    reps,
                    "shots":   args.shots,
                    "backend": backend_name,
                    "mode":    "fixed_params_no_session",
                    "fixed_params_source": str(args.fixed_params),
                },
                "beta_used":       [round(b, 8) for b in betas],
                "gamma_used":      [round(g, 8) for g in gammas],
                "runtime_seconds": round(runtime_s, 4),
            }
            out_path = out_dir / f"qaoa_hardware_n{args.n_sites}_{args.cost_model}_p{reps}_{timestamp}_TIMEOUT.json"
            with open(out_path, "w") as f:
                json.dump(partial_out, f, indent=2)
            print(f"\n[Export] Partial result saved: {out_path}")
            print("         Re-run with the job_id above to fetch the result.")
            return

        result_dict, top_samples, probs_dict = decode_counts_result(
            counts, Q, meta, site_ids, args.shots
        )

        ar = result_dict["energy"] / e_best if e_best != 0 else 0.0

        print()
        print("=" * 65)
        print(f"  Hardware Results (backend={backend_name}, p={reps})")
        print("=" * 65)
        print(f"  Energy             : {result_dict['energy']:.6f}")
        print(f"  Ground truth       : {e_best:.6f}")
        print(f"  Approximation ratio: {ar:.4f}")
        print(f"  Selected sites     : {result_dict['selected_sites']}")
        print(f"  Runtime (incl wait): {runtime_s:.1f}s")
        print(f"  Job ID             : {job_id}")
        print()
        print("  NOTE: Hardware AR is typically lower than statevector sim")
        print("        due to shot noise and gate errors;")
        print("        report as exploratory rather than as a central result.")

        out = {
            "task":          "qaoa_hardware",
            "method":        "qaoa_hardware_fixed_params",
            "n_sites":       meta["n_sites"],
            "cost_model":    args.cost_model,
            "run_timestamp": timestamp,
            "parameters": {
                "reps":    reps,
                "shots":   args.shots,
                "backend": backend_name,
                "mode":    "fixed_params_no_session",
                "fixed_params_source": str(args.fixed_params),
            },
            "runtime_seconds": round(runtime_s, 4),
            "result":          result_dict,
            "metrics": {
                "approximation_ratio": round(ar, 6),
                "n_qubits":            meta["n_sites"],
                "search_space":        2 ** meta["n_sites"],
                "portfolio_cost":      round(result_dict["total_cost"], 6),
                "budget_B":            round(meta["budget"], 6),
                "budget_feasible":     result_dict["total_cost"] <= meta["budget"] + 1e-9,
            },
            "extra_metrics": {
                "optimal_beta":  [round(b, 8) for b in betas],
                "optimal_gamma": [round(g, 8) for g in gammas],
                "job_id":        job_id,
            },
            "top_samples":         top_samples,
            "ground_truth_energy": e_best,
            "hardware_note": (
                "Fixed-params mode: β/γ loaded from simulator result, "
                "one hardware job submitted without Session (free plan compatible). "
                "Results include shot noise and gate errors. "
                "Report results as exploratory, not a central claim."
            ),
        }

    # ─────────────────────────────────────────────────────────────────────
    # MODE B: Session mode (full COBYLA on hardware, Standard/Premium plan)
    # ─────────────────────────────────────────────────────────────────────
    else:
        result, betas, gammas, conv_curve, runtime_s, backend_name = \
            run_qaoa_hardware_session(qp, backend, args.shots, args.reps, args.maxiter)

        x_vals = result.x
        bits   = "".join(str(int(v)) for v in x_vals)
        sel_idx        = [i for i, v in enumerate(x_vals) if v > 0.5]
        selected_sites = [site_ids[i] for i in sel_idx]
        w = np.array(meta["w"])
        C = np.array(meta["C"])

        result_dict = {
            "energy":         float(result.fval),
            "bitstring":      bits,
            "n_selected":     len(selected_sites),
            "selected_sites": selected_sites,
            "total_benefit":  float(np.sum(w[sel_idx])),
            "total_cost":     float(np.sum(C[sel_idx])),
            "excluded_used":  0,
            "budget_B":       meta["budget"],
        }

        ar = result_dict["energy"] / e_best if e_best != 0 else 0.0

        top_samples = [
            {
                "bitstring":   "".join(str(int(v)) for v in s.x),
                "energy":      round(float(s.fval), 8),
                "probability": round(float(s.probability), 8),
            }
            for s in sorted(result.samples, key=lambda s: s.fval)[:TOP_N_SAMPLES]
        ]

        print()
        print("=" * 65)
        print(f"  Hardware Results (backend={backend_name}, p={args.reps})")
        print("=" * 65)
        print(f"  Energy             : {result_dict['energy']:.6f}")
        print(f"  Ground truth       : {e_best:.6f}")
        print(f"  Approximation ratio: {ar:.4f}")
        print(f"  Selected sites     : {result_dict['selected_sites']}")
        print(f"  Runtime (incl wait): {runtime_s:.1f}s")
        print(f"  COBYLA nfev        : {len(conv_curve)}")
        print()
        print("  NOTE: Hardware AR is typically lower than statevector sim")
        print("        due to shot noise and gate errors;")
        print("        report as exploratory rather than as a central result.")

        out = {
            "task":          "qaoa_hardware",
            "method":        "qaoa_hardware_session",
            "n_sites":       meta["n_sites"],
            "cost_model":    args.cost_model,
            "run_timestamp": timestamp,
            "parameters": {
                "reps":      args.reps,
                "optimizer": "COBYLA",
                "maxiter":   args.maxiter,
                "shots":     args.shots,
                "backend":   backend_name,
                "mode":      "session",
            },
            "runtime_seconds": round(runtime_s, 4),
            "result":          result_dict,
            "metrics": {
                "approximation_ratio": round(ar, 6),
                "n_qubits":            meta["n_sites"],
                "search_space":        2 ** meta["n_sites"],
                "portfolio_cost":      round(result_dict["total_cost"], 6),
                "budget_B":            round(meta["budget"], 6),
                "budget_feasible":     result_dict["total_cost"] <= meta["budget"] + 1e-9,
            },
            "extra_metrics": {
                "optimal_beta":      [round(b, 8) for b in betas],
                "optimal_gamma":     [round(g, 8) for g in gammas],
                "optimizer_nfev":    len(conv_curve),
                "convergence_curve": [round(e, 8) for e in conv_curve
                                      if e is not None],
            },
            "top_samples":         top_samples,
            "ground_truth_energy": e_best,
            "hardware_note": (
                "Session mode: full COBYLA optimization on hardware. "
                "Results include shot noise and gate errors. "
                "Report results as exploratory, not a central claim."
            ),
        }

    # ── Export ────────────────────────────────────────────────────────────
    out_dir  = qubo_results_dir(args.base_dir, args.cost_model, args.n_sites)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qaoa_hardware_n{args.n_sites}_{args.cost_model}_p{out['parameters']['reps']}_{timestamp}.json"

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print(f"[Export] {out_path}")
    print()
    print("=" * 65)
    print("  Run COMPLETE")
    print(f"  AR = {ar:.4f}  |  Runtime = {runtime_s:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
