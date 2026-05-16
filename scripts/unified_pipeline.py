#!/usr/bin/env python3
"""
VQ-MAR — Unified Pipeline
=========================
Merges two implementations of the site-selection pipeline:

  Core pipeline:
    + Brute-force ground truth
    + Domain-informed normalization (ksat/50, depth/30)
    + Bounds verification, sanity checks
    + Sequential JSON outputs for traceability

  Enhancements:
    + Richer benefit formula with subtractive risk terms
    + Real cost model (Eq. 15)
    + Genetic algorithm solver

Reads the processed per-domain score files and produces a complete
set of outputs (site scores, QUBO matrices, solver results, sanity
report) for downstream consumption.

Usage:
    python unified_pipeline.py [--cost_model flat|real] [--base_dir PATH]
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_BASE = os.environ.get('VQMAR_BASE', _REPO_ROOT)

# QUBO parameters. Normalized [0,1] scores are used so brute-force
# ground truth remains interpretable; the --cost_model toggle provides
# a dollar-scale cost when needed.
LAMBDA = 2.0    # budget penalty (Eq. 11)
MU     = 5.0    # exclusion penalty (Eq. 11)
ETA    = 0.3    # spatial interaction (Eq. 8)
ELL_KM = 10.0   # decay length km (Eq. 8)
SEED   = 42

# Budget ceiling
# Flat cost: B=3.0 allows ~6 sites at Ci=0.5
# Real cost: B recalibrated to mean(Ci)*6
BUDGET_FLAT = 3.0

# SA parameters
SA_T0       = 10.0
SA_ALPHA    = 0.995
SA_STEPS    = 10_000
SA_RESTARTS = 10

# GA parameters
GA_POP      = 100
GA_GENS     = 200
GA_MUTATION = 0.02
GA_CROSSOVER = 0.8


# ─── 1. DATA LOADING ─────────────────────────────────────────────────────────

def load_processed_scores(base_dir, grid=20):
    """
    Load processed score files for the specified grid size.
    Returns dict of DataFrames keyed by domain.

    grid=20 reads the committed 20-site files (*_scores.csv, no suffix).
    grid=50 reads the 50-site files (*_scores_50.csv).
    """
    proc = Path(base_dir) / "georgia" / "processed"
    grid_file = Path(base_dir) / "georgia" / "raw" / "ssurgo" / f"candidate_grid_{grid}.csv"
    suffix = "" if grid == 20 else f"_{grid}"

    print(f"[Load] Reading processed score files for grid={grid}...")

    grid_df = pd.read_csv(grid_file, dtype={"site_id": str})
    print(f"  Grid: {len(grid_df)} sites")

    scores = {"grid": grid_df}

    files = {
        "ssurgo": proc / "ssurgo" / f"ssurgo_scores{suffix}.csv",
        "nwis":   proc / "nwis"   / f"nwis_scores{suffix}.csv",
        "nlcd":   proc / "nlcd"   / f"nlcd_scores{suffix}.csv",
        "noaa":   proc / "noaa"   / f"noaa_scores{suffix}.csv",
        "osm":    proc / "osm"    / f"osm_scores{suffix}.csv",
    }

    for domain, path in files.items():
        if path.exists():
            df = pd.read_csv(path, dtype={"site_id": str})
            scores[domain] = df
            print(f"  {domain}: {len(df)} rows, cols={list(df.columns)}")
        else:
            print(f"  [WARN] {domain} not found at {path}")
            scores[domain] = None

    return scores


# ─── 2. UNIFIED SCORING ENGINE ───────────────────────────────────────────────
#
# Scoring design notes:
#
# 1. Normalization: domain-informed reference values rather than
#    min-max. This prevents outlier sensitivity and produces stable
#    scores across different grid sizes.
#
# 2. Benefit formula: 5-domain additive structure as the base, with
#    risk adjustments applied as a modifier. Captures both core domain
#    scoring and richer risk-aware modeling.
#
#    Enhanced formula:
#      w_i = base_benefit * (1 - risk_penalty)
#    where:
#      base_benefit = 0.30*Si + 0.25*Ni + 0.15*Li + 0.15*Cclim + 0.15*Ai
#      risk_penalty = 0.10 * impervious_norm + 0.05 * maintenance_norm
#    Both terms clipped to [0, 0.3] so risk can reduce benefit by up to 30%
#    but never flip the sign.
#
# 3. Exclusion criteria (conservative union):
#    NLCD: 11 (water), 23 (dev med), 24 (dev high), 95 (wetlands)
#    NLCD: 21 (dev open), 22 (dev low)
#    Water table < 300 cm (3 m) from SSURGO
#    Impervious > 50%
#
# 4. Cost model: Configurable. Default is flat Ci=0.5.
#    --cost_model real activates the dollar-scale formula (Eq. 15),
#    normalized to [0,1] to stay compatible with the QUBO parameter
#    scale.
#

def compute_unified_scores(scores, cost_model="flat"):
    """
    Build the unified site table from the per-domain processed scores.

    Returns DataFrame with columns:
        site_id, latitude, longitude,
        Si, Ni, Li, Cclim_i, Ai,
        wi_base, risk_penalty, wi,
        Ci, ei_ssurgo, ei_nlcd, ei_imperv, ei
    """
    grid = scores["grid"].copy()
    n = len(grid)

    # ── Merge domain scores ──────────────────────────────────────────────
    # Start with grid as the canonical site list
    df = grid[["site_id", "latitude", "longitude"]].copy()

    # SSURGO
    if scores["ssurgo"] is not None:
        ssurgo = scores["ssurgo"][["site_id", "Si", "ei_ssurgo"]].copy()
        if "wtdepannmin" in scores["ssurgo"].columns:
            ssurgo["wtdepannmin"] = scores["ssurgo"]["wtdepannmin"]
        df = df.merge(ssurgo, on="site_id", how="left")
    else:
        df["Si"] = 0.5
        df["ei_ssurgo"] = 0

    # NWIS
    if scores["nwis"] is not None:
        nwis_cols = ["site_id", "Ni"]
        if "well_mean_depth_ft" in scores["nwis"].columns:
            nwis_cols.append("well_mean_depth_ft")
        df = df.merge(scores["nwis"][nwis_cols], on="site_id", how="left")
    else:
        df["Ni"] = 0.5

    # NLCD
    if scores["nlcd"] is not None:
        nlcd_cols = ["site_id", "Li", "ei_nlcd"]
        nlcd = scores["nlcd"].copy()
        if "nlcd_class" in nlcd.columns:
            nlcd_cols.append("nlcd_class")
        if "nlcd_class_name" in nlcd.columns:
            nlcd_cols.append("nlcd_class_name")
        df = df.merge(nlcd[[c for c in nlcd_cols if c in nlcd.columns]],
                      on="site_id", how="left")
    else:
        df["Li"] = 0.5
        df["ei_nlcd"] = 0

    # NOAA
    if scores["noaa"] is not None:
        df = df.merge(scores["noaa"][["site_id", "Cclim_i"]],
                      on="site_id", how="left")
    else:
        df["Cclim_i"] = 0.65

    # OSM
    if scores["osm"] is not None:
        osm_cols = ["site_id", "Ai"]
        if "dist_road_m" in scores["osm"].columns:
            osm_cols.append("dist_road_m")
        df = df.merge(scores["osm"][[c for c in osm_cols
                                      if c in scores["osm"].columns]],
                      on="site_id", how="left")
    else:
        df["Ai"] = 0.5

    # Fill any NaN from failed joins
    for col in ["Si", "Ni", "Li", "Cclim_i", "Ai"]:
        df[col] = df[col].fillna(0.5)
    for col in ["ei_ssurgo", "ei_nlcd"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
        else:
            df[col] = 0

    # ── Base benefit (Eq. 2) ─────────────────────────────────────────────
    df["wi_base"] = (0.30 * df["Si"] +
                     0.25 * df["Ni"] +
                     0.15 * df["Li"] +
                     0.15 * df["Cclim_i"] +
                     0.15 * df["Ai"])

    # ── Risk penalty ─────────────────────────────────────────────────────
    # Impervious risk: sites with high impervious surface nearby
    # are harder to recharge through. Use NLCD score as proxy:
    # low Li implies high impervious/developed land.
    impervious_risk = np.clip(1.0 - df["Li"].values, 0, 1)

    # Maintenance risk: far from roads = harder to maintain.
    # Use inverse of Ai (which is already 1=close, 0=far).
    maintenance_risk = np.clip(1.0 - df["Ai"].values, 0, 1)

    # Weighted risk penalty, capped at 30% reduction
    df["risk_penalty"] = np.clip(
        0.10 * impervious_risk + 0.05 * maintenance_risk, 0, 0.30
    )

    # ── Enhanced benefit ─────────────────────────────────────────────────
    df["wi"] = df["wi_base"] * (1.0 - df["risk_penalty"])

    # Ensure [0, 1] bounds
    df["wi"] = df["wi"].clip(0, 1)

    # ── Exclusion flags (union of both approaches) ───────────────────────
    # Additional NLCD exclusions: classes 21, 22 (developed open/low)
    df["ei_nlcd_extended"] = 0
    if "nlcd_class" in df.columns:
        additional_exclusions = {21, 22}  # developed open/low
        df["ei_nlcd_extended"] = df["nlcd_class"].isin(
            additional_exclusions
        ).astype(int)

    # Impervious-surface exclusion (> 50%): not available in current
    # processed data, so default to 0.
    df["ei_imperv"] = 0

    # Combined exclusion: union of all flags
    df["ei"] = np.maximum.reduce([
        df["ei_ssurgo"].values,
        df["ei_nlcd"].values,
        df["ei_nlcd_extended"].values,
        df["ei_imperv"].values,
    ])

    # ── Cost model ───────────────────────────────────────────────────────
    if cost_model == "real":
        # Eq. (15), normalized to [0,1]
        # Ci = c0 + c1*D_wt_norm + c2*(1-Li) + c3*Ai_inv
        # Then normalize: Ci_norm = (Ci - Ci_min) / (Ci_max - Ci_min)
        # Scale so mean(Ci) ~ 0.5 (preserving QUBO parameter balance)
        c0, c1, c2, c3 = 0.20, 0.30, 0.30, 0.20

        # Water table depth proxy (deeper = more expensive to build)
        if "wtdepannmin" in df.columns:
            wt = df["wtdepannmin"].fillna(df["wtdepannmin"].median())
            wt_norm = np.clip(wt / 600.0, 0, 1)  # 600cm reference
        else:
            wt_norm = 0.5

        land_penalty = 1.0 - df["Li"].values  # less suitable = costlier
        access_penalty = 1.0 - df["Ai"].values  # farther = costlier

        Ci_raw = c0 + c1 * wt_norm + c2 * land_penalty + c3 * access_penalty
        df["Ci"] = np.clip(Ci_raw, 0.1, 1.0)
        print(f"[Scoring] Real cost model: Ci range "
              f"[{df['Ci'].min():.3f}, {df['Ci'].max():.3f}]")
    else:
        df["Ci"] = 0.5

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n[Scoring] Unified scores for {len(df)} sites:")
    print(f"  wi_base: [{df['wi_base'].min():.4f}, {df['wi_base'].max():.4f}]")
    print(f"  risk:    [{df['risk_penalty'].min():.4f}, {df['risk_penalty'].max():.4f}]")
    print(f"  wi:      [{df['wi'].min():.4f}, {df['wi'].max():.4f}]")
    print(f"  Ci:      [{df['Ci'].min():.4f}, {df['Ci'].max():.4f}]")
    print(f"  excluded: {df['ei'].sum()} / {len(df)}")

    return df


# ─── 3. PAIRWISE INTERACTION ─────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def compute_pairwise(lats, lons, eta=ETA, ell_km=ELL_KM):
    n = len(lats)
    dist_km = np.zeros((n, n))
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            dist_km[i, j] = dist_km[j, i] = d
            mij = eta * np.exp(-d / ell_km)
            M[i, j] = M[j, i] = mij
    return dist_km, M


# ─── 4. QUBO ASSEMBLY ────────────────────────────────────────────────────────

def assemble_qubo(wi, Ci, ei, M, budget, lam=LAMBDA, mu=MU):
    n = len(wi)
    Q = np.zeros((n, n))

    # Diagonal: Qii = -wi + lambda*(Ci^2 - 2*B*Ci) + mu*ei
    for i in range(n):
        Q[i, i] = -wi[i] + lam * (Ci[i]**2 - 2*budget*Ci[i]) + mu * ei[i]

    # Off-diagonal: Qij = 2*lambda*Ci*Cj + Mij
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = 2 * lam * Ci[i] * Ci[j] + M[i, j]
            Q[j, i] = Q[i, j]

    const = lam * budget**2
    return Q, const


# ─── 5. SOLVERS ──────────────────────────────────────────────────────────────

def qubo_energy(x, Q):
    return float(x @ Q @ x)


def brute_force(Q, wi, Ci, ei, budget):
    """Exhaustive enumeration — ground truth for n <= 20."""
    n = Q.shape[0]
    total = 2 ** n
    best_energy, best_x = np.inf, None
    feas_energy, feas_x = np.inf, None

    print(f"[Brute] Enumerating {total:,} solutions...")
    t0 = time.time()

    for val in range(total):
        x = np.array([(val >> i) & 1 for i in range(n)], dtype=float)
        e = qubo_energy(x, Q)

        if e < best_energy:
            best_energy, best_x = e, x.copy()

        cost = Ci @ x
        excl = ei @ x
        if cost <= budget + 1e-9 and excl == 0 and e < feas_energy:
            feas_energy, feas_x = e, x.copy()

    dt = time.time() - t0
    print(f"[Brute] Done in {dt:.1f}s. Global={best_energy:.4f}, "
          f"Feasible={feas_energy:.4f}")

    return {
        "method": "brute_force",
        "energy": float(best_energy),
        "x": best_x,
        "feasible_energy": float(feas_energy),
        "feasible_x": feas_x,
        "runtime": dt,
    }


def greedy_solve(Q, wi, Ci, ei, budget):
    """Rank by wi/Ci, add sequentially."""
    n = len(wi)
    t0 = time.time()

    ratios = np.full(n, -np.inf)
    for i in range(n):
        if ei[i] == 0 and Ci[i] > 0:
            ratios[i] = wi[i] / Ci[i]

    ranked = np.argsort(-ratios)
    x = np.zeros(n)
    remaining = budget

    for idx in ranked:
        if ei[idx] == 1:
            continue
        if Ci[idx] <= remaining + 1e-9:
            x[idx] = 1
            remaining -= Ci[idx]

    dt = time.time() - t0
    return {
        "method": "greedy",
        "energy": qubo_energy(x, Q),
        "x": x,
        "runtime": dt,
    }


def sa_solve(Q, wi, Ci, ei, budget, seed=SEED):
    """Simulated annealing with geometric cooling."""
    n = Q.shape[0]
    rng = np.random.default_rng(seed)
    t0 = time.time()

    global_best_x, global_best_e = None, np.inf

    for r in range(SA_RESTARTS):
        x = rng.integers(0, 2, size=n).astype(float)
        current_e = qubo_energy(x, Q)
        best_x, best_e = x.copy(), current_e
        T = SA_T0

        for _ in range(SA_STEPS):
            flip = rng.integers(0, n)
            x[flip] = 1.0 - x[flip]
            new_e = qubo_energy(x, Q)
            delta = new_e - current_e

            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_e = new_e
                if current_e < best_e:
                    best_e, best_x = current_e, x.copy()
            else:
                x[flip] = 1.0 - x[flip]

            T *= SA_ALPHA

        if best_e < global_best_e:
            global_best_e, global_best_x = best_e, best_x.copy()

    dt = time.time() - t0
    return {
        "method": "simulated_annealing",
        "energy": float(global_best_e),
        "x": global_best_x,
        "runtime": dt,
    }


def ga_solve(Q, wi, Ci, ei, budget, seed=SEED):
    """Genetic algorithm."""
    n = Q.shape[0]
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # Initialize population
    pop = rng.integers(0, 2, size=(GA_POP, n)).astype(float)
    fitness = np.array([-qubo_energy(ind, Q) for ind in pop])

    best_x = pop[np.argmax(fitness)].copy()
    best_e = -np.max(fitness)

    for gen in range(GA_GENS):
        # Tournament selection
        idx = rng.integers(0, GA_POP, size=(GA_POP, 2))
        parents = np.where(
            fitness[idx[:, 0]] > fitness[idx[:, 1]],
            idx[:, 0], idx[:, 1]
        )

        new_pop = []
        for k in range(0, GA_POP - 1, 2):
            p1, p2 = pop[parents[k]], pop[parents[k + 1]]

            if rng.random() < GA_CROSSOVER:
                pt = rng.integers(1, n)
                c1 = np.concatenate([p1[:pt], p2[pt:]])
                c2 = np.concatenate([p2[:pt], p1[pt:]])
            else:
                c1, c2 = p1.copy(), p2.copy()

            for c in [c1, c2]:
                mask = rng.random(n) < GA_MUTATION
                c[mask] = 1 - c[mask]
            new_pop.extend([c1, c2])

        pop = np.array(new_pop[:GA_POP])
        fitness = np.array([-qubo_energy(ind, Q) for ind in pop])

        gen_best_idx = np.argmax(fitness)
        gen_best_e = -fitness[gen_best_idx]
        if gen_best_e < best_e:
            best_e = gen_best_e
            best_x = pop[gen_best_idx].copy()

    dt = time.time() - t0
    return {
        "method": "genetic_algorithm",
        "energy": float(best_e),
        "x": best_x,
        "runtime": dt,
    }


# ─── 6. SANITY CHECK ─────────────────────────────────────────────────────────

def sanity_check(results, Q, wi, Ci, ei, budget, site_ids):
    """Validate all results. Returns (passed, report_dict)."""
    print("\n" + "=" * 65)
    print("  SANITY CHECK")
    print("=" * 65)

    checks = []
    all_pass = True

    def check(name, cond, detail=""):
        nonlocal all_pass
        ok = "PASS" if cond else "FAIL"
        if not cond:
            all_pass = False
        checks.append({"check": name, "status": ok, "detail": detail})
        sym = "\u2713" if cond else "\u2717"
        print(f"  [{sym}] {name}: {ok}  {detail}")

    brute = results.get("brute_force")  # None if brute-force was skipped

    # ── AR reference selection ──────────────────────────────────────────
    # At small N, brute-force is the true global minimum.
    # At N > BRUTE_FORCE_MAX_N (~24), brute-force is skipped and we fall
    # back to the best classical heuristic as the AR reference. Log the
    # choice explicitly so downstream consumers know exactly what the AR
    # column means for each run.
    if brute is not None:
        ar_reference_energy = brute["energy"]
        ar_reference_source = "brute_force"
        print(f"\n  [AR REFERENCE] Using brute-force optimum "
              f"(energy = {ar_reference_energy:.6f}). "
              f"Approximation ratios are against the true global minimum.")
    else:
        classical_candidates = {
            "simulated_annealing": results.get("simulated_annealing"),
            "genetic_algorithm":   results.get("genetic_algorithm"),
            "greedy":              results.get("greedy"),
        }
        valid_candidates = {n: r for n, r in classical_candidates.items()
                            if r is not None and "energy" in r}
        if not valid_candidates:
            raise RuntimeError(
                "Brute-force skipped AND no classical heuristic produced a "
                "result. Cannot compute approximation ratios. Check solver "
                "configuration."
            )
        ar_reference_source = min(valid_candidates,
                                  key=lambda k: valid_candidates[k]["energy"])
        ar_reference_energy = valid_candidates[ar_reference_source]["energy"]
        print(f"\n  [AR REFERENCE] Brute-force skipped at N={len(wi)}. "
              f"Using best classical heuristic: {ar_reference_source} "
              f"(energy = {ar_reference_energy:.6f}).")
        print(f"  [AR REFERENCE] Approximation ratios are RELATIVE to this "
              f"heuristic, NOT to the true global minimum. Interpret "
              f"accordingly in Tables I/II and downstream AR comparisons.")

    # ── Energy ordering (only applicable when brute-force ground truth exists) ─
    if brute is not None:
        for name in ["greedy", "simulated_annealing", "genetic_algorithm"]:
            r = results[name]
            check(
                f"brute <= {name}",
                brute["energy"] <= r["energy"] + 1e-9,
                f"brute={brute['energy']:.4f}, {name}={r['energy']:.4f}"
            )
    else:
        print("  [Skip] 'brute <= solver' ordering checks not applicable "
              "when brute-force is skipped.")

    # Random baseline
    rng = np.random.default_rng(123)
    rand_energies = [qubo_energy(rng.integers(0, 2, size=len(wi)).astype(float), Q)
                     for _ in range(1000)]
    rand_mean = np.mean(rand_energies)

    for name, r in results.items():
        if name == "brute_force":
            continue
        check(f"{name} < random", r["energy"] < rand_mean,
              f"{name}={r['energy']:.4f}, rand_mean={rand_mean:.4f}")

    # Selection count, feasibility, energy reproducibility
    for name, r in results.items():
        x = r["x"]
        n_sel = int(np.sum(x > 0.5))
        cost = float(Ci @ x)
        excl = int(ei @ x)
        recomputed = qubo_energy(x, Q)

        check(f"{name}: 3-10 sites", 3 <= n_sel <= 10, f"selected={n_sel}")
        check(f"{name}: cost <= B", cost <= budget + 1e-9,
              f"cost={cost:.4f}, B={budget}")
        check(f"{name}: no excluded", excl == 0, f"excluded={excl}")
        check(f"{name}: energy match",
              abs(recomputed - r["energy"]) < 1e-6,
              f"stored={r['energy']:.6f}, recomputed={recomputed:.6f}")

    # Comparison table
    print("\n" + "=" * 65)
    print("  COMPARISON TABLE")
    print("=" * 65)
    header = (f"  {'Method':<22} {'Energy':>10} {'Sites':>6} "
              f"{'Benefit':>8} {'Cost':>6} {'Excl':>5} {'Time':>8}")
    print(header)
    print("  " + "-" * 63)

    rows = []
    for name, r in results.items():
        x = r["x"]
        n_sel = int(np.sum(x > 0.5))
        benefit = float(wi @ x)
        cost = float(Ci @ x)
        excl = int(ei @ x)
        sel_sites = [site_ids[i] for i in range(len(x)) if x[i] > 0.5]
        row = {
            "method": name,
            "energy": r["energy"],
            "n_selected": n_sel,
            "selected_sites": sel_sites,
            "total_benefit": benefit,
            "total_cost": cost,
            "excluded_used": excl,
            "runtime": r["runtime"],
            "ar_reference_source": ar_reference_source,
            "ar_reference_energy": ar_reference_energy,
        }
        rows.append(row)
        print(f"  {name:<22} {r['energy']:>10.4f} {n_sel:>6} "
              f"{benefit:>8.4f} {cost:>6.3f} {excl:>5} {r['runtime']:>7.2f}s")

    # Approximation ratios (against the reference chosen above)
    print(f"\n  Approximation Ratios (vs {ar_reference_source}):")
    for name, r in results.items():
        if name == ar_reference_source:
            continue
        ar = (r["energy"] / ar_reference_energy
              if ar_reference_energy != 0 else float("inf"))
        print(f"    AR({name}) = {ar:.6f}")

    verdict = "ALL PASSED" if all_pass else "SOME FAILED"
    print(f"\n  OVERALL: {verdict}")
    print("=" * 65)

    return all_pass, {
        "overall": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "comparison": rows,
        "random_baseline_mean": float(rand_mean),
        "ar_reference_source": ar_reference_source,
        "ar_reference_energy": float(ar_reference_energy),
    }


# ─── 7. EXPORT ────────────────────────────────────────────────────────────────

def export_all(df, dist_km, M, Q, const, results, sanity, budget,
               base_dir, cost_model, grid=20):
    """Save all outputs: per-task JSON files plus numpy/CSV exports."""
    out_qubo    = Path(base_dir) / "georgia" / "qiskit-ready" / "qubo_matrices" / cost_model
    inputs_dir  = out_qubo / f"n{grid}" / "inputs"
    results_dir = out_qubo / f"n{grid}" / "results"
    out_meta    = Path(base_dir) / "georgia" / "qiskit-ready" / "site_metadata" / cost_model
    out_proc    = Path(base_dir) / "georgia" / "processed"

    for d in [inputs_dir, results_dir, out_meta]:
        d.mkdir(parents=True, exist_ok=True)

    n = len(df)
    site_ids = df["site_id"].tolist()
    tag = "unified"

    # Pull AR reference info from the sanity report (computed during
    # sanity_check). Used to populate the results CSV columns below.
    ar_reference_source = sanity.get("ar_reference_source", "brute_force")
    ar_reference_energy = sanity.get("ar_reference_energy", 0.0)

    # ── Site scores (enhanced version of sites_{grid}_scored.csv) ────────
    score_file = out_proc / f"sites_{grid}_scored_{cost_model}.csv"
    df.to_csv(score_file, index=False)
    print(f"\n[Export] Site scores → {score_file}")

    # Also copy to qiskit-ready for downstream QAOA scripts
    df.to_csv(out_meta / f"sites_{grid}.csv", index=False)

    # ── Pairwise matrix (JSON) ───────────────────────────────────────────
    pairwise_out = {
        "task": "unified_pipeline_pairwise",
        "parameters": {"eta": ETA, "ell_km": ELL_KM},
        "n_sites": n,
        "n_pairs": n * (n - 1) // 2,
        "site_ids": site_ids,
        "distance_matrix_km": dist_km.tolist(),
        "interaction_matrix_M": M.tolist(),
    }
    with open(inputs_dir / f"pairwise_{grid}.json", "w") as f:
        json.dump(pairwise_out, f, indent=2)

    # ── QUBO matrix (both formats) ──────────────────────────────────────
    qubo_out = {
        "task": "unified_pipeline_qubo_assembly",
        "description": f"Unified QUBO (cost_model={cost_model})",
        "parameters": {"lambda": LAMBDA, "mu": MU, "budget_B": budget},
        "n_sites": n,
        "site_ids": site_ids,
        "Q_matrix": Q.tolist(),
        "wi": df["wi"].tolist(),
        "Ci": df["Ci"].tolist(),
        "ei": df["ei"].tolist(),
    }
    with open(inputs_dir / f"qubo_{grid}.json", "w") as f:
        json.dump(qubo_out, f, indent=2)

    # Numpy and CSV exports
    np.save(inputs_dir / f"Q_{grid}.npy", Q)
    np.savetxt(inputs_dir / f"Q_{grid}.csv", Q, delimiter=",")

    meta_json = {
        "n_sites": n, "budget": budget, "qubo_const": const,
        "w": df["wi"].tolist(), "C": df["Ci"].tolist(),
        "cost_model": cost_model,
    }
    with open(inputs_dir / f"meta_{grid}.json", "w") as f:
        json.dump(meta_json, f, indent=2)

    # ── Solver results (per-method JSON) ────────────────────────────────
    for name, r in results.items():
        out = {
            "task": {"brute_force":         "unified_pipeline_brute_force",
                     "greedy":              "unified_pipeline_greedy",
                     "simulated_annealing": "unified_pipeline_simulated_annealing",
                     "genetic_algorithm":   "unified_pipeline_genetic_algorithm"}[name],
            "method": name,
            "n_sites": n,
            "runtime_seconds": r["runtime"],
            "result": {
                "energy": r["energy"],
                "bitstring": "".join(str(int(b)) for b in r["x"]),
                "n_selected": int(np.sum(r["x"] > 0.5)),
                "selected_sites": [site_ids[i] for i in range(n)
                                   if r["x"][i] > 0.5],
                "total_benefit": float(df["wi"].values @ r["x"]),
                "total_cost": float(df["Ci"].values @ r["x"]),
                "excluded_used": int(df["ei"].values @ r["x"]),
                "budget_B": budget,
            },
        }
        fname = f"{name}_{grid}.json"
        with open(results_dir / fname, "w") as f:
            json.dump(out, f, indent=2)

    # ── Sanity check ────────────────────────────────────────────────────
    with open(results_dir / f"sanity_check_{grid}.json", "w") as f:
        json.dump(sanity, f, indent=2, default=str)

    # ── Flat results CSV ────────────────────────────────────────────────
    rows = []
    for name, r in results.items():
        x = r["x"]
        ar_value = (r["energy"] / ar_reference_energy
                    if ar_reference_energy != 0 else float("inf"))
        rows.append({
            "Method": name,
            "QUBO Energy": round(r["energy"], 4),
            "Total Benefit": round(float(df["wi"].values @ x), 4),
            "Total Cost": round(float(df["Ci"].values @ x), 4),
            "Sites Selected": int(np.sum(x > 0.5)),
            "Excluded Selected": int(df["ei"].values @ x),
            "Feasible": float(df["Ci"].values @ x) <= budget + 1e-9,
            "Runtime (s)": round(r["runtime"], 4),
            "AR": round(ar_value, 6),
            "AR Reference Source": ar_reference_source,
            "AR Reference Energy": round(ar_reference_energy, 6),
        })
    pd.DataFrame(rows).to_csv(
        Path(base_dir) / "georgia" / "processed" / f"results_{grid}_{cost_model}.csv",
        index=False
    )

    print(f"[Export] All outputs saved to {out_qubo}/")


# ─── 8. MAIN ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VQ-MAR Unified Pipeline")
    parser.add_argument("--base_dir", default=DEFAULT_BASE,
                        help="Project root directory")
    parser.add_argument("--cost_model", choices=["flat", "real"], default="flat",
                        help="Cost model: flat (Ci=0.5) or real (Eq. 15)")
    parser.add_argument("--grid", type=int, choices=[20, 50], default=20,
                        help="Number of candidate sites. 20 is the default "
                             "pilot grid; 50 is the scaling grid. "
                             "Brute-force is auto-skipped at N > 24.")
    parser.add_argument("--skip_brute", action="store_true",
                        help="Skip brute-force (for testing only)")
    args = parser.parse_args()

    print("=" * 65)
    print("  VQ-MAR Unified Pipeline")
    print(f"  Cost model: {args.cost_model}")
    print(f"  Grid size:  {args.grid}")
    print("=" * 65)

    # ── Load processed scores ────────────────────────────────────────────
    scores = load_processed_scores(args.base_dir, args.grid)

    # ── Compute unified scores ───────────────────────────────────────────
    df = compute_unified_scores(scores, cost_model=args.cost_model)
    n = len(df)

    # ── Determine budget ─────────────────────────────────────────────────
    if args.cost_model == "real":
        budget = df["Ci"].mean() * 6  # calibrate to ~6 sites
        print(f"[QUBO] Real cost budget: B = {budget:.4f}")
    else:
        budget = BUDGET_FLAT
        print(f"[QUBO] Flat cost budget: B = {budget}")

    # ── Validate bounds ──────────────────────────────────────────────────
    for col in ["Si", "Ni", "Li", "Cclim_i", "Ai", "wi", "Ci"]:
        if col in df.columns:
            lo, hi = df[col].min(), df[col].max()
            if lo < -1e-9 or hi > 1.0 + 1e-9:
                raise ValueError(f"{col} out of [0,1]: [{lo:.4f}, {hi:.4f}]")

    # ── Pairwise interactions ────────────────────────────────────────────
    print("\n[Pairwise] Computing pairwise interactions...")
    dist_km, M = compute_pairwise(
        df["latitude"].values, df["longitude"].values
    )
    print(f"[Pairwise] {n*(n-1)//2} pairs. "
          f"M range: [{M[M>0].min():.6f}, {M.max():.6f}]")

    # ── QUBO assembly ────────────────────────────────────────────────────
    print("\n[Assembly] Assembling QUBO matrix...")
    Q, const = assemble_qubo(
        df["wi"].values, df["Ci"].values, df["ei"].values, M, budget
    )
    print(f"[Assembly] Q shape: {Q.shape}. Const offset: {const:.4f}")
    print(f"[Assembly] Diagonal: [{np.diag(Q).min():.4f}, {np.diag(Q).max():.4f}]")

    # ── Solvers ──────────────────────────────────────────────────────────
    site_ids = df["site_id"].tolist()
    wi = df["wi"].values
    Ci = df["Ci"].values
    ei = df["ei"].values

    results = {}

    # Auto-skip brute-force when N exceeds the tractable threshold.
    # 2^24 = 16.7M combinations is still seconds on a modern CPU; anything
    # larger becomes impractical. The --skip_brute flag remains as a manual
    # escape hatch for testing.
    BRUTE_FORCE_MAX_N = 24
    if args.skip_brute:
        print(f"\n[Brute] Brute-force skipped (--skip_brute flag set)")
    elif args.grid > BRUTE_FORCE_MAX_N:
        print(f"\n[Brute] Brute-force skipped: N={args.grid} > "
              f"BRUTE_FORCE_MAX_N={BRUTE_FORCE_MAX_N} "
              f"(2^{args.grid} = {2**args.grid:,} combinations is infeasible)")
    else:
        print(f"\n[Brute] Brute-force enumeration "
              f"(2^{args.grid} = {2**args.grid:,} combinations)...")
        results["brute_force"] = brute_force(Q, wi, Ci, ei, budget)

    print("\n[Greedy] Greedy baseline...")
    results["greedy"] = greedy_solve(Q, wi, Ci, ei, budget)

    print("\n[SA] Simulated annealing...")
    results["simulated_annealing"] = sa_solve(Q, wi, Ci, ei, budget)

    print("\n[GA] Genetic algorithm...")
    results["genetic_algorithm"] = ga_solve(Q, wi, Ci, ei, budget)

    # ── Sanity check ─────────────────────────────────────────────────────
    passed, sanity = sanity_check(results, Q, wi, Ci, ei, budget, site_ids)

    # ── Export ────────────────────────────────────────────────────────────
    export_all(df, dist_km, M, Q, const, results, sanity, budget,
               args.base_dir, args.cost_model, args.grid)

    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE — {'ALL CHECKS PASSED' if passed else 'CHECK FAILURES'}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
