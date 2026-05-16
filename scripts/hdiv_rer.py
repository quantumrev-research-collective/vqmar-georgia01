"""
VQ-MAR — Portfolio Characterization

Reads top-10 bitstring JSON files from QAOA variants and computes
portfolio-level diversity metrics for Tables III and IV of the
IEEE QCE 2026 paper.

Metrics computed:
  Table III: energy, benefit, AR, runtime (per solver)
  Table IV:  portfolio overlap (Jaccard), site diversity,
             benefit spread, cost utilization

Usage:
  # --data_dir is required. The repo layout splits results into four
  # slices: flat|real x n20|n50. Point --data_dir at the specific
  # slice you want to analyze. All top10_*.json files in the slice
  # are discovered and grouped by their internal 'variant' field.

  # Discover all variants in the N=20 flat-cost slice:
  python scripts/hdiv_rer.py --n_sites 20 \
      --data_dir georgia/qiskit-ready/qubo_matrices/flat/n20/results

  # Restrict to specific variants in the N=50 flat-cost slice:
  python scripts/hdiv_rer.py --n_sites 50 \
      --data_dir georgia/qiskit-ready/qubo_matrices/flat/n50/results \
      --only_variants baseline choice_a choice_b
"""

import json
import os
import sys
import csv
import glob
import argparse
from itertools import combinations


# ============================================================
# Configuration
# ============================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_BASE = os.environ.get("VQMAR_BASE", _REPO_ROOT)

DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_BASE, "georgia", "qiskit-ready", "portfolio_analysis")

DEFAULT_BUDGET = 3.0
DEFAULT_N_SITES = 20


# ============================================================
# File Loading
# ============================================================

def load_variant(filepath):
    """Load a top-10 bitstring JSON file and return the parsed dict."""
    if not os.path.exists(filepath):
        print(f"  WARNING: File not found: {filepath}")
        return None
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def load_variants_auto_discover(data_dir, n_sites=None, only_variants=None):
    """
    Discover variant files by scanning for top10_*.json and grouping by
    the 'variant' field inside each file.

    Parameters:
      data_dir:       directory to scan
      n_sites:        if provided, only include files whose 'n_sites' field
                      matches this value. Essential when flat/real
                      directories contain both N=20 and N=50 JSONs.
      only_variants:  if provided (list of variant labels), restrict the
                      result to just those variants. Use this to produce
                      a specific subset (e.g. ['baseline', 'choice_a',
                      'choice_b']) for a paper table.

    If multiple files share the same 'variant' label after filtering, the
    most recently modified one wins. Returns dict of {label: parsed_json}.
    """
    variants = {}
    pattern = os.path.join(data_dir, "top10_*.json")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime)

    if not candidates:
        print(f"  No top10_*.json files found in {data_dir}")
        return variants

    only_set = set(only_variants) if only_variants else None
    skipped_n = 0
    skipped_variant = 0

    for filepath in candidates:
        data = load_variant(filepath)
        if data is None:
            continue

        # Filter by n_sites when requested. Files missing the field are
        # conservatively skipped so we never silently mix problem sizes.
        if n_sites is not None:
            file_n = data.get("n_sites")
            if file_n != n_sites:
                skipped_n += 1
                continue

        label = data.get("variant", "unknown")

        # Filter by requested variant subset.
        if only_set is not None and label not in only_set:
            skipped_variant += 1
            continue

        variants[label] = data  # newest-wins for duplicate labels
        n_entries = len(data.get("top10_bitstrings", []))
        print(f"  Discovered {label}: {os.path.basename(filepath)}")
        print(f"    -> {n_entries} bitstrings loaded")

    if skipped_n > 0:
        print(f"  Skipped {skipped_n} file(s) with n_sites != {n_sites}")
    if skipped_variant > 0:
        print(f"  Skipped {skipped_variant} file(s) outside --only_variants")

    return variants


# ============================================================
# Table III Metrics (per-solver performance)
# ============================================================

def extract_table3_metrics(data, label):
    """
    Extract Table III row data from a variant JSON.

    Returns a dict with:
      variant, energy, benefit, cost, ar, feasible_count,
      n_sites_selected, runtime
    """
    top10 = data.get("top10_bitstrings", [])
    if not top10:
        return None

    # Filter to entries that have a valid energy value
    valid = [e for e in top10 if e.get("energy") is not None]
    if not valid:
        return None

    # Best solution is the lowest energy in the top 10
    best = min(valid, key=lambda x: x["energy"])

    # Runtime may or may not be in the JSON
    runtime = data.get("runtime_seconds", data.get("runtime", data.get("wall_time", "N/A")))

    return {
        "variant": label,
        "cost_model": data.get("cost_model", "unknown"),
        "p_depth": data.get("p_depth", "N/A"),
        "simulator": data.get("simulator", "unknown"),
        "best_energy": best.get("energy"),
        "best_benefit": best.get("benefit"),
        "best_cost": best.get("cost"),
        "best_feasible": best.get("feasible"),
        "best_sites": best.get("sites") or [],
        "n_selected": len(best.get("sites") or []),
        "approximation_ratio": data.get("approximation_ratio", "N/A"),
        "runtime": runtime,
    }


# ============================================================
# Table IV Metrics (portfolio diversity)
# ============================================================

def compute_site_diversity(data, n_sites):
    """
    Site diversity: fraction of unique candidate sites appearing
    across all top-10 solutions within a single variant.

    If all 10 solutions select the exact same 4 sites, diversity
    is 4/20 = 0.20. If they collectively touch 15 different sites,
    diversity is 15/20 = 0.75.
    """
    top10 = data.get("top10_bitstrings", [])
    all_sites = set()
    for entry in top10:
        sites = entry.get("sites") or []
        all_sites.update(sites)
    return len(all_sites) / n_sites if n_sites > 0 else 0.0


def compute_benefit_spread(data):
    """
    Benefit spread: standard deviation of total benefit across
    the top-10 solutions within a variant.

    High spread = the variant finds qualitatively different portfolios.
    Low spread = the top solutions are nearly identical.
    """
    top10 = data.get("top10_bitstrings", [])
    benefits = [entry["benefit"] for entry in top10 if entry.get("benefit") is not None]
    if len(benefits) < 2:
        return 0.0
    mean_b = sum(benefits) / len(benefits)
    variance = sum((b - mean_b) ** 2 for b in benefits) / (len(benefits) - 1)
    return variance ** 0.5


def compute_cost_utilization(data, budget=DEFAULT_BUDGET):
    """
    Cost utilization: average of (total_cost / budget) across
    the top-10 solutions.

    Shows how fully each variant uses the available budget.
    A value near 1.0 means the variant is spending nearly the
    full budget; a low value means it's being conservative.
    """
    top10 = data.get("top10_bitstrings", [])
    if not top10 or budget == 0:
        return 0.0
    utilizations = [entry["cost"] / budget for entry in top10 if entry.get("cost") is not None]
    if not utilizations:
        return 0.0
    return sum(utilizations) / len(utilizations)


def compute_feasibility_rate(data):
    """
    Feasibility rate: fraction of top-10 solutions that satisfy
    the budget constraint (feasible == true).
    """
    top10 = data.get("top10_bitstrings", [])
    if not top10:
        return 0.0
    feasible_count = sum(1 for entry in top10 if entry.get("feasible"))
    return feasible_count / len(top10)


def compute_pairwise_jaccard(variants):
    """
    Portfolio overlap (Jaccard similarity) between each pair of variants.

    For each variant, we take the union of all sites selected across
    its top-10 solutions (the variant's "site footprint"). Then for
    each pair of variants, Jaccard = |intersection| / |union|.

    High Jaccard = the two variants are selecting similar sites.
    Low Jaccard = they're finding different parts of the landscape.
    """
    footprints = {}
    for label, data in variants.items():
        top10 = data.get("top10_bitstrings", [])
        sites = set()
        for entry in top10:
            entry_sites = entry.get("sites") or []
            sites.update(entry_sites)
        footprints[label] = sites

    results = {}
    for (a, b) in combinations(footprints.keys(), 2):
        intersection = footprints[a] & footprints[b]
        union = footprints[a] | footprints[b]
        jaccard = len(intersection) / len(union) if union else 0.0
        results[f"{a}_vs_{b}"] = jaccard

    return results


def compute_table4_metrics(variants, n_sites, budget=DEFAULT_BUDGET):
    """
    Compute all Table IV metrics for each variant and pairwise overlaps.

    Returns:
      per_variant: list of dicts with per-variant diversity metrics
      pairwise: dict of Jaccard similarities between variant pairs
    """
    per_variant = []
    for label, data in variants.items():
        metrics = {
            "variant": label,
            "site_diversity": round(compute_site_diversity(data, n_sites), 4),
            "benefit_spread": round(compute_benefit_spread(data), 6),
            "cost_utilization": round(compute_cost_utilization(data, budget=budget), 4),
            "feasibility_rate": round(compute_feasibility_rate(data), 4),
        }
        per_variant.append(metrics)

    pairwise = compute_pairwise_jaccard(variants)

    return per_variant, pairwise


# ============================================================
# Synthetic Unit Tests
# ============================================================

def run_synthetic_tests(n_sites):
    """
    Three synthetic-portfolio unit tests verifying expected metric ordering.

    Test 1 (Clustered): all top-10 pick the same 3 sites -> low diversity
    Test 2 (Medium):    top-10 pick from a pool of 10 sites -> medium diversity
    Test 3 (Distributed): top-10 pick from 18 of 20 sites -> high diversity

    Expected ordering: clustered < medium < distributed for site_diversity
    """
    print("\n" + "=" * 60)
    print("SYNTHETIC UNIT TESTS")
    print("=" * 60)

    def make_synthetic(site_pools, label):
        """Build a fake variant JSON from a list of site pools."""
        top10 = []
        for i, sites in enumerate(site_pools):
            top10.append({
                "bitstring": "0" * n_sites,
                "energy": -10.0 + i * 0.1,
                "sites": sites,
                "benefit": 2.0 + i * 0.05,
                "cost": 0.8,
                "feasible": True,
                "probability": 0.01,
            })
        return {"top10_bitstrings": top10, "variant": label}

    # Test 1: Clustered — all 10 solutions pick the same 3 sites
    clustered_sites = [["GA_001", "GA_002", "GA_003"]] * 10
    clustered = make_synthetic(clustered_sites, "clustered")

    # Test 2: Medium — 10 solutions drawing from a pool of 10 sites
    medium_pools = [
        ["GA_001", "GA_002", "GA_003"],
        ["GA_004", "GA_005", "GA_006"],
        ["GA_007", "GA_008", "GA_009"],
        ["GA_010", "GA_001", "GA_005"],
        ["GA_002", "GA_006", "GA_009"],
        ["GA_003", "GA_007", "GA_010"],
        ["GA_001", "GA_004", "GA_008"],
        ["GA_005", "GA_009", "GA_002"],
        ["GA_006", "GA_010", "GA_003"],
        ["GA_007", "GA_001", "GA_004"],
    ]
    medium = make_synthetic(medium_pools, "medium")

    # Test 3: Distributed — 10 solutions touching 18 of 20 sites
    distributed_pools = [
        ["GA_001", "GA_002"],
        ["GA_003", "GA_004"],
        ["GA_005", "GA_006"],
        ["GA_007", "GA_008"],
        ["GA_009", "GA_010"],
        ["GA_011", "GA_012"],
        ["GA_013", "GA_014"],
        ["GA_015", "GA_016"],
        ["GA_017", "GA_018"],
        ["GA_001", "GA_019"],
    ]
    distributed = make_synthetic(distributed_pools, "distributed")

    # Compute diversity for each
    d_clustered = compute_site_diversity(clustered, n_sites)
    d_medium = compute_site_diversity(medium, n_sites)
    d_distributed = compute_site_diversity(distributed, n_sites)

    print(f"\n  Clustered site diversity:    {d_clustered:.4f}  (expected: low)")
    print(f"  Medium site diversity:       {d_medium:.4f}  (expected: medium)")
    print(f"  Distributed site diversity:  {d_distributed:.4f}  (expected: high)")

    # Verify ordering
    if d_clustered < d_medium < d_distributed:
        print("\n  PASS: Ordering is clustered < medium < distributed")
    else:
        print("\n  FAIL: Ordering violated!")
        print(f"    Expected: {d_clustered} < {d_medium} < {d_distributed}")

    # Also test benefit spread
    bs_clustered = compute_benefit_spread(clustered)
    bs_distributed = compute_benefit_spread(distributed)
    print(f"\n  Clustered benefit spread:    {bs_clustered:.6f}")
    print(f"  Distributed benefit spread:  {bs_distributed:.6f}")

    # Test Jaccard between clustered and distributed (should be low)
    test_variants = {"clustered": clustered, "distributed": distributed}
    jaccard = compute_pairwise_jaccard(test_variants)
    for pair, j in jaccard.items():
        print(f"\n  Jaccard ({pair}): {j:.4f}  (expected: low)")

    print("\n" + "=" * 60)
    print("SYNTHETIC TESTS COMPLETE")
    print("=" * 60)


# ============================================================
# Output
# ============================================================

def save_table3_csv(rows, output_path):
    """Save Table III metrics to CSV."""
    if not rows:
        print("  No Table III data to save.")
        return

    fieldnames = [
        "variant", "cost_model", "p_depth", "simulator",
        "best_energy", "best_benefit", "best_cost", "best_feasible",
        "n_selected", "approximation_ratio", "runtime",
    ]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            filtered = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered)
    print(f"  Table III saved to {output_path}")


def save_table4_csv(per_variant, pairwise, output_path):
    """Save Table IV metrics to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Per-variant metrics
        writer.writerow(["=== Per-Variant Diversity Metrics ==="])
        writer.writerow([
            "variant", "site_diversity", "benefit_spread",
            "cost_utilization", "feasibility_rate"
        ])
        for row in per_variant:
            writer.writerow([
                row["variant"],
                row["site_diversity"],
                row["benefit_spread"],
                row["cost_utilization"],
                row["feasibility_rate"],
            ])

        writer.writerow([])

        # Pairwise Jaccard
        writer.writerow(["=== Pairwise Portfolio Overlap (Jaccard) ==="])
        writer.writerow(["pair", "jaccard_similarity"])
        for pair, jaccard in sorted(pairwise.items()):
            writer.writerow([pair, round(jaccard, 4)])

    print(f"  Table IV saved to {output_path}")


def print_summary(table3_rows, per_variant, pairwise, n_sites):
    """Print a human-readable summary to the terminal."""

    print(f"\n" + "=" * 70)
    print(f"TABLE III — Per-Solver Performance (Best of Top-10) [N={n_sites}]")
    print("=" * 70)
    print(f"  {'Variant':<12} {'Energy':>10} {'Benefit':>10} {'Cost':>8} "
          f"{'AR':>8} {'Feasible':>9} {'Sites':>6}")
    print("  " + "-" * 65)
    for row in table3_rows:
        ar = row['approximation_ratio']
        ar_str = f"{ar:.4f}" if isinstance(ar, (int, float)) else str(ar)
        energy = row['best_energy']
        energy_str = f"{energy:.4f}" if energy is not None else "N/A"
        benefit = row['best_benefit']
        benefit_str = f"{benefit:.4f}" if benefit is not None else "N/A"
        cost = row['best_cost']
        cost_str = f"{cost:.4f}" if cost is not None else "N/A"
        feasible = row['best_feasible']
        feasible_str = "Yes" if feasible else ("No" if feasible is not None else "N/A")
        print(f"  {row['variant']:<12} {energy_str:>10} "
              f"{benefit_str:>10} {cost_str:>8} "
              f"{ar_str:>8} {feasible_str:>9} "
              f"{row['n_selected']:>6}")

    print(f"\n" + "=" * 70)
    print(f"TABLE IV — Portfolio Diversity Metrics [N={n_sites}]")
    print("=" * 70)
    print(f"  {'Variant':<12} {'Site Div':>10} {'Benefit Spread':>15} "
          f"{'Cost Util':>10} {'Feas Rate':>10}")
    print("  " + "-" * 60)
    for row in per_variant:
        print(f"  {row['variant']:<12} {row['site_diversity']:>10.4f} "
              f"{row['benefit_spread']:>15.6f} "
              f"{row['cost_utilization']:>10.4f} "
              f"{row['feasibility_rate']:>10.4f}")

    print(f"\n  {'Pairwise Overlap (Jaccard)':<40}")
    print("  " + "-" * 40)
    for pair, jaccard in sorted(pairwise.items()):
        print(f"  {pair:<35} {jaccard:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="VQ-MAR Portfolio Characterization — Tables III and IV"
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Directory containing top-10 JSON files. Required unless --test "
             "or --test_only is used. Results are split into four slices "
             "(qubo_matrices/{flat,real}/n{20,50}/results/) — point this flag "
             "at the specific slice to analyze."
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CSV files (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET,
        help=f"Budget value for cost utilization (default: {DEFAULT_BUDGET})"
    )
    parser.add_argument(
        "--n_sites",
        type=int,
        default=DEFAULT_N_SITES,
        help=f"Number of candidate sites (default: {DEFAULT_N_SITES})"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run synthetic unit tests before processing real data"
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Run synthetic unit tests only (no real data processing)"
    )
    parser.add_argument(
        "--only_variants",
        nargs="+",
        default=None,
        help="Restrict to a specific list of variant labels "
             "(e.g. baseline choice_a choice_b). Variant labels are "
             "read from the JSON's internal 'variant' field."
    )
    args = parser.parse_args()

    n_sites = args.n_sites

    print("=" * 70)
    print(f"VQ-MAR — Portfolio Characterization [N={n_sites}]")
    print("=" * 70)

    # Run synthetic tests if requested
    if args.test or args.test_only:
        run_synthetic_tests(n_sites)
        if args.test_only:
            return

    # --data_dir is required for real-data analysis (but not for --test_only)
    if not args.data_dir:
        print("\nERROR: --data_dir is required (unless running with --test_only).")
        print("  Valid slices:")
        print("    georgia/qiskit-ready/qubo_matrices/flat/n20/results")
        print("    georgia/qiskit-ready/qubo_matrices/flat/n50/results")
        print("    georgia/qiskit-ready/qubo_matrices/real/n20/results")
        print("    georgia/qiskit-ready/qubo_matrices/real/n50/results")
        sys.exit(1)

    # Load variant files
    print(f"\nLoading variants from: {args.data_dir}")
    variants = load_variants_auto_discover(
        args.data_dir,
        n_sites=args.n_sites,
        only_variants=args.only_variants,
    )

    if not variants:
        print("\nERROR: No variant files found. Check --data_dir path.")
        print(f"  Looked in: {args.data_dir}")
        print(f"  Pattern searched: top10_*.json")
        if args.n_sites:
            print(f"  n_sites filter:    {args.n_sites}")
        if args.only_variants:
            print(f"  only_variants:     {args.only_variants}")
        sys.exit(1)

    print(f"\n  Loaded {len(variants)} variants")

    # Compute Table III metrics
    print("\nComputing Table III metrics...")
    table3_rows = []
    for label, data in variants.items():
        row = extract_table3_metrics(data, label)
        if row:
            table3_rows.append(row)

    # Compute Table IV metrics
    print("Computing Table IV metrics...")
    per_variant, pairwise = compute_table4_metrics(variants, n_sites, budget=args.budget)

    # Print summary
    print_summary(table3_rows, per_variant, pairwise, n_sites)

    # Save outputs with n_sites in filename
    os.makedirs(args.output_dir, exist_ok=True)

    table3_path = os.path.join(args.output_dir, f"table3_solver_performance_n{n_sites}.csv")
    table4_path = os.path.join(args.output_dir, f"table4_portfolio_diversity_n{n_sites}.csv")

    print(f"\nSaving outputs to: {args.output_dir}")
    save_table3_csv(table3_rows, table3_path)
    save_table4_csv(per_variant, pairwise, table4_path)

    # Also save a combined JSON for downstream tools
    combined_path = os.path.join(args.output_dir, f"portfolio_characterization_n{n_sites}.json")
    combined = {
        "n_sites": n_sites,
        "table3": table3_rows,
        "table4_per_variant": per_variant,
        "table4_pairwise_jaccard": pairwise,
        "config": {
            "data_dir": args.data_dir,
            "budget": args.budget,
            "n_sites": n_sites,
            "only_variants": args.only_variants,
        }
    }
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"  Combined JSON saved to {combined_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()