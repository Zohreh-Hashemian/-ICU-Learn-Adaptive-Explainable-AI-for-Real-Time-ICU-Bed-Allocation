"""
ICU-Learn: Adaptive & Explainable AI for Real-Time ICU Bed Allocation

Module: analyze_results.py
Purpose:
  - Load the consolidated policy metrics CSV
  - Compare Efficiency (Total Reward), Safety (Violation Rate), and Fairness (Admission Gap)
  - Save simple comparison plots for the paper

Inputs (one of):
  - outputs/policy_comparison_with_bandit.csv  (preferred; includes LinUCB)
  - outputs/policy_comparison.csv              (fallback; baselines only)

Outputs:
  - outputs/figures/comparison_total_reward.png
  - outputs/figures/comparison_safety.png
  - outputs/figures/comparison_fairness.png
"""

from pathlib import Path
from typing import List
import sys

import pandas as pd
import matplotlib.pyplot as plt


def load_results() -> pd.DataFrame:
    """
    Load the policy comparison CSV. Prefer the file that includes bandit results.

    Returns
    -------
    pd.DataFrame
    """
    preferred = Path("outputs/policy_comparison_with_bandit.csv")
    fallback = Path("outputs/policy_comparison.csv")

    if preferred.exists():
        df = pd.read_csv(preferred)
    elif fallback.exists():
        df = pd.read_csv(fallback)
    else:
        sys.exit(
            "No results file found. Run layer 3/4 first to produce:\n"
            "  - outputs/policy_comparison_with_bandit.csv (preferred)\n"
            "  - or outputs/policy_comparison.csv (fallback)"
        )

    # Basic sanity checks
    required_cols: List[str] = [
        "policy",
        "total_reward",
        "safety_viol_rate",
        "admitted_gap_g1_minus_g0",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns in results CSV: {missing}")

    return df


def print_tables(df: pd.DataFrame) -> None:
    """Pretty-print key tables to the console."""
    # Ensure consistent policy ordering if present
    order = ["FCFS", "SeverityThreshold", "GreedyRisk", "LinUCB_Safe"]
    if set(order).issuperset(set(df["policy"].unique())):
        df = df.set_index("policy").loc[order].reset_index()

    print("=== Full Results ===")
    print(df)

    print("\n=== Efficiency: Total Reward ===")
    print(df[["policy", "total_reward"]])

    print("\n=== Safety: Violation Rate ===")
    print(df[["policy", "safety_viol_rate"]])

    print("\n=== Fairness: Admission Gap (G1 - G0) ===")
    print(df[["policy", "admitted_gap_g1_minus_g0"]])


def save_plots(df: pd.DataFrame) -> None:
    """
    Save bar plots for the three key metrics.
    """
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set index for plotting convenience
    dfi = df.set_index("policy")

    # Total Reward
    plt.figure()
    dfi[["total_reward"]].plot(kind="bar", legend=False)
    plt.ylabel("Total Reward")
    plt.title("Efficiency: Total Reward Comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_total_reward.png")
    plt.close()

    # Safety Violations
    plt.figure()
    dfi[["safety_viol_rate"]].plot(kind="bar", legend=False)
    plt.ylabel("Violation Rate")
    plt.title("Safety: Violation Rate Comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_safety.png")
    plt.close()

    # Fairness Gap
    plt.figure()
    dfi[["admitted_gap_g1_minus_g0"]].plot(kind="bar", legend=False)
    plt.ylabel("Gap (Group1 - Group0)")
    plt.title("Fairness: Admission Gap")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_fairness.png")
    plt.close()


def main() -> None:
    df = load_results()
    print_tables(df)
    save_plots(df)


if __name__ == "__main__":
    main()
