"""
ICU-Learn: Adaptive & Explainable AI for Real-Time ICU Bed Allocation

Module: train_and_sim.py
Purpose:
  - L2: Train a calibrated risk model (GradientBoosting + isotonic calibration)
  - L3: Run a minimal ICU queueing simulator and compare fixed policies
  - Save metrics and figures for use in the paper

Outputs:
  - outputs/policy_comparison.csv
  - outputs/figures/predictive_reliability.png
  - outputs/figures/policy_avg_wait.png
  - outputs/figures/policy_utilization.png
  - outputs/figures/policy_safety_viol.png
  - outputs/figures/policy_total_reward.png

Pipeline position:
  L1 Data Prep → L2/L3 Risk Model & Baselines → L4 Safe LinUCB → L5 XAI
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss


# --------------------------- Minimal ICU Simulator ---------------------------
class ICUSimulatorMinimal:
    """
    A simple FIFO queue simulator for ICU bed allocation.

    Mechanics per time step:
      - New arrivals ~ Poisson(arrival_rate)
      - Up to k patients from the head of queue are evaluated
      - If a bed is free and the policy permits → admit (occupy a bed)
      - Random discharges free up some occupied beds
      - If a patient waits ≥ max_wait without admission → they leave the queue

    Policies:
      - 'fcfs'      : admit whenever a bed is free
      - 'threshold' : admit if predicted risk ≥ threshold (and bed is free)
      - 'greedy'    : same admission rule as 'threshold' (used as a simple risk-based baseline)

    Safety constraint (hard rule):
      - If predicted risk ≥ safety_threshold and a bed is available → force admit.
    """

    def __init__(self, n_beds: int = 20, max_wait: int = 4, random_state: int = 42, k_process: int = 3):
        self.n_beds = n_beds
        self.max_wait = max_wait
        self.k_process = k_process
        self.rs = np.random.RandomState(random_state)
        self.reset()

    def reset(self) -> None:
        from collections import deque
        self.beds_occupied = 0
        self.queue = deque()
        self.time = 0
        self.logs: List[Dict] = []
        self.cursor = 0

    def arrivals(self, lam: float) -> int:
        """Number of new patients in a time step ~ Poisson(lam)."""
        return self.rs.poisson(lam)

    def run(
        self,
        steps: int,
        model,
        X: np.ndarray,
        y: np.ndarray,
        g: np.ndarray,
        arrival_rate: float = 4.5,
        safety_threshold: float = 0.9,
        policy: str = "fcfs",
        threshold: float = 0.75,
    ) -> pd.DataFrame:
        """
        Execute the simulator under a fixed policy.

        Parameters
        ----------
        steps : int
            Number of time steps to simulate.
        model : sklearn-like classifier with predict_proba
            Calibrated risk model (probabilities used as predicted risk).
        X, y, g : arrays
            Test feature matrix, binary labels, and fairness group (0/1).
        arrival_rate : float
            Mean of Poisson arrivals per time step.
        safety_threshold : float
            Clinical hard threshold: if prisk ≥ this and a bed is free → force admit.
        policy : {'fcfs','threshold','greedy'}
            Decision policy (apart from the hard safety constraint).
        threshold : float
            Risk cutoff for 'threshold'/'greedy' policies.

        Returns
        -------
        pd.DataFrame
            A log of per-patient decisions and outcomes across time.
        """
        self.reset()

        def decide(prisk: float, beds_free: int, wait: int, policy: str) -> int:
            """Policy rule excluding the safety constraint."""
            if policy == "fcfs":
                return 1 if beds_free > 0 else 0
            if policy in ["threshold", "greedy"]:
                return 1 if (prisk >= threshold and beds_free > 0) else 0
            return 0

        for _ in range(steps):
            # Random discharges proportional to occupancy
            discharges = int(max(0, self.rs.normal(0.1, 0.05) * self.beds_occupied))
            self.beds_occupied = max(0, self.beds_occupied - discharges)

            # New arrivals
            for _ in range(self.arrivals(arrival_rate)):
                if self.cursor < len(X):
                    self.queue.append({"idx": self.cursor, "wait": 0})
                    self.cursor += 1

            # Increase waiting time
            for p in self.queue:
                p["wait"] += 1

            # Process the head of queue (up to k patients)
            for _ in range(min(self.k_process, len(self.queue))):
                p = self.queue[0]
                i = p["idx"]
                beds_free = self.n_beds - self.beds_occupied
                prisk = float(model.predict_proba(X[i:i + 1])[:, 1][0])

                # Hard safety constraint
                if prisk >= safety_threshold and beds_free > 0:
                    action = 1
                else:
                    action = decide(prisk, beds_free, p["wait"], policy)

                # Apply action
                admitted = 0
                if action == 1 and beds_free > 0:
                    admitted = 1
                    self.beds_occupied += 1
                    self.queue.popleft()
                elif action == 0 and p["wait"] >= self.max_wait:
                    # Patient leaves due to timeout
                    self.queue.popleft()

                # Reward design (for qualitative comparison only)
                y_true = int(y[i])
                if admitted and y_true == 1:
                    reward = 1.0
                elif admitted and y_true == 0:
                    reward = -0.2
                elif (not admitted) and y_true == 1:
                    reward = -1.0
                else:
                    reward = 0.0

                # Safety violation if a high-risk patient is not admitted
                violation = int(prisk >= safety_threshold and action == 0)

                self.logs.append(dict(
                    time=self.time,
                    admitted=admitted,
                    beds_occupied=self.beds_occupied,
                    wait=p["wait"],
                    group=int(g[i]) if len(g) > 0 else 0,
                    y_true=y_true,
                    prisk=prisk,
                    action=action,
                    reward=reward,
                    violation=violation,
                ))

            self.time += 1

        return pd.DataFrame(self.logs)


# --------------------------- Reliability Diagram ---------------------------
def reliability_plot(probs: np.ndarray, y: np.ndarray, out_path: str) -> None:
    """
    Plot a simple reliability diagram by binning predicted probabilities into 10 buckets
    and computing the empirical positive rate per bucket.

    The closer the curve to the diagonal, the better the calibration.
    """
    bins = np.linspace(0, 1, 11)
    idxs = np.digitize(probs, bins) - 1
    bin_centers, emp_rate = [], []
    for b in range(10):
        mask = (idxs == b)
        if np.any(mask):
            bin_centers.append((bins[b] + bins[b + 1]) / 2)
            emp_rate.append(np.mean(y[mask]))

    plt.figure()
    plt.plot(bin_centers, emp_rate, marker="o", label="Empirical")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Positive Rate")
    plt.title("Reliability Plot (Test)")
    plt.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)


# --------------------- Train calibrated model & simulate ---------------------
def train_and_simulate(
    prepared_csv: str,
    n_beds: int = 20,
    time_steps: int = 240,
    arrival_rate: float = 4.5,
    k_process: int = 3,
    safety_threshold: float = 0.9,
    seed: int = 42,
) -> None:
    """
    L2/L3 pipeline:
      1) Load prepared data and create a time-aware split if possible.
      2) Train GradientBoosting and calibrate with isotonic regression.
      3) Evaluate AUROC/Brier; plot a reliability diagram (test set).
      4) Run the ICU simulator with baseline policies (FCFS / Threshold / Greedy).
      5) Compute and save metrics and comparison plots.

    Parameters
    ----------
    prepared_csv : str
        Path to the cleaned dataset produced by L1.
    n_beds : int
        Number of ICU beds in the simulator.
    time_steps : int
        Number of simulation steps.
    arrival_rate : float
        Mean arrivals per time step.
    k_process : int
        Patients processed per step from head of queue.
    safety_threshold : float
        Clinical hard threshold for forced admissions.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    None
    """
    df = pd.read_csv(prepared_csv)

    # 1) Train/Test split (time-based if 'adm_time' exists)
    if "adm_time" in df.columns:
        t = pd.to_datetime(df["adm_time"], errors="coerce")
        order = np.argsort(t.fillna(t.min()).values.astype("int64"))
    else:
        order = np.arange(len(df))
        np.random.default_rng(seed).shuffle(order)

    n = len(df)
    tr_end = int(0.7 * n)  # 70% train / 30% test
    tr_idx, te_idx = order[:tr_end], order[tr_end:]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ["label", "group"]]

    Xtr = df.iloc[tr_idx][feature_cols].to_numpy(dtype=float)
    ytr = df.iloc[tr_idx]["label"].astype(int).to_numpy()
    Xte = df.iloc[te_idx][feature_cols].to_numpy(dtype=float)
    yte = df.iloc[te_idx]["label"].astype(int).to_numpy()
    gte = df.iloc[te_idx]["group"].astype(int).to_numpy() if "group" in df.columns else np.zeros(len(te_idx), dtype=int)

    # 2) Train base model + isotonic calibration
    base = GradientBoostingClassifier(random_state=seed)
    base.fit(Xtr, ytr)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(Xtr, ytr)

    # 3) Predictive evaluation (test)
    probs = clf.predict_proba(Xte)[:, 1]
    auroc = roc_auc_score(yte, probs)
    brier = brier_score_loss(yte, probs)
    print({"auroc": float(auroc), "brier": float(brier)})

    # 4) Reliability diagram
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    reliability_plot(probs, yte, "outputs/figures/predictive_reliability.png")

    # 5) Run simulator under baseline policies
    sim = ICUSimulatorMinimal(n_beds=n_beds, max_wait=4, random_state=seed, k_process=k_process)
    results: List[Dict[str, float]] = []

    policies: List[Tuple[str, str, float]] = [
        ("FCFS", "fcfs", None),             # admit if bed free
        ("SeverityThreshold", "threshold", 0.75),
        ("GreedyRisk", "greedy", 0.70),
    ]

    for name, policy, thr in policies:
        df_log = sim.run(
            steps=time_steps,
            model=clf,
            X=Xte,
            y=yte,
            g=gte,
            arrival_rate=arrival_rate,
            safety_threshold=safety_threshold,
            policy=policy,
            threshold=(thr if thr is not None else 0.5),
        )

        # Aggregate metrics
        avg_wait = df_log["wait"].mean()
        util = df_log["beds_occupied"].mean() / max(1, df_log["beds_occupied"].max())
        adm_rate = df_log["admitted"].mean()
        viol_rate = df_log["violation"].mean()
        total_reward = df_log["reward"].sum()

        # Simple group gap: mean(group=1) - mean(group=0)
        def gap(col: str) -> float:
            g0 = df_log[df_log["group"] == 0][col].mean() if (df_log["group"] == 0).any() else np.nan
            g1 = df_log[df_log["group"] == 1][col].mean() if (df_log["group"] == 1).any() else np.nan
            return float(g1 - g0) if (np.isfinite(g0) and np.isfinite(g1)) else np.nan

        res = dict(
            policy=name,
            avg_wait=float(avg_wait),
            utilization=float(util),
            admit_rate=float(adm_rate),
            safety_viol_rate=float(viol_rate),
            total_reward=float(total_reward),
            admitted_gap_g1_minus_g0=gap("admitted"),
            violation_gap_g1_minus_g0=gap("violation"),
        )
        results.append(res)
        print(name, res)

    # 6) Save comparison table and plots
    out_dir = Path("outputs")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results).set_index("policy")
    table.to_csv(out_dir / "policy_comparison.csv")
    print("\nSaved metrics to outputs/policy_comparison.csv")
    print(table)

    # Plots
    plt.figure()
    table["avg_wait"].plot(kind="bar")
    plt.ylabel("Average Wait")
    plt.title("Avg Wait")
    plt.tight_layout()
    plt.savefig("outputs/figures/policy_avg_wait.png")

    plt.figure()
    table["utilization"].plot(kind="bar")
    plt.ylabel("Utilization")
    plt.title("Utilization")
    plt.tight_layout()
    plt.savefig("outputs/figures/policy_utilization.png")

    plt.figure()
    table["safety_viol_rate"].plot(kind="bar")
    plt.ylabel("Safety Violations")
    plt.title("Safety Violations")
    plt.tight_layout()
    plt.savefig("outputs/figures/policy_safety_viol.png")

    plt.figure()
    table["total_reward"].plot(kind="bar")
    plt.ylabel("Total Reward")
    plt.title("Total Reward")
    plt.tight_layout()
    plt.savefig("outputs/figures/policy_total_reward.png")


if __name__ == "__main__":
    # Adjust path/params to your environment and run.
    train_and_simulate(
        prepared_csv=r"D:\Apply\ICU_Learn\data\prepared.csv",
        n_beds=20,
        time_steps=240,
        arrival_rate=4.5,
        k_process=3,
        safety_threshold=0.9,
        seed=42,
    )
