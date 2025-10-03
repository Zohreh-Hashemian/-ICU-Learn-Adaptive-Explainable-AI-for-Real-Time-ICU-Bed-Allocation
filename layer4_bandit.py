"""
ICU-Learn: Adaptive & Explainable AI for Real-Time ICU Bed Allocation

Module: layer4_bandit.py
Purpose:
  - Train a calibrated risk model (GradientBoosting + isotonic)
  - Simulate an ICU queue with a Safe LinUCB policy (adaptive)
  - Enforce a hard clinical safety constraint (force admit if risk ≥ threshold)
  - Log efficiency / safety / fairness metrics and save results

Notes:
  Pipeline: L1 Data Prep → L2/L3 Risk Model & Baselines → L4 Safe LinUCB → L5 XAI
"""

from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss


# ----------------------------- Safe LinUCB -----------------------------
class SafeLinUCB:
    """
    Linear UCB bandit with two actions: 0 = defer/reject, 1 = admit.

    Context vector:
      x = [bias, predicted_risk, normalized_wait, normalized_free_beds, group]

    Scoring:
      score(a) = θ_a^T x + α * sqrt(x^T A_a^{-1} x)

    Updates:
      A_a ← A_a + x x^T
      b_a ← b_a + r x   (with reward r clipped to [-1, +1])
    """
    def __init__(self, d: int, alpha: float = 0.5, n_actions: int = 2, seed: int = 42):
        self.d = d
        self.alpha = alpha
        self.n_actions = n_actions
        self.rs = np.random.RandomState(seed)
        # Per-action ridge matrices and linear terms
        self.As = [np.eye(d) for _ in range(n_actions)]
        self.bs = [np.zeros((d, 1)) for _ in range(n_actions)]

    def _theta(self, a: int):
        A_inv = np.linalg.inv(self.As[a])
        return A_inv @ self.bs[a], A_inv  # (theta, A_inv)

    def select(self, x: np.ndarray) -> int:
        """
        Select an action via LinUCB. x must be shaped (d, 1).
        """
        scores = []
        for a in range(self.n_actions):
            theta, A_inv = self._theta(a)
            mean = float((theta.T @ x)[0, 0])
            ucb = self.alpha * float(np.sqrt(x.T @ A_inv @ x))
            scores.append(mean + ucb)
        return int(np.argmax(scores))

    def update(self, a: int, x: np.ndarray, r: float) -> None:
        """
        Update parameters with observation (x, a, r).
        Reward is clipped to [-1, +1] for numerical stability.
        """
        r = np.clip(r, -1.0, 1.0)
        self.As[a] += x @ x.T
        self.bs[a] += r * x


# ---------------------- Train calibrated risk model --------------------
def train_calibrated_model(prepared_csv: str, seed: int = 42):
    """
    Train GradientBoosting on training split, then calibrate (isotonic).
    Returns the calibrated classifier and the held-out arrays.
    """
    df = pd.read_csv(prepared_csv)

    # Time-based split if 'adm_time' exists; otherwise random split
    if "adm_time" in df.columns:
        t = pd.to_datetime(df["adm_time"], errors="coerce")
        order = np.argsort(t.fillna(t.min()).values.astype("int64"))
    else:
        order = np.arange(len(df))
        np.random.default_rng(seed).shuffle(order)

    n = len(df)
    tr_end = int(0.7 * n)
    tr_idx, te_idx = order[:tr_end], order[tr_end:]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ["label", "group"]]

    Xtr = df.iloc[tr_idx][feature_cols].to_numpy(dtype=float)
    ytr = df.iloc[tr_idx]["label"].astype(int).to_numpy()
    Xte = df.iloc[te_idx][feature_cols].to_numpy(dtype=float)
    yte = df.iloc[te_idx]["label"].astype(int).to_numpy()
    gte = df.iloc[te_idx]["group"].astype(int).to_numpy() if "group" in df.columns else np.zeros(len(te_idx), dtype=int)

    base = GradientBoostingClassifier(random_state=seed)
    base.fit(Xtr, ytr)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(Xtr, ytr)

    probs = clf.predict_proba(Xte)[:, 1]
    auroc = roc_auc_score(yte, probs)
    brier = brier_score_loss(yte, probs)
    print({"auroc": float(auroc), "brier": float(brier)})

    return clf, Xte, yte, gte


# ---------------------------- LinUCB simulator -------------------------
def run_linucb_sim(
    prepared_csv: str,
    n_beds: int = 20,
    max_wait: int = 4,
    time_steps: int = 240,
    arrival_rate: float = 4.5,
    k_process: int = 4,
    safety_threshold: float = 0.90,
    alpha: float = 0.30,
    seed: int = 42,
):
    """
    Run a queueing simulation with a Safe LinUCB policy.

    Safety constraint (hard rule):
      If predicted risk ≥ safety_threshold and there is a free bed ⇒ force admit.

    Context fed to LinUCB:
      x = [1.0, prisk, wait_norm, beds_free_norm, group]
    """
    clf, X, y, g = train_calibrated_model(prepared_csv, seed=seed)
    rs = np.random.RandomState(seed)

    beds_occupied = 0
    queue = deque()
    logs = []
    cursor = 0
    time = 0

    bandit = SafeLinUCB(d=5, alpha=alpha, n_actions=2, seed=seed)

    def arrivals(lam: float) -> int:
        return rs.poisson(lam)

    for _ in range(time_steps):
        # Random discharges proportional to current occupancy
        discharges = int(max(0, rs.normal(0.1, 0.05) * beds_occupied))
        beds_occupied = max(0, beds_occupied - discharges)

        # Patient arrivals
        for _ in range(arrivals(arrival_rate)):
            if cursor < len(X):
                queue.append({"idx": cursor, "wait": 0})
                cursor += 1

        # Increase waiting time for all queued patients
        for p in queue:
            p["wait"] += 1

        # Process up to k patients from the head of the queue
        for _ in range(min(k_process, len(queue))):
            p = queue[0]
            i = p["idx"]
            beds_free = n_beds - beds_occupied

            # Predicted risk from the calibrated model
            prisk = float(clf.predict_proba(X[i:i + 1])[:, 1][0])

            # Hard safety constraint: force admit if high risk and bed available
            if prisk >= safety_threshold and beds_free > 0:
                action = 1
                used_bandit = False
            else:
                # Build bandit context
                wait_norm = min(p["wait"] / max_wait, 1.0)
                beds_free_norm = max(beds_free, 0) / max(1, n_beds)
                group_val = int(g[i]) if len(g) > 0 else 0
                x = np.array([[1.0], [prisk], [wait_norm], [beds_free_norm], [group_val]], dtype=float)

                # If no bed is free, action must be 0; otherwise query LinUCB
                if beds_free <= 0:
                    action = 0
                    used_bandit = False
                else:
                    action = bandit.select(x)  # 0 or 1
                    used_bandit = True

            # Apply action
            admitted = 0
            if action == 1 and beds_free > 0:
                admitted = 1
                beds_occupied += 1
                queue.popleft()
            elif action == 0 and p["wait"] >= max_wait:
                # Timed-out patient leaves the queue
                queue.popleft()

            # --------------------- Reward shaping (tuned) ---------------------
            #  - True-positive admission (admit & y=1):  +1.0
            #  - False-positive admission (admit & y=0):  0.0
            #  - False-negative (not admit & y=1):       -0.8
            #  - True-negative (not admit & y=0):         0.0
            # Intuition: emphasize safety while keeping efficiency by not
            # over-penalizing low-risk admissions.
            y_true = int(y[i])
            if admitted and y_true == 1:
                reward = 1.0
            elif admitted and y_true == 0:
                reward = 0.0
            elif (not admitted) and y_true == 1:
                reward = -0.8
            else:
                reward = 0.0

            # A safety violation occurs if a high-risk patient is not admitted
            violation = int(prisk >= safety_threshold and action == 0)

            # Update bandit only when its own decision was used
            if used_bandit:
                bandit.update(
                    action,
                    np.array([[1.0], [prisk], [wait_norm], [beds_free_norm], [group_val]], dtype=float),
                    reward
                )

            # Logging
            logs.append(dict(
                time=time,
                admitted=admitted,
                beds_occupied=beds_occupied,
                wait=p["wait"],
                group=int(g[i]) if len(g) > 0 else 0,
                y_true=y_true,
                prisk=prisk,
                action=action,
                reward=reward,
                violation=violation,
            ))

        time += 1

    # Aggregate metrics
    df_log = pd.DataFrame(logs)
    avg_wait = df_log["wait"].mean()
    util = df_log["beds_occupied"].mean() / max(1, df_log["beds_occupied"].max())
    adm_rate = df_log["admitted"].mean()
    viol_rate = df_log["violation"].mean()
    total_reward = df_log["reward"].sum()

    def gap(col: str) -> float:
        g0 = df_log[df_log["group"] == 0][col].mean() if (df_log["group"] == 0).any() else np.nan
        g1 = df_log[df_log["group"] == 1][col].mean() if (df_log["group"] == 1).any() else np.nan
        return float(g1 - g0) if np.isfinite(g0) and np.isfinite(g1) else np.nan

    res = dict(
        policy="LinUCB_Safe",
        avg_wait=float(avg_wait),
        utilization=float(util),
        admit_rate=float(adm_rate),
        safety_viol_rate=float(viol_rate),
        total_reward=float(total_reward),
        admitted_gap_g1_minus_g0=gap("admitted"),
        violation_gap_g1_minus_g0=gap("violation"),
    )

    print("LinUCB_Safe", res)

    # Persist results next to previous baselines if present
    out_dir = Path("outputs")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    prev_csv = out_dir / "policy_comparison.csv"
    if prev_csv.exists():
        prev = pd.read_csv(prev_csv)
        combined = pd.concat([prev, pd.DataFrame([res])], ignore_index=True)
        combined.set_index("policy", inplace=True)
        combined.to_csv(out_dir / "policy_comparison_with_bandit.csv")
        print("\nSaved metrics to outputs/policy_comparison_with_bandit.csv")
        print(combined)

        # Simple bar plot for total_reward comparison
        plt.figure()
        combined["total_reward"].plot(kind="bar")
        plt.ylabel("Total Reward")
        plt.title("Policies (with LinUCB)")
        plt.tight_layout()
        plt.savefig("outputs/figures/policies_total_reward_with_linucb.png")
    else:
        pd.DataFrame([res]).set_index("policy").to_csv(out_dir / "policy_comparison_with_bandit.csv")
        print("\nSaved metrics to outputs/policy_comparison_with_bandit.csv (only bandit result present)")

    return df_log, res


if __name__ == "__main__":
    # Configure and run a single simulation (seeded, reproducible)
    df_log, res = run_linucb_sim(
        prepared_csv=r"D:\Apply\ICU_Learn\data\prepared.csv",
        n_beds=20,
        max_wait=4,
        time_steps=240,
        arrival_rate=4.5,
        k_process=4,
        safety_threshold=0.90,  # tuned in experiments
        alpha=0.30,             # exploration parameter
        seed=42,
    )
