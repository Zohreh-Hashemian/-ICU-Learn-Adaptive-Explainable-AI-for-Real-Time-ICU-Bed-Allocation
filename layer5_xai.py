"""
ICU-Learn: Adaptive & Explainable AI for Real-Time ICU Bed Allocation

Module: layer5_xai.py
Purpose:
  - Train a calibrated risk model (GradientBoosting + isotonic calibration)
  - Produce SHAP-based explanations for global and local interpretability
  - Save figures and top-10 features for inclusion in the paper

Global plots (test set):
  - outputs/figures/shap_beeswarm.png
  - outputs/figures/shap_bar.png

Local plots (individual cases):
  - outputs/figures/shap_waterfall_sample_0.png
  - outputs/figures/shap_waterfall_sample_1.png
  - outputs/figures/shap_waterfall_sample_2.png

Tabular export:
  - outputs/figures/shap_top10_features.csv  (feature, mean|SHAP|)

Notes:
  - We explain the uncalibrated tree model (GradientBoosting) with TreeExplainer,
    as calibration layers can complicate exact SHAP attributions.
  - Waterfall plots may require a suitable Matplotlib backend; code falls back
    to legacy API if needed and continues gracefully on errors.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

import shap


# --------------------------- Data utilities ---------------------------
def load_train_test(prepared_csv: str, seed: int = 42) -> Tuple[pd.DataFrame, List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the prepared dataset and create a time-aware split if 'adm_time' exists.

    Returns
    -------
    df : pd.DataFrame
    feat_cols : list[str]
    Xtr, ytr : np.ndarray
    Xte, yte : np.ndarray
    """
    df = pd.read_csv(prepared_csv)

    # Time-based split if available; otherwise random
    if "adm_time" in df.columns:
        t = pd.to_datetime(df["adm_time"], errors="coerce")
        order = np.argsort(t.fillna(t.min()).values.astype("int64"))
    else:
        order = np.arange(len(df))
        rng = np.random.default_rng(seed)
        rng.shuffle(order)

    n = len(df)
    tr_end = int(0.7 * n)
    tr_idx, te_idx = order[:tr_end], order[tr_end:]

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in ["label", "group"]]

    Xtr = df.iloc[tr_idx][feat_cols].to_numpy(float)
    ytr = df.iloc[tr_idx]["label"].astype(int).to_numpy()
    Xte = df.iloc[te_idx][feat_cols].to_numpy(float)
    yte = df.iloc[te_idx]["label"].astype(int).to_numpy()

    return df, feat_cols, Xtr, ytr, Xte, yte


# --------------------------- Model utilities ---------------------------
def train_calibrated_gb(Xtr: np.ndarray, ytr: np.ndarray, seed: int = 42):
    """
    Train a GradientBoosting classifier and calibrate its probabilities via isotonic regression.

    Returns
    -------
    clf : CalibratedClassifierCV
        Calibrated model (for probability-quality).
    base : GradientBoostingClassifier
        Uncalibrated tree model (for SHAP explanations).
    """
    base = GradientBoostingClassifier(random_state=seed)
    base.fit(Xtr, ytr)

    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(Xtr, ytr)
    return clf, base


# ------------------------ SHAP helper utilities ------------------------
def to_scalar_expected_value(exp_val) -> float:
    """
    SHAP's expected_value can be scalar or array depending on version.
    Normalize to a scalar.
    """
    arr = np.array(exp_val)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.ravel()[0])


def to_2d_shap_values(shap_values) -> np.ndarray:
    """
    Normalize SHAP values to a 2D array of shape (n_samples, n_features).
    For single-output models, SHAP may return a list; return the first element.
    """
    if isinstance(shap_values, list):
        return np.array(shap_values[0])
    return np.array(shap_values)


# --------------------------------- Main ---------------------------------
if __name__ == "__main__":
    # Adjust path to your environment if needed
    prepared_csv = r"D:\Apply\ICU_Learn\data\prepared.csv"

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & train
    df, feat_cols, Xtr, ytr, Xte, yte = load_train_test(prepared_csv, seed=42)
    clf, base = train_calibrated_gb(Xtr, ytr, seed=42)

    # 2) Predictive evaluation on the test set
    probs = clf.predict_proba(Xte)[:, 1]
    auroc = roc_auc_score(yte, probs)
    brier = brier_score_loss(yte, probs)
    print({"auroc": float(auroc), "brier": float(brier)})

    # 3) SHAP on the uncalibrated tree (global + local explainability)
    explainer = shap.TreeExplainer(base)
    shap_values = explainer.shap_values(Xte)         # (n_samples, n_features) or list
    expected_value = explainer.expected_value        # scalar or array depending on SHAP version

    # 4) Global plots (beeswarm & bar)
    # Beeswarm: distribution of SHAP values per feature
    shap.summary_plot(shap_values, features=Xte, feature_names=feat_cols, show=False)
    plt.tight_layout()
    (out_dir / "shap_beeswarm.png").unlink(missing_ok=True)
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=200)
    plt.close()

    # Bar: mean absolute SHAP value per feature
    shap.summary_plot(shap_values, features=Xte, feature_names=feat_cols, plot_type="bar", show=False)
    plt.tight_layout()
    (out_dir / "shap_bar.png").unlink(missing_ok=True)
    plt.savefig(out_dir / "shap_bar.png", dpi=200)
    plt.close()

    # 5) Top-10 features by mean |SHAP| (useful for the paper's tables)
    sv_matrix = to_2d_shap_values(shap_values)               # (n_samples, n_features)
    mean_abs_shap = np.abs(sv_matrix).mean(axis=0)           # per-feature
    top_idx = np.argsort(-mean_abs_shap)[:10]
    top_features = [(feat_cols[j], float(mean_abs_shap[j])) for j in top_idx]

    print("\nTop-10 features by mean |SHAP|:")
    for name, score in top_features:
        print(f"  {name:30s}  {score:.6f}")

    pd.DataFrame({
        "feature": [feat_cols[j] for j in top_idx],
        "mean_abs_shap": [float(mean_abs_shap[j]) for j in top_idx],
    }).to_csv(out_dir / "shap_top10_features.csv", index=False)

    # 6) Local explanations: Waterfall plots for a few representative samples
    exp_val_scalar = to_scalar_expected_value(expected_value)
    for i in [0, 1, 2]:
        try:
            sv_i = sv_matrix[i]  # (n_features,)
            # Try the new API first
            try:
                shap.plots.waterfall(exp_val_scalar, sv_i, feature_names=feat_cols, max_display=12, show=False)
            except Exception:
                # Fallback to legacy API for older SHAP versions
                shap.plots._waterfall.waterfall_legacy(exp_val_scalar, sv_i, feature_names=feat_cols, max_display=12, show=False)

            plt.tight_layout()
            (out_dir / f"shap_waterfall_sample_{i}.png").unlink(missing_ok=True)
            plt.savefig(out_dir / f"shap_waterfall_sample_{i}.png", dpi=200)
            plt.close()
            print(f"[saved] shap_waterfall_sample_{i}.png")
        except Exception as e:
            print(f"[warn] waterfall for sample {i} failed: {e}")

    print(f"[done] saved SHAP figures in: {out_dir.resolve()}")
