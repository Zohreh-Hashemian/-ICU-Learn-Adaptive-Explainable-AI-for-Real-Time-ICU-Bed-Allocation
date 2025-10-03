"""
ICU-Learn: Adaptive & Explainable AI for Real-Time ICU Bed Allocation

Module: prepare_dataset.py
Purpose:
  - Load a raw hospital dataset (CSV/Excel) and output a modeling-ready CSV.
  - Build a binary label from LOS, create a fairness group, normalize time column,
    keep numeric features, drop leakage columns, and median-impute numerics.

Pipeline position:
  L1 Data Prep → L2/L3 Risk Model & Baselines → L4 Safe LinUCB → L5 XAI
"""

from pathlib import Path
from typing import Iterable, Optional
import pandas as pd
import numpy as np


def load_df(path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a DataFrame.

    Parameters
    ----------
    path : str
        Path to the input file (.csv, .xlsx, .xls).

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    elif p.suffix.lower() in [".xlsx", ".xls"]:
        try:
            import openpyxl  # pip install openpyxl
        except ImportError as e:
            raise RuntimeError(
                "Reading Excel requires 'openpyxl'. Install via: pip install openpyxl"
            ) from e
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def first_present(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """
    Return the first matching column name from `candidates` present in `cols`.
    Matches exact names first; if none, returns the first 'contains' match.

    Examples
    --------
    first_present(df.columns, ["los", "length_of_stay"]) -> "lengthofstay"

    Parameters
    ----------
    cols : Iterable[str]
        Available column names.
    candidates : Iterable[str]
        Preferred names (order matters).

    Returns
    -------
    Optional[str]
        Matched column name or None if not found.
    """
    cols_lower = {c.lower() for c in cols}
    # exact
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cand
    # contains
    for c in cols:
        for cand in candidates:
            if cand.lower() in str(c).lower():
                return c
    return None


def prepare_dataset(
    input_path: str,
    output_path: str,
    los_col: Optional[str] = None,
    los_threshold: Optional[float] = None,
    time_col: Optional[str] = None,
    group_from: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Clean and standardize a raw dataset and write a modeling-ready CSV.

    Steps
    -----
    1) Load CSV/Excel and infer key columns (LOS, admission time, gender/sex) if not provided.
    2) Build a binary `label` from LOS: label=1 if LOS > threshold, else 0.
       - If `los_threshold` is None, infer it:
         * If mean(LOS) is small → assume days → use 3 days
         * Otherwise assume hours → use 72 hours
    3) Build a `group` column for fairness from gender/sex: F→0, M→1 (fallback to 0 if missing).
    4) Normalize time column to `adm_time` (string acceptable; used for time-based splits).
    5) Anti-leakage: drop label-generating LOS column and apparent outcome columns
       (e.g., 'discharged', 'death', 'mortality', 'outcome').
    6) Keep only numeric features plus {label, group, adm_time}; drop all-NaN/constant columns.
    7) Median-impute numeric missing values (critical for sklearn pipelines).
    8) Save the cleaned table to `output_path`.

    Parameters
    ----------
    input_path : str
        Raw CSV/Excel path.
    output_path : str
        Destination CSV path.
    los_col : Optional[str]
        LOS column name; inferred if None.
    los_threshold : Optional[float]
        Threshold to binarize LOS; inferred if None.
    time_col : Optional[str]
        Admission/record time column; inferred if None.
    group_from : Optional[str]
        Gender/sex/group source column; inferred if None.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    None
    """
    # 1) Load
    df = load_df(input_path)
    if verbose:
        print("=== Columns ===")
        print(list(df.columns))
        print(df.head(3))

    # 2) Infer column names if not provided
    if los_col is None:
        los_col = first_present(
            df.columns,
            ["los", "length_of_stay", "lengthofstay", "los_hours", "stay", "los_days"],
        )
        if verbose:
            print(f"[info] guessed LOS column: {los_col}")
    if time_col is None:
        time_col = first_present(
            df.columns, ["adm_time", "vdate", "charttime", "time", "admittime", "intime"]
        )
        if verbose:
            print(f"[info] guessed time column: {time_col}")
    if group_from is None:
        group_from = first_present(df.columns, ["group", "gender", "sex"])
        if verbose:
            print(f"[info] guessed group source: {group_from}")

    out = df.copy()

    # 3) Build label from LOS (if not already present)
    if "label" not in out.columns:
        if los_col is None:
            if verbose:
                print("[warn] No LOS-like column found. Creating label=0 (dummy). "
                      "Later set los_col/threshold explicitly.")
            out["label"] = 0
        else:
            col = pd.to_numeric(out[los_col], errors="coerce")
            if los_threshold is not None:
                thr = los_threshold
            else:
                # Heuristic: small mean → likely days; else hours
                mean_val = col.dropna().mean()
                thr = 3.0 if (pd.notna(mean_val) and mean_val < 20) else 72.0
                if verbose:
                    unit = "days" if thr == 3.0 else "hours"
                    print(f"[info] inferred LOS threshold = {thr} ({unit})")
            out["label"] = (col > thr).astype(int)

    # 5) Anti-leakage: drop columns that can leak the target
    leakage_cols = []
    if los_col is not None and los_col in out.columns:
        leakage_cols.append(los_col)
    for c in ["discharged", "death", "mortality", "outcome"]:
        if c in out.columns:
            leakage_cols.append(c)
    if leakage_cols:
        if verbose:
            print(f"[leakage] dropping target-related columns: {leakage_cols}")
        out = out.drop(columns=leakage_cols, errors="ignore")

    # 4) Build fairness group from gender/sex (F→0, M→1)
    if "group" not in out.columns:
        if group_from and group_from in out.columns:
            s = out[group_from]
            if s.dtype == object:
                up = s.astype(str).strip().str.upper()
                # Normalize common values; fallback to 0
                out["group"] = (
                    up.map({"M": 1, "MALE": 1, "F": 0, "FEMALE": 0})
                      .fillna(0)
                      .astype(int)
                )
            else:
                out["group"] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
        else:
            out["group"] = 0

    # 5) Normalize time column name
    if time_col and time_col in out.columns:
        out = out.rename(columns={time_col: "adm_time"})

    # 6) Keep numeric features + {label, group, adm_time}
    num_df = out.select_dtypes(include=[np.number]).copy()
    for c in ["label", "group"]:
        if c not in num_df.columns and c in out.columns:
            num_df[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    if "adm_time" in out.columns:
        # Keep as-is (string is fine; used only for splitting)
        num_df["adm_time"] = out["adm_time"]

    # 7) Drop all-NaN or constant columns
    drop_cols = []
    for c in list(num_df.columns):
        series = num_df[c]
        if series.notna().sum() == 0 or series.nunique(dropna=True) <= 1:
            drop_cols.append(c)
    if drop_cols and verbose:
        print(f"[note] dropping all-NaN/constant cols: {drop_cols}")
    num_df = num_df.drop(columns=drop_cols, errors="ignore")

    # 8) Median-impute numeric columns (critical for sklearn)
    numeric_cols = num_df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        if c != "label":  # never impute the label
            median_val = num_df[c].median()
            num_df[c] = num_df[c].fillna(median_val)

    # 9) Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    num_df.to_csv(output_path, index=False)
    if verbose:
        print(f"[done] wrote: {output_path}, shape={num_df.shape}")
        print(num_df.head(5))


if __name__ == "__main__":
    # Adjust paths for your environment and run.
    print("=== Running prepare_dataset main block ===")
    prepare_dataset(
        input_path=r"D:\Apply\ICU_Learn\data\LengthOfStay.csv",   # raw file
        output_path=r"D:\Apply\ICU_Learn\data\prepared.csv",      # cleaned output
        los_col="lengthofstay",        # LOS column
        los_threshold=3,               # dataset is in days → threshold = 3 days
        time_col="vdate",              # time column → standardized to 'adm_time'
        group_from="gender",           # fairness group (F→0, M→1)
        verbose=True,
    )
