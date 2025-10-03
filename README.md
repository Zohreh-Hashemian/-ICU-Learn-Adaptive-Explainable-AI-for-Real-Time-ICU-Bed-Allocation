# ICU-Learn: Adaptive & Explainable AI for Real-Time ICU Bed Allocation

This repository provides a modular and reproducible proof-of-concept for **adaptive** (contextual bandit) and **explainable** (SHAP) AI to support **real-time ICU bed allocation**.

---

## Pipeline Overview

- **L1 – Data Preparation** → `prepare_dataset.py`
- **L2 – Risk Model** → `train_and_sim.py` (training + calibration + reliability)
- **L3 – Fixed Policies** → `train_and_sim.py` (FCFS / Threshold / Greedy baselines)
- **L4 – Adaptive Policy** → `layer4_bandit.py` (Safe LinUCB with safety constraint)
- **L5 – Explainability** → `layer5_xai.py` (SHAP global & local explanations)
- **L6 – Reporting** → `analyze_results.py` (compare efficiency / safety / fairness)

---

## Repository Structure

```text
ICU_Learn/
├─ README.md
├─ prepare_dataset.py
├─ train_and_sim.py
├─ layer4_bandit.py
├─ layer5_xai.py
├─ analyze_results.py
├─ data/
│  ├─ LengthOfStay.csv            # raw dataset (example; replaceable)
│  └─ prepared.csv                # produced by L1
└─ outputs/
   ├─ policy_comparison.csv
   ├─ policy_comparison_with_bandit.csv
   └─ figures/
      ├─ predictive_reliability.png
      ├─ policy_avg_wait.png
      ├─ policy_utilization.png
      ├─ policy_safety_viol.png
      ├─ policy_total_reward.png
      ├─ policies_total_reward_with_linucb.png
      ├─ shap_beeswarm.png
      ├─ shap_bar.png
      ├─ shap_waterfall_sample_0.png
      ├─ shap_waterfall_sample_1.png
      ├─ shap_waterfall_sample_2.png
      └─ shap_top10_features.csv
```

---

## Environment Setup

```bash
conda create -n icu_poc python=3.10 -y
conda activate icu_poc
pip install numpy<2 pandas scikit-learn shap matplotlib tqdm openpyxl
```

---

## How to Run

### 1. Data Preparation (L1)

```bash
python prepare_dataset.py
```

Cleans raw data, generates numeric features, and creates `label`, `group`, and `adm_time`.
**Output:** `data/prepared.csv`

---

### 2. Risk Model & Fixed Policies (L2, L3)

```bash
python train_and_sim.py
```

- Trains GradientBoosting + isotonic calibration
- Evaluates AUROC, Brier, and reliability curve
- Runs FCFS, Threshold, and Greedy policies

**Output:** `outputs/policy_comparison.csv` + figures

---

### 3. Adaptive Policy (L4)

```bash
python layer4_bandit.py
```

- Runs Safe LinUCB with safety constraint
- Appends results to baseline policies

**Output:** `outputs/policy_comparison_with_bandit.csv` + figures

---

### 4. Explainability (L5)

```bash
python layer5_xai.py
```

- Generates global SHAP plots (beeswarm, bar)
- Generates local SHAP plots (waterfall for sample patients)
- Produces top-10 feature CSV for reporting

**Output:** SHAP figures in `outputs/figures/`

---

### 5. Reporting & Analysis (L6)

```bash
python analyze_results.py
```

- Compares **Efficiency (Total Reward)**, **Safety (Violation Rate)**, and **Fairness (Group Gap)**
- Exports comparison plots

**Output:** figures in `outputs/figures/`

---

## Key Concepts

- **Calibration:** AUROC, Brier, and reliability plots to validate probability quality
- **Safety Constraint:** Automatic admission if predicted risk ≥ threshold and a bed is free
- **Adaptive Decision-making:** Safe LinUCB balances exploration vs. exploitation
- **Explainability:** SHAP global & local analyses for transparency
- **Fairness:** Group-based gap metrics (`admitted_gap`, `violation_gap`)

---

## Notes

- Use de-identified or synthetic datasets for experiments
- Outputs are reproducible with `seed=42`
- For clinical deployment: IRB approval, bias audits, drift monitoring, and **human-in-the-loop** decision support are required
