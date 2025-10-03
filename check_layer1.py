import pandas as pd, numpy as np, os
p = r"D:\Apply\ICU_Learn\data\prepared.csv"
print(os.path.exists(p))
df = pd.read_csv(p)

# 1) نشتی نداشته باشیم
bad = [c for c in df.columns if c.lower() in ["lengthofstay","discharged","death","mortality","outcome"]]
print("leakage:", bad)

# 2) ستون‌های کلیدی
print("has label/group:", "label" in df.columns, "group" in df.columns)

# 3) همه‌چیز عددی (به‌جز adm_time)
nonnum = [c for c in df.columns if c!="adm_time" and not pd.api.types.is_numeric_dtype(df[c])]
print("non-numeric cols:", nonnum)

# 4) تعادل برچسب
print(df["label"].value_counts(normalize=True))
