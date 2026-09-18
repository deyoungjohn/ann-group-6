"""
Phase 1 & 2 — Data Audit for P6: Forest Cover Type
====================================================
Loads the Covertype dataset, performs a comprehensive audit,
and saves all findings to CSV / text for the report.

Run:  python phase1_2_data_audit.py
"""

import pandas as pd
import numpy as np
import os
import sys
import io
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Configuration ──────────────────────────────────────────────────
DATA_FILE = r"c:\Users\DELL\Downloads\ANN\covtype.data.gz"
OUTPUT_DIR = r"c:\Users\DELL\Downloads\ANN\audit_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Column names (from covtype.info) ──────────────────────────────
QUANT_COLS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

WILDERNESS_COLS = [f"Wilderness_Area_{i}" for i in range(1, 5)]

SOIL_COLS = [f"Soil_Type_{i}" for i in range(1, 41)]

TARGET_COL = "Cover_Type"

ALL_COLS = QUANT_COLS + WILDERNESS_COLS + SOIL_COLS + [TARGET_COL]

COVER_TYPE_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}

# ══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 2 — DATA AUDIT: Forest Cover Type (P6)")
print("=" * 70)
print(f"\nTimestamp  : {datetime.now().isoformat()}")
print(f"Data file : {DATA_FILE}")

df = pd.read_csv(DATA_FILE, header=None, names=ALL_COLS)

print(f"\n{'─' * 70}")
print("1. BASIC SHAPE")
print(f"{'─' * 70}")
print(f"  Rows    : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")
print(f"  Memory  : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB (in-memory)")

# ── Data types ────────────────────────────────────────────────────
print(f"\n{'─' * 70}")
print("2. FEATURE TYPES")
print(f"{'─' * 70}")
print(f"  Quantitative features   : {len(QUANT_COLS):>3d}  (continuous)")
print(f"  Wilderness area (binary): {len(WILDERNESS_COLS):>3d}  (one-hot)")
print(f"  Soil type (binary)      : {len(SOIL_COLS):>3d}  (one-hot)")
print(f"  Target column           :   1  (integer 1–7)")
print(f"  Total columns           : {df.shape[1]:>3d}")
print(f"\n  Pandas dtypes:")
for dtype, count in df.dtypes.value_counts().items():
    print(f"    {dtype}: {count}")

# ── Missing values ────────────────────────────────────────────────
print(f"\n{'─' * 70}")
print("3. MISSING VALUES")
print(f"{'─' * 70}")
total_missing = df.isnull().sum().sum()
print(f"  Total missing cells: {total_missing}")
if total_missing == 0:
    print("  ✓ Confirmed: NO missing values in the dataset.")
else:
    print("  ✗ WARNING: Missing values found!")
    print(df.isnull().sum()[df.isnull().sum() > 0])

# ── Duplicates ────────────────────────────────────────────────────
print(f"\n{'─' * 70}")
print("4. DUPLICATE ROWS")
print(f"{'─' * 70}")
n_dupes = df.duplicated().sum()
print(f"  Exact duplicate rows: {n_dupes:,}")
if n_dupes > 0:
    pct = n_dupes / len(df) * 100
    print(f"  ({pct:.2f}% of the dataset)")
    print("  NOTE: These represent different 30×30m cells that happen to share")
    print("  identical cartographic measurements. Removing them would discard")
    print("  valid geographic observations. We retain them.")
else:
    print("  ✓ No duplicate rows found.")

# ── Class distribution ────────────────────────────────────────────
print(f"\n{'─' * 70}")
print("5. CLASS DISTRIBUTION (Target: Cover_Type)")
print(f"{'─' * 70}")
class_counts = df[TARGET_COL].value_counts().sort_index()
class_pct = (class_counts / len(df) * 100).round(2)

class_df = pd.DataFrame({
    "Cover_Type_ID": class_counts.index,
    "Name": [COVER_TYPE_NAMES[i] for i in class_counts.index],
    "Count": class_counts.values,
    "Percentage": class_pct.values,
})
print(class_df.to_string(index=False))
print(f"\n  Total: {class_counts.sum():,}")

# Imbalance ratio
majority = class_counts.max()
minority = class_counts.min()
print(f"\n  Majority class : {COVER_TYPE_NAMES[class_counts.idxmax()]} ({majority:,} = {majority/len(df)*100:.2f}%)")
print(f"  Minority class : {COVER_TYPE_NAMES[class_counts.idxmin()]} ({minority:,} = {minority/len(df)*100:.2f}%)")
print(f"  Imbalance ratio: {majority / minority:.1f} : 1")

# Cross-check with UCI published numbers
print(f"\n  Cross-check against UCI published counts:")
uci_counts = {1: 211840, 2: 283301, 3: 35754, 4: 2747, 5: 9493, 6: 17367, 7: 20510}
all_match = True
for ct_id, uci_n in uci_counts.items():
    our_n = class_counts[ct_id]
    match = "✓" if our_n == uci_n else "✗ MISMATCH"
    if our_n != uci_n:
        all_match = False
    print(f"    Type {ct_id} ({COVER_TYPE_NAMES[ct_id]:20s}): UCI={uci_n:>7,}  Ours={our_n:>7,}  {match}")
if all_match:
    print("  ✓ All counts match UCI published values.")

# ── Quantitative feature statistics ──────────────────────────────
print(f"\n{'─' * 70}")
print("6. QUANTITATIVE FEATURE STATISTICS")
print(f"{'─' * 70}")
quant_stats = df[QUANT_COLS].describe().T
quant_stats["range"] = quant_stats["max"] - quant_stats["min"]
quant_stats["iqr"] = quant_stats["75%"] - quant_stats["25%"]
print(quant_stats[["mean", "std", "min", "25%", "50%", "75%", "max", "range"]].round(2).to_string())

# Check for suspicious zeros or sentinel values in quantitative columns
print(f"\n  Checking for zeros in quantitative columns:")
for col in QUANT_COLS:
    n_zeros = (df[col] == 0).sum()
    if n_zeros > 0:
        pct = n_zeros / len(df) * 100
        print(f"    {col:45s}: {n_zeros:>7,} zeros ({pct:.1f}%)")

# ── Binary indicator validation ───────────────────────────────────
print(f"\n{'─' * 70}")
print("7. BINARY INDICATOR VALIDATION")
print(f"{'─' * 70}")

# Wilderness area: each row should have exactly one 1
wa_sums = df[WILDERNESS_COLS].sum(axis=1)
print(f"  Wilderness Area columns (4):")
print(f"    Rows with exactly one '1': {(wa_sums == 1).sum():,} / {len(df):,}")
print(f"    Rows with zero '1's      : {(wa_sums == 0).sum():,}")
print(f"    Rows with multiple '1's   : {(wa_sums > 1).sum():,}")
print(f"    Distribution:")
for col in WILDERNESS_COLS:
    n = df[col].sum()
    print(f"      {col}: {n:>7,} ({n/len(df)*100:.2f}%)")

# Soil type: each row should have exactly one 1
soil_sums = df[SOIL_COLS].sum(axis=1)
print(f"\n  Soil Type columns (40):")
print(f"    Rows with exactly one '1': {(soil_sums == 1).sum():,} / {len(df):,}")
print(f"    Rows with zero '1's      : {(soil_sums == 0).sum():,}")
print(f"    Rows with multiple '1's   : {(soil_sums > 1).sum():,}")

# Which soil types are actually used?
soil_usage = df[SOIL_COLS].sum().sort_values(ascending=False)
unused_soils = soil_usage[soil_usage == 0]
print(f"    Soil types with zero observations: {len(unused_soils)}")
if len(unused_soils) > 0:
    print(f"      {', '.join(unused_soils.index.tolist())}")
rare_soils = soil_usage[soil_usage < 100]
print(f"    Soil types with < 100 observations: {len(rare_soils)}")
if len(rare_soils) > 0:
    for s, c in rare_soils.items():
        print(f"      {s}: {c}")

# ── Quantitative feature correlations ─────────────────────────────
print(f"\n{'─' * 70}")
print("8. QUANTITATIVE FEATURE CORRELATIONS (|r| > 0.5)")
print(f"{'─' * 70}")
corr = df[QUANT_COLS].corr()
high_corr_pairs = []
for i in range(len(QUANT_COLS)):
    for j in range(i + 1, len(QUANT_COLS)):
        r = corr.iloc[i, j]
        if abs(r) > 0.5:
            high_corr_pairs.append((QUANT_COLS[i], QUANT_COLS[j], r))

if high_corr_pairs:
    for f1, f2, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
        print(f"  {f1:45s} ↔ {f2:45s}  r = {r:+.3f}")
else:
    print("  No pairs with |r| > 0.5")

# ── Leakage risk check ───────────────────────────────────────────
print(f"\n{'─' * 70}")
print("9. LEAKAGE RISK CHECK")
print(f"{'─' * 70}")
print("  ✓ No identifier/index column (data has no row IDs)")
print("  ✓ No post-outcome variables (all features are cartographic, measured before cover type)")
print("  ✓ No temporal ordering issues (spatial data, not time series)")
print("  NOTE: Blackard & Dean (1999) scaled inputs across train+val+test combined.")
print("        This is a minor data leak. Our pipeline will scale inside each CV fold.")

# ── Anchor paper key findings (Phase 1 summary) ──────────────────
print(f"\n{'─' * 70}")
print("10. PHASE 1 — ANCHOR PAPER KEY FINDINGS (Blackard & Dean, 1999)")
print(f"{'─' * 70}")
print("""
  Paper: "Comparative accuracies of artificial neural networks and
          discriminant analysis in predicting forest cover types
          from cartographic variables"
  
  Methods compared:
    • Feedforward ANN (54-120-7, backpropagation, LR=0.05, MR=0.5)
    • Linear Discriminant Analysis (LDA)
    • Quadratic DA (QDA) — unstable with qualitative vars, limited use
  
  Data protocol:
    • Fixed split: 11,340 train (1,620/class, balanced) / 3,780 val
      (540/class) / 565,892 test (natural proportions)
    • Inputs scaled to [0,1] across ALL three sets combined (leak!)
    • No cross-validation — single fixed split
  
  Key results:
    • ANN overall accuracy: 70.58% (mean of 30 runs: 70.52%, 95% CI: 70.26–70.80%)
    • LDA overall accuracy: 58.38%
    • ANN outperformed LDA on every individual cover type
    • Main confusions: ponderosa pine ↔ Douglas-fir ↔ cottonwood/willow
      (geographic proximity); krummholz → spruce/fir (elevation);
      aspen ↔ lodgepole pine (same altitudinal zone)
  
  Protocol differences we will note:
    1. They used a fixed split; we use stratified k-fold CV (more robust)
    2. They scaled globally; we scale inside each fold (no leak)
    3. They balanced the training set; we preserve natural proportions + stratify
    4. They compared 2 models; we compare ≥4 from ≥3 families
    5. They reported only overall accuracy; we report Macro-F1 + per-class recall
""")

# ── Save outputs ──────────────────────────────────────────────────
print(f"\n{'─' * 70}")
print("SAVING AUDIT OUTPUTS")
print(f"{'─' * 70}")

# Save class distribution
class_df.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"), index=False)
print(f"  ✓ class_distribution.csv")

# Save quantitative stats
quant_stats.round(4).to_csv(os.path.join(OUTPUT_DIR, "quantitative_feature_stats.csv"))
print(f"  ✓ quantitative_feature_stats.csv")

# Save correlation matrix
corr.round(4).to_csv(os.path.join(OUTPUT_DIR, "quantitative_correlations.csv"))
print(f"  ✓ quantitative_correlations.csv")

# Save soil type usage
soil_usage.to_csv(os.path.join(OUTPUT_DIR, "soil_type_usage.csv"), header=["count"])
print(f"  ✓ soil_type_usage.csv")

# Save binary indicator check
indicator_check = {
    "wilderness_rows_exactly_one": int((wa_sums == 1).sum()),
    "wilderness_rows_zero": int((wa_sums == 0).sum()),
    "wilderness_rows_multiple": int((wa_sums > 1).sum()),
    "soil_rows_exactly_one": int((soil_sums == 1).sum()),
    "soil_rows_zero": int((soil_sums == 0).sum()),
    "soil_rows_multiple": int((soil_sums > 1).sum()),
    "unused_soil_types": len(unused_soils),
    "total_rows": int(len(df)),
    "total_columns": int(df.shape[1]),
    "total_missing_cells": int(total_missing),
    "total_duplicate_rows": int(n_dupes),
}
pd.Series(indicator_check).to_csv(os.path.join(OUTPUT_DIR, "data_integrity_checks.csv"), header=["value"])
print(f"  ✓ data_integrity_checks.csv")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("=" * 70)
print("PHASE 1 & 2 COMPLETE")
print("=" * 70)
