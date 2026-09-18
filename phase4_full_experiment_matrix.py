"""
Phase 4 — Full Experiment Matrix Execution
===========================================
Executes the controlled empirical comparison for P6 (Forest Cover Type):
- Protocol: Stratified 5-Fold × 3 Repeats (15 paired outer folds)
- Sample: Documented N=50,000 stratified subsample (seed=42)
- Preprocessing: StandardScaler fitted strictly inside each training fold (cols 0-9),
                 44 binary indicators passed through unscaled (cols 10-53).
- Algorithms (5 models from 4 families + Majority Class baseline):
    1. MajorityClass_ZeroRule (Trivial reference)
    2. LogisticRegression (Linear/Probabilistic)
    3. DecisionTree (Tree/Rule-based)
    4. RandomForest (Ensemble)
    5. LightGBM (Ensemble)
    6. MLP (Neural/Kernel)
- Hyperparameters: Frozen from Phase 3 inner-CV tuning (appendix_c_frozen_hyperparameters.json)
- Outputs:
    - raw_fold_results.csv (Per-fold, per-algorithm, per-metric results)
    - summary_results_table.csv (Mean ± Std for Table 4)
    - per_class_recall_summary.csv (Mean ± Std recall per cover type)
    - aggregate_confusion_matrices.json (Aggregated confusion matrix for Figure 2)
    - requirements.txt (Pinned library versions for reproducibility)

Run: python phase4_full_experiment_matrix.py
"""

import os
import sys
import io
import time
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Fix console encoding with line buffering for real-time output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True, errors="replace")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    recall_score,
    confusion_matrix,
)

# ── Paths and Config ──────────────────────────────────────────────
DATA_FILE = r"c:\Users\DELL\Downloads\ANN\covtype.data.gz"
FROZEN_PARAMS_FILE = r"c:\Users\DELL\Downloads\ANN\protocol_outputs\appendix_c_frozen_hyperparameters.json"
RESULTS_DIR = r"c:\Users\DELL\Downloads\ANN\experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
SUBSAMPLE_SIZE = 50000
N_SPLITS = 5
N_REPEATS = 3
N_FOLDS = N_SPLITS * N_REPEATS

COVER_TYPE_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}

FAMILY_MAP = {
    "MajorityClass_ZeroRule": "Trivial Baseline",
    "LogisticRegression": "Linear/Probabilistic",
    "DecisionTree": "Tree/Rule-based",
    "RandomForest": "Ensemble",
    "LightGBM": "Ensemble",
    "MLP": "Neural/Kernel",
}

def main():
    print("=" * 80)
    print("PHASE 4 — FULL EXPERIMENT MATRIX EXECUTION (P6: FOREST COVER TYPE)")
    print("=" * 80)
    print(f"Start Timestamp     : {datetime.now().isoformat()}")
    print(f"Hardware            : Intel 16 logical cores, 32 GB RAM, Windows 11")
    print(f"Data File           : {DATA_FILE}")
    print(f"Subsample Size      : {SUBSAMPLE_SIZE:,} (stratified, seed={RANDOM_SEED})")
    print(f"Outer Validation    : Stratified {N_SPLITS}-Fold × {N_REPEATS} Repeats ({N_FOLDS} paired folds)")
    print(f"Results Directory   : {RESULTS_DIR}")

    # ── 1. Load Data and Extract Documented Subsample ─────────────
    print(f"\n{'─' * 80}")
    print("1. LOADING RAW DATA AND EXTRACTING DOCUMENTED SUBSAMPLE")
    print(f"{'─' * 80}")
    
    df = pd.read_csv(DATA_FILE, header=None)
    X_raw = df.iloc[:, :54].values
    y_raw = df.iloc[:, 54].values

    X_sub, _, y_sub, _ = train_test_split(
        X_raw, y_raw,
        train_size=SUBSAMPLE_SIZE,
        stratify=y_raw,
        random_state=RANDOM_SEED,
    )
    print(f"Loaded {X_sub.shape[0]:,} samples × {X_sub.shape[1]} features.")
    print("Class counts in subsample:")
    for ct_id in sorted(COVER_TYPE_NAMES.keys()):
        cnt = (y_sub == ct_id).sum()
        pct = cnt / len(y_sub) * 100
        print(f"  Class {ct_id} ({COVER_TYPE_NAMES[ct_id]:20s}): {cnt:>6,} ({pct:>5.2f}%)")

    # ── 2. Load Frozen Hyperparameters ────────────────────────────
    print(f"\n{'─' * 80}")
    print("2. LOADING FROZEN HYPERPARAMETERS (PHASE 3 ARTIFACT)")
    print(f"{'─' * 80}")
    
    with open(FROZEN_PARAMS_FILE, "r") as f:
        frozen_params = json.load(f)
    
    print("Loaded configurations from appendix_c_frozen_hyperparameters.json:")
    for algo, params in frozen_params.items():
        print(f"  • {algo:20s}: {params}")

    # ── 3. Instantiate Algorithms with Preprocessor ───────────────
    print(f"\n{'─' * 80}")
    print("3. BUILDING LEAK-FREE PIPELINES")
    print(f"{'─' * 80}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(range(10))),
            ("bin", "passthrough", list(range(10, 54))),
        ]
    )

    models = {
        "MajorityClass_ZeroRule": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            C=frozen_params["LogisticRegression"]["C"],
            penalty=frozen_params["LogisticRegression"]["penalty"],
            solver=frozen_params["LogisticRegression"]["solver"],
            max_iter=frozen_params["LogisticRegression"]["max_iter"],
            random_state=RANDOM_SEED,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=frozen_params["DecisionTree"]["max_depth"],
            min_samples_split=frozen_params["DecisionTree"]["min_samples_split"],
            min_samples_leaf=frozen_params["DecisionTree"]["min_samples_leaf"],
            criterion=frozen_params["DecisionTree"]["criterion"],
            random_state=RANDOM_SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=frozen_params["RandomForest"]["n_estimators"],
            max_depth=frozen_params["RandomForest"]["max_depth"],
            max_features=frozen_params["RandomForest"]["max_features"],
            min_samples_split=frozen_params["RandomForest"]["min_samples_split"],
            min_samples_leaf=frozen_params["RandomForest"]["min_samples_leaf"],
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=frozen_params["LightGBM"]["n_estimators"],
            num_leaves=frozen_params["LightGBM"]["num_leaves"],
            learning_rate=frozen_params["LightGBM"]["learning_rate"],
            max_depth=frozen_params["LightGBM"]["max_depth"],
            subsample=frozen_params["LightGBM"]["subsample"],
            colsample_bytree=frozen_params["LightGBM"]["colsample_bytree"],
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=tuple(frozen_params["MLP"]["hidden_layer_sizes"]),
            activation=frozen_params["MLP"]["activation"],
            alpha=frozen_params["MLP"]["alpha"],
            learning_rate_init=frozen_params["MLP"]["learning_rate_init"],
            max_iter=frozen_params["MLP"]["max_iter"],
            early_stopping=frozen_params["MLP"]["early_stopping"],
            random_state=RANDOM_SEED,
        ),
    }

    # Wrap models needing scaling in pipeline
    pipelines = {}
    for name, model in models.items():
        if name == "MajorityClass_ZeroRule":
            pipelines[name] = model
        else:
            pipelines[name] = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model),
            ])

    # ── 4. Execute 15 Outer Folds ──────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"4. EXECUTING 15 OUTER FOLDS (5-FOLD × 3 REPEATS)")
    print(f"{'─' * 80}")

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)

    raw_results = []
    # Accumulate confusion matrices across all 15 evaluations per algorithm
    cum_confusion_matrices = {name: np.zeros((7, 7), dtype=int) for name in pipelines.keys()}

    fold_counter = 0
    t_experiment_start = time.time()

    for repeat_idx in range(N_REPEATS):
        for split_idx in range(N_SPLITS):
            fold_counter += 1
            fold_id = f"Rep{repeat_idx+1}_Fold{split_idx+1}"
            
            # Extract train/test splits for this fold
            for i, (tr_idx, te_idx) in enumerate(rskf.split(X_sub, y_sub)):
                if i == fold_counter - 1:
                    break
            
            X_tr, y_tr = X_sub[tr_idx], y_sub[tr_idx]
            X_te, y_te = X_sub[te_idx], y_sub[te_idx]

            print(f"\n── Fold {fold_counter:>2d}/{N_FOLDS} [{fold_id}] (Train: {len(tr_idx):,}, Test: {len(te_idx):,}) ──")

            for algo_name, pipe in pipelines.items():
                # Measure training time (including fold-level scaling)
                t0 = time.time()
                pipe.fit(X_tr, y_tr)
                t_fit = time.time() - t0

                # Measure prediction time on test fold
                t0 = time.time()
                y_pred = pipe.predict(X_te)
                t_pred = time.time() - t0

                # Compute metrics
                acc = accuracy_score(y_te, y_pred)
                mf1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
                kappa = cohen_kappa_score(y_te, y_pred)
                latency_ms_per_1k = (t_pred / len(te_idx)) * 1000 * 1000

                # Per-class recall (classes 1 to 7)
                recalls = recall_score(y_te, y_pred, labels=list(range(1, 8)), average=None, zero_division=0)

                # Update cumulative confusion matrix
                cm = confusion_matrix(y_te, y_pred, labels=list(range(1, 8)))
                cum_confusion_matrices[algo_name] += cm

                result_row = {
                    "fold_index": fold_counter,
                    "fold_id": fold_id,
                    "repeat": repeat_idx + 1,
                    "split": split_idx + 1,
                    "algorithm": algo_name,
                    "family": FAMILY_MAP[algo_name],
                    "macro_f1": round(mf1, 6),
                    "accuracy": round(acc, 6),
                    "cohen_kappa": round(kappa, 6),
                    "recall_spruce_fir": round(recalls[0], 6),
                    "recall_lodgepole_pine": round(recalls[1], 6),
                    "recall_ponderosa_pine": round(recalls[2], 6),
                    "recall_cottonwood_willow": round(recalls[3], 6),
                    "recall_aspen": round(recalls[4], 6),
                    "recall_douglas_fir": round(recalls[5], 6),
                    "recall_krummholz": round(recalls[6], 6),
                    "fit_time_sec": round(t_fit, 4),
                    "pred_time_sec": round(t_pred, 4),
                    "pred_latency_ms_per_1k": round(latency_ms_per_1k, 4),
                }
                raw_results.append(result_row)

                print(f"  {algo_name:24s} | Macro-F1: {mf1:.4f} | Acc: {acc:.4f} | Kappa: {kappa:.4f} | Fit: {t_fit:6.2f}s | Lat: {latency_ms_per_1k:5.2f}ms/1k")

    total_experiment_time = time.time() - t_experiment_start
    print(f"\n{'=' * 80}")
    print(f"All 15 folds completed in {total_experiment_time/60:.2f} minutes.")
    print(f"{'=' * 80}")

    # ── 5. Save Raw Per-Fold Results CSV ───────────────────────────
    raw_df = pd.DataFrame(raw_results)
    raw_csv_path = os.path.join(RESULTS_DIR, "raw_fold_results.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"\n✓ Saved raw fold results ({len(raw_df)} rows) to:")
    print(f"  {raw_csv_path}")

    # ── 6. Compute Table 4: Main Results (Mean ± Std across 15 folds)
    print(f"\n{'─' * 80}")
    print("5. COMPUTING TABLE 4: MAIN EXPERIMENT RESULTS (MEAN ± STD ACROSS 15 FOLDS)")
    print(f"{'─' * 80}")

    summary_rows = []
    # Rank algorithms by mean Macro-F1
    ranked_algos = (
        raw_df.groupby("algorithm")["macro_f1"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    for rank, algo in enumerate(ranked_algos, start=1):
        sub = raw_df[raw_df["algorithm"] == algo]
        summary_rows.append({
            "Rank": rank,
            "Algorithm": algo,
            "Family": FAMILY_MAP[algo],
            "Macro-F1 (Mean ± Std)": f"{sub['macro_f1'].mean():.4f} ± {sub['macro_f1'].std():.4f}",
            "Accuracy (Mean ± Std)": f"{sub['accuracy'].mean():.4f} ± {sub['accuracy'].std():.4f}",
            "Cohen's Kappa (Mean ± Std)": f"{sub['cohen_kappa'].mean():.4f} ± {sub['cohen_kappa'].std():.4f}",
            "Training Time (s)": f"{sub['fit_time_sec'].mean():.2f} ± {sub['fit_time_sec'].std():.2f}",
            "Prediction Latency (ms/1k)": f"{sub['pred_latency_ms_per_1k'].mean():.2f} ± {sub['pred_latency_ms_per_1k'].std():.2f}",
            "raw_macro_f1_mean": sub["macro_f1"].mean(),
            "raw_macro_f1_std": sub["macro_f1"].std(),
            "raw_accuracy_mean": sub["accuracy"].mean(),
            "raw_accuracy_std": sub["accuracy"].std(),
            "raw_fit_time_mean": sub["fit_time_sec"].mean(),
            "raw_pred_lat_mean": sub["pred_latency_ms_per_1k"].mean(),
        })

    table_4_df = pd.DataFrame(summary_rows)
    table_4_path = os.path.join(RESULTS_DIR, "table_4_main_results.csv")
    table_4_df.to_csv(table_4_path, index=False)
    print(table_4_df[["Rank", "Algorithm", "Family", "Macro-F1 (Mean ± Std)", "Accuracy (Mean ± Std)", "Training Time (s)", "Prediction Latency (ms/1k)"]].to_string(index=False))
    print(f"\n✓ Saved Table 4 main results to:\n  {table_4_path}")

    # ── 7. Compute Per-Class Recall Summary ────────────────────────
    print(f"\n{'─' * 80}")
    print("6. COMPUTING PER-CLASS RECALL SUMMARY (MEAN ± STD ACROSS 15 FOLDS)")
    print(f"{'─' * 80}")

    recall_cols = [
        ("recall_spruce_fir", "Spruce/Fir (36.5%)"),
        ("recall_lodgepole_pine", "Lodgepole Pine (48.8%)"),
        ("recall_ponderosa_pine", "Ponderosa Pine (6.2%)"),
        ("recall_cottonwood_willow", "Cottonwood/Willow (0.5%)"),
        ("recall_aspen", "Aspen (1.6%)"),
        ("recall_douglas_fir", "Douglas-fir (3.0%)"),
        ("recall_krummholz", "Krummholz (3.5%)"),
    ]

    recall_summary_rows = []
    for algo in ranked_algos:
        sub = raw_df[raw_df["algorithm"] == algo]
        row = {"Algorithm": algo, "Family": FAMILY_MAP[algo]}
        for col_key, col_label in recall_cols:
            row[col_label] = f"{sub[col_key].mean():.4f} ± {sub[col_key].std():.4f}"
        recall_summary_rows.append(row)

    recall_df = pd.DataFrame(recall_summary_rows)
    recall_path = os.path.join(RESULTS_DIR, "per_class_recall_summary.csv")
    recall_df.to_csv(recall_path, index=False)
    print(recall_df.to_string(index=False))
    print(f"\n✓ Saved Per-Class Recall Summary to:\n  {recall_path}")

    # ── 8. Save Aggregate Confusion Matrices (Figure 2 source) ─────
    print(f"\n{'─' * 80}")
    print("7. SAVING AGGREGATED CONFUSION MATRICES (FIGURE 2 SOURCE)")
    print(f"{'─' * 80}")

    cm_serializable = {
        algo: {
            "classes": [COVER_TYPE_NAMES[i] for i in range(1, 8)],
            "matrix": cum_confusion_matrices[algo].tolist(),
        }
        for algo in cum_confusion_matrices
    }
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrices.json")
    with open(cm_path, "w") as f:
        json.dump(cm_serializable, f, indent=4)
    print(f"✓ Saved cumulative confusion matrices to:\n  {cm_path}")

    # ── 9. Generate Pinned Requirements File ───────────────────────
    print(f"\n{'─' * 80}")
    print("8. PINNING REPRODUCIBILITY ENVIRONMENT")
    print(f"{'─' * 80}")
    
    import sklearn
    import scipy
    import lightgbm
    import matplotlib
    
    reqs = [
        f"python=={sys.version.split()[0]}",
        f"numpy=={np.__version__}",
        f"pandas=={pd.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"scipy=={scipy.__version__}",
        f"lightgbm=={lightgbm.__version__}",
        f"matplotlib=={matplotlib.__version__}",
    ]
    reqs_path = os.path.join(RESULTS_DIR, "requirements.txt")
    with open(reqs_path, "w") as f:
        f.write("\n".join(reqs) + "\n")
    print(f"✓ Saved requirements.txt to:\n  {reqs_path}")
    for r in reqs:
        print(f"  • {r}")

    print("\n" + "=" * 80)
    print("PHASE 4 COMPLETE: EXPERIMENTAL MATRIX EVALUATED & ARCHIVED")
    print("=" * 80)

if __name__ == "__main__":
    main()
