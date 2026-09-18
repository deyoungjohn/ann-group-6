# Forest Cover Type — Empirical Machine Learning Benchmark

**Dataset**: UCI Forest Cover Type (#31) · **DOI**: `10.24432/C50K5N`  
**Anchor Paper**: Blackard, J. A., & Dean, D. J. (1999). *Computers and Electronics in Agriculture*, 24(3), 131–151.

---

## 1. Quick Start: Full Reproduction in One Command

To reproduce all audit findings, hyperparameter tuning, 15-fold outer cross-validation, statistical significance tests, and figures from the raw dataset (`covtype.data.gz`):

```bash
# 1. Install pinned dependencies
pip install -r experiment_results/requirements.txt

# 2. Run the master replication pipeline
python run_all.py
```

To run the automated verification test independently:
```bash
python verify_reproducibility.py
```

---

## 2. Directory Structure & Deliverable Manifest

```
c:\Users\DELL\Downloads\ANN\
├── covtype.data.gz                     # Raw dataset from UCI (581,012 rows × 55 columns)
├── covtype.info                        # UCI dataset metadata and documentation
│
├── phase1_2_data_audit.py              # Phase 1 & 2: Anchor paper analysis & data audit
├── phase3_protocol_and_tuning.py       # Phase 3: Pipeline construction & inner CV tuning
├── phase4_full_experiment_matrix.py    # Phase 4: 15-fold outer CV matrix execution
├── phase5_statistical_analysis_and_plots.py # Phase 5: Friedman, Wilcoxon+Holm & figure generation
├── phase6_7_generate_report.py         # Phase 6 & 7: Word & PDF report generation
├── verify_reproducibility.py           # Phase 8: Automated parity audit (76/76 checks passed)
├── run_all.py                          # Master one-command replication pipeline
│
├── audit_outputs/                      # Phase 2 audit outputs
│   ├── class_distribution.csv          # Full class breakdown (103:1 imbalance ratio)
│   ├── quantitative_feature_stats.csv  # Feature ranges, standard deviations, and zeros
│   ├── quantitative_correlations.csv   # Pearson correlation matrix (|r| > 0.5 flagged)
│   ├── soil_type_usage.csv             # Soil type counts (Soil_Type_15 = 3 observations)
│   └── data_integrity_checks.csv       # Row/col counts, zero missing values, zero duplicates
│
├── protocol_outputs/                   # Phase 3 experimental design outputs
│   ├── table_1_prior_work.csv          # Prior-work protocol comparison (Table 1)
│   ├── table_3_tuning_protocol.csv     # Equal tuning budgets: 30 iterations each (Table 3)
│   ├── appendix_c_frozen_hyperparameters.json # Frozen optimal hyperparameters per algorithm
│   ├── baseline_fold_results.csv       # Raw baseline fold scores (15 folds)
│   └── baseline_summary_table.csv      # Baseline summary (Mean ± Std)
│
├── experiment_results/                 # Phase 4 raw experimental results
│   ├── raw_fold_results.csv            # Per-fold, per-algorithm, per-metric records (90 rows)
│   ├── table_4_main_results.csv        # Main comparison table: Mean ± Std (Table 4)
│   ├── per_class_recall_summary.csv    # 7-class recall breakdown for all models
│   ├── confusion_matrices.json         # Raw cumulative confusion matrices (N=150,000 predictions)
│   └── requirements.txt                # Pinned library versions for reproduction
│
├── statistical_outputs/                # Phase 5 statistical analysis
│   ├── table_5_statistical_tests.csv   # Wilcoxon tests, Holm p-values & effect sizes (Table 5)
│   └── friedman_and_wilcoxon_details.json # Omnibus Friedman test & rank details
│
└── report_figures/                     # Publication-ready figures (PNG 300 DPI + vector PDF)
    ├── figure1_fold_dispersion.png     # Figure 1: Fold-level Macro-F1 & Accuracy boxplots
    ├── figure1_fold_dispersion.pdf     # Vector PDF format
    ├── figure2_confusion_matrices.png  # Figure 2: Error structure heatmaps with raw counts
    ├── figure2_confusion_matrices.pdf  # Vector PDF format
    ├── figure3_cost_vs_performance.png # Figure 3: Pareto frontier (Macro-F1 vs Training Time)
    └── figure3_cost_vs_performance.pdf # Vector PDF format
```

---

## 3. Pinned Environment & Hardware Specifications

- **Operating System**: Windows 11
- **CPU**: Intel Core i9-10885H
- **RAM**: 32.0 GB
- **Python Version**: 3.11.6
- **Key Dependencies**:
  - `numpy==2.4.6`
  - `pandas==3.0.6`
  - `scikit-learn==1.9.1`
  - `scipy==1.17.1`
  - `lightgbm==4.7.0`
  - `matplotlib==3.11.1`
  - `seaborn==0.13.2`
  - `python-docx==1.1.2`
  - `statsmodels==0.14.6`

---

## 4. Experimental Design Summary

- **Subsampling**: Documented stratified subsample of $N = 50,000$ from the full 581,012 dataset, and preserved class balance down to 0.00% deviation.
- **Validation Protocol**: Stratified 5-Fold cross-validation repeated 3 times.
- **Tuning Protocol**: Inner Stratified 3-Fold CV on training partitions using `RandomizedSearchCV` with an identical budget of exactly **30 evaluations** per algorithm, optimizing `f1_macro`.
- **Data Leakage Safeguard**: `ColumnTransformer` embedded inside the `Pipeline`. Continuous features are standardized using `StandardScaler` fitted **only on the training fold**. Binary indicator features are passed through unscaled. Test folds are evaluated exactly once at the end.
- **Evaluation Criteria**: Primary: unweighted Macro-Averaged F1 and per-class recall. Secondary: overall accuracy, Cohen's kappa ($\kappa$), training runtime, and prediction latency.
- **Statistical Rigor**: Omnibus Friedman test followed by post-hoc pairwise Wilcoxon signed-rank tests with Holm–Bonferroni step-down correction and non-parametric effect sizes $r = Z / \sqrt{N}$.
