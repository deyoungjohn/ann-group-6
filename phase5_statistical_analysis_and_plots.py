"""
Phase 5 — Statistical Significance Analysis & Publication-Quality Visualization
==============================================================================
Performs the rigorous statistical analysis and generates the required figures:
1. Loads raw fold-level results from Phase 4 (N=15 paired folds).
2. Runs Friedman's test across algorithms on Macro-F1 and Accuracy.
3. Runs post-hoc pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.
4. Computes effect sizes r = Z / sqrt(N) for each pairwise comparison.
5. Generates Table 5 (Statistical Comparison Table) saved to CSV.
6. Generates publication-ready figures (PNG 300 DPI and vector PDF):
   - Figure 1: Fold-level dispersion (Macro-F1 and Accuracy across 15 folds).
   - Figure 2: Error structure confusion matrices for the best model in each of
     the 4 families (Random Forest, MLP, Decision Tree, Logistic Regression).
   - Figure 3: Cost vs. Performance trade-off (Macro-F1 vs Training Time on log scale).

Run: python phase5_statistical_analysis_and_plots.py
"""

import os
import sys
import io
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import friedmanchisquare, wilcoxon, norm
from statsmodels.stats.multitest import multipletests

# Fix console encoding with line buffering
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True, errors="replace")
warnings.filterwarnings("ignore")

# ── Paths and Config ──────────────────────────────────────────────
RESULTS_DIR = r"c:\Users\DELL\Downloads\ANN\experiment_results"
RAW_RESULTS_FILE = os.path.join(RESULTS_DIR, "raw_fold_results.csv")
CONFUSION_FILE = os.path.join(RESULTS_DIR, "confusion_matrices.json")

OUTPUT_DIR = r"c:\Users\DELL\Downloads\ANN\statistical_outputs"
PLOTS_DIR = r"c:\Users\DELL\Downloads\ANN\report_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

COVER_TYPE_NAMES = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]

# Set consistent matplotlib plotting style
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def main():
    print("=" * 80)
    print("PHASE 5 — STATISTICAL SIGNIFICANCE ANALYSIS & VISUALIZATION (P6)")
    print("=" * 80)
    print(f"Timestamp        : {datetime.now().isoformat()}")
    print(f"Raw results file : {RAW_RESULTS_FILE}")
    print(f"Confusion file   : {CONFUSION_FILE}")
    print(f"Statistical dir  : {OUTPUT_DIR}")
    print(f"Figures dir      : {PLOTS_DIR}")

    # ── 1. Load Raw Results ─────────────────────────────────────────
    raw_df = pd.read_csv(RAW_RESULTS_FILE)
    print(f"\nLoaded {len(raw_df)} fold observations across {raw_df['algorithm'].nunique()} algorithms.")

    algorithms_order = [
        "RandomForest",
        "LightGBM",
        "MLP",
        "DecisionTree",
        "LogisticRegression",
        "MajorityClass_ZeroRule",
    ]

    # ── 2. Friedman Test across All Algorithms ──────────────────────
    print(f"\n{'─' * 80}")
    print("1. OMNIBUS FRIEDMAN TESTS (N=15 PAIRED FOLDS)")
    print(f"{'─' * 80}")

    # Pivot scores into paired matrices (15 folds x algorithms)
    pivot_f1 = raw_df.pivot(index="fold_id", columns="algorithm", values="macro_f1")[algorithms_order]
    pivot_acc = raw_df.pivot(index="fold_id", columns="algorithm", values="accuracy")[algorithms_order]

    # Friedman test on Macro-F1
    stat_f1, p_f1 = friedmanchisquare(*[pivot_f1[col] for col in algorithms_order])
    ranks_f1 = pivot_f1.rank(axis=1, ascending=False).mean()

    print(f"Friedman Test on Macro-F1:")
    print(f"  Chi-squared statistic : {stat_f1:.4f}")
    print(f"  Degrees of freedom    : {len(algorithms_order) - 1}")
    print(f"  Asymptotic p-value    : {p_f1:.4e} ({'Significant p < 0.001' if p_f1 < 0.001 else 'Not significant'})")
    print("\nMean Ranks (1 = Best):")
    for algo, rank in ranks_f1.items():
        print(f"  • {algo:24s}: {rank:.2f}")

    # Friedman test on Accuracy
    stat_acc, p_acc = friedmanchisquare(*[pivot_acc[col] for col in algorithms_order])
    ranks_acc = pivot_acc.rank(axis=1, ascending=False).mean()
    print(f"\nFriedman Test on Overall Accuracy:")
    print(f"  Chi-squared statistic : {stat_acc:.4f}")
    print(f"  Degrees of freedom    : {len(algorithms_order) - 1}")
    print(f"  Asymptotic p-value    : {p_acc:.4e}")

    # ── 3. Post-Hoc Pairwise Wilcoxon Signed-Rank Tests with Holm Correction
    print(f"\n{'─' * 80}")
    print("2. POST-HOC PAIRWISE WILCOXON SIGNED-RANK TESTS & EFFECT SIZES (MACRO-F1)")
    print(f"{'─' * 80}")

    # Define key hypothesis-driven comparisons
    comparisons = [
        ("RandomForest", "LightGBM", "Top ensemble comparison (Bagging vs Boosting)"),
        ("RandomForest", "MLP", "Top ensemble vs Neural anchor"),
        ("LightGBM", "MLP", "Gradient boosting vs Neural anchor"),
        ("RandomForest", "DecisionTree", "Ensemble gain over single tree"),
        ("MLP", "DecisionTree", "Neural network vs Single decision tree"),
        ("DecisionTree", "LogisticRegression", "Non-linear greedy tree vs Linear model"),
        ("LogisticRegression", "MajorityClass_ZeroRule", "Linear model vs Trivial baseline"),
        ("RandomForest", "LogisticRegression", "Global best vs Linear model"),
    ]

    pairwise_records = []
    raw_p_values = []

    for a1, a2, desc in comparisons:
        s1 = pivot_f1[a1].values
        s2 = pivot_f1[a2].values
        diff = s1 - s2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)

        # Wilcoxon signed-rank test
        # Handle exact zero differences if any
        if np.all(diff == 0):
            stat = 0.0
            p_val = 1.0
            z_score = 0.0
        else:
            res = wilcoxon(s1, s2, zero_method="wilcox")
            stat = res.statistic
            p_val = res.pvalue
            # Normal approximation for Z score:
            if p_val == 1.0:
                z_score = 0.0
            else:
                z_score = norm.ppf(1 - p_val / 2)

        effect_size_r = z_score / np.sqrt(len(s1))  # N = 15

        raw_p_values.append(p_val)
        pairwise_records.append({
            "Pair": f"{a1} vs. {a2}",
            "Algorithm_1": a1,
            "Algorithm_2": a2,
            "Description": desc,
            "Mean_Diff": mean_diff,
            "Std_Diff": std_diff,
            "Statistic_W": stat,
            "p_value_raw": p_val,
            "Z_score": z_score,
            "Effect_Size_r": effect_size_r,
        })

    # Apply Holm-Bonferroni correction
    rejects, corrected_pvals, _, _ = multipletests(raw_p_values, method="holm")

    table_5_rows = []
    for idx, rec in enumerate(pairwise_records):
        adj_p = corrected_pvals[idx]
        rec["p_value_holm"] = adj_p
        rec["Significant"] = bool(rejects[idx])

        # Plain-English interpretation of effect size and significance
        r_val = rec["Effect_Size_r"]
        if not rejects[idx]:
            interp = f"Within fold noise (p_holm = {adj_p:.4f} > 0.05, r = {r_val:.2f}); difference not reliable."
        else:
            if r_val >= 0.5:
                magnitude = "large practical effect"
            elif r_val >= 0.3:
                magnitude = "medium practical effect"
            else:
                magnitude = "small/trivial practical effect"
            better_algo = rec["Algorithm_1"] if rec["Mean_Diff"] > 0 else rec["Algorithm_2"]
            interp = f"Statistically significant ({better_algo} superior, p_holm = {adj_p:.4e}, {magnitude}, r = {r_val:.2f})."

        rec["Interpretation"] = interp

        table_5_rows.append({
            "Comparison Pair": rec["Pair"],
            "Description": rec["Description"],
            "Mean Difference (A1 - A2)": f"{rec['Mean_Diff']:+.4f} ± {rec['Std_Diff']:.4f}",
            "Wilcoxon W": f"{rec['Statistic_W']:.1f}",
            "Raw p-value": f"{rec['p_value_raw']:.4e}" if rec['p_value_raw'] < 0.001 else f"{rec['p_value_raw']:.4f}",
            "Holm-Adjusted p": f"{adj_p:.4e}" if adj_p < 0.001 else f"{adj_p:.4f}",
            "Effect Size (r)": f"{r_val:.3f}",
            "Interpretation": interp,
        })

    table_5_df = pd.DataFrame(table_5_rows)
    table_5_path = os.path.join(OUTPUT_DIR, "table_5_statistical_tests.csv")
    table_5_df.to_csv(table_5_path, index=False)

    print("\nTable 5: Statistical Comparison Table (Macro-F1, N=15 Paired Folds):")
    for row in table_5_rows:
        print(f"\n  • {row['Comparison Pair']:35s}")
        print(f"    Diff: {row['Mean Difference (A1 - A2)']} | W: {row['Wilcoxon W']} | p_holm: {row['Holm-Adjusted p']} | r: {row['Effect Size (r)']}")
        print(f"    → {row['Interpretation']}")

    print(f"\n✓ Saved Table 5 to:\n  {table_5_path}")

    # Also save detailed statistical JSON
    with open(os.path.join(OUTPUT_DIR, "friedman_and_wilcoxon_details.json"), "w") as f:
        json.dump({
            "friedman_macro_f1": {"statistic": stat_f1, "p_value": p_f1, "mean_ranks": ranks_f1.to_dict()},
            "friedman_accuracy": {"statistic": stat_acc, "p_value": p_acc, "mean_ranks": ranks_acc.to_dict()},
            "pairwise_wilcoxon": pairwise_records,
        }, f, indent=4, default=str)

    # ── 4. Generate Figure 1: Fold-Level Dispersion ─────────────────
    print(f"\n{'─' * 80}")
    print("3. GENERATING FIGURE 1: FOLD-LEVEL DISPERSION PLOTS")
    print(f"{'─' * 80}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    display_names = {
        "RandomForest": "Random Forest\n(Ensemble)",
        "LightGBM": "LightGBM\n(Ensemble)",
        "MLP": "MLP Neural Net\n(Neural/Kernel)",
        "DecisionTree": "Decision Tree\n(Tree-based)",
        "LogisticRegression": "Logistic Reg.\n(Linear)",
        "MajorityClass_ZeroRule": "Zero-Rule\n(Baseline)",
    }

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728", "#7f7f7f"]

    # Panel A: Macro-F1 Dispersion
    ax1 = axes[0]
    plot_data_f1 = [pivot_f1[algo].values for algo in algorithms_order]
    bp1 = ax1.boxplot(
        plot_data_f1,
        tick_labels=[display_names[a] for a in algorithms_order],
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
        medianprops={"color": "darkred", "linewidth": 1.5},
        boxprops={"facecolor": "#e6f2ff", "edgecolor": "#1f77b4", "linewidth": 1.2},
    )
    for i, col_data in enumerate(plot_data_f1):
        x = np.random.normal(i + 1, 0.04, size=len(col_data))
        ax1.scatter(x, col_data, alpha=0.7, color=colors[i], s=25, edgecolors="none", zorder=3)

    ax1.set_title("(a) Macro-Averaged F1 Score (Primary Metric)")
    ax1.set_ylabel("Macro-F1")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.set_ylim(-0.02, 1.0)

    # Panel B: Overall Accuracy Dispersion
    ax2 = axes[1]
    plot_data_acc = [pivot_acc[algo].values for algo in algorithms_order]
    bp2 = ax2.boxplot(
        plot_data_acc,
        tick_labels=[display_names[a] for a in algorithms_order],
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
        medianprops={"color": "darkred", "linewidth": 1.5},
        boxprops={"facecolor": "#eaf7ea", "edgecolor": "#2ca02c", "linewidth": 1.2},
    )
    for i, col_data in enumerate(plot_data_acc):
        x = np.random.normal(i + 1, 0.04, size=len(col_data))
        ax2.scatter(x, col_data, alpha=0.7, color=colors[i], s=25, edgecolors="none", zorder=3)

    ax2.set_title("(b) Overall Classification Accuracy (Secondary Metric)")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.set_ylim(0.40, 1.0)

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)

    plt.suptitle("Figure 1: Fold-to-Fold Metric Dispersion Across 15 Paired Folds (Stratified 5-Fold × 3 Repeats, N=50,000)", fontsize=11, y=1.02)
    plt.tight_layout()

    fig1_png = os.path.join(PLOTS_DIR, "figure1_fold_dispersion.png")
    fig1_pdf = os.path.join(PLOTS_DIR, "figure1_fold_dispersion.pdf")
    fig.savefig(fig1_png)
    fig.savefig(fig1_pdf)
    plt.close(fig)
    print(f"✓ Figure 1 saved to:\n  {fig1_png}\n  {fig1_pdf}")

    # ── 5. Generate Figure 2: Confusion Matrices for Best in Each Family
    print(f"\n{'─' * 80}")
    print("4. GENERATING FIGURE 2: ERROR STRUCTURE CONFUSION MATRICES")
    print(f"{'─' * 80}")

    with open(CONFUSION_FILE, "r") as f:
        cm_data = json.load(f)

    # 4 representative models spanning the 4 families:
    # Ensemble: RandomForest
    # Neural: MLP
    # Tree: DecisionTree
    # Linear: LogisticRegression
    cm_models = [
        ("RandomForest", "Random Forest (Ensemble Family) — Rank 1"),
        ("MLP", "MLP Neural Network (Neural Family) — Rank 3"),
        ("DecisionTree", "Decision Tree CART (Tree-Based Family) — Rank 4"),
        ("LogisticRegression", "Logistic Regression (Linear Family) — Rank 5"),
    ]

    short_labels = ["Spruce", "Lodgepole", "Ponderosa", "Cottonwood", "Aspen", "Douglas-fir", "Krummholz"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (m_key, title) in enumerate(cm_models):
        ax = axes[idx]
        mat = np.array(cm_data[m_key]["matrix"])
        
        # Plot normalized heatmap with raw count annotations
        mat_norm = mat.astype("float") / mat.sum(axis=1)[:, np.newaxis]
        
        annot_text = np.empty_like(mat, dtype=object)
        for r in range(7):
            for c in range(7):
                cnt = mat[r, c]
                pct = mat_norm[r, c] * 100
                if cnt >= 1000:
                    annot_text[r, c] = f"{cnt:,}\n({pct:.1f}%)"
                elif cnt > 0:
                    annot_text[r, c] = f"{cnt}\n({pct:.1f}%)"
                else:
                    annot_text[r, c] = "0\n(0%)"

        sns.heatmap(
            mat_norm,
            annot=annot_text,
            fmt="",
            cmap="Blues" if idx in [0, 1] else "YlGnBu",
            cbar=True,
            cbar_kws={"label": "Normalized Recall (Row %)"},
            xticklabels=short_labels,
            yticklabels=short_labels,
            ax=ax,
            vmin=0,
            vmax=1,
            annot_kws={"size": 7.5},
        )
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Forest Cover Type")
        ax.set_ylabel("True Forest Cover Type")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.suptitle("Figure 2: Error Structure Confusion Matrices by Model Class (Aggregated over 15 Test Evaluations, N=150,000 Predictions)", fontsize=12, y=1.01)
    plt.tight_layout()

    fig2_png = os.path.join(PLOTS_DIR, "figure2_confusion_matrices.png")
    fig2_pdf = os.path.join(PLOTS_DIR, "figure2_confusion_matrices.pdf")
    fig.savefig(fig2_png)
    fig.savefig(fig2_pdf)
    plt.close(fig)
    print(f"✓ Figure 2 saved to:\n  {fig2_png}\n  {fig2_pdf}")

    # ── 6. Generate Figure 3: Cost vs Performance Trade-off ─────────
    print(f"\n{'─' * 80}")
    print("5. GENERATING FIGURE 3: PERFORMANCE VS COMPUTATIONAL COST")
    print(f"{'─' * 80}")

    table_4_df = pd.read_csv(os.path.join(RESULTS_DIR, "table_4_main_results.csv"))

    fig, ax = plt.subplots(figsize=(9, 6))

    family_palette = {
        "Ensemble": "#1f77b4",
        "Neural/Kernel": "#ff7f0e",
        "Tree/Rule-based": "#2ca02c",
        "Linear/Probabilistic": "#d62728",
        "Trivial Baseline": "#7f7f7f",
    }

    for _, row in table_4_df.iterrows():
        algo = row["Algorithm"]
        fam = row["Family"]
        f1_mean = row["raw_macro_f1_mean"]
        f1_std = row["raw_macro_f1_std"]
        fit_mean = max(row["raw_fit_time_mean"], 0.01)  # avoid log(0)
        lat_mean = row["raw_pred_lat_mean"]

        color = family_palette.get(fam, "#333333")
        # Scatter point: size corresponds to prediction latency
        ax.errorbar(
            fit_mean, f1_mean, yerr=f1_std,
            fmt="o", color=color, ecolor=color, elinewidth=1.5,
            capsize=4, markersize=8 + min(lat_mean, 40), alpha=0.85,
            label=fam if fam not in [h.get_label() for h in ax.get_legend_handles_labels()[0]] else "",
        )

        # Label algorithm names with slight offsets
        offset_x = 1.15
        offset_y = 0.0
        if algo == "LightGBM":
            offset_y = -0.02
        elif algo == "RandomForest":
            offset_y = +0.015
        elif algo == "MLP":
            offset_y = -0.015

        ax.annotate(
            f"{algo}\n({f1_mean:.3f}, {fit_mean:.2f}s)",
            (fit_mean * offset_x, f1_mean + offset_y),
            fontsize=8.5,
            fontweight="semibold",
            color=color,
            va="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training Time per Fold (Seconds, Log Scale)")
    ax.set_ylabel("Macro-Averaged F1 Score (Primary Metric)")
    ax.set_title("Figure 3: Empirical Pareto Frontier: Predictive Performance vs. Computational Training Cost", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_ylim(0.0, 0.90)

    # Note bubble size meaning in annotation
    ax.annotate(
        "Note: Marker size is proportional to prediction latency (ms/1,000 samples).\nError bars represent ±1 standard deviation across 15 folds.",
        xy=(0.03, 0.05), xycoords="axes fraction",
        fontsize=8, style="italic", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc"),
    )

    # Deduplicate legend handles
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right", framealpha=0.9)

    plt.tight_layout()

    fig3_png = os.path.join(PLOTS_DIR, "figure3_cost_vs_performance.png")
    fig3_pdf = os.path.join(PLOTS_DIR, "figure3_cost_vs_performance.pdf")
    fig.savefig(fig3_png)
    fig.savefig(fig3_pdf)
    plt.close(fig)
    print(f"✓ Figure 3 saved to:\n  {fig3_png}\n  {fig3_pdf}")

    print("\n" + "=" * 80)
    print("PHASE 5 COMPLETE: STATISTICAL ANALYSIS EXECUTED & FIGURES PRODUCED")
    print("=" * 80)

if __name__ == "__main__":
    main()
