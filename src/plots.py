"""Publication-style figures saved to outputs/figures."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.tree import plot_tree

from config import FIGURES_DIR, LABELS

sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
PALETTE = {"naive": "#7f7f7f", "logreg": "#1f77b4", "tree": "#d62728", "rf": "#2ca02c", "logreg_demo": "#9467bd"}


def _save(fig, name: str) -> str:
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def roc_pr_curves(y, probas: dict, names: dict, prevalence: float) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for key, p in probas.items():
        fpr, tpr, _ = roc_curve(y, p)
        from sklearn.metrics import auc, average_precision_score
        axes[0].plot(fpr, tpr, label=f"{names[key]} (AUC = {auc(fpr, tpr):.3f})", color=PALETTE.get(key))
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=f"{names[key]} (AP = {average_precision_score(y, p):.3f})", color=PALETTE.get(key))
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    axes[0].set(xlabel="False-positive rate (1 - specificity)", ylabel="True-positive rate (recall)",
                title="ROC curves - hold-out districts")
    axes[1].axhline(prevalence, color="k", ls="--", lw=1, label=f"Chance (prevalence = {prevalence:.3f})")
    axes[1].set(xlabel="Recall (share of out-of-school children caught)", ylabel="Precision (share of flags that are correct)",
                title="Precision-recall curves - hold-out districts")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "upper right", fontsize=9)
    return _save(fig, "fig1_roc_pr_curves")


def confusion_heatmaps(cms: dict, names: dict, suffix: str = "") -> str:
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6))
    axes = np.atleast_1d(axes)
    for ax, (key, c) in zip(axes, cms.items()):
        mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        row = mat.sum(axis=1, keepdims=True)
        pct = mat / np.where(row == 0, 1, row)
        annot = np.array([[f"{int(mat[i, j]):,}\n({pct[i, j]:.1%})" for j in range(2)] for i in range(2)])
        sns.heatmap(pct, annot=annot, fmt="", cmap="Blues", cbar=False, vmin=0, vmax=1, ax=ax,
                    xticklabels=["Pred. enrolled", "Pred. out of school"],
                    yticklabels=["Actual enrolled", "Actual out of school"])
        ax.set_title(f"{names[key]}\nthreshold = {c.get('threshold', 0.5):.2f}", fontsize=11)
    fig.suptitle("Confusion matrices on hold-out districts (counts; row % = share of each true class)", y=1.02, fontsize=12)
    return _save(fig, f"fig2_confusion_matrices{suffix}")


def threshold_curves(sweeps_oof: dict, sweeps_holdout: dict, chosen: dict, names: dict, ratio: int) -> str:
    """Left: recall/precision vs threshold on out-of-fold TRAINING predictions (used to choose the threshold).
    Right: expected cost on OOF (solid, used for selection) and on the untouched hold-out (dashed, reported only)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for key, sw in sweeps_oof.items():
        axes[0].plot(sw.threshold, sw.recall, color=PALETTE.get(key), label=f"{names[key]} recall")
        axes[0].plot(sw.threshold, sw.precision, color=PALETTE.get(key), ls=":", label=f"{names[key]} precision")
        axes[1].plot(sw.threshold, sw[f"cost_per_1000_ratio{ratio}"], color=PALETTE.get(key), label=f"{names[key]} - training OOF (selection)")
        axes[1].plot(sweeps_holdout[key].threshold, sweeps_holdout[key][f"cost_per_1000_ratio{ratio}"], color=PALETTE.get(key), ls="--", lw=1.2,
                     label=f"{names[key]} - hold-out (not used)")
        for ax in axes:
            ax.axvline(chosen[key], color=PALETTE.get(key), lw=1, ls="-.")
    axes[0].axvline(0.5, color="k", lw=1, ls="--")
    axes[0].set(xlabel="Classification threshold on P(out of school)", ylabel="Metric value",
                title="Recall and precision vs threshold (training out-of-fold)")
    axes[1].axvline(0.5, color="k", lw=1, ls="--")
    axes[1].set(xlabel="Classification threshold on P(out of school)", ylabel=f"Expected cost per 1,000 children\n(1 FN = {ratio} FP)",
                title=f"Expected cost (FN:FP = {ratio}:1); dash-dot = chosen threshold")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    return _save(fig, "fig3_threshold_cost_curves")


def importance_bars(lr: pd.DataFrame, tree: pd.DataFrame, perm_lr: pd.DataFrame, perm_tree: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    d = lr.sort_values("std_coef")
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in d.std_coef]
    axes[0, 0].barh(d.label, d.std_coef, color=colors)
    axes[0, 0].axvline(0, color="k", lw=1)
    axes[0, 0].set(title="Logistic regression: standardised coefficient\n(red raises odds of being out of school, blue lowers)", xlabel="Log-odds change per 1 SD")
    d = tree.sort_values("gini_importance")
    axes[0, 1].barh(d.label, d.gini_importance, color="#d62728")
    axes[0, 1].set(title="Decision tree: Gini importance", xlabel="Share of impurity reduction")
    for ax, pm, ttl, col in ((axes[1, 0], perm_lr, "Logistic regression: permutation importance", "#1f77b4"),
                             (axes[1, 1], perm_tree, "Decision tree: permutation importance", "#d62728")):
        d = pm.sort_values("auc_drop_mean")
        ax.barh(d.label, d.auc_drop_mean, xerr=d.auc_drop_std, color=col, ecolor="k", capsize=3)
        ax.set(title=ttl, xlabel="Drop in hold-out ROC-AUC when feature is shuffled")
    return _save(fig, "fig4_feature_importance")


def odds_ratio_forest(ors: pd.DataFrame) -> str:
    d = ors.sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.errorbar(d.odds_ratio, d.label, xerr=[d.odds_ratio - d.or_ci_low, d.or_ci_high - d.odds_ratio],
                fmt="o", color="#1f77b4", ecolor="k", capsize=3)
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set(xlabel="Odds ratio for being out of school (log scale, 95% CI)", title="Logistic regression odds ratios (raw units, training districts)")
    return _save(fig, "fig5_odds_ratios")


def h5_auc_folds(folds_demo: np.ndarray, folds_full: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    k = np.arange(1, len(folds_demo) + 1)
    ax.plot(k, folds_demo, "o-", color=PALETTE["logreg_demo"], label="Demographic model (age, female, urban)")
    ax.plot(k, folds_full, "o-", color=PALETTE["logreg"], label="Full model (13 features)")
    for i in range(len(k)):
        ax.plot([k[i], k[i]], [folds_demo[i], folds_full[i]], color="grey", lw=1)
    ax.set(xlabel="Grouped CV fold", ylabel="ROC-AUC", title="H5: AUC gain from poverty, B-form, disability and controls", xticks=k)
    ax.legend(fontsize=9)
    return _save(fig, "fig6_h5_auc_by_fold")


def gender_area_plot(cells: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    piv = cells.pivot(index="urban", columns="female", values="p_out_of_school")
    piv.index = ["Rural", "Urban"]; piv.columns = ["Boys", "Girls"]
    piv.plot(kind="bar", ax=ax, color=["#1f77b4", "#d62728"], rot=0)
    ax.set(ylabel="Predicted P(out of school)", xlabel="Area", title="H4: predicted out-of-school probability by gender and area\n(other features at median)")
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f", fontsize=9)
    return _save(fig, "fig7_gender_area_cells")


def subgroup_recall_plot(sub: pd.DataFrame, title: str) -> str:
    d = sub.copy()
    d["group"] = d["variable"] + " = " + d["level"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#bbbbbb" if s else "#1f77b4" for s in d.flag_small]
    ax.barh(d.group, d.recall, color=colors)
    for i, (r, n, p) in enumerate(zip(d.recall, d.n, d.positives)):
        ax.text(r + 0.01, i, f"{r:.2f}  (n={n:,}, pos={p:,})", va="center", fontsize=8)
    ax.set(xlim=(0, 1.15), xlabel="Recall (share of out-of-school children caught)", title=title)
    return _save(fig, "fig8_subgroup_recall")


def tree_top_levels(pipe, features) -> str:
    clf = pipe.named_steps["clf"]
    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(clf, max_depth=3, feature_names=[LABELS.get(f, f) for f in features],
              class_names=["enrolled", "out of school"], filled=True, impurity=False, proportion=True,
              rounded=True, fontsize=8, ax=ax)
    ax.set_title(f"Tuned decision tree - first three levels (full depth = {clf.get_depth()}, leaves = {clf.get_n_leaves()})")
    return _save(fig, "fig9_tree_top_levels")


def rf_importance_bars(gini_rf: pd.DataFrame, perm_rf: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    d = gini_rf.sort_values("gini_importance")
    axes[0].barh(d.label, d.gini_importance, color="#2ca02c")
    axes[0].set(title="Random forest: Gini importance (averaged over 200 trees)", xlabel="Share of impurity reduction")
    d = perm_rf.sort_values("auc_drop_mean")
    axes[1].barh(d.label, d.auc_drop_mean, xerr=d.auc_drop_std, color="#2ca02c", ecolor="k", capsize=3)
    axes[1].set(title="Random forest: permutation importance", xlabel="Drop in hold-out ROC-AUC when feature is shuffled")
    return _save(fig, "fig10_rf_importance")


# ----------------------------------------------------------------------------- Milestone 4 additions
PALETTE.update({"xgb": "#ff7f0e", "mlp": "#8c564b"})


def benchmark_bars(holdout: pd.DataFrame, names: dict, order: list) -> str:
    """Hold-out ROC-AUC and PR-AUC per model; M3 baselines hatched, M4 models solid."""
    d = holdout.set_index("model").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, col, ttl in ((axes[0], "roc_auc", "ROC-AUC (hold-out districts)"), (axes[1], "pr_auc", "PR-AUC (hold-out districts)")):
        bars = ax.barh([names[m] for m in order], d[col], color=[PALETTE.get(m, "#999") for m in order])
        for b, m in zip(bars, order):
            if m in ("naive", "logreg", "tree", "rf"):
                b.set_hatch("//"); b.set_alpha(0.75)
        for b, v in zip(bars, d[col]):
            ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
        ax.set(title=ttl, xlim=(0, min(1, d[col].max() + 0.12)))
    fig.suptitle("Milestone 3 baselines (hatched) vs Milestone 4 advanced models (solid)", fontsize=12)
    return _save(fig, "fig14_benchmark")


def fairness_panels(rates: pd.DataFrame, attr_label: str, names: dict, fname: str, ci: pd.DataFrame | None = None) -> str:
    """Selection rate, TPR and FPR by protected group, one bar per model. rates: columns model, group, selection_rate, tpr, fpr, small."""
    models = list(dict.fromkeys(rates.model))
    groups = list(dict.fromkeys(rates.group))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    width = 0.8 / len(models)
    for ax, col, ttl in ((axes[0], "selection_rate", "Share flagged (selection rate)"),
                         (axes[1], "tpr", "True-positive rate (recall): share of out-of-school found"),
                         (axes[2], "fpr", "False-positive rate: share of enrolled wrongly flagged")):
        for i, m in enumerate(models):
            d = rates[rates.model == m].set_index("group").reindex(groups)
            x = np.arange(len(groups)) + (i - (len(models) - 1) / 2) * width
            cols = [("#bbbbbb" if s else PALETTE.get(m, "#999")) for s in d.small]
            ax.bar(x, d[col], width=width, color=cols, label=names[m], edgecolor="k", linewidth=0.3)
        ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels(groups, rotation=0, fontsize=9)
        ax.set(title=ttl, ylim=(0, 1))
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Fairness audit by {attr_label} at each model's cost-tuned threshold (grey = fewer than 100 positives)", fontsize=12)
    return _save(fig, fname)


def disparity_forest(aud: pd.DataFrame, metric: str, title: str, fname: str, ref: float, names: dict) -> str:
    """Point + 95% CI of one disparity metric: one row per attribute, models offset vertically within the row."""
    models = list(dict.fromkeys(aud.model)); attrs = list(dict.fromkeys(aud.label))
    fig, ax = plt.subplots(figsize=(10.5, 0.95 * len(attrs) + 1.8))
    step = 0.8 / max(len(models), 1)
    for ai, a in enumerate(attrs):
        for mi, m in enumerate(models):
            r = aud[(aud.label == a) & (aud.model == m)]
            if r.empty:
                continue
            r = r.iloc[0]; yy = ai + (mi - (len(models) - 1) / 2) * step
            lo, hi = r.get(f"{metric}_ci_low", np.nan), r.get(f"{metric}_ci_high", np.nan)
            ax.errorbar(r[metric], yy, xerr=[[r[metric] - lo], [hi - r[metric]]] if not np.isnan(lo) else None,
                        fmt="o", ms=5, color="#bbbbbb" if r.small_group else PALETTE.get(m, "#999"), ecolor="k", elinewidth=0.8, capsize=2,
                        label=names[m] if ai == 0 else None)
        if ai < len(attrs) - 1:
            ax.axhline(ai + 0.5, color="#dddddd", lw=0.8)
    ax.axvline(ref, color="k", ls="--", lw=1)
    if metric == "di_ratio":
        ax.axvspan(0.8, 1.25, color="green", alpha=0.08)
    first = aud.drop_duplicates("label").set_index("label")
    ax.set_yticks(range(len(attrs)))
    ax.set_yticklabels([a if " vs " in a else f"{a}: {first.loc[a, 'unpriv_name']} vs {first.loc[a, 'priv_name']}" for a in attrs], fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel=title, title=f"{title}\nwith district-bootstrap 95% intervals (grey = fewer than 100 positives)")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, fname)


def pareto_plot(frontiers: dict, attr_label: str, fname: str, names: dict, chosen_gap: float) -> str:
    """Hold-out cost vs |TPR gap| for each allowed-gap setting; Pareto points filled, dominated hollow; single-threshold point starred."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for m, df in frontiers.items():
        col = PALETTE.get(m, "#999")
        g = df[df.point == "group thresholds"].sort_values("holdout_abs_gap")
        ax.plot(g.holdout_abs_gap, g.holdout_cost_per_1000, "-", color=col, lw=1, alpha=0.6)
        ax.scatter(g[g.pareto].holdout_abs_gap, g[g.pareto].holdout_cost_per_1000, color=col, s=40, label=f"{names[m]} (Pareto points)")
        ax.scatter(g[~g.pareto].holdout_abs_gap, g[~g.pareto].holdout_cost_per_1000, facecolors="none", edgecolors=col, s=40)
        s = df[df.point != "group thresholds"]
        if not s.empty:
            ax.scatter(s.holdout_abs_gap, s.holdout_cost_per_1000, marker="*", s=220, color=col, edgecolors="k", label=f"{names[m]} single threshold")
        c = g[g.max_tpr_gap == chosen_gap]
        if not c.empty:
            ax.annotate(f"gap <= {chosen_gap:.2f}", (c.holdout_abs_gap.iloc[0], c.holdout_cost_per_1000.iloc[0]),
                        textcoords="offset points", xytext=(8, 8), fontsize=8, color=col)
    ax.set(xlabel=f"|TPR gap| between groups ({attr_label}) on hold-out  (0 = equal opportunity)",
           ylabel="Expected cost per 1,000 children (1 FN = 5 FP)",
           title=f"Fairness-accuracy trade-off by {attr_label}: thresholds chosen on training OOF, scored on hold-out")
    ax.legend(fontsize=8)
    return _save(fig, fname)


def before_after_plot(tbl: pd.DataFrame, fname: str) -> str:
    """TPR by group before vs after mitigation for the recommended model; one panel per attribute."""
    attrs = list(dict.fromkeys(tbl.attribute))
    fig, axes = plt.subplots(1, len(attrs), figsize=(4.2 * len(attrs), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, attrs):
        d = tbl[tbl.attribute == a]
        x = np.arange(2)
        ax.bar(x - 0.2, [d.tpr_priv_before.iloc[0], d.tpr_unpriv_before.iloc[0]], 0.4, label="before (single threshold)", color="#9ecae1", edgecolor="k", linewidth=0.3)
        ax.bar(x + 0.2, [d.tpr_priv_after.iloc[0], d.tpr_unpriv_after.iloc[0]], 0.4, label="after (group thresholds)", color="#3182bd", edgecolor="k", linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels([d.priv_name.iloc[0], d.unpriv_name.iloc[0]], fontsize=9)
        ax.set(title=d.label.iloc[0], ylim=(0, 1))
        for xx, v in zip(list(x - 0.2) + list(x + 0.2), [d.tpr_priv_before.iloc[0], d.tpr_unpriv_before.iloc[0], d.tpr_priv_after.iloc[0], d.tpr_unpriv_after.iloc[0]]):
            ax.text(xx, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axes[0].set_ylabel("True-positive rate (share of out-of-school children found)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Post-processing mitigation: recall by group before and after group-specific thresholds (hold-out)", fontsize=11)
    return _save(fig, fname)


def ensemble_importance_bars(gini: pd.DataFrame, perm: pd.DataFrame, model_name: str, color: str, fname: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    d = gini.sort_values("gini_importance")
    axes[0].barh(d.label, d.gini_importance, color=color)
    axes[0].set(title=f"{model_name}: split-gain importance", xlabel="Share of total gain")
    d = perm.sort_values("auc_drop_mean")
    axes[1].barh(d.label, d.auc_drop_mean, xerr=d.auc_drop_std, color=color, ecolor="k", capsize=3)
    axes[1].set(title=f"{model_name}: permutation importance", xlabel="Drop in hold-out ROC-AUC when feature is shuffled")
    return _save(fig, fname)
