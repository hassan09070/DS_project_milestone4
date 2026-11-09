"""Write HEADLINE.md (summary tables for the README) from outputs/results.json."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "outputs" / "results.json"))
N = {"naive": "Majority class", "logreg": "Logistic regression", "tree": "Decision tree", "rf": "Random forest", "xgb": "XGBoost", "mlp": "Neural network (MLP)"}
h = R["holdout_05"]; tc = {(r["model"], r["threshold_label"]): r for r in R["threshold_comparison"]}; cv = {r["model"]: r for r in R["cv_best"]}
rows = ["# Headline results", "", "Hold-out: 32,516 children in 29 districts (identical to Milestone 3). Cut-offs chosen on training out-of-fold predictions, cost ratio 5:1.", "",
        "| Model | Source | CV ROC-AUC (mean +/- sd) | Hold-out ROC-AUC | Hold-out PR-AUC | Cost-tuned cut-off | Recall / precision there | Flagged per 1,000 | Cost per 1,000 |", "|---|---|---|---|---|---|---|---|---|"]
for m in ("naive", "logreg", "tree", "rf", "xgb", "mlp"):
    src = "M3" if m in ("naive", "logreg", "tree", "rf") else "M4"
    c = cv.get(m); cvs = f"{c['cv_roc_auc_mean']:.3f} +/- {c['cv_roc_auc_sd']:.3f}" if c else "-"
    r = tc.get((m, "recommended"))
    rows.append(f"| {N[m]} | {src} | {cvs} | {h[m]['roc_auc']:.3f} | {h[m]['pr_auc']:.3f} | " + (f"{r['threshold']:.2f} | {r['recall']:.2f} / {r['precision']:.2f} | {r['flagged_per_1000']:.0f} | {r['cost_per_1000_ratio5']:.0f} |" if r else f"- | - | - | {tc[('naive', '0.50 (default)')]['cost_per_1000_ratio5']:.0f} |"))
b = pd.DataFrame(R["auc_gain_bootstrap"])
rows += ["", "## Bootstrap AUC gains (1,000 district re-samples)", "", "| Model | vs | Gain | 95% CI |", "|---|---|---|---|"]
for _, r in b.iterrows(): rows.append(f"| {N[r.model]} | {N[r.baseline]} | {r.diff_point:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] |")
f = R["mitigation_focus"]; a = pd.DataFrame(R["fairness_audit"]); a = a[a.model == f]
rows += ["", f"## Fairness audit, {N[f]} at cut-off {R['recommended_threshold'][f]:.2f} (unprivileged vs privileged)", "", "| Attribute | DI ratio | TPR priv. | TPR unpriv. | Equal-opp. gap [95% CI] | Pred.-equality gap |", "|---|---|---|---|---|---|"]
for _, r in a.iterrows(): rows.append(f"| {r.label} | {r.di_ratio:.2f} | {r.tpr_priv:.2f} | {r.tpr_unpriv:.2f} | {r.eopp_gap:+.2f} [{r.eopp_gap_ci_low:+.2f}, {r.eopp_gap_ci_high:+.2f}] | {r.pe_gap:+.2f} |")
m = pd.DataFrame(R["mitigation_before_after"]); m = m[m.model == f]
rows += ["", f"## Post-processing: group-specific cut-offs, {N[f]} (allowed TPR gap {R['pareto'][0]['max_tpr_gap'] if False else 0.05})", "", "| Attribute | Cut-offs after {unpriv/priv} | EO gap before -> after | Cost/1,000 before -> after | Flagged/1,000 before -> after |", "|---|---|---|---|---|"]
for _, r in m.iterrows(): rows.append(f"| {r.label} | {r.thresholds_after} | {r.eopp_gap_before:+.2f} -> {r.eopp_gap_after:+.2f} | {r.cost_before:.0f} -> {r.cost_after:.0f} | {r.flagged_per_1000_before:.0f} -> {r.flagged_per_1000_after:.0f} |")
rw = pd.DataFrame(R["reweighing"])
rows += ["", "## Pre-processing: reweighing", "", "| Model | Attribute | EO gap before -> after | Cost/1,000 before -> after |", "|---|---|---|---|"]
for _, r in rw.iterrows(): rows.append(f"| {N[r.model]} | {r.label} | {r.eopp_gap_before:+.2f} -> {r.eopp_gap_after:+.2f} | {r.cost_before:.0f} -> {r.cost_after:.0f} |")
rows += ["", f"Mitigation focus model: {N[f]} (distinct hold-out scores: " + ", ".join(f"{N[k]} {v:,}" for k, v in R["n_distinct_scores"].items()) + f"). Runtime {R['runtime_sec']/60:.0f} min."]
(ROOT / "HEADLINE.md").write_text("\n".join(rows) + "\n"); print("wrote HEADLINE.md")
