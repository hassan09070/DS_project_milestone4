"""Generate Milestone04_AdvancedModels.ipynb from cell definitions, then execute it with nbconvert.

The notebook imports src/ and replays the pipeline in the rubric's order: setup, baselines reproduced, advanced tuning,
comparison + bootstrap, fairness audit, mitigation + threshold optimisation, Pareto, export. Every code cell starts with a
one-line comment stating its purpose.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Milestone 04 - Advanced Models, Fairness Audit and Bias Mitigation (ASER 2025, Pakistan)

**Research question.** Among children aged 5-16 in Pakistan, can we predict whether a child is out of school from household
poverty indicators, legal identity, disability and demographic / geographic characteristics - and does the model work equally
well for girls, children with disabilities and rural children? (SDG 4.1 access, SDG 4.5 disparities.)

**What this notebook does.** Builds on Milestone 3 (logistic regression, decision tree, random forest on a district-grouped
hold-out). It (1) reproduces those baselines on the identical split, (2) tunes two advanced models - XGBoost with early stopping
and a regularised MLP - by random search inside grouped stratified 5-fold CV, (3) compares all models with district-bootstrap
confidence intervals, (4) audits group fairness (demographic parity / disparate impact, equal opportunity, equalized odds,
predictive equality) over gender, area, disability, legal identity, poverty and province, (5) mitigates bias with group-specific
cost-tuned cut-offs (post-processing) and reweighing (pre-processing), (6) maps the fairness-accuracy Pareto frontier, and
(7) exports every table and figure used in the manuscript. All modelling code lives in `src/`; `src/run_all.py` reproduces
everything from the command line and `src/make_report.py` builds the Word report from `outputs/results.json`.

**Leakage rules.** District never crosses a split; scaling and early-stopping validation slices are fitted inside each training
fold; every cut-off is chosen on training out-of-fold predictions and applied once to the hold-out.
""")

md("## 1. Setup, data loading, seeds and the saved split")
code("""
# Imports, display options, and the src/ modules (config reads column roles from data/feature_manifest.json)
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import Image, display
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 3)
sys.path.insert(0, str(Path("src").resolve()))
import config as C
from data import load_training, split_xy, describe_split, province_of
from cv import make_holdout, make_inner_cv, tune_random, cv_table, fold_scores, fold_assignment
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
import models as M, evaluate as ev, fairness as F, importance as imp, plots
print("random state:", C.RANDOM_STATE, "| target:", C.TARGET, "| features:", len(C.FEATURES))
print("Milestone 3 best params carried over:", C.M3_BEST_PARAMS)
""")
code("""
# Load the Milestone 2 training file with integrity assertions, and re-create the exact Milestone 3 hold-out (fold 0 of a seeded StratifiedGroupKFold by district)
df = load_training()
X, y, groups, w = split_xy(df)
train_idx, test_idx = make_holdout(y, groups)
split_desc = describe_split(df, train_idx, test_idx)
m3_test = C.M3["split"][1]
assert int(split_desc.loc[1, "rows"]) == m3_test["rows"] and int(split_desc.loc[1, "districts"]) == m3_test["districts"], "hold-out differs from Milestone 3"
Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]; ytr, yte = y[train_idx], y[test_idx]
gtr, gte = groups[train_idx], groups[test_idx]; wte = w[test_idx]
df_tr = df.iloc[train_idx].reset_index(drop=True); df_te = df.iloc[test_idx].reset_index(drop=True)
display(split_desc)
# Save the district -> fold assignment so every teammate gets the identical split
fa = fold_assignment(df, y, groups, train_idx, test_idx, C.GROUP_COL)
fa.to_csv(C.TABLES_DIR / "t00_fold_assignment.csv", index=False)
fa.groupby(["partition", "inner_fold"]).size()
""")

md("""
## 2. Milestone 3 baselines reproduced
The three Milestone 3 models are re-fitted with their tuned settings (no re-tuning) so the comparison is like-for-like. The
assertion checks that their hold-out ROC-AUCs match the Milestone 3 report.
""")
code("""
# Re-fit LR / tree / RF with the Milestone 3 parameters and verify hold-out AUCs against data/m3_results.json
probas, fitted = {}, {}
fitted["naive"] = M.build_naive().fit(Xtr, ytr); probas["naive"] = fitted["naive"].predict_proba(Xte)[:, 1]
rows = []
for key in M.BASELINES:
    fitted[key] = M.build_from_m3(key, C.M3_BEST_PARAMS[key]).fit(Xtr, ytr)
    probas[key] = fitted[key].predict_proba(Xte)[:, 1]
    rows.append({"model": key, "auc_now": ev.metrics(yte, probas[key])["roc_auc"], "auc_m3": C.M3["holdout_05"][key]["roc_auc"]})
repro = pd.DataFrame(rows); repro["abs_diff"] = (repro.auc_now - repro.auc_m3).abs()
assert (repro.abs_diff < 2e-3).all()
repro
""")

md("""
## 3. Advanced model tuning (random search inside grouped stratified CV)
**XGBoost** (gradient boosting) is wrapped so that early stopping uses a stratified 10% slice carved from *its own training fold* -
never the CV test fold, never the hold-out. **MLP** uses L2 weight decay and early stopping, with the scaler inside the pipeline.
Both are tuned with `RandomizedSearchCV` on the same five district-grouped folds, refit on ROC-AUC with PR-AUC recorded. Search
spaces are in `models.XGB_DIST` / `models.MLP_DIST` and documented in the report (Table 12). Search results are saved to CSV.
""")
code("""
# XGBoost random search (30 draws x 5 folds); full results -> outputs/tables/t03c_cv_random_xgb.csv
rs_xgb = tune_random(M.build_xgb(), M.XGB_DIST, Xtr, ytr, gtr, n_iter=M.XGB_N_ITER)
cv_table(rs_xgb).to_csv(C.TABLES_DIR / "t03c_cv_random_xgb.csv", index=False)
fitted["xgb"] = rs_xgb.best_estimator_; probas["xgb"] = rs_xgb.predict_proba(Xte)[:, 1]
print("best:", rs_xgb.best_params_, "| CV ROC-AUC %.4f" % rs_xgb.best_score_, "| trees used:", fitted["xgb"].named_steps["clf"].best_iteration_ + 1)
cv_table(rs_xgb).head(8)
""")
code("""
# MLP random search (12 draws x 5 folds); full results -> outputs/tables/t03d_cv_random_mlp.csv
rs_mlp = tune_random(M.build_mlp(), M.MLP_DIST, Xtr, ytr, gtr, n_iter=M.MLP_N_ITER)
cv_table(rs_mlp).to_csv(C.TABLES_DIR / "t03d_cv_random_mlp.csv", index=False)
fitted["mlp"] = rs_mlp.best_estimator_; probas["mlp"] = rs_mlp.predict_proba(Xte)[:, 1]
print("best:", rs_mlp.best_params_, "| CV ROC-AUC %.4f" % rs_mlp.best_score_)
cv_table(rs_mlp).head(6)
""")
code("""
# Cross-validated summary (mean +/- SD over folds) for the new models next to the Milestone 3 ones
cv_rows = [r for r in C.M3["cv_best"] if r["model"] in M.BASELINES]
for key, rs in (("xgb", rs_xgb), ("mlp", rs_mlp)):
    row = {"model": key}
    for m in ("roc_auc", "average_precision", "recall", "precision", "f1", "accuracy"):
        f = fold_scores(rs, m); row[f"cv_{m}_mean"], row[f"cv_{m}_sd"] = f.mean(), f.std(ddof=1)
    cv_rows.append(row)
pd.DataFrame(cv_rows)[["model", "cv_roc_auc_mean", "cv_roc_auc_sd", "cv_average_precision_mean", "cv_average_precision_sd"]]
""")

md("""
## 4. Comparison tables and district bootstrap
Hold-out metrics for every model at the default 0.5 cut-off and (below, section 6) at the cost-tuned cut-off. AUC gains are
given 95% intervals by re-sampling the 29 held-out districts 1,000 times, because children within a district are not independent.
""")
code("""
# Hold-out metrics at the 0.5 cut-off, unweighted and ASER-weighted
rows = []
for key in ("naive", *M.ALL_MODELS):
    rows.append({"model": key, "weighted": False, **ev.metrics(yte, probas[key], 0.5)})
    rows.append({"model": key, "weighted": True, **ev.metrics(yte, probas[key], 0.5, w=wte)})
holdout = pd.DataFrame(rows)
holdout[~holdout.weighted][["model", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "flagged_rate"]]
""")
code("""
# District-cluster bootstrap of hold-out ROC-AUC gains (advanced model minus baseline)
boots = []
for a, b in (("rf", "xgb"), ("logreg", "xgb"), ("tree", "xgb"), ("rf", "mlp"), ("logreg", "mlp"), ("xgb", "mlp")):
    boots.append({"baseline": a, "model": b, **ev.cluster_bootstrap_auc_diff(yte, probas[a], probas[b], gte)})
pd.DataFrame(boots)[["model", "baseline", "diff_point", "ci_low", "ci_high", "share_positive"]]
""")
code("""
# ROC / PR curves and the benchmark bars (hatched = Milestone 3 baselines)
display(Image(plots.roc_pr_curves(yte, {k: probas[k] for k in M.ALL_MODELS}, M.MODEL_NAMES, float(yte.mean()))))
display(Image(plots.benchmark_bars(holdout[~holdout.weighted].assign(model_name=lambda d: d.model.map(M.MODEL_NAMES)), M.MODEL_NAMES, ["naive", *M.ALL_MODELS])))
""")

md("""
### 4.1 Cost-tuned cut-offs on training out-of-fold predictions
As in Milestone 3: a false negative costs r false positives (r = 1, 3, 5, 10; 5 recommended). Each tuned model is re-fitted on the
grouped folds to obtain out-of-fold probabilities for every training child; the cost-minimising cut-off is chosen there and applied
once to the hold-out. The "oracle" column shows the hold-out's own optimum for transparency.
""")
code("""
# Out-of-fold probabilities, cost sweeps, chosen cut-offs and hold-out performance at them
oof = {key: cross_val_predict(clone(fitted[key]), Xtr, ytr, groups=gtr, cv=make_inner_cv(), method="predict_proba", n_jobs=-1)[:, 1] for key in M.ALL_MODELS}
sweeps_oof, sweeps_ho, best_t = {}, {}, []
for key in M.ALL_MODELS:
    sw = ev.threshold_sweep(ytr, oof[key]); sh = ev.threshold_sweep(yte, probas[key]); sweeps_oof[key], sweeps_ho[key] = sw, sh
    bt = ev.best_thresholds(sw); bt.insert(0, "model", key)
    for i, r in bt.iterrows():
        c = ev.confusion(yte, (probas[key] >= r["threshold"]).astype(int)); m = ev.metrics_from_counts(c)
        bt.loc[i, "holdout_recall"], bt.loc[i, "holdout_precision"], bt.loc[i, "holdout_cost_per_1000"] = m["recall"], m["precision"], ev.expected_cost(c, r["cost_ratio_fn_to_fp"])
        col = f"cost_per_1000_ratio{int(r['cost_ratio_fn_to_fp'])}"; bt.loc[i, "holdout_oracle_threshold"] = sh.loc[sh[col].idxmin(), "threshold"]
    best_t.append(bt)
best_t = pd.concat(best_t, ignore_index=True)
rec_thr = {k: float(best_t[(best_t.model == k) & (best_t.cost_ratio_fn_to_fp == C.RECOMMENDED_COST_RATIO)].threshold.item()) for k in M.ALL_MODELS}
at5 = best_t[best_t.cost_ratio_fn_to_fp == C.RECOMMENDED_COST_RATIO].set_index("model")
rec_model = at5.holdout_cost_per_1000.idxmin()
# Mitigation focus: best ranker among models within 1% of the lowest cost (group cut-offs need a smooth score; a tree gives one score per leaf)
near = [k for k in M.ALL_MODELS if at5.loc[k, "holdout_cost_per_1000"] <= 1.01 * at5.holdout_cost_per_1000.min()]
focus = max(near, key=lambda k: ev.metrics(yte, probas[k])["roc_auc"])
print("cut-offs:", rec_thr, "| cheapest:", rec_model, "| mitigation focus:", focus, "| distinct hold-out scores:", {k: len(np.unique(probas[k].round(6))) for k in M.ALL_MODELS})
at5[["threshold", "recall", "precision", "holdout_recall", "holdout_precision", "holdout_cost_per_1000", "holdout_oracle_threshold"]]
""")
code("""
# Threshold / cost curves (solid = training OOF used for selection, dashed = hold-out)
display(Image(plots.threshold_curves(sweeps_oof, sweeps_ho, rec_thr, M.MODEL_NAMES, C.RECOMMENDED_COST_RATIO)))
""")

md("""
## 5. Fairness metric functions and the audit
**Protected attributes**: gender (girls), area (rural), disability, legal identity (unregistered), poverty (bottom asset tercile),
province (each vs Punjab; Islamabad excluded - no hold-out rows). **Framing**: a positive prediction is a flag for outreach (help),
so the harm is a false negative and the disparate-impact concern is a group being *under-flagged* relative to need. Metrics are
nevertheless reported in the conventional direction (unprivileged relative to privileged). Functions are hand-coded in
`src/fairness.py` on top of the 2x2 confusion counts; see `F.disparity` for the definitions.
""")
code("""
# Build the attribute arrays for train and test, including the poverty flag (tercile cut computed on training only)
poor_cut = float(np.quantile(df_tr["assets_n"], 1 / 3))
def attr_arrays(d):
    a = {k: d[k].to_numpy() for k in ("female", "urban", "any_disability", "bform")}
    a["poor"] = (d["assets_n"] <= poor_cut).astype(int).to_numpy(); a["province"] = province_of(d).to_numpy(); return a
A_tr, A_te = attr_arrays(df_tr), attr_arrays(df_te)
specs = {}
for k, s in C.PROTECTED.items():
    specs[k] = {"values": A_te[k], "priv": s["priv"], "unpriv": s["unpriv"], "label": s["label"], "priv_name": s["names"][s["priv"]], "unpriv_name": s["names"][s["unpriv"]]}
for prov in sorted(set(A_te["province"]) - {"Punjab"} - set(C.PROVINCE_ATTR["exclude"])):
    specs[f"province:{prov}"] = {"values": A_te["province"], "priv": "Punjab", "unpriv": prov, "label": f"Province: {prov} vs Punjab", "priv_name": "Punjab", "unpriv_name": prov}
print("poverty flag: assets_n <=", poor_cut, "| attributes:", list(specs))
""")
code("""
# Worked example of the metrics for one model and one attribute (area), then the full audit with district-bootstrap CIs
yhat = (probas[focus] >= rec_thr[focus]).astype(int)
display(F.group_rates(yte, yhat, A_te["urban"], C.PROTECTED["urban"]["names"]))
d = F.disparity(yte, yhat, A_te["urban"], priv=1, unpriv=0)
print({k: round(v, 3) if isinstance(v, float) else v for k, v in d.items() if k in ("sel_priv", "sel_unpriv", "di_ratio", "di_within_band", "tpr_priv", "tpr_unpriv", "eopp_gap", "fpr_priv", "fpr_unpriv", "pe_gap", "eodds_diff")})
aud_rows = []
for key in M.ALL_MODELS:
    a = F.audit(yte, (probas[key] >= rec_thr[key]).astype(int), specs, clusters=gte, n_boot=1000); a.insert(0, "model", key); aud_rows.append(a)
aud = pd.concat(aud_rows, ignore_index=True)
aud[aud.model.isin([focus, rec_model])][["model", "label", "di_ratio", "di_within_band", "tpr_priv", "tpr_unpriv", "eopp_gap", "eopp_gap_ci_low", "eopp_gap_ci_high", "pe_gap", "eodds_diff", "small_group"]]
""")
code("""
# Forest plots of the equal-opportunity gap and the disparate-impact ratio across all models and attributes
display(Image(plots.disparity_forest(aud, "eopp_gap", "Equal-opportunity gap: TPR(unprivileged) - TPR(privileged)", "fig11b_eopp_gap_forest", 0.0, M.MODEL_NAMES)))
display(Image(plots.disparity_forest(aud, "di_ratio", "Disparate-impact ratio", "fig11c_di_ratio_forest", 1.0, M.MODEL_NAMES)))
""")
code("""
# Did the advanced models fix, preserve or worsen the Milestone 3 recall gaps? (M3 logistic regression as the reference)
m3 = pd.DataFrame(C.M3["subgroups_logreg"])
def m3_recall(var, lvl):
    r = m3[(m3.variable == var) & (m3.level.astype(str) == str(lvl))]; return float(r.recall.iloc[0]) if len(r) else np.nan
rows = []
for k in ("female", "urban", "any_disability", "bform", "province:Sindh", "province:Balochistan"):
    s = specs[k]; var = "province" if k.startswith("province:") else k
    g3 = m3_recall(var, s["unpriv"]) - m3_recall(var, s["priv"]); row = {"attribute": k, "m3_gap": g3}
    for key in M.ALL_MODELS:
        g = float(aud[(aud.model == key) & (aud.attribute == k)].eopp_gap.iloc[0]); row[key] = g
        row[f"{key}_verdict"] = "reduced" if abs(g) < abs(g3) - 0.05 else ("worsened" if abs(g) > abs(g3) + 0.05 else "preserved")
    rows.append(row)
pd.DataFrame(rows)
""")

md("""
## 6. Mitigation and asymmetric threshold optimisation
**Post-processing (main)**: a cut-off per group, chosen on training OOF predictions to minimise expected cost (1 FN = 5 FP) subject to
|TPR gap| <= 0.05, applied once to the hold-out. **Pre-processing**: Kamiran-Calders reweighing, model re-trained with sample weights,
single cut-off re-chosen on OOF. **Policy knob**: a missed girl / missed child with a disability costs 1.5x.
""")
code("""
# Group-specific cut-offs for the mitigation-focus model (fine 0.0025 grid), before/after on the hold-out
rows = []
for k in C.PROTECTED:
    s = specs[k]; g = F.group_thresholds(ytr, oof[focus], A_tr[k], C.RECOMMENDED_COST_RATIO, C.RECOMMENDED_TPR_GAP, grid=F.FINE_GRID)
    yb = (probas[focus] >= rec_thr[focus]).astype(int); ya = F.apply_group_thresholds(probas[focus], A_te[k], g["thresholds"])
    db, da = F.disparity(yte, yb, A_te[k], s["priv"], s["unpriv"]), F.disparity(yte, ya, A_te[k], s["priv"], s["unpriv"])
    cb, ca = ev.confusion(yte, yb), ev.confusion(yte, ya)
    rows.append({"attribute": k, "thresholds": {str(a): b for a, b in g["thresholds"].items()}, "feasible_oof": g["feasible"],
                 "tpr_priv_before": db["tpr_priv"], "tpr_unpriv_before": db["tpr_unpriv"], "tpr_priv_after": da["tpr_priv"], "tpr_unpriv_after": da["tpr_unpriv"],
                 "eopp_gap_before": db["eopp_gap"], "eopp_gap_after": da["eopp_gap"], "fpr_unpriv_before": db["fpr_unpriv"], "fpr_unpriv_after": da["fpr_unpriv"],
                 "cost_before": ev.expected_cost(cb, 5), "cost_after": ev.expected_cost(ca, 5),
                 "flagged_before": ev.per_1000(cb)["flagged_per_1000"], "flagged_after": ev.per_1000(ca)["flagged_per_1000"]})
ba = pd.DataFrame(rows); ba
""")
code("""
# Reweighing (pre-processing) for area, gender and legal identity on logistic regression and XGBoost
rows = []
for k in ("urban", "female", "bform"):
    s = specs[k]
    for key in ("logreg", "xgb"):
        wts = F.reweighing_weights(ytr, A_tr[k]); pipe = clone(fitted[key]).fit(Xtr, ytr, clf__sample_weight=wts)
        oof_w = cross_val_predict(clone(fitted[key]), Xtr, ytr, groups=gtr, cv=make_inner_cv(), method="predict_proba", n_jobs=-1, params={"clf__sample_weight": wts})[:, 1]
        thr_w = float(ev.best_thresholds(ev.threshold_sweep(ytr, oof_w)).set_index("cost_ratio_fn_to_fp").loc[5, "threshold"])
        p_w = pipe.predict_proba(Xte)[:, 1]
        db = F.disparity(yte, (probas[key] >= rec_thr[key]).astype(int), A_te[k], s["priv"], s["unpriv"]); da = F.disparity(yte, (p_w >= thr_w).astype(int), A_te[k], s["priv"], s["unpriv"])
        rows.append({"model": key, "attribute": k, "auc_before": ev.metrics(yte, probas[key])["roc_auc"], "auc_after": ev.metrics(yte, p_w)["roc_auc"],
                     "eopp_gap_before": db["eopp_gap"], "eopp_gap_after": da["eopp_gap"], "di_before": db["di_ratio"], "di_after": da["di_ratio"],
                     "cost_before": ev.expected_cost(ev.confusion(yte, (probas[key] >= rec_thr[key]).astype(int)), 5), "cost_after": ev.expected_cost(ev.confusion(yte, (p_w >= thr_w).astype(int)), 5)})
pd.DataFrame(rows)
""")
code("""
# Equity cost knob: a missed girl / missed child with a disability costs 1.5x (no gap constraint)
rows = []
for k, mult in C.GROUP_FN_MULTIPLIER.items():
    s = specs[k]; g = F.group_thresholds(ytr, oof[focus], A_tr[k], 5, 1.0, fn_multiplier={s["unpriv"]: mult, s["priv"]: 1.0}, grid=F.FINE_GRID)
    ya = F.apply_group_thresholds(probas[focus], A_te[k], g["thresholds"]); d = F.disparity(yte, ya, A_te[k], s["priv"], s["unpriv"])
    rows.append({"attribute": k, "thresholds": {str(a): b for a, b in g["thresholds"].items()}, "tpr_priv": d["tpr_priv"], "tpr_unpriv": d["tpr_unpriv"], "eopp_gap": d["eopp_gap"],
                 "cost_standard": ev.expected_cost(ev.confusion(yte, ya), 5)})
pd.DataFrame(rows)
""")

md("""
## 7. Fairness-accuracy Pareto frontier
For each allowed TPR gap the cost-minimising group cut-offs are found on training OOF and then scored on the hold-out; filled points
are non-dominated (no other point is both fairer and cheaper), the star is the single group-blind cut-off.
""")
code("""
# Frontiers for gender, area and disability (recommended model and XGBoost)
frontiers = {a: {} for a in C.PARETO_ATTRS}
for key in dict.fromkeys([focus, rec_model, "tree"]):   # Milestone 3 tree kept as a contrast
    for k in C.PARETO_ATTRS:
        s = specs[k]
        frontiers[k][key] = F.pareto_frontier(ytr, oof[key], A_tr[k], yte, probas[key], A_te[k], s["priv"], s["unpriv"], 5, single_threshold=rec_thr[key])
for k in C.PARETO_ATTRS:
    display(Image(plots.pareto_plot(frontiers[k], C.PROTECTED[k]["label"], f"fig12_pareto_{k}", M.MODEL_NAMES, C.RECOMMENDED_TPR_GAP)))
frontiers["urban"][focus][["max_tpr_gap", "oof_tpr_gap", "holdout_abs_gap", "holdout_cost_per_1000", "holdout_recall", "pareto", "point"]]
""")

md("""
## 8. Export
`src/run_all.py` is the authoritative exporter (it writes every t*.csv, fig*.png and results.json, and `src/make_report.py` builds
the Word report from them). The cell below writes the notebook's own copies of the main tables so the two can be diffed.
""")
code("""
# Write the notebook's tables next to the pipeline's (suffix _nb) and list the figures produced
aud.to_csv(C.TABLES_DIR / "nb_fairness_audit.csv", index=False); ba.to_csv(C.TABLES_DIR / "nb_mitigation_before_after.csv", index=False)
best_t.to_csv(C.TABLES_DIR / "nb_cost_thresholds.csv", index=False)
print(sorted(p.name for p in C.FIGURES_DIR.glob("*.png")))
""")
code("""
# Environment for reproducibility
import sklearn, xgboost, platform
print("python", platform.python_version(), "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__, "| xgboost", xgboost.__version__)
""")

nb.cells = cells
out = Path(__file__).resolve().parents[1] / "Milestone04_AdvancedModels.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
