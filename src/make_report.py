"""Build report/Milestone04_Report.docx from outputs/results.json and the saved figures.

    .venv/bin/python src/make_report.py

Manuscript enhancement for Milestone 4: Sections 5-9 to be inserted after the Milestone 3 Section 4 draft, replacing the old
"Plan for Milestone 04" bullets. Same conventions as Milestone 3: professional but plain language, every technical term
followed by a short explanation the first time it appears, every number read from results.json / outputs/tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

R = json.load(open(C.OUT_DIR / "results.json"))
M3 = C.M3
T = C.TABLES_DIR
NAMES = {"naive": "Majority-class baseline", "logreg": "Logistic regression", "tree": "Decision tree", "rf": "Random forest",
         "xgb": "XGBoost", "mlp": "Neural network (MLP)"}
BASE = ("logreg", "tree", "rf"); ADV = ("xgb", "mlp"); ALL = (*BASE, *ADV)
RATIO = R["recommended_cost_ratio"]


def fmt(x, nd=3):
    if isinstance(x, (bool, np.bool_)):
        return "Yes" if x else "No"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, (float, np.floating)):
        if pd.isna(x):
            return "-"
        return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.{nd}f}"
    return str(x)


def add_table(doc, df: pd.DataFrame, nd=3, font=8.5, caption: str | None = None):
    if caption:
        p = doc.add_paragraph(); r = p.add_run(caption); r.bold = True; r.font.size = Pt(9.5)
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]; cell.text = str(c)
        for p in cell.paragraphs:
            for r in p.runs: r.bold = True; r.font.size = Pt(font)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = fmt(v, nd)
            for p in cells[i].paragraphs:
                for r in p.runs: r.font.size = Pt(font)
    doc.add_paragraph()


def add_fig(doc, path, caption, width=6.3):
    doc.add_picture(str(path), width=Inches(width)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(); r = p.add_run(caption); r.italic = True; r.font.size = Pt(9); p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def bullets(doc, items):
    for it in items: doc.add_paragraph(it, style="List Bullet")


def para(doc, text): doc.add_paragraph(text)


def main():
    h05 = R["holdout_05"]; cvb = {r["model"]: r for r in R["cv_best"]}; bp = R["best_params"]; rec_thr = R["recommended_threshold"]
    best = R["recommended_model"]; rank = R["model_rank_by_holdout_cost"]; ct = {(r["model"], r["cost_ratio_fn_to_fp"]): r for r in R["cost_thresholds"]}
    tc = {(r["model"], r["threshold_label"]): r for r in R["threshold_comparison"]}
    boots = pd.DataFrame(R["auc_gain_bootstrap"]).set_index(["baseline", "model"])
    aud = pd.DataFrame(R["fairness_audit"]); aud05 = pd.DataFrame(R["fairness_audit_05"])
    A = {(r["model"], r["attribute"]): r for r in aud.to_dict(orient="records")}
    gaps = pd.DataFrame(R["gap_change_vs_m3"]); ba = pd.DataFrame(R["mitigation_before_after"]); rw = pd.DataFrame(R["reweighing"])
    knob = pd.DataFrame(R["group_fn_cost_knob"]); pareto = pd.DataFrame(R["pareto"]); prov = R["province_thresholds"]
    figs = R["figures"]; split = {r["partition"]: r for r in R["split"]}; ds = R["dataset"]
    top_auc = max(ALL, key=lambda m: h05[m]["roc_auc"]); best_base = max(BASE, key=lambda m: h05[m]["roc_auc"])
    m3_best = M3["recommended_model"]
    focus = R["mitigation_focus"]; nds = R["n_distinct_scores"]; contrast = best if best != focus else m3_best
    ba_best = ba[ba.model == focus].set_index("attribute"); rw_x = rw[rw.model == "xgb"].set_index("attribute")
    urb = A[(focus, "urban")]; dis = A[(focus, "any_disability")]; fem = A[(focus, "female")]; bf = A[(focus, "bform")]; po = A[(focus, "poor")]
    sindh = A[(focus, "province:Sindh")]; bal = A[(focus, "province:Balochistan")]
    g_best = {r.attribute: r for r in gaps.itertuples()}
    def gapcol(r, m): return getattr(r, f"{m}_eopp_gap")
    def verdict(r, m): return getattr(r, f"{m}_verdict")

    doc = Document(); st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
    doc.add_heading("Milestone 04 - Advanced Models, Fairness Audit, Mitigation and Governance", 0)
    doc.add_paragraph("Manuscript enhancement: Sections 5-9 below follow the Milestone 3 Section 4 draft and replace its closing "
                      "'Plan for Milestone 04' bullets. Same data, same district-grouped hold-out, same cost framework.").italic = True
    para(doc, "How to read this section. Each technical term is explained in brackets the first time it appears and collected in the glossary "
              "at the end. Every number is produced by the accompanying code (milestone4/src) and traceable to a table in outputs/tables. "
              f"Two figures that recur: {ds['enrolled_share_full_data']:.1%} is the share of enrolled children in the full data set, and "
              f"{h05['naive']['accuracy']:.1%} is the accuracy of the majority-class baseline on the held-out districts; they differ because the "
              "held-out districts have a slightly lower out-of-school share than the whole sample.")

    # ================================================================== 5
    doc.add_heading("5. Advanced models and how they compare with the Milestone 3 baselines", 1)
    doc.add_heading("5.1 Which models were added and why", 2)
    bullets(doc, [
        "XGBoost (gradient boosting: an ensemble that grows decision trees one after another, each new tree correcting the mistakes of "
        "the trees before it; usually the strongest learner for tabular data like a household survey). It was chosen because the random "
        "forest in Milestone 3 showed that tree ensembles rank children better than a single tree or a linear model, and boosting is the "
        "next step up. Boosting can overfit (memorise noise), so the number of trees is set by early stopping (training stops when the score "
        "on a validation slice stops improving) and the tree depth, learning rate (how much each new tree is allowed to change the "
        "prediction) and regularisation (penalties that keep the trees simple) are tuned.",
        "A regularised neural network (a multi-layer perceptron, MLP: layers of simple units that each combine their inputs and pass the "
        "result on, so the model can learn curved and interacting patterns). It covers the 'neural network' option in the brief and is a "
        "different family from trees altogether, which makes it a useful check on whether the trees are missing structure. L2 weight decay "
        "(a penalty that keeps the connection weights small) and early stopping guard against overfitting. Inputs are standardised inside "
        "the pipeline, so the scaler never sees the checking data.",
        "The random forest from Milestone 3 is kept as the third ensemble, and the logistic regression and decision tree are the baselines "
        "to beat. All five are re-fitted on exactly the same training districts and judged on exactly the same "
        f"{split['test']['rows']:,} held-out children in {split['test']['districts']} districts as in Milestone 3; the code asserts that "
        "the hold-out is identical and that the three Milestone 3 models reproduce their published ROC-AUCs (Table 11).",
        "Models considered but not used: a support-vector machine with a radial kernel does not scale to 129,000 rows in the time "
        "available, and LightGBM would duplicate what XGBoost does.",
    ])
    rep = pd.DataFrame(R["baselines_reproduced"]); rep["model"] = rep["model"].map(NAMES)
    rep = rep[["model", "params", "holdout_auc_m3", "holdout_auc_m4_refit", "abs_diff"]]
    rep.columns = ["Milestone 3 model", "Tuned settings carried over", "Hold-out ROC-AUC (Milestone 3)", "Hold-out ROC-AUC (re-fitted here)", "Difference"]
    add_table(doc, rep, nd=4, caption="Table 11. The Milestone 3 baselines re-created on the identical split")

    doc.add_heading("5.2 Hyperparameter search and leakage control", 2)
    para(doc,
        "Hyperparameters (settings chosen by the analyst rather than learned from the data) were tuned with random search (sampling "
        "settings at random from a range instead of trying every combination; it covers wide ranges cheaply) inside the same district-grouped, "
        "stratified 5-fold cross-validation as Milestone 3, choosing the draw with the best mean ROC-AUC and recording PR-AUC alongside. "
        "Table 12 documents the search spaces.")
    ss = R["search_space"]
    rows = [{"Model": NAMES["xgb"], "Setting": k, "Range searched": v} for k, v in ss["xgb"].items()] + \
           [{"Model": NAMES["mlp"], "Setting": k, "Range searched": v} for k, v in ss["mlp"].items()]
    add_table(doc, pd.DataFrame(rows), caption="Table 12. Hyperparameter search spaces")
    xp, mp = bp["xgb"], bp["mlp"]
    para(doc,
        f"The best XGBoost draw used a depth of {xp['max_depth']}, a learning rate of {xp['learning_rate']:.3f}, a minimum child weight of "
        f"{xp['min_child_weight']}, subsample {xp['subsample']:.2f}, column sample {xp['colsample_bytree']:.2f}, L2 penalty {xp['reg_lambda']:.1f} and a "
        f"positive-class weight of {xp['scale_pos_weight']:.1f}; " + (f"the validation PR-AUC was still creeping upward at the cap of {xp['n_estimators_used']:,} trees, so early stopping did not trigger and the cap was used" if xp['n_estimators_used'] >= 1499 else f"early stopping settled on {xp['n_estimators_used']} trees") + ". The best network had "
        f"hidden layers {mp['hidden_layer_sizes']}, L2 penalty {mp['alpha']:.4f} and a learning rate of {mp['learning_rate_init']}.")
    para(doc,
        "Leakage check (information from the test data influencing the model). Four safeguards are in place and asserted in code: "
        "no district appears on both sides of any split; the standardisation step is fitted inside each fold; the validation slice used "
        "for early stopping is carved from the training fold only, never from the fold being checked and never from the hold-out; and every "
        "cut-off is chosen on training out-of-fold predictions (a prediction for every training child made by a model that did not see that "
        "child) and applied once to the hold-out.")

    doc.add_heading("5.3 Comparative performance", 2)
    cvdf = pd.DataFrame(R["cv_best"])[["source", "model", "cv_roc_auc_mean", "cv_roc_auc_sd", "cv_average_precision_mean", "cv_average_precision_sd"]]
    cvdf["model"] = cvdf["model"].map(NAMES)
    cvdf.columns = ["Source", "Model", "CV ROC-AUC", "Spread (SD)", "CV PR-AUC", "Spread (SD)"]
    add_table(doc, cvdf, caption="Table 13. Cross-validated ranking quality for the best settings (mean over 5 district-grouped folds)")
    both = pd.read_csv(T / "t07b_holdout_comparison_both_cutoffs.csv")
    both["model"] = both["model"].map(NAMES)
    both = both[["model", "cutoff", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "flagged_rate"]]
    both.columns = ["Model", "Cut-off", "Value", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Flagged"]
    add_table(doc, both, caption="Table 14. Hold-out performance of every model at the default 0.5 cut-off and at its cost-tuned cut-off "
                                 f"(cost ratio {RATIO}:1, cut-off chosen on training out-of-fold predictions)", font=8)
    add_fig(doc, figs["benchmark"], "Figure 10. Hold-out ROC-AUC and PR-AUC. Hatched bars are the Milestone 3 baselines, solid bars the Milestone 4 models.")
    add_fig(doc, figs["roc_pr"], "Figure 11. ROC and precision-recall curves on the held-out districts for all five models.")
    bt = pd.DataFrame(R["auc_gain_bootstrap"])
    bt["baseline"] = bt["baseline"].map(NAMES); bt["model"] = bt["model"].map(NAMES)
    bt = bt[["model", "baseline", "diff_point", "ci_low", "ci_high", "share_positive"]]
    bt.columns = ["Model", "Compared with", "ROC-AUC gain on hold-out", "95% CI low", "95% CI high", "Share of re-samples with a gain"]
    add_table(doc, bt, caption=f"Table 15. Hold-out ROC-AUC gains with district-cluster bootstrap intervals ({bt.shape[0]} comparisons, 1,000 re-samples of the "
                               f"{split['test']['districts']} held-out districts)")
    xr = boots.loc[("rf", "xgb")]; xl = boots.loc[("logreg", "xgb")]; xt = boots.loc[("tree", "xgb")]; mr = boots.loc[("rf", "mlp")]
    para(doc,
        f"What the comparison shows. XGBoost has the highest hold-out ROC-AUC ({h05['xgb']['roc_auc']:.3f}) and PR-AUC ({h05['xgb']['pr_auc']:.3f}). "
        f"Against the best Milestone 3 model, the random forest ({h05['rf']['roc_auc']:.3f}), the gain is {xr['diff_point']:+.3f} with a 95% interval of "
        f"[{xr['ci_low']:+.3f}, {xr['ci_high']:+.3f}]; against the logistic regression it is {xl['diff_point']:+.3f} [{xl['ci_low']:+.3f}, {xl['ci_high']:+.3f}] "
        f"and against the tree {xt['diff_point']:+.3f} [{xt['ci_low']:+.3f}, {xt['ci_high']:+.3f}]. The network reaches {h05['mlp']['roc_auc']:.3f}, "
        f"{mr['diff_point']:+.3f} [{mr['ci_low']:+.3f}, {mr['ci_high']:+.3f}] relative to the forest. "
        + ("The boosting gain over the forest is small and its interval " + ("excludes" if xr['ci_low'] > 0 else "includes") + " zero: "
           "the thirteen household features set a ceiling that a stronger learner cannot lift far, as the modest pseudo-R-squared "
           f"({M3['pseudo_r2']:.2f}) in Milestone 3 already suggested. We state this plainly rather than overstating it. ")
        + "The clear and well-supported gain is over the linear model, which confirms that the pattern has interactions and non-linear "
          "pieces a weighted sum cannot express.")
    low = [m for m in ALL if h05[m]["recall"] < 0.1]
    para(doc,
        "The cut-off matters more than the model. At the default 0.5 cut-off the " + ", ".join(NAMES[m].lower() for m in low) +
        (" find almost nobody" if len(low) > 1 else " finds almost nobody") + " (Table 14), for the reason given in Milestone 3 section 3: they were tuned on "
        "ROC-AUC without class weighting, so few leaves hold a majority of out-of-school children. XGBoost's search chose a positive-class weight of "
        f"{xp['scale_pos_weight']:.1f}, which centres its scores near 0.5, so its default-cut-off recall is high; the other models' useful operating points are "
        "low cut-offs. All cut-offs are chosen on cost below.")

    doc.add_heading("5.4 Cost-tuned cut-offs and the recommended operating point", 2)
    para(doc,
        f"As in Milestone 3, a missed child (false negative) is priced at r false alarms (false positives), for r = 1, 3, 5 and 10, and the "
        f"cut-off that minimises expected cost per 1,000 children is found on training out-of-fold predictions and applied once to the hold-out. "
        f"Table 16 gives the result at the recommended ratio of {RATIO}:1, including the 'oracle' cut-off (the hold-out's own best value, which "
        "we could only know by peeking) so the reader can see how little selection-on-test would have flattered the numbers.")
    bt9 = pd.read_csv(T / "t09_cost_minimising_thresholds.csv"); bt9 = bt9[bt9.cost_ratio_fn_to_fp == RATIO]
    bt9["model"] = bt9["model"].map(NAMES)
    bt9 = bt9[["model", "threshold", "oof_recall", "oof_precision", "holdout_recall", "holdout_precision", "holdout_flagged_rate", "holdout_cost_per_1000", "holdout_oracle_threshold", "holdout_oracle_cost_per_1000"]]
    bt9.columns = ["Model", "Cut-off (training)", "Train recall", "Train precision", "Hold-out recall", "Hold-out precision", "Hold-out flagged", "Hold-out cost/1,000", "Oracle cut-off", "Oracle cost/1,000"]
    add_table(doc, bt9, caption=f"Table 16. Cost-minimising cut-offs at cost ratio {RATIO}:1", font=8)
    add_fig(doc, figs["threshold"], f"Figure 12. Recall, precision and expected cost against the cut-off (ratio {RATIO}:1). Solid: training out-of-fold "
            "(used for selection); dashed: hold-out (reported only); dash-dot: chosen cut-offs.")
    add_fig(doc, figs["cm_rec"], "Figure 13. Confusion matrices at the recommended cut-offs.")
    br = tc[(best, "recommended")]
    para(doc,
        f"Ranked by hold-out cost per 1,000 at training-chosen cut-offs the models come out as: " + "; ".join(f"{NAMES[m]} {rank[m]:.0f}" for m in rank) + ". "
        f"The recommended operating point is therefore the {NAMES[best]} at a cut-off of {rec_thr[best]:.2f}: for every 1,000 children screened "
        f"it flags {br['flagged_per_1000']:.0f}, finds {br['caught_per_1000']:.0f} of the {br['caught_per_1000'] + br['missed_per_1000']:.0f} who are out of "
        f"school and misses {br['missed_per_1000']:.0f} (recall {br['recall']:.2f}, precision {br['precision']:.2f}). "
        + (f"This replaces the Milestone 3 recommendation ({NAMES[m3_best]} at {M3['recommended_threshold'][m3_best]:.2f}, "
           f"{M3['model_rank_by_holdout_cost'][m3_best]:.0f} per 1,000). " if best != m3_best else
           f"This confirms the Milestone 3 recommendation; the advanced models rank children better but do not lower the cost at this ratio. ")
        + f"The differences between the top models ({', '.join(f'{rank[m]:.0f}' for m in list(rank)[:3])} per 1,000) are within fold-to-fold variation, so the "
          "choice among them should also weigh explainability and the fairness results below.")
    para(doc,
        f"One practical consideration decides which model the mitigation work in section 7 is built on. Group-specific cut-offs need a score "
        f"that varies smoothly, so that a cut-off can be moved by a small amount and have a small effect. A decision tree produces one score per "
        f"leaf: on the hold-out the tree gives only {nds['tree']:,} distinct values, against {nds['xgb']:,} for XGBoost, {nds['rf']:,} for the forest and "
        f"{nds['logreg']:,} for the logistic regression. Among the models within 1% of the lowest cost ({', '.join(NAMES[m] for m in R['mitigation_focus_candidates'])}), "
        f"we take the best ranker, {NAMES[focus]}, as the model for the fairness mitigation and the operating policy" +
        (f"; it is also the cheapest, so the two criteria agree. The decision tree is carried through the audit for contrast." if focus == best else
         f", and report the {NAMES[best]} alongside it. The two differ by {abs(rank[best] - rank[focus]):.1f} per 1,000 in cost, well inside fold-to-fold noise.") )

    doc.add_heading("5.5 What drives the XGBoost predictions", 2)
    gx = pd.read_csv(T / "t17c_xgb_gain_importance.csv")[["feature", "label", "gini_importance"]] \
        .merge(pd.read_csv(T / "t19c_permutation_xgb.csv")[["feature", "auc_drop_mean"]], on="feature") \
        .merge(pd.DataFrame(M3["perm_logreg"])[["feature", "auc_drop_mean"]].rename(columns={"auc_drop_mean": "perm_lr"}), on="feature") \
        .merge(pd.DataFrame(M3["gini_rf"])[["feature", "gini_importance"]].rename(columns={"gini_importance": "gini_rf"}), on="feature")
    gx = gx[["label", "gini_importance", "gini_rf", "auc_drop_mean", "perm_lr"]].sort_values("gini_importance", ascending=False)
    gx.columns = ["Factor", "Gain share (XGBoost)", "Gini share (forest, M3)", "Permutation drop (XGBoost)", "Permutation drop (logistic, M3)"]
    add_table(doc, gx, caption="Table 17. Feature importance of XGBoost next to the Milestone 3 forest and logistic model", font=8)
    add_fig(doc, figs["importance_xgb"], "Figure 14. XGBoost split-gain and hold-out permutation importance.")
    top3 = list(gx["Factor"].iloc[:3])
    para(doc,
        f"The ordering is the same as in Milestone 3: {top3[0]}, {top3[1]} and {top3[2]} carry most of the signal, with disability and B-form "
        "behind them. The advanced models did not discover new drivers; they combine the same ones more flexibly. This matters for the fairness "
        "audit, because the strongest predictors are markers of poverty and connectivity, so the model's view of who is at risk is a view of "
        "who is poor and disconnected.")

    # ================================================================== 6
    doc.add_heading("6. Algorithmic bias and fairness audit", 1)
    doc.add_heading("6.1 Sensitive attributes and how to read a 'positive' here", 2)
    pa = R["protected_attributes"]
    attr_tbl = pd.DataFrame([
        {"Attribute": pa["female"]["label"], "Column": "female", "Privileged": pa["female"]["privileged"], "Unprivileged": pa["female"]["unprivileged"],
         "Why it is sensitive": "SDG 4.5 names gender parity explicitly; Milestone 3 found a rural gender gap."},
        {"Attribute": pa["urban"]["label"], "Column": "urban", "Privileged": pa["urban"]["privileged"], "Unprivileged": pa["urban"]["unprivileged"],
         "Why it is sensitive": "Rural children face higher out-of-school rates; SDG 4.5 covers rural-urban disparity. Milestone 3 found the model finds rural children far more often."},
        {"Attribute": pa["any_disability"]["label"], "Column": "any_disability", "Privileged": pa["any_disability"]["privileged"], "Unprivileged": pa["any_disability"]["unprivileged"],
         "Why it is sensitive": "SDG 4.5 names children with disabilities; the research question asks whether the model misses them."},
        {"Attribute": pa["bform"]["label"], "Column": "bform", "Privileged": pa["bform"]["privileged"], "Unprivileged": pa["bform"]["unprivileged"],
         "Why it is sensitive": "Legal identity; unregistered children are often invisible to services. Milestone 3 found the largest recall gap here."},
        {"Attribute": pa["poor"]["label"], "Column": f"assets_n <= {R['poverty_cut_assets_n_le']:.0f} (bottom tercile)", "Privileged": pa["poor"]["privileged"], "Unprivileged": pa["poor"]["unprivileged"],
         "Why it is sensitive": "Socio-economic class; the asset count is also a model input, so this checks whether the model treats the poorest differently rather than whether it uses poverty at all."},
        {"Attribute": "Province", "Column": "province dummies", "Privileged": "Punjab (reference)", "Unprivileged": "Sindh, KP, Balochistan in turn",
         "Why it is sensitive": "Geographic and ethnic-linguistic identity; Balochistan has the highest out-of-school rate. " + R["ict_excluded_reason"]},
    ])
    add_table(doc, attr_tbl, caption="Table 18. Protected attributes used in the audit", font=8)
    para(doc,
        "One point changes how every fairness number below must be read. In most fairness audits a positive prediction is a burden (a loan "
        "refused, a person stopped). Here a positive prediction means a child is flagged for an enrolment visit, which is help. The harm of "
        "concern is therefore a false negative - an out-of-school child who is never visited - and the disparate-impact worry is a group being "
        "under-flagged relative to its need, not over-flagged. We still report every metric in the conventional direction (unprivileged "
        "relative to privileged) so that both readings are visible, and we treat the equal-opportunity gap as the primary metric because the "
        "true-positive rate is exactly the share of out-of-school children in a group who get found.")
    doc.add_heading("6.2 The metrics", 2)
    bullets(doc, [
        "Selection rate: the share of a group that the model flags.",
        "Demographic parity difference: selection rate of the unprivileged group minus that of the privileged group. Disparate-impact ratio: "
        f"the same two rates divided; the four-fifths rule treats values between {C.DI_BAND[0]} and {C.DI_BAND[1]} as acceptable.",
        "Equal opportunity gap: true-positive rate (recall: the share of truly out-of-school children who are flagged) of the unprivileged "
        "group minus that of the privileged group. Negative means the unprivileged group is found less often. This is the primary metric.",
        "Predictive equality gap: false-positive rate (the share of enrolled children wrongly flagged) of the unprivileged group minus that of "
        "the privileged group. A positive value means enrolled children in that group bear more unnecessary visits.",
        "Equalized odds difference: the larger of the two gaps above in absolute value; zero means both error rates are equal across groups.",
        f"Every gap carries a 95% interval from a district-cluster bootstrap (re-sampling the {split['test']['districts']} held-out districts "
        f"1,000 times), and any group with fewer than {C.SMALL_GROUP_POSITIVES} out-of-school children is marked small and greyed in the figures.",
    ])
    doc.add_heading("6.3 Results", 2)
    para(doc,
        f"Table 19 reports the audit for the mitigation-focus model ({NAMES[focus]}) and, for contrast, the {NAMES[contrast]} at their cost-tuned cut-offs; the same table for "
        "all five models and for the default 0.5 cut-off is in outputs/tables/t22_fairness_audit.csv. Figures 15 to 17 show the pattern.")
    show = aud[aud.model.isin([focus, contrast])].copy()
    show["model"] = show["model"].map(NAMES)
    show = show[["model", "label", "sel_priv", "sel_unpriv", "di_ratio", "di_within_band", "tpr_priv", "tpr_unpriv", "eopp_gap", "eopp_gap_ci_low", "eopp_gap_ci_high", "fpr_priv", "fpr_unpriv", "pe_gap", "eodds_diff", "small_group"]]
    show.columns = ["Model", "Attribute", "Flag rate (priv.)", "Flag rate (unpriv.)", "DI ratio", "Within 0.8-1.25", "TPR (priv.)", "TPR (unpriv.)", "Equal-opp. gap", "CI low", "CI high", "FPR (priv.)", "FPR (unpriv.)", "Pred.-equality gap", "Eq. odds diff.", "Small group"]
    add_table(doc, show, nd=2, caption=f"Table 19. Fairness audit at the cost-tuned cut-offs ({NAMES[focus]} at {rec_thr[focus]:.2f}, {NAMES[contrast]} at {rec_thr[contrast]:.2f}); unprivileged relative to privileged", font=7)
    add_fig(doc, figs["eopp_forest"], "Figure 15. Equal-opportunity gap (TPR unprivileged minus privileged) with 95% district-bootstrap intervals, all models. Negative = unprivileged group found less often.")
    add_fig(doc, figs["di_forest"], "Figure 16. Disparate-impact ratio with intervals; the green band is the four-fifths rule.")
    add_fig(doc, figs["fairness"]["urban"], "Figure 17. Selection rate, true-positive rate and false-positive rate by area for all five models.")
    add_fig(doc, figs["fairness"]["any_disability"], "Figure 17b. The same by disability status.")
    add_fig(doc, figs["fairness"]["province"], "Figure 17c. The same by province (Islamabad excluded: no hold-out rows).")
    def sgn(v): return "found more often" if v > 0 else "found less often"
    bullets(doc, [
        f"Area. Rural children are {sgn(urb['eopp_gap'])} than urban children by {abs(urb['eopp_gap']):.2f} in true-positive rate "
        f"(TPR {urb['tpr_unpriv']:.2f} rural versus {urb['tpr_priv']:.2f} urban; interval [{urb['eopp_gap_ci_low']:+.2f}, {urb['eopp_gap_ci_high']:+.2f}]). "
        f"The disparate-impact ratio is {urb['di_ratio']:.2f}: rural children are flagged {urb['di_ratio']:.1f} times as often. In the conventional reading "
        "this fails the four-fifths rule against urban children; in the help-allocation reading it is partly legitimate because rural prevalence "
        "is higher, but the TPR gap says urban out-of-school children are genuinely under-found, which is the Milestone 3 weakness persisting.",
        f"Gender. Girls are {sgn(fem['eopp_gap'])} than boys ({fem['tpr_unpriv']:.2f} versus {fem['tpr_priv']:.2f}, gap {fem['eopp_gap']:+.2f}, interval "
        f"[{fem['eopp_gap_ci_low']:+.2f}, {fem['eopp_gap_ci_high']:+.2f}]), and flagged more often (DI ratio {fem['di_ratio']:.2f}). Enrolled girls also bear a higher "
        f"false-positive rate ({fem['fpr_unpriv']:.2f} versus {fem['fpr_priv']:.2f}).",
        f"Disability. Children with a disability are {sgn(dis['eopp_gap'])} than children without ({dis['tpr_unpriv']:.2f} versus {dis['tpr_priv']:.2f}, "
        f"gap {dis['eopp_gap']:+.2f}, interval [{dis['eopp_gap_ci_low']:+.2f}, {dis['eopp_gap_ci_high']:+.2f}]), with a DI ratio of {dis['di_ratio']:.2f}. "
        + ("This is a reversal of Milestone 3, where the logistic model found disabled children more often (gap "
           f"{g_best['any_disability'].m3_logreg_eopp_gap:+.2f}). The tree ensembles lean on housing and digital access, and disability, a weaker split "
           "variable, loses influence. For SDG 4.5 this is the most important finding of the audit." if dis["eopp_gap"] < 0 else
           "The direction favours children with disabilities, consistent with Milestone 3."),
        f"Legal identity. Unregistered children are found more often ({bf['tpr_unpriv']:.2f} versus {bf['tpr_priv']:.2f}, gap {bf['eopp_gap']:+.2f}) and flagged "
        f"{bf['di_ratio']:.1f} times as often. Registered out-of-school children, who are more likely to be dropouts than never-enrolled, are the ones missed.",
        f"Poverty. The poorest tercile is found far more often ({po['tpr_unpriv']:.2f} versus {po['tpr_priv']:.2f}) and flagged {po['di_ratio']:.1f} times as often, "
        "with a much higher false-positive rate: enrolled poor children receive most of the unnecessary visits. This is the model's reliance on "
        "poverty markers made visible.",
        f"Province. Out-of-school children in Balochistan are found at {bal['tpr_unpriv']:.2f} against {bal['tpr_priv']:.2f} in Punjab, those in Sindh at "
        f"{sindh['tpr_unpriv']:.2f}. Sindh is the one province with a disparate-impact ratio below 0.8 ({sindh['di_ratio']:.2f}): its children are "
        "under-flagged in both readings.",
    ])
    doc.add_heading("6.4 Base rates and why not every fairness criterion can hold at once", 2)
    para(doc,
        f"Out-of-school prevalence differs sharply between groups ({M3['subgroup_counts_file'][8]['prevalence']:.0%} in Balochistan against "
        f"{M3['subgroup_counts_file'][12]['prevalence']:.0%} in Sindh; {M3['subgroup_counts_file'][2]['prevalence']:.0%} rural against {M3['subgroup_counts_file'][3]['prevalence']:.0%} urban). "
        "A known impossibility result (Chouldechova 2017; Kleinberg, Mullainathan and Raghavan 2016) shows that when base rates differ, a "
        "classifier cannot simultaneously equalise selection rates (demographic parity), both error rates (equalized odds) and the precision "
        "of its flags (calibration within groups), except in trivial cases. A choice has to be made. Because the harm here is a missed child, "
        "we prioritise equal opportunity (equal true-positive rates) and accept unequal selection rates, which will naturally be higher where "
        "need is higher. Demographic parity is reported for completeness, not targeted.")
    doc.add_heading("6.5 Did the advanced models fix, preserve or worsen the Milestone 3 gaps?", 2)
    gtab = gaps.copy(); gtab = gtab[["label", "m3_logreg_eopp_gap"] + [c for m in ("tree", "rf", "xgb", "mlp") for c in (f"{m}_eopp_gap", f"{m}_verdict")]]
    gtab.columns = ["Attribute", "M3 logistic gap"] + [f"{NAMES[m]} {w}" for m in ("tree", "rf", "xgb", "mlp") for w in ("gap", "verdict")]
    add_table(doc, gtab, nd=2, caption="Table 20. Equal-opportunity gap (TPR unprivileged minus privileged) in Milestone 3 and for each model now; 'reduced' / 'worsened' means the absolute gap moved by more than 0.05", font=7.5)
    para(doc,
        f"Milestone 3 flagged three gaps as starting hypotheses: urban versus rural recall ({M3['subgroups_logreg'][3]['recall']:.2f} versus {M3['subgroups_logreg'][2]['recall']:.2f}), "
        f"registered versus unregistered, and Sindh versus Balochistan. The tree ensembles {verdict(g_best['urban'], 'xgb')} the area gap "
        f"(XGBoost {gapcol(g_best['urban'], 'xgb'):+.2f} against {g_best['urban'].m3_logreg_eopp_gap:+.2f}) and {verdict(g_best['bform'], 'xgb')} the legal-identity gap "
        f"({gapcol(g_best['bform'], 'xgb'):+.2f} against {g_best['bform'].m3_logreg_eopp_gap:+.2f}), but none closed them: better ranking does not by itself produce fair "
        f"operating points. The disability gap moved from {g_best['any_disability'].m3_logreg_eopp_gap:+.2f} to {gapcol(g_best['any_disability'], 'xgb'):+.2f} for XGBoost and "
        f"{gapcol(g_best['any_disability'], 'mlp'):+.2f} for the network, which the verdict column records as a reduction in size but which is in fact a change of sign: "
        "children with disabilities went from being found more often to being found less often. A fairness audit is needed precisely to catch "
        "a change like this, which no accuracy metric would show.")
    doc.add_heading("6.6 Fairness-accuracy trade-off: the Pareto frontier", 2)
    para(doc,
        "To map the trade-off we enforce a fairness constraint of varying strength and record what it costs. The constraint is a maximum "
        "allowed gap in true-positive rate between the two groups of an attribute; for each allowed gap the cost-minimising pair of "
        "group-specific cut-offs is found on training out-of-fold predictions (section 7.1 explains the method) and then scored on the hold-out. "
        "Plotting hold-out cost against the hold-out gap gives one point per setting; the Pareto frontier (the set of points where no other "
        "point is both fairer and cheaper) is the curve of best achievable trade-offs. The star marks the single, group-blind cut-off from Table 16.")
    for k in C.PARETO_ATTRS:
        add_fig(doc, figs["pareto"][k], f"Figure 18{'abc'[C.PARETO_ATTRS.index(k)]}. Fairness-accuracy frontier by {C.PROTECTED[k]['label'].lower()}: expected cost per 1,000 against |TPR gap| on the hold-out; filled points are Pareto-optimal, the star is the single cut-off.", 5.8)
    pt = pareto[(pareto.model == focus) & (pareto.attribute == "urban")]
    single = pt[pt.point != "group thresholds"].iloc[0]; gp = pt.dropna(subset=["max_tpr_gap"]).reset_index(drop=True)
    tight = gp.iloc[gp.max_tpr_gap.idxmin()]; loose = gp.iloc[gp.max_tpr_gap.idxmax()]
    chosen = gp.iloc[(gp.max_tpr_gap - C.RECOMMENDED_TPR_GAP).abs().idxmin()]
    para(doc,
        f"Reading the area frontier for the {NAMES[focus]}: the single cut-off sits at a hold-out gap of {single['holdout_abs_gap']:.2f} and a cost of "
        f"{single['holdout_cost_per_1000']:.0f} per 1,000. Letting the two cut-offs differ with only a loose constraint (allowed gap {loose['max_tpr_gap']:.2f}) gives "
        f"a gap of {loose['holdout_abs_gap']:.2f} at {loose['holdout_cost_per_1000']:.0f}; tightening to an allowed gap of {chosen['max_tpr_gap']:.2f} gives "
        f"{chosen['holdout_abs_gap']:.2f} at {chosen['holdout_cost_per_1000']:.0f}; forcing the gap towards zero (allowed {tight['max_tpr_gap']:.3f}) costs "
        f"{tight['holdout_cost_per_1000']:.0f} for a hold-out gap of {tight['holdout_abs_gap']:.2f}" +
        (", which is both fairer and cheaper than the single cut-off: on these districts the group-blind cut-off was simply mis-set for urban children. " if tight['holdout_cost_per_1000'] < single['holdout_cost_per_1000'] else ". ") +
        f"The frontier is shallow: {1 - tight['holdout_abs_gap'] / single['holdout_abs_gap']:.0%} of the gap can be removed for a cost change of "
        f"{abs(tight['holdout_cost_per_1000'] / single['holdout_cost_per_1000'] - 1):.1%}, because the model's ranking within each group is already reasonable and only the cut-offs are misaligned. "
        f"Our recommended operating point is the allowed gap of {chosen['max_tpr_gap']:.2f}: on the hold-out it removes "
        f"{1 - chosen['holdout_abs_gap'] / single['holdout_abs_gap']:.0%} of the disparity at a cost within fold-to-fold "
        "noise, and the hold-out gap it actually achieves is reported rather than assumed. Note that the hold-out gap is usually larger than the "
        "training gap the constraint was set on; this is the honest number.")

    # ================================================================== 7
    doc.add_heading("7. Bias mitigation and asymmetric threshold tuning", 1)
    doc.add_heading("7.1 Methods", 2)
    bullets(doc, [
        "Post-processing: group-specific cut-offs (main method). The trained model is left untouched; each group gets its own cut-off, chosen "
        "on training out-of-fold predictions to minimise the expected cost (a missed child = 5 false alarms) subject to the true-positive rates "
        f"of the groups differing by at most {C.RECOMMENDED_TPR_GAP:.2f}. For a two-level attribute every pair of cut-offs on a 0.0025 grid is "
        "evaluated; for province a coordinate search is used. The chosen cut-offs are applied once to the hold-out. This is asymmetric threshold "
        "tuning in both senses the brief asks for: the error costs are asymmetric and the cut-offs differ across demographic groups.",
        "Pre-processing: reweighing (Kamiran and Calders 2012). Each training child is given a weight equal to P(group) x P(label) / P(group and "
        "label), which makes the protected attribute statistically independent of the label in the training data. The model is re-trained with "
        "these weights and its single cut-off re-chosen on out-of-fold predictions. Applied to the logistic regression and XGBoost for area, "
        "gender and legal identity.",
        "In-processing (changing the learning objective itself, for example fairlearn's ExponentiatedGradient with a true-positive-rate "
        "parity constraint) was not implemented in the time available and is listed as future work.",
        f"A policy extension: group-specific false-negative costs. SDG 4.5 gives extra weight to girls and children with disabilities, so we also "
        f"show the cut-offs that result when a missed child in those groups costs {C.GROUP_FN_MULTIPLIER['female']:.1f} times as much, with no gap "
        "constraint. This is a knob for officials, not a recommendation.",
    ])
    doc.add_heading("7.2 Group-specific cut-offs: before and after", 2)
    bb = ba[ba.model == focus].copy()
    bb = bb[["label", "threshold_before", "thresholds_after", "tpr_priv_before", "tpr_unpriv_before", "tpr_priv_after", "tpr_unpriv_after", "eopp_gap_before", "eopp_gap_after",
             "fpr_priv_after", "fpr_unpriv_after", "pe_gap_before", "pe_gap_after", "di_before", "di_after", "cost_before", "cost_after", "flagged_per_1000_before", "flagged_per_1000_after"]]
    bb.columns = ["Attribute", "Cut-off before", "Cut-offs after {unpriv/priv}", "TPR priv. before", "TPR unpriv. before", "TPR priv. after", "TPR unpriv. after", "EO gap before", "EO gap after",
                  "FPR priv. after", "FPR unpriv. after", "PE gap before", "PE gap after", "DI before", "DI after", "Cost/1,000 before", "Cost/1,000 after", "Flagged/1,000 before", "Flagged/1,000 after"]
    add_table(doc, bb, nd=2, caption=f"Table 21. {NAMES[focus]}: single cut-off versus group-specific cut-offs (allowed TPR gap {C.RECOMMENDED_TPR_GAP:.2f}), hold-out districts", font=6.5)
    add_fig(doc, figs["before_after"], f"Figure 19. True-positive rate by group before and after group-specific cut-offs, {NAMES[focus]}.")
    u = ba_best.loc["urban"]; d = ba_best.loc["any_disability"]; b_ = ba_best.loc["bform"]
    bullets(doc, [
        f"Area: against the single cut-off of {rec_thr[focus]:.2f}, the group cut-offs become {u['thresholds_after']} (rural / urban); urban recall moves from {u['tpr_priv_before']:.2f} to {u['tpr_priv_after']:.2f} "
        f"and rural from {u['tpr_unpriv_before']:.2f} to {u['tpr_unpriv_after']:.2f}; the gap shrinks from {u['eopp_gap_before']:+.2f} to {u['eopp_gap_after']:+.2f} at a cost change from "
        f"{u['cost_before']:.0f} to {u['cost_after']:.0f} per 1,000 and {u['flagged_per_1000_after'] - u['flagged_per_1000_before']:+.0f} children flagged per 1,000.",
        f"Disability: cut-offs {d['thresholds_after']}; recall for children with a disability moves from {d['tpr_unpriv_before']:.2f} to {d['tpr_unpriv_after']:.2f} and the gap from "
        f"{d['eopp_gap_before']:+.2f} to {d['eopp_gap_after']:+.2f}, cost {d['cost_before']:.0f} to {d['cost_after']:.0f}." +
        (" The gap is wider on the hold-out after mitigation than before, even though it was closed on training: children with a disability are a small group whose scores shift between districts, so a cut-off tuned for them on training does not transfer. This is the clearest failure case in the exercise and is why Table 19 shows a wide interval for this attribute; the equity cost knob in section 7.5 is the more robust lever here." if abs(d['eopp_gap_after']) > abs(d['eopp_gap_before']) else
         (" The gap is not closed on the hold-out even though it was on training: the disability group is small enough that its cut-off does not transfer cleanly, which is why the interval in Table 19 is wide." if abs(d['eopp_gap_after']) > C.RECOMMENDED_TPR_GAP else "")),
        f"Legal identity: cut-offs {b_['thresholds_after']}; the gap moves from {b_['eopp_gap_before']:+.2f} to {b_['eopp_gap_after']:+.2f}, cost {b_['cost_before']:.0f} to {b_['cost_after']:.0f}.",
        (lambda t: f"The same procedure on the {NAMES['tree']} illustrates why a smooth score matters: its disability cut-offs {t['thresholds_after']} satisfied the "
                   f"gap on training but moved the hold-out gap from {t['eopp_gap_before']:+.2f} to {t['eopp_gap_after']:+.2f}, and its cost rose from {t['cost_before']:.0f} to "
                   f"{t['cost_after']:.0f}. With one score per leaf, a per-group cut-off jumps whole leaves at a time and does not transfer to new districts.")
        (ba[ba.model == "tree"].set_index("attribute").loc["any_disability"]) if (ba.model == "tree").any() and focus != "tree" else
        "Group cut-offs on the single decision tree were also computed (outputs/tables/t24_mitigation_before_after.csv) and behave less stably because the tree gives one score per leaf.",
        "The burden shifts. Lowering a group's cut-off raises its false-positive rate: more enrolled children in that group receive an "
        "unnecessary visit (the 'FPR after' columns). Equalising the chance of being found therefore means accepting unequal rates of false "
        "alarms. This is the trade-off the frontier in section 6.6 quantifies, and it should be decided by the programme, not hidden in a default.",
    ])
    doc.add_heading("7.3 Province: the levelling-down problem", 2)
    pv = pd.DataFrame(prov["table"]); pv = pv[["group", "n", "positives", "threshold_after", "tpr_before", "tpr_after", "fpr_before", "fpr_after", "selection_rate_before", "selection_rate_after"]]
    pv.columns = ["Province", "Children (test)", "Out of school", "Cut-off after", "TPR before", "TPR after", "FPR before", "FPR after", "Flag rate before", "Flag rate after"]
    add_table(doc, pv, nd=2, caption=f"Table 22. Province-specific cut-offs for the {NAMES[prov['model']]} (allowed TPR gap {C.RECOMMENDED_TPR_GAP:.2f}); cost per 1,000 {prov['cost_before']:.0f} before, {prov['cost_after']:.0f} after", font=8)
    para(doc,
        "Equalising true-positive rates across four provinces is achieved mostly by raising Balochistan's cut-off, so that fewer of its "
        "out-of-school children are found, rather than by finding more in Punjab and Sindh. This is levelling down: the metric improves while "
        "fewer children are helped overall. We report it as a warning. A one-sided rule (no group may be found less often than the best-found "
        "group minus the tolerance, but groups may exceed it) avoids the problem at a higher cost, and is the form we would recommend if "
        "province-specific cut-offs were ever used in practice.")
    doc.add_heading("7.4 Reweighing", 2)
    rwt = rw.copy(); rwt["model"] = rwt["model"].map(NAMES)
    rwt = rwt[["model", "label", "auc_before", "auc_after", "threshold_before", "threshold_after", "tpr_priv_before", "tpr_unpriv_before", "tpr_priv_after", "tpr_unpriv_after", "eopp_gap_before", "eopp_gap_after", "di_before", "di_after", "cost_before", "cost_after"]]
    rwt.columns = ["Model", "Attribute", "ROC-AUC before", "ROC-AUC after", "Cut-off before", "Cut-off after", "TPR priv. before", "TPR unpriv. before", "TPR priv. after", "TPR unpriv. after", "EO gap before", "EO gap after", "DI before", "DI after", "Cost/1,000 before", "Cost/1,000 after"]
    add_table(doc, rwt, nd=2, caption="Table 23. Pre-processing by reweighing: the model is re-trained on weights that decouple the attribute from the label; single cut-off re-chosen on training", font=7)
    ru = rw_x.loc["urban"]; rb = rw_x.loc["bform"]
    para(doc,
        f"Reweighing works through the model rather than around it. For area it reduces the XGBoost gap from {ru['eopp_gap_before']:+.2f} to {ru['eopp_gap_after']:+.2f} "
        f"at a cost change from {ru['cost_before']:.0f} to {ru['cost_after']:.0f} per 1,000, with ROC-AUC {ru['auc_before']:.3f} to {ru['auc_after']:.3f}. For legal identity it "
        f"overshoots (gap {rb['eopp_gap_before']:+.2f} to {rb['eopp_gap_after']:+.2f}) at a cost of {rb['cost_after']:.0f}. It removes the attribute's association with the label "
        "in training, which is a blunter instrument than a cut-off tuned to a target gap, and it cannot be aimed at one attribute without affecting "
        "the others. Group-specific cut-offs remain our main method; reweighing is the fallback where using the attribute at decision time is "
        "not permitted (section 8).")
    doc.add_heading("7.5 The equity cost knob", 2)
    kt = knob.copy(); kt["model"] = kt["model"].map(NAMES)
    kt = kt[["model", "label", "fn_multiplier_unpriv", "thresholds", "tpr_priv", "tpr_unpriv", "eopp_gap", "di_ratio", "cost_standard_per_1000", "flagged_per_1000"]]
    kt.columns = ["Model", "Attribute", "FN cost multiplier (unpriv.)", "Cut-offs {unpriv/priv}", "TPR priv.", "TPR unpriv.", "EO gap", "DI ratio", "Cost/1,000 (standard)", "Flagged/1,000"]
    add_table(doc, kt, nd=2, caption=f"Table 24. Cut-offs when a missed girl or missed child with a disability costs {C.GROUP_FN_MULTIPLIER['female']:.1f} times a missed child elsewhere (no gap constraint)", font=8)
    para(doc,
        "Pricing a missed girl or a missed child with a disability above other missed children lowers their cut-off and raises their recall, "
        "deliberately creating a gap in their favour. This is the SDG 4.5 equity priority written as a number. It is presented as a dial for "
        "officials, with the standard-cost column showing what the choice costs in the usual accounting.")
    doc.add_heading("7.6 Recommended operating policy", 2)
    bullets(doc, [
        f"Model: {NAMES[focus]} (sections 5.4 and 7.2): the best ranker, the cheapest at its cost-tuned cut-off, and with a score smooth enough for group cut-offs, unlike the tree's. Where child-by-child explanations are required, the logistic regression remains the fallback.",
        f"Cut-offs: area-specific, chosen on training out-of-fold predictions at cost ratio {RATIO}:1 with an allowed TPR gap of {C.RECOMMENDED_TPR_GAP:.2f} "
        f"({u['thresholds_after']} for rural/urban). On the hold-out this finds {u['tpr_priv_after']:.0%} of urban and {u['tpr_unpriv_after']:.0%} of rural out-of-school children "
        f"and flags {u['flagged_per_1000_after']:.0f} per 1,000.",
        "Monitor, do not constrain, the other attributes in the first instance: report the disability, gender, legal-identity and province gaps "
        "with intervals every cycle; act when an interval excludes zero on the wrong side, as it does for disability under the tree ensembles.",
        "Using a protected attribute (area) to set the cut-off is a deliberate, documented policy choice with legal and ethical weight; section 8 "
        "treats it as a decision for the programme's governance body rather than a technical default.",
    ])

    # ================================================================== 8
    doc.add_heading("8. Governance: FATEN alignment and living-lab deployment", 1)
    doc.add_heading("8.1 FATEN scorecard", 2)
    faten = pd.DataFrame([
        {"Principle": "Fairness", "What it means here": "Out-of-school children have a similar chance of being found whatever their sex, area, disability, registration or province.",
         "Evidence": f"Area gap {urb['eopp_gap']:+.2f} and disability gap {dis['eopp_gap']:+.2f} at the single cut-off; area gap {u['eopp_gap_after']:+.2f} after group cut-offs; Sindh DI ratio {sindh['di_ratio']:.2f}.",
         "Status": "Partly met", "Remaining gap": "Disability and Sindh gaps not yet mitigated; equal opportunity cannot be met jointly with demographic parity."},
        {"Principle": "Autonomy", "What it means here": "Families and field officers keep the decision; the score advises, it does not decide.",
         "Evidence": "The output is a ranked outreach list with a documented cut-off; no automatic action is attached to a flag.",
         "Status": "Met in design", "Remaining gap": "An appeal/contest channel and a right to decline a visit must be written into the pilot protocol."},
        {"Principle": "Trust", "What it means here": "Stakeholders can understand and verify what the model does.",
         "Evidence": f"Logistic model fully transparent (odds ratios); {NAMES[focus]} explained through importance tables (Table 17) and this audit; all code and seeds published; hold-out never used for choices.",
         "Status": "Partly met", "Remaining gap": "A model card and a plain-language leaflet for communities; calibration check so a '30%' means 30%."},
        {"Principle": "Equity", "What it means here": "The system reaches the groups SDG 4.5 names: girls, children with disabilities, the rural poor.",
         "Evidence": f"Girls found at {fem['tpr_unpriv']:.2f} vs boys {fem['tpr_priv']:.2f}; poorest tercile {po['tpr_unpriv']:.2f} vs {po['tpr_priv']:.2f}; disabled {dis['tpr_unpriv']:.2f} vs {dis['tpr_priv']:.2f}.",
         "Status": "Partly met", "Remaining gap": "Children with disabilities are the group the advanced models serve worst; the equity cost knob (Table 24) is the lever."},
        {"Principle": "Non-maleficence", "What it means here": "No harm from being flagged (stigma, scrutiny) or from being missed.",
         "Evidence": f"Precision about {tc[(focus, 'recommended')]['precision']:.2f}: roughly {1 - tc[(focus, 'recommended')]['precision']:.0%} of flagged children are enrolled and would receive an unnecessary visit; {tc[(focus, 'recommended')]['missed_per_1000']:.0f} per 1,000 are missed.",
         "Status": "Partly met", "Remaining gap": "Visits must be framed as support, never enforcement; no punitive use; data minimisation for children's records."},
    ])
    add_table(doc, faten, caption=f"Table 25. FATEN scorecard for the {NAMES[focus]} and the operating policy of section 7.6", font=7.5)
    doc.add_heading("8.2 Living lab: how this would behave in the field", 2)
    para(doc,
        "A living lab (a real-world pilot run with the people who would use the system, measured as it runs) is the right next step before any "
        "scale-up. We propose a pilot in three to five districts of mixed type (one urban, one rural, one in Balochistan) with district education "
        "officers, in which the model's outreach list is used alongside the officers' usual lists for one survey cycle and both are followed up.")
    para(doc, "Failure modes we expect, from the evidence above:")
    bullets(doc, [
        "Missed urban dropouts. The model has learned 'poor, rural, disconnected household means out of school'. Urban and better-off children "
        "who dropped out look enrolled on paper and are the children it misses most (Milestone 3 Table 6; area gap here). A list built only from "
        "the model would systematically skip them.",
        "Wrongly flagged poor families. The poorest tercile bears a false-positive rate of "
        f"{po['fpr_unpriv']:.2f}: many enrolled children in poor households would be visited. Repeated unnecessary visits can be experienced as "
        "scrutiny, and the burden falls on the group least able to object.",
        "Drift. Asset ownership and digital access change fast; a model trained on 2025 asset and phone patterns will mis-rank households within "
        "a few years. Retraining on each ASER round and a drift check on the input distributions are needed.",
        "Gaming and reporting effects. If households learn that certain answers trigger a visit or a stipend, survey answers will shift. The model "
        "should never be the sole trigger for a benefit.",
        "Reinforcing existing outreach geography. The model concentrates flags where prevalence is already known to be high (Balochistan, rural "
        "areas). Used naively it would confirm where officers already go and withdraw attention from Sindh and urban areas, where its disparate-impact "
        "ratio is lowest.",
        "Digital marginalisation. The two strongest predictors are housing quality and digital use. A system that scores children by their "
        "household's connectivity risks a feedback loop in which the least connected are both most flagged and least able to contest a flag, "
        "while connected households with out-of-school children are overlooked. This is the specific digital-marginalisation risk the brief asks about.",
    ])
    para(doc, "Governance safeguards for the pilot:")
    bullets(doc, [
        "Human review of every list before use, with officers able to add and remove names and the reasons logged.",
        "An appeal and opt-out channel for families, in local languages, and an explicit rule that a flag never carries a penalty or conditions a benefit.",
        "A monitoring dashboard that reports, each cycle, recall and false-positive rate by sex, area, disability, registration and province with "
        "intervals, and the disparate-impact ratio against the four-fifths band; a pre-agreed threshold at which the cut-offs are revisited.",
        "A written policy decision on whether group-specific cut-offs are permitted, taken by the programme's governance body, with the "
        "frontier (Figure 18) and the levelling-down warning (Table 22) in front of it.",
        "Periodic re-audit and retraining triggers tied to each ASER round, with the hold-out protocol repeated so that reported numbers remain honest.",
        "Data protection for children's records: minimisation (only the thirteen features), no re-identification of households beyond the outreach "
        "list, access limited to the district team, and deletion schedules.",
        "A published model card summarising intended use, data, performance by group, known failure modes and the cost ratio in force.",
    ])

    # ================================================================== 9
    doc.add_heading("9. Limitations and conclusions", 1)
    bullets(doc, [
        f"Modest ceiling. The best hold-out ROC-AUC is {h05[top_auc]['roc_auc']:.3f}; the advanced models add {xr['diff_point']:+.3f} over the forest and "
        f"{xl['diff_point']:+.3f} over the logistic regression. Thirteen household features cannot separate enrolled from dropped-out children in "
        "similar households; school-side variables (distance, supply, cost) and the child's history are the missing information.",
        "Cross-sectional, associational evidence (Milestone 3 section 7): no driver has an estimated causal effect, and the fairness numbers "
        "describe how the model's errors fall, not why children are out of school.",
        f"Small groups and one missing province. Islamabad has no hold-out rows; disability intervals are wide; any group with fewer than "
        f"{C.SMALL_GROUP_POSITIVES} out-of-school children is marked and should not drive decisions.",
        "The impossibility result means equal opportunity was chosen over demographic parity; a different programme could reasonably choose otherwise.",
        "Cut-offs chosen on training out-of-fold predictions transfer imperfectly: hold-out gaps are larger than the training gaps the constraints "
        "were set on, and the oracle columns show the small flattery that selection on test would have introduced. Both are reported.",
        "In-processing mitigation and calibration were not implemented in the time available.",
    ])
    para(doc,
        f"Conclusions. Gradient boosting is the best ranker of out-of-school risk on unseen districts, but the step up from Milestone 3 is modest and "
        "the cut-off still decides who gets found. The fairness audit shows that out-of-school children in urban areas, in Sindh and, under the "
        "tree ensembles, with disabilities are found less often than others, while the rural poor are flagged most and bear most unnecessary visits. "
        "Group-specific cut-offs chosen on training data remove most of the area gap at a cost within noise; reweighing is a blunter alternative; "
        "the frontier shows the trade-off is shallow. The remaining decisions - whether to use a protected attribute at decision time, how to price "
        "a missed girl or disabled child, and how to avoid levelling down across provinces - are governance choices, and the pilot design above is "
        "built to surface them.")

    doc.add_heading("Reproducibility", 1)
    para(doc, "Code: milestone4/src (config, data, cv, models, fairness, evaluate, importance, plots, run_all, make_report) and the executed notebook "
              "Milestone04_AdvancedModels.ipynb. src/run_all.py regenerates every table and figure from data/aser2025_training.csv with random seed 42; "
              "the Milestone 3 baselines are re-created from data/m3_results.json and the district-to-fold assignment is saved in outputs/tables/t00_fold_assignment.csv "
              f"so the team can reproduce the identical split. Runtime about {R['runtime_sec'] / 60:.0f} minutes.")

    doc.add_heading("Glossary of new terms", 1)
    gl = pd.DataFrame([
        ("Gradient boosting / XGBoost", "An ensemble that grows trees one after another, each correcting the errors of the last; XGBoost is a widely used implementation."),
        ("Learning rate", "How much each new boosting tree is allowed to change the prediction; smaller is slower but safer."),
        ("Early stopping", "Stopping training when performance on a validation slice stops improving, to avoid overfitting."),
        ("Neural network / multi-layer perceptron", "Layers of simple units that combine their inputs and pass results on, able to learn curved and interacting patterns."),
        ("L2 penalty / weight decay", "A term that keeps model weights small so the model does not over-react to noise."),
        ("Random search", "Sampling hyperparameter settings at random from ranges instead of trying every combination."),
        ("Out-of-fold prediction", "A prediction for a training child made by a model that did not see that child."),
        ("Protected / sensitive attribute", "A characteristic such as sex, disability or area along which fairness is assessed."),
        ("Privileged / unprivileged group", "The reference group and the group whose treatment is compared with it."),
        ("Selection rate", "The share of a group the model flags."),
        ("Demographic parity / disparate-impact ratio", "Equal selection rates across groups / the ratio of the two selection rates; 0.8-1.25 is the four-fifths band."),
        ("Equal opportunity", "Equal true-positive rates across groups: each group's out-of-school children have the same chance of being found."),
        ("Equalized odds", "Equal true-positive and false-positive rates across groups."),
        ("Predictive equality", "Equal false-positive rates across groups."),
        ("Impossibility result", "When base rates differ, demographic parity, equalized odds and calibration cannot all hold at once."),
        ("Pareto frontier", "The set of trade-off points where no other point is both fairer and cheaper."),
        ("Post- / pre- / in-processing", "Fixing bias after training (cut-offs), before training (reweighing) or inside training (constrained objective)."),
        ("Reweighing", "Giving each training row a weight so the protected attribute and the label become independent."),
        ("Levelling down", "Improving a fairness metric by making the better-served group worse off rather than the worse-served group better off."),
        ("FATEN", "Fairness, Autonomy, Trust, Equity, Non-maleficence: five principles for responsible deployment."),
        ("Living lab", "A real-world pilot run with the intended users and measured as it runs."),
        ("Model card", "A short public document describing a model's intended use, data, performance by group and known limitations."),
    ], columns=["Term", "Meaning"])
    add_table(doc, gl, font=8.5)

    out = C.REPORT_DIR / "Milestone04_Report.docx"; doc.save(out); print("wrote", out)


if __name__ == "__main__":
    main()
