"""End-to-end Milestone 4 pipeline: advanced models, fairness audit, mitigation, Pareto frontier.

    .venv/bin/python src/run_all.py

Writes outputs/tables/*.csv, outputs/figures/*.png and outputs/results.json. Every number in the report comes from results.json.
Milestone 3 baselines are re-created with their tuned parameters (data/m3_results.json) on the identical district split.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C                                                      # noqa: E402
from cv import cv_table, fold_assignment, fold_scores, make_holdout, make_inner_cv, tune_random  # noqa: E402
from data import describe_split, load_subgroup_counts, load_training, province_of, split_xy      # noqa: E402
import evaluate as ev                                                    # noqa: E402
import fairness as F                                                     # noqa: E402
import importance as imp                                                 # noqa: E402
import models as M                                                       # noqa: E402
import plots                                                             # noqa: E402

T = C.TABLES_DIR
QUICK = bool(int(__import__("os").environ.get("M4_QUICK", "0")))     # M4_QUICK=1 -> tiny searches / bootstraps for a smoke test
N_BOOT = 50 if QUICK else 1000
if QUICK:
    M.XGB_N_ITER, M.MLP_N_ITER = 2, 2
def save(df: pd.DataFrame, name: str) -> None: df.to_csv(T / f"{name}.csv", index=False)
def pyk(d: dict) -> dict: return {str(k): (v.item() if hasattr(v, "item") else v) for k, v in d.items()}


def main() -> dict:
    t0 = time.time()
    R: dict = {"random_state": C.RANDOM_STATE, "features": C.FEATURES, "target": C.TARGET,
               "m3_source": "data/m3_results.json (frozen copy of Milestone 3 results.json)"}

    # ================================================================== 1. data + identical split to Milestone 3
    df = load_training()
    X, y, groups, w = split_xy(df)
    train_idx, test_idx = make_holdout(y, groups)
    split_desc = describe_split(df, train_idx, test_idx); save(split_desc, "t01_split_description")
    m3_test = C.M3["split"][1]
    assert int(split_desc.loc[1, "rows"]) == m3_test["rows"] and int(split_desc.loc[1, "positives"]) == m3_test["positives"] \
        and int(split_desc.loc[1, "districts"]) == m3_test["districts"], "hold-out differs from Milestone 3"
    fa = fold_assignment(df, y, groups, train_idx, test_idx, C.GROUP_COL); save(fa, "t00_fold_assignment")
    R["split"] = split_desc.to_dict(orient="records"); R["split_identical_to_m3"] = True
    R["dataset"] = {"rows": int(len(df)), "positives": int(y.sum()), "prevalence": float(y.mean()), "districts": int(df[C.GROUP_COL].nunique()),
                    "enrolled_share_full_data": float(1 - y.mean())}
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]; ytr, yte = y[train_idx], y[test_idx]
    gtr, gte = groups[train_idx], groups[test_idx]; wte = w[test_idx]
    df_te = df.iloc[test_idx].reset_index(drop=True); df_tr = df.iloc[train_idx].reset_index(drop=True)
    print(f"[split] train {len(ytr):,} / test {len(yte):,} rows; test districts {split_desc.loc[1,'districts']}; identical to M3")

    # ================================================================== 2. Milestone 3 baselines reproduced (no re-tuning)
    probas, fitted = {}, {}
    naive = M.build_naive().fit(Xtr, ytr); probas["naive"] = naive.predict_proba(Xte)[:, 1]; fitted["naive"] = naive
    repro = []
    for key in M.BASELINES:
        pipe = M.build_from_m3(key, C.M3_BEST_PARAMS[key]).fit(Xtr, ytr)
        fitted[key] = pipe; probas[key] = pipe.predict_proba(Xte)[:, 1]
        auc = ev.metrics(yte, probas[key])["roc_auc"]; auc_m3 = C.M3["holdout_05"][key]["roc_auc"]
        repro.append({"model": key, "params": json.dumps(C.M3_BEST_PARAMS[key]), "holdout_auc_m4_refit": auc, "holdout_auc_m3": auc_m3, "abs_diff": abs(auc - auc_m3)})
        print(f"[baseline] {key}: AUC {auc:.4f} (M3 {auc_m3:.4f})")
    repro = pd.DataFrame(repro); save(repro, "t02_baselines_reproduced")
    assert (repro.abs_diff < 2e-3).all(), "baseline reproduction drifted from Milestone 3"
    R["baselines_reproduced"] = repro.to_dict(orient="records")

    # ================================================================== 3. advanced models: random search on grouped stratified CV
    print("[tune] XGBoost random search ...")
    rs_xgb = tune_random(M.build_xgb(), M.XGB_DIST, Xtr, ytr, gtr, n_iter=M.XGB_N_ITER)
    save(cv_table(rs_xgb), "t03c_cv_random_xgb"); fitted["xgb"] = rs_xgb.best_estimator_; probas["xgb"] = rs_xgb.predict_proba(Xte)[:, 1]
    print(f"        best {rs_xgb.best_params_}  CV AUC {rs_xgb.best_score_:.4f}  trees used {fitted['xgb'].named_steps['clf'].best_iteration_ + 1}")
    print("[tune] MLP random search ...")
    rs_mlp = tune_random(M.build_mlp(), M.MLP_DIST, Xtr, ytr, gtr, n_iter=M.MLP_N_ITER)
    save(cv_table(rs_mlp), "t03d_cv_random_mlp"); fitted["mlp"] = rs_mlp.best_estimator_; probas["mlp"] = rs_mlp.predict_proba(Xte)[:, 1]
    print(f"        best {rs_mlp.best_params_}  CV AUC {rs_mlp.best_score_:.4f}")
    R["best_params"] = {**C.M3_BEST_PARAMS,
                        "xgb": {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in rs_xgb.best_params_.items()},
                        "mlp": {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else str(v)) for k, v in rs_mlp.best_params_.items()}}
    R["best_params"]["xgb"]["n_estimators_used"] = int(fitted["xgb"].named_steps["clf"].best_iteration_ + 1)
    R["search_space"] = {
        "xgb": {"method": f"RandomizedSearchCV, {M.XGB_N_ITER} draws x 5 grouped folds", "max_depth": "3-8", "learning_rate": "0.02-0.15 (log-uniform)",
                "n_estimators": "up to 1,500, early stopping (50 rounds, PR-AUC) on a stratified 10% slice of each training fold", "min_child_weight": "20-200",
                "subsample": "0.6-1.0", "colsample_bytree": "0.6-1.0", "reg_lambda": "0-10", "scale_pos_weight": "{1, 6.6}"},
        "mlp": {"method": f"RandomizedSearchCV, {M.MLP_N_ITER} draws x 5 grouped folds", "hidden_layer_sizes": "(32,) or (64, 32)", "alpha (L2)": "1e-4-1e-1 (log-uniform)",
                "learning_rate_init": "{0.001, 0.003}", "early_stopping": "10% validation slice of each training fold, patience 10", "max_iter": "300"},
    }
    cv_rows = [r for r in C.M3["cv_best"] if r["model"] in M.BASELINES]
    for key, rs in (("xgb", rs_xgb), ("mlp", rs_mlp)):
        row = {"model": key, "best_params": json.dumps(R["best_params"][key])}
        for m in ("roc_auc", "average_precision", "recall", "precision", "f1", "accuracy", "balanced_accuracy"):
            f = fold_scores(rs, m); row[f"cv_{m}_mean"], row[f"cv_{m}_sd"] = float(f.mean()), float(f.std(ddof=1))
        cv_rows.append(row)
    cv_summary = pd.DataFrame(cv_rows); cv_summary.insert(1, "source", ["Milestone 3"] * len(M.BASELINES) + ["Milestone 4"] * 2)
    save(cv_summary, "t05_cv_best_summary"); R["cv_best"] = cv_summary.to_dict(orient="records")

    # ================================================================== 4. hold-out metrics @0.5 + bootstrap gains
    rows = []
    for key in ("naive", *M.ALL_MODELS):
        rows.append({"model": key, "model_name": M.MODEL_NAMES[key], "weighted": False, **ev.metrics(yte, probas[key], 0.5)})
        rows.append({"model": key, "model_name": M.MODEL_NAMES[key], "weighted": True, **ev.metrics(yte, probas[key], 0.5, w=wte)})
    holdout = pd.DataFrame(rows); save(holdout, "t07_holdout_metrics_threshold_0.5")
    R["holdout_05"] = holdout[~holdout.weighted].set_index("model").to_dict(orient="index")
    R["holdout_05_weighted"] = holdout[holdout.weighted].set_index("model").to_dict(orient="index")
    boots = []
    for a, b in (("rf", "xgb"), ("logreg", "xgb"), ("tree", "xgb"), ("rf", "mlp"), ("logreg", "mlp"), ("xgb", "mlp")):
        bs = ev.cluster_bootstrap_auc_diff(yte, probas[a], probas[b], gte, n_boot=N_BOOT)
        boots.append({"baseline": a, "model": b, **bs}); print(f"[bootstrap] AUC({b}) - AUC({a}) = {bs['diff_point']:+.4f} [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]")
    boots = pd.DataFrame(boots); save(boots, "t06b_auc_gain_bootstrap"); R["auc_gain_bootstrap"] = boots.to_dict(orient="records")

    # ================================================================== 5. threshold selection on training OOF + asymmetric cost
    print("[threshold] out-of-fold training predictions ...")
    oof = {key: cross_val_predict(clone(fitted[key]), Xtr, ytr, groups=gtr, cv=make_inner_cv(), method="predict_proba", n_jobs=-1)[:, 1] for key in M.ALL_MODELS}
    sweeps_oof, sweeps_ho, best_t, cms_05, cms_rec = {}, {}, [], {}, {}
    for key in M.ALL_MODELS:
        sw = ev.threshold_sweep(ytr, oof[key]); sw.insert(0, "model", key); sw.insert(1, "data", "train_oof"); sweeps_oof[key] = sw
        sh = ev.threshold_sweep(yte, probas[key]); sh.insert(0, "model", key); sh.insert(1, "data", "holdout"); sweeps_ho[key] = sh
        bt = ev.best_thresholds(sw); bt.insert(0, "model", key)
        for i, r in bt.iterrows():
            c = ev.confusion(yte, (probas[key] >= r["threshold"]).astype(int)); m = ev.metrics_from_counts(c)
            bt.loc[i, "holdout_recall"], bt.loc[i, "holdout_precision"], bt.loc[i, "holdout_specificity"] = m["recall"], m["precision"], m["specificity"]
            bt.loc[i, "holdout_flagged_rate"] = m["flagged_rate"]; bt.loc[i, "holdout_cost_per_1000"] = ev.expected_cost(c, r["cost_ratio_fn_to_fp"])
            col = f"cost_per_1000_ratio{int(r['cost_ratio_fn_to_fp'])}"
            bt.loc[i, "holdout_oracle_threshold"] = sh.loc[sh[col].idxmin(), "threshold"]; bt.loc[i, "holdout_oracle_cost_per_1000"] = sh[col].min()
        best_t.append(bt)
    save(pd.concat(list(sweeps_oof.values()) + list(sweeps_ho.values())), "t08_threshold_sweep")
    best_t = pd.concat(best_t, ignore_index=True).rename(columns={"recall": "oof_recall", "precision": "oof_precision", "specificity": "oof_specificity",
                                                                   "f1": "oof_f1", "flagged_rate": "oof_flagged_rate", "cost_per_1000": "oof_cost_per_1000"})
    save(best_t, "t09_cost_minimising_thresholds"); R["cost_thresholds"] = best_t.to_dict(orient="records")
    rec_thr = {key: float(best_t[(best_t.model == key) & (best_t.cost_ratio_fn_to_fp == C.RECOMMENDED_COST_RATIO)].threshold.item()) for key in M.ALL_MODELS}
    for key in M.ALL_MODELS:
        cms_05[key] = {**ev.confusion(yte, (probas[key] >= 0.5).astype(int)), "threshold": 0.5}
        cms_rec[key] = {**ev.confusion(yte, (probas[key] >= rec_thr[key]).astype(int)), "threshold": rec_thr[key]}
    cms_05["naive"] = {**ev.confusion(yte, (probas["naive"] >= 0.5).astype(int)), "threshold": 0.5}
    R["recommended_threshold"] = rec_thr; R["recommended_cost_ratio"] = C.RECOMMENDED_COST_RATIO
    at_ratio = best_t[best_t.cost_ratio_fn_to_fp == C.RECOMMENDED_COST_RATIO].set_index("model")
    R["recommended_model"] = str(at_ratio["holdout_cost_per_1000"].idxmin()); R["model_rank_by_holdout_cost"] = at_ratio["holdout_cost_per_1000"].sort_values().round(2).to_dict()
    R["m3_recommended_model"] = C.M3["recommended_model"]; R["m3_model_rank_by_holdout_cost"] = C.M3["model_rank_by_holdout_cost"]
    print(f"[threshold] recommended thresholds {rec_thr}; best by cost: {R['recommended_model']} ({R['model_rank_by_holdout_cost']})")
    comp_rows = []
    for key in ("naive", *M.ALL_MODELS):
        for label, cm in (("0.50 (default)", cms_05[key]), ("recommended", cms_rec.get(key))):
            if cm is None: continue
            mm = ev.metrics_from_counts(cm)
            comp_rows.append({"model": key, "threshold_label": label, "threshold": cm["threshold"], **{k: cm[k] for k in ("tn", "fp", "fn", "tp")}, **mm,
                              **ev.per_1000(cm), **{f"cost_per_1000_ratio{r}": ev.expected_cost(cm, r) for r in C.COST_RATIOS}})
    comp = pd.DataFrame(comp_rows); save(comp, "t10_confusion_default_vs_recommended"); R["threshold_comparison"] = comp.to_dict(orient="records")
    for key, cm in {**{f"{k}_05": v for k, v in cms_05.items()}, **{f"{k}_rec": v for k, v in cms_rec.items()}}.items():
        ev.confusion_table(cm).to_csv(T / f"t11_confusion_{key}.csv")
    # comparison at both cut-offs, one tidy table for the report
    both = []
    for key in ("naive", *M.ALL_MODELS):
        for lab, thr in (("0.5", 0.5), ("cost-tuned", rec_thr.get(key))):
            if thr is None: continue
            m = ev.metrics(yte, probas[key], thr)
            both.append({"model": key, "model_name": M.MODEL_NAMES[key], "source": "Milestone 3" if key in ("naive", *M.BASELINES) else "Milestone 4",
                         "cutoff": lab, "threshold": thr, **{k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity", "flagged_rate")}})
    both = pd.DataFrame(both); save(both, "t07b_holdout_comparison_both_cutoffs"); R["holdout_both_cutoffs"] = both.to_dict(orient="records")

    # ================================================================== 6. fairness audit
    print("[fairness] audit ...")
    poor_cut = float(np.quantile(df_tr["assets_n"], 1 / 3)); R["poverty_cut_assets_n_le"] = poor_cut
    def attr_arrays(d: pd.DataFrame) -> dict:
        a = {k: d[k].to_numpy() for k in ("female", "urban", "any_disability", "bform")}
        a["poor"] = (d["assets_n"] <= poor_cut).astype(int).to_numpy(); a["province"] = province_of(d).to_numpy(); return a
    A_te, A_tr = attr_arrays(df_te), attr_arrays(df_tr)
    def audit_specs(A: dict) -> dict:
        specs = {}
        for k, s in C.PROTECTED.items():
            specs[k] = {"values": A[k], "priv": s["priv"], "unpriv": s["unpriv"], "label": s["label"], "priv_name": s["names"][s["priv"]], "unpriv_name": s["names"][s["unpriv"]]}
        for prov in sorted(set(A["province"]) - {C.PROVINCE_ATTR["priv"]} - set(C.PROVINCE_ATTR["exclude"])):
            specs[f"province:{prov}"] = {"values": A["province"], "priv": C.PROVINCE_ATTR["priv"], "unpriv": prov, "label": f"Province: {prov} vs Punjab",
                                         "priv_name": "Punjab", "unpriv_name": prov}
        return specs
    specs_te = audit_specs(A_te)
    R["protected_attributes"] = {k: {"label": v["label"], "privileged": v["priv_name"], "unprivileged": v["unpriv_name"]} for k, v in specs_te.items()}
    R["ict_excluded_reason"] = "Islamabad Capital Territory is a single district that fell into the training side; it has no hold-out rows (52 positives overall)."
    aud_rows, rate_rows = [], []
    for key in M.ALL_MODELS:
        for lab, thr in (("cost-tuned", rec_thr[key]), ("0.5", 0.5)):
            yhat = (probas[key] >= thr).astype(int)
            a = F.audit(yte, yhat, specs_te, clusters=gte if lab == "cost-tuned" else None, n_boot=N_BOOT)
            a.insert(0, "model", key); a.insert(1, "cutoff", lab); a.insert(2, "threshold", thr); aud_rows.append(a)
            if lab == "cost-tuned":
                for k, s in C.PROTECTED.items():
                    g = F.group_rates(yte, yhat, A_te[k], s["names"]); g.insert(0, "model", key); g.insert(1, "attribute", k); g.insert(2, "label", s["label"]); rate_rows.append(g)
                g = F.group_rates(yte, yhat, A_te["province"]); g = g[~g.level.isin(C.PROVINCE_ATTR["exclude"])]
                g.insert(0, "model", key); g.insert(1, "attribute", "province"); g.insert(2, "label", "Province"); rate_rows.append(g)
    aud = pd.concat(aud_rows, ignore_index=True); save(aud, "t22_fairness_audit")
    rates = pd.concat(rate_rows, ignore_index=True); save(rates, "t22b_group_rates_cost_tuned")
    R["fairness_audit"] = aud[aud.cutoff == "cost-tuned"].round(4).to_dict(orient="records")
    R["fairness_audit_05"] = aud[aud.cutoff == "0.5"].round(4).to_dict(orient="records")
    R["group_rates"] = rates.round(4).to_dict(orient="records")
    # M3 starting hypotheses: does the gap shrink, hold or grow?
    m3_sub = pd.DataFrame(C.M3["subgroups_logreg"])
    def m3_recall(var, lvl):
        r = m3_sub[(m3_sub.variable == var) & (m3_sub.level.astype(str) == str(lvl))]; return float(r.recall.iloc[0]) if len(r) else np.nan
    gap_rows = []
    for k, s in list(C.PROTECTED.items()) + [("province:Sindh", None), ("province:Balochistan", None), ("province:Khyber Pakhtuknkhwa", None)]:
        spec = specs_te[k]
        if s is not None:
            m3_gap = m3_recall(k, spec["unpriv"]) - m3_recall(k, spec["priv"]) if k != "poor" else np.nan
        else:
            m3_gap = m3_recall("province", spec["unpriv"]) - m3_recall("province", spec["priv"])
        row = {"attribute": k, "label": spec["label"], "m3_logreg_eopp_gap": m3_gap}
        for key in M.ALL_MODELS:
            g = float(aud[(aud.model == key) & (aud.cutoff == "cost-tuned") & (aud.attribute == k)].eopp_gap.iloc[0])
            row[f"{key}_eopp_gap"] = g
            if np.isnan(m3_gap): row[f"{key}_verdict"] = "n/a"
            else: row[f"{key}_verdict"] = "reduced" if abs(g) < abs(m3_gap) - 0.05 else ("worsened" if abs(g) > abs(m3_gap) + 0.05 else "preserved")
        gap_rows.append(row)
    gaps = pd.DataFrame(gap_rows); save(gaps, "t23_gap_change_vs_m3"); R["gap_change_vs_m3"] = gaps.round(4).to_dict(orient="records")

    # ================================================================== 7. mitigation
    print("[mitigation] group thresholds, Pareto, reweighing ...")
    rec = R["recommended_model"]
    # Mitigation focus: group-specific cut-offs need a smooth, well-ranked score. Among models within 1% of the lowest hold-out cost,
    # take the one with the highest hold-out ROC-AUC; also record how many distinct scores each model produces (a tree gives one per leaf).
    R["n_distinct_scores"] = {k: int(len(np.unique(np.round(probas[k], 6)))) for k in M.ALL_MODELS}
    near = [k for k in M.ALL_MODELS if at_ratio.loc[k, "holdout_cost_per_1000"] <= 1.01 * at_ratio["holdout_cost_per_1000"].min()]
    focus = max(near, key=lambda k: R["holdout_05"][k]["roc_auc"])
    R["mitigation_focus"] = focus; R["mitigation_focus_candidates"] = near
    mit_models = list(dict.fromkeys([focus, rec, "xgb", C.M3["recommended_model"]]))   # M3 tree kept as a contrast (one score per leaf)
    R["mitigation_models"] = mit_models
    print(f"[mitigation] focus model {focus} (candidates within 1% cost: {near}); distinct scores {R['n_distinct_scores']}")
    ba_rows, thr_rows, pareto_tabs, frontiers = [], [], [], {a: {} for a in C.PARETO_ATTRS}
    for key in mit_models:
        for k, spec in specs_te.items():
            if k.startswith("province:"): continue
            tr_vals, te_vals = A_tr[k], A_te[k]
            g = F.group_thresholds(ytr, oof[key], tr_vals, C.RECOMMENDED_COST_RATIO, C.RECOMMENDED_TPR_GAP, grid=F.FINE_GRID)
            yhat_b = (probas[key] >= rec_thr[key]).astype(int); yhat_a = F.apply_group_thresholds(probas[key], te_vals, g["thresholds"])
            db = F.disparity(yte, yhat_b, te_vals, spec["priv"], spec["unpriv"]); da = F.disparity(yte, yhat_a, te_vals, spec["priv"], spec["unpriv"])
            cb, ca = ev.confusion(yte, yhat_b), ev.confusion(yte, yhat_a); mb, ma = ev.metrics_from_counts(cb), ev.metrics_from_counts(ca)
            ba_rows.append({"model": key, "method": "post-processing: group thresholds", "attribute": k, "label": spec["label"], "priv_name": spec["priv_name"], "unpriv_name": spec["unpriv_name"],
                            "threshold_before": rec_thr[key], "thresholds_after": json.dumps(pyk(g["thresholds"])), "feasible_oof": g["feasible"], "oof_tpr_gap_after": g["oof_tpr_gap"],
                            "tpr_priv_before": db["tpr_priv"], "tpr_unpriv_before": db["tpr_unpriv"], "tpr_priv_after": da["tpr_priv"], "tpr_unpriv_after": da["tpr_unpriv"],
                            "fpr_priv_before": db["fpr_priv"], "fpr_unpriv_before": db["fpr_unpriv"], "fpr_priv_after": da["fpr_priv"], "fpr_unpriv_after": da["fpr_unpriv"],
                            "di_before": db["di_ratio"], "di_after": da["di_ratio"], "eopp_gap_before": db["eopp_gap"], "eopp_gap_after": da["eopp_gap"],
                            "pe_gap_before": db["pe_gap"], "pe_gap_after": da["pe_gap"], "eodds_before": db["eodds_diff"], "eodds_after": da["eodds_diff"],
                            "accuracy_before": mb["accuracy"], "accuracy_after": ma["accuracy"], "recall_before": mb["recall"], "recall_after": ma["recall"],
                            "cost_before": ev.expected_cost(cb, C.RECOMMENDED_COST_RATIO), "cost_after": ev.expected_cost(ca, C.RECOMMENDED_COST_RATIO),
                            "flagged_per_1000_before": ev.per_1000(cb)["flagged_per_1000"], "flagged_per_1000_after": ev.per_1000(ca)["flagged_per_1000"]})
            # policy knob: group-specific FN cost
            if k in C.GROUP_FN_MULTIPLIER:
                mult = {spec["unpriv"]: C.GROUP_FN_MULTIPLIER[k], spec["priv"]: 1.0}
                gm = F.group_thresholds(ytr, oof[key], tr_vals, C.RECOMMENDED_COST_RATIO, 1.0, fn_multiplier=mult, grid=F.FINE_GRID)   # no gap constraint, cost knob only
                yhat_m = F.apply_group_thresholds(probas[key], te_vals, gm["thresholds"]); dm = F.disparity(yte, yhat_m, te_vals, spec["priv"], spec["unpriv"]); cm_ = ev.confusion(yte, yhat_m)
                fn_rows = np.where(te_vals == spec["unpriv"], C.RECOMMENDED_COST_RATIO * C.GROUP_FN_MULTIPLIER[k], C.RECOMMENDED_COST_RATIO)
                thr_rows.append({"model": key, "attribute": k, "label": spec["label"], "fn_multiplier_unpriv": C.GROUP_FN_MULTIPLIER[k], "thresholds": json.dumps(pyk(gm["thresholds"])),
                                 "tpr_priv": dm["tpr_priv"], "tpr_unpriv": dm["tpr_unpriv"], "eopp_gap": dm["eopp_gap"], "di_ratio": dm["di_ratio"],
                                 "cost_standard_per_1000": ev.expected_cost(cm_, C.RECOMMENDED_COST_RATIO), "cost_weighted_per_1000": ev.expected_cost_rows(yte, yhat_m, fn_rows),
                                 "flagged_per_1000": ev.per_1000(cm_)["flagged_per_1000"]})
            if k in C.PARETO_ATTRS:
                pf = F.pareto_frontier(ytr, oof[key], tr_vals, yte, probas[key], te_vals, spec["priv"], spec["unpriv"], C.RECOMMENDED_COST_RATIO, single_threshold=rec_thr[key])
                pf.insert(0, "model", key); pf.insert(1, "attribute", k); pareto_tabs.append(pf); frontiers[k][key] = pf
        # province (multi-level) group thresholds for the mitigation-focus model only
        if key == focus:
            keep = ~np.isin(A_tr["province"], C.PROVINCE_ATTR["exclude"])
            gp = F.group_thresholds(ytr[keep], oof[key][keep], A_tr["province"][keep], C.RECOMMENDED_COST_RATIO, C.RECOMMENDED_TPR_GAP, grid=F.FINE_GRID)
            yhat_p = F.apply_group_thresholds(probas[key], A_te["province"], {**gp["thresholds"], **{e: rec_thr[key] for e in C.PROVINCE_ATTR["exclude"]}})
            gr_b = F.group_rates(yte, (probas[key] >= rec_thr[key]).astype(int), A_te["province"]); gr_a = F.group_rates(yte, yhat_p, A_te["province"])
            prov_tbl = gr_b[["group", "n", "positives", "tpr", "fpr", "selection_rate"]].merge(gr_a[["group", "tpr", "fpr", "selection_rate"]], on="group", suffixes=("_before", "_after"))
            prov_tbl["threshold_after"] = prov_tbl.group.map(pyk(gp["thresholds"])); prov_tbl.insert(0, "model", key)
            save(prov_tbl, "t24c_province_group_thresholds")
            R["province_thresholds"] = {"model": key, "thresholds": pyk(gp["thresholds"]), "feasible_oof": gp["feasible"], "oof_tpr_gap": gp["oof_tpr_gap"],
                                        "cost_before": ev.expected_cost(ev.confusion(yte, (probas[key] >= rec_thr[key]).astype(int)), C.RECOMMENDED_COST_RATIO),
                                        "cost_after": ev.expected_cost(ev.confusion(yte, yhat_p), C.RECOMMENDED_COST_RATIO), "table": prov_tbl.round(4).to_dict(orient="records")}
    ba = pd.DataFrame(ba_rows); save(ba, "t24_mitigation_before_after"); R["mitigation_before_after"] = ba.round(4).to_dict(orient="records")
    knob = pd.DataFrame(thr_rows); save(knob, "t24b_group_fn_cost_knob"); R["group_fn_cost_knob"] = knob.round(4).to_dict(orient="records")
    pareto = pd.concat(pareto_tabs, ignore_index=True); save(pareto, "t25_pareto_frontier"); R["pareto"] = pareto.round(4).to_dict(orient="records")

    # pre-processing: reweighing (Kamiran-Calders) for the two attributes with the largest M3 gaps, LR and XGB
    rw_rows = []
    for k in ("urban", "female", "bform"):
        spec = specs_te[k]
        for key in ("logreg", "xgb"):
            wts = F.reweighing_weights(ytr, A_tr[k])
            pipe = clone(fitted[key]); pipe.fit(Xtr, ytr, clf__sample_weight=wts)
            oof_w = cross_val_predict(clone(fitted[key]), Xtr, ytr, groups=gtr, cv=make_inner_cv(), method="predict_proba", n_jobs=-1, params={"clf__sample_weight": wts})[:, 1]
            thr_w = float(ev.best_thresholds(ev.threshold_sweep(ytr, oof_w)).set_index("cost_ratio_fn_to_fp").loc[C.RECOMMENDED_COST_RATIO, "threshold"])
            p_w = pipe.predict_proba(Xte)[:, 1]; yhat_w = (p_w >= thr_w).astype(int); yhat_b = (probas[key] >= rec_thr[key]).astype(int)
            db = F.disparity(yte, yhat_b, A_te[k], spec["priv"], spec["unpriv"]); da = F.disparity(yte, yhat_w, A_te[k], spec["priv"], spec["unpriv"])
            cb, ca = ev.confusion(yte, yhat_b), ev.confusion(yte, yhat_w)
            rw_rows.append({"model": key, "method": "pre-processing: reweighing", "attribute": k, "label": spec["label"], "priv_name": spec["priv_name"], "unpriv_name": spec["unpriv_name"],
                            "threshold_before": rec_thr[key], "threshold_after": thr_w, "auc_before": ev.metrics(yte, probas[key])["roc_auc"], "auc_after": ev.metrics(yte, p_w)["roc_auc"],
                            "tpr_priv_before": db["tpr_priv"], "tpr_unpriv_before": db["tpr_unpriv"], "tpr_priv_after": da["tpr_priv"], "tpr_unpriv_after": da["tpr_unpriv"],
                            "di_before": db["di_ratio"], "di_after": da["di_ratio"], "eopp_gap_before": db["eopp_gap"], "eopp_gap_after": da["eopp_gap"], "pe_gap_before": db["pe_gap"], "pe_gap_after": da["pe_gap"],
                            "cost_before": ev.expected_cost(cb, C.RECOMMENDED_COST_RATIO), "cost_after": ev.expected_cost(ca, C.RECOMMENDED_COST_RATIO),
                            "flagged_per_1000_before": ev.per_1000(cb)["flagged_per_1000"], "flagged_per_1000_after": ev.per_1000(ca)["flagged_per_1000"]})
            print(f"[reweigh] {key}/{k}: gap {db['eopp_gap']:+.3f} -> {da['eopp_gap']:+.3f}, cost {rw_rows[-1]['cost_before']:.0f} -> {rw_rows[-1]['cost_after']:.0f}")
    rw = pd.DataFrame(rw_rows); save(rw, "t26_reweighing_before_after"); R["reweighing"] = rw.round(4).to_dict(orient="records")
    R["in_processing_note"] = "Not implemented (time): fairlearn ExponentiatedGradient with a TruePositiveRateParity constraint is the natural next step."

    # ================================================================== 8. importance for XGBoost
    print("[importance] xgb ...")
    gini_x = imp.tree_gini(fitted["xgb"]); save(gini_x, "t17c_xgb_gain_importance")
    perm_x = imp.permutation(fitted["xgb"], Xte, yte); save(perm_x, "t19c_permutation_xgb")
    R["gini_xgb"] = gini_x.round(5).to_dict(orient="records"); R["perm_xgb"] = perm_x.round(5).to_dict(orient="records")
    R["perm_logreg_m3"] = C.M3["perm_logreg"]; R["gini_rf_m3"] = C.M3["gini_rf"]

    # ================================================================== 9. figures
    print("[plots] ...")
    names = M.MODEL_NAMES
    figs = {
        "roc_pr": plots.roc_pr_curves(yte, {k: probas[k] for k in M.ALL_MODELS}, names, float(yte.mean())),
        "cm_05": plots.confusion_heatmaps({k: cms_05[k] for k in ("naive", *M.ALL_MODELS)}, names, "_default"),
        "cm_rec": plots.confusion_heatmaps(cms_rec, names, "_recommended"),
        "threshold": plots.threshold_curves(sweeps_oof, sweeps_ho, rec_thr, names, C.RECOMMENDED_COST_RATIO),
        "benchmark": plots.benchmark_bars(holdout[~holdout.weighted], names, ["naive", *M.ALL_MODELS]),
        "importance_xgb": plots.ensemble_importance_bars(gini_x, perm_x, names["xgb"], plots.PALETTE["xgb"], "fig15_xgb_importance"),
        "pareto": {}, "fairness": {},
    }
    for k, s in C.PROTECTED.items():
        figs["fairness"][k] = plots.fairness_panels(rates[rates.attribute == k], s["label"], names, f"fig11_fairness_{k}")
    figs["fairness"]["province"] = plots.fairness_panels(rates[rates.attribute == "province"], "Province", names, "fig11_fairness_province")
    aud_ct = aud[aud.cutoff == "cost-tuned"]
    figs["eopp_forest"] = plots.disparity_forest(aud_ct, "eopp_gap", "Equal-opportunity gap: TPR(unprivileged) - TPR(privileged)", "fig11b_eopp_gap_forest", 0.0, names)
    figs["di_forest"] = plots.disparity_forest(aud_ct, "di_ratio", "Disparate-impact ratio: flag rate(unprivileged) / flag rate(privileged)", "fig11c_di_ratio_forest", 1.0, names)
    for k in C.PARETO_ATTRS:
        figs["pareto"][k] = plots.pareto_plot(frontiers[k], C.PROTECTED[k]["label"], f"fig12_pareto_{k}", names, C.RECOMMENDED_TPR_GAP)
    figs["before_after"] = plots.before_after_plot(ba[ba.model == focus], "fig13_before_after")
    R["figures"] = figs
    R["runtime_sec"] = round(time.time() - t0, 1)
    with open(C.OUT_DIR / "results.json", "w") as fh:
        json.dump(R, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[done] {R['runtime_sec']} s -> {C.OUT_DIR / 'results.json'}")
    return R


if __name__ == "__main__":
    main()
