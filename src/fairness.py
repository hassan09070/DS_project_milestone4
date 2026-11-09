"""Group-fairness audit, group-specific threshold mitigation, Pareto frontier and reweighing.

Everything is hand-coded on top of evaluate.confusion / metrics_from_counts so each number is traceable to a 2x2 table.

Framing used throughout: a positive prediction means a child is FLAGGED for enrolment outreach (a benefit). The harm of
concern is therefore a false negative (a missed child), and the disparate-impact worry is a group being UNDER-flagged
relative to its need. Metrics are nevertheless reported in the conventional direction (unprivileged / privileged) too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (COST_RATIOS, DI_BAND, RECOMMENDED_COST_RATIO, SMALL_GROUP_POSITIVES, THRESHOLD_GRID,
                    TPR_GAP_GRID)
from evaluate import confusion, expected_cost, metrics_from_counts

FINE_GRID = [round(x, 3) for x in np.arange(0.02, 0.951, 0.0025)]   # finer cut-off grid for the Pareto sweep


# ----------------------------------------------------------------------------- per-group rates
def group_rates(y, yhat, attr, names: dict | None = None) -> pd.DataFrame:
    """One row per level of a protected attribute: n, positives, selection rate, TPR, FPR, FNR, precision."""
    y = np.asarray(y); yhat = np.asarray(yhat); attr = np.asarray(attr)
    rows = []
    for lvl in pd.unique(attr):
        m = attr == lvl
        c = confusion(y[m], yhat[m]); k = metrics_from_counts(c)
        pos = int(c["tp"] + c["fn"])
        rows.append({
            "level": lvl, "group": (names or {}).get(lvl, str(lvl)), "n": int(m.sum()), "positives": pos,
            "prevalence": pos / m.sum(), "selection_rate": k["flagged_rate"], "tpr": k["recall"],
            "fpr": 1 - k["specificity"] if not np.isnan(k["specificity"]) else np.nan,
            "fnr": 1 - k["recall"] if not np.isnan(k["recall"]) else np.nan, "precision": k["precision"],
            "small": pos < SMALL_GROUP_POSITIVES,
        })
    return pd.DataFrame(rows)


def disparity(y, yhat, attr, priv, unpriv) -> dict:
    """Group-fairness metrics for one unprivileged vs privileged pair.

    di_ratio      = selection rate(unpriv) / selection rate(priv)      (four-fifths rule: within DI_BAND)
    dp_diff       = selection rate(unpriv) - selection rate(priv)      (demographic parity difference)
    eopp_gap      = TPR(unpriv) - TPR(priv)                            (equal opportunity; negative = unprivileged group under-found)
    pe_gap        = FPR(unpriv) - FPR(priv)                            (predictive equality)
    eodds_diff    = max(|eopp_gap|, |pe_gap|)                          (equalized odds difference)
    """
    y = np.asarray(y); yhat = np.asarray(yhat); attr = np.asarray(attr)
    out = {}
    for tag, lvl in (("priv", priv), ("unpriv", unpriv)):
        m = attr == lvl
        c = confusion(y[m], yhat[m]); k = metrics_from_counts(c)
        out[f"n_{tag}"] = int(m.sum()); out[f"pos_{tag}"] = int(c["tp"] + c["fn"])
        out[f"sel_{tag}"] = k["flagged_rate"]; out[f"tpr_{tag}"] = k["recall"]
        out[f"fpr_{tag}"] = 1 - k["specificity"]; out[f"prec_{tag}"] = k["precision"]
    out["di_ratio"] = out["sel_unpriv"] / out["sel_priv"] if out["sel_priv"] > 0 else np.nan
    out["di_within_band"] = bool(DI_BAND[0] <= out["di_ratio"] <= DI_BAND[1]) if not np.isnan(out["di_ratio"]) else False
    out["dp_diff"] = out["sel_unpriv"] - out["sel_priv"]
    out["eopp_gap"] = out["tpr_unpriv"] - out["tpr_priv"]
    out["pe_gap"] = out["fpr_unpriv"] - out["fpr_priv"]
    out["eodds_diff"] = max(abs(out["eopp_gap"]), abs(out["pe_gap"]))
    out["small_group"] = bool(min(out["pos_priv"], out["pos_unpriv"]) < SMALL_GROUP_POSITIVES)
    return out


def bootstrap_disparity(y, yhat, attr, priv, unpriv, clusters, n_boot: int = 1000, seed: int = 42,
                        keys=("di_ratio", "eopp_gap", "pe_gap", "eodds_diff", "dp_diff")) -> dict:
    """District-cluster bootstrap 95% intervals for the disparity metrics (districts resampled with replacement)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); yhat = np.asarray(yhat); attr = np.asarray(attr); cl = np.asarray(clusters)
    uniq = np.unique(cl); idx_by = {c: np.flatnonzero(cl == c) for c in uniq}
    draws = {k: [] for k in keys}
    tries = 0
    while len(draws[keys[0]]) < n_boot and tries < n_boot * 3:
        tries += 1
        idx = np.concatenate([idx_by[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
        a = attr[idx]
        if (a == priv).sum() == 0 or (a == unpriv).sum() == 0 or y[idx][a == priv].sum() == 0 or y[idx][a == unpriv].sum() == 0:
            continue
        d = disparity(y[idx], yhat[idx], a, priv, unpriv)
        for k in keys:
            draws[k].append(d[k])
    out = {}
    for k in keys:
        arr = np.array(draws[k], dtype=float); arr = arr[~np.isnan(arr)]
        out[f"{k}_ci_low"] = float(np.percentile(arr, 2.5)) if len(arr) else np.nan
        out[f"{k}_ci_high"] = float(np.percentile(arr, 97.5)) if len(arr) else np.nan
    out["n_boot_effective"] = len(draws[keys[0]])
    return out


def audit(y, yhat, attrs: dict, clusters=None, n_boot: int = 1000) -> pd.DataFrame:
    """Disparity table over several protected attributes.

    attrs: {name: {"values": array, "priv": level, "unpriv": level, "label": str}}; multi-level attributes are passed
    as several (unpriv vs priv) pairs by the caller.
    """
    rows = []
    for name, spec in attrs.items():
        d = disparity(y, yhat, spec["values"], spec["priv"], spec["unpriv"])
        if clusters is not None:
            d.update(bootstrap_disparity(y, yhat, spec["values"], spec["priv"], spec["unpriv"], clusters, n_boot))
        rows.append({"attribute": name, "label": spec.get("label", name), "priv": spec["priv"], "unpriv": spec["unpriv"],
                     "priv_name": spec.get("priv_name", str(spec["priv"])), "unpriv_name": spec.get("unpriv_name", str(spec["unpriv"])), **d})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- post-processing: group thresholds
def _sweep_counts(y, p, grid=THRESHOLD_GRID) -> pd.DataFrame:
    y = np.asarray(y); p = np.asarray(p)
    pos = y == 1
    rows = []
    for t in grid:
        yhat = p >= t
        tp = int((yhat & pos).sum()); fn = int((~yhat & pos).sum()); fp = int((yhat & ~pos).sum()); tn = int((~yhat & ~pos).sum())
        rows.append({"threshold": t, "tp": tp, "fn": fn, "fp": fp, "tn": tn, "tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1)})
    return pd.DataFrame(rows)


def apply_group_thresholds(p, attr, thresholds: dict) -> np.ndarray:
    p = np.asarray(p); attr = np.asarray(attr)
    thr = np.array([thresholds[a] for a in attr], dtype=float)
    return (p >= thr).astype(int)


def _group_arrays(y, p, attr, levels, grid):
    """Per group: arrays over the threshold grid of fn, fp, tpr (vectorised)."""
    y = np.asarray(y); p = np.asarray(p); attr = np.asarray(attr); g = np.asarray(grid, dtype=float)
    out = {}
    for lv in levels:
        m = attr == lv; pp = p[m]; yy = y[m]
        pos = pp[yy == 1]; neg = pp[yy == 0]
        # counts of scores >= t for each t: sort and searchsorted
        pos_s = np.sort(pos); neg_s = np.sort(neg)
        tp = len(pos) - np.searchsorted(pos_s, g, side="left"); fp = len(neg) - np.searchsorted(neg_s, g, side="left")
        fn = len(pos) - tp
        out[lv] = {"fn": fn.astype(float), "fp": fp.astype(float), "tpr": tp / max(len(pos), 1), "n_pos": len(pos)}
    return out


def group_thresholds(y, p, attr, cost_ratio: float = RECOMMENDED_COST_RATIO, max_tpr_gap: float = 0.05,
                     grid=THRESHOLD_GRID, fn_multiplier: dict | None = None) -> dict:
    """Cost-minimising threshold per group subject to max(TPR_g) - min(TPR_g) <= max_tpr_gap.

    Binary attribute: exhaustive vectorised search over all threshold pairs. More levels: coordinate descent from the
    pooled optimum, each pass re-optimising one group with the others fixed. If no combination satisfies the gap, the
    smallest-gap combination (ties broken by cost) is returned and `feasible` is False. fn_multiplier lets the FN cost
    differ by group (policy knob). Thresholds are chosen on the data passed in (training OOF), never on the hold-out.
    """
    y = np.asarray(y); attr = np.asarray(attr); n = len(y); grid = np.asarray(grid, dtype=float)
    levels = list(pd.unique(attr))
    arr = _group_arrays(y, p, attr, levels, grid)
    mult = {g: (fn_multiplier or {}).get(g, 1.0) for g in levels}
    if len(levels) == 2:
        a, b = levels
        cost = 1000.0 * ((cost_ratio * mult[a] * arr[a]["fn"] + arr[a]["fp"])[:, None] + (cost_ratio * mult[b] * arr[b]["fn"] + arr[b]["fp"])[None, :]) / n
        gap = np.abs(arr[a]["tpr"][:, None] - arr[b]["tpr"][None, :])
        ok = gap <= max_tpr_gap + 1e-12
        if ok.any():
            flat = np.where(ok, cost, np.inf).argmin(); feasible = True
        else:
            # smallest gap, ties by cost (lexicographic)
            flat = np.lexsort((cost.ravel(), gap.ravel()))[0]; feasible = False
        i, j = np.unravel_index(flat, cost.shape); choice = {a: int(i), b: int(j)}
    else:
        pooled_fn = sum(arr[g]["fn"] for g in levels); pooled_fp = sum(arr[g]["fp"] for g in levels)
        start = int(np.argmin(cost_ratio * pooled_fn + pooled_fp)); choice = {g: start for g in levels}
        def total(ch):
            c = 1000.0 * sum(cost_ratio * mult[g] * arr[g]["fn"][ch[g]] + arr[g]["fp"][ch[g]] for g in levels) / n
            t = [arr[g]["tpr"][ch[g]] for g in levels]; return c, max(t) - min(t)
        for _ in range(8):
            changed = False
            for g in levels:
                others_c = sum(cost_ratio * mult[h] * arr[h]["fn"][choice[h]] + arr[h]["fp"][choice[h]] for h in levels if h != g)
                others_t = np.array([arr[h]["tpr"][choice[h]] for h in levels if h != g])
                c_vec = 1000.0 * (others_c + cost_ratio * mult[g] * arr[g]["fn"] + arr[g]["fp"]) / n
                gap_vec = np.maximum(others_t.max(), arr[g]["tpr"]) - np.minimum(others_t.min(), arr[g]["tpr"])
                ok = gap_vec <= max_tpr_gap + 1e-12
                best_i = int(np.where(ok, c_vec, np.inf).argmin()) if ok.any() else int(np.lexsort((c_vec, gap_vec))[0])
                if best_i != choice[g]:
                    choice[g] = best_i; changed = True
            if not changed:
                break
        _, gap_v = total(choice); feasible = gap_v <= max_tpr_gap + 1e-12
    cost_v, gap_v = (1000.0 * sum(cost_ratio * mult[g] * arr[g]["fn"][choice[g]] + arr[g]["fp"][choice[g]] for g in levels) / n,
                     max(arr[g]["tpr"][choice[g]] for g in levels) - min(arr[g]["tpr"][choice[g]] for g in levels))
    thresholds = {g: float(grid[choice[g]]) for g in levels}
    return {"thresholds": thresholds, "oof_cost_per_1000": float(cost_v), "oof_tpr_gap": float(gap_v), "feasible": bool(feasible),
            "max_tpr_gap": max_tpr_gap, "cost_ratio": cost_ratio, "tpr_by_group": {g: float(arr[g]["tpr"][choice[g]]) for g in levels}}


def pareto_frontier(y_oof, p_oof, attr_oof, y_ho, p_ho, attr_ho, priv, unpriv, cost_ratio: float = RECOMMENDED_COST_RATIO,
                    gaps=TPR_GAP_GRID, single_threshold: float | None = None) -> pd.DataFrame:
    """For each allowed TPR gap: choose group thresholds on training OOF, then score cost and gap on hold-out.

    Adds the unconstrained single-threshold point and marks non-dominated (Pareto) points by (hold-out gap, hold-out cost).
    """
    rows = []
    for tau in gaps:
        g = group_thresholds(y_oof, p_oof, attr_oof, cost_ratio, tau, grid=FINE_GRID)
        yhat_ho = apply_group_thresholds(p_ho, attr_ho, g["thresholds"])
        d = disparity(y_ho, yhat_ho, attr_ho, priv, unpriv)
        c = confusion(y_ho, yhat_ho); k = metrics_from_counts(c)
        rows.append({"max_tpr_gap": tau, "feasible_oof": g["feasible"], **{f"thr_{lv}": v for lv, v in g["thresholds"].items()},
                     "oof_cost_per_1000": g["oof_cost_per_1000"], "oof_tpr_gap": g["oof_tpr_gap"],
                     "holdout_cost_per_1000": expected_cost(c, cost_ratio), "holdout_eopp_gap": d["eopp_gap"],
                     "holdout_abs_gap": abs(d["eopp_gap"]), "holdout_di_ratio": d["di_ratio"], "holdout_pe_gap": d["pe_gap"],
                     "holdout_accuracy": k["accuracy"], "holdout_f1": k["f1"], "holdout_recall": k["recall"],
                     "holdout_flagged_rate": k["flagged_rate"], "point": "group thresholds"})
    if single_threshold is not None:
        yhat_ho = (np.asarray(p_ho) >= single_threshold).astype(int)
        d = disparity(y_ho, yhat_ho, attr_ho, priv, unpriv); c = confusion(y_ho, yhat_ho); k = metrics_from_counts(c)
        rows.append({"max_tpr_gap": np.nan, "feasible_oof": True, "oof_cost_per_1000": np.nan, "oof_tpr_gap": np.nan,
                     "holdout_cost_per_1000": expected_cost(c, cost_ratio), "holdout_eopp_gap": d["eopp_gap"],
                     "holdout_abs_gap": abs(d["eopp_gap"]), "holdout_di_ratio": d["di_ratio"], "holdout_pe_gap": d["pe_gap"],
                     "holdout_accuracy": k["accuracy"], "holdout_f1": k["f1"], "holdout_recall": k["recall"],
                     "holdout_flagged_rate": k["flagged_rate"], "point": f"single threshold {single_threshold:.2f}"})
    df = pd.DataFrame(rows)
    # non-dominated: no other point has both lower |gap| and lower cost
    dom = []
    for i, r in df.iterrows():
        others = df.drop(i)
        dominated = ((others.holdout_abs_gap <= r.holdout_abs_gap) & (others.holdout_cost_per_1000 <= r.holdout_cost_per_1000)
                     & ((others.holdout_abs_gap < r.holdout_abs_gap) | (others.holdout_cost_per_1000 < r.holdout_cost_per_1000))).any()
        dom.append(not dominated)
    df["pareto"] = dom
    return df


# ----------------------------------------------------------------------------- pre-processing: reweighing
def reweighing_weights(y, attr) -> np.ndarray:
    """Kamiran & Calders reweighing: w(a, y) = P(a) P(y) / P(a, y), making attribute and label independent in training."""
    y = np.asarray(y); attr = np.asarray(attr); n = len(y)
    w = np.ones(n)
    for a in np.unique(attr):
        for yy in (0, 1):
            m = (attr == a) & (y == yy)
            if m.sum() == 0:
                continue
            w[m] = ((attr == a).mean() * (y == yy).mean()) / (m.sum() / n)
    return w
