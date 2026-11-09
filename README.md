# Milestone 04 - Advanced Models, Fairness Audit, Bias Mitigation and Governance

Out-of-school prediction for children aged 5-16 in Pakistan (ASER 2025), continued from Milestone 3. This folder is
self-contained: the Milestone 2 inputs and a frozen copy of the Milestone 3 results are in `data/`, the brief and the team
plan in `docs/`, and every Milestone 4 deliverable is produced from them.

## Deliverables

| Rubric item | Where |
|---|---|
| Executable notebook | `Milestone04_AdvancedModels.ipynb` (executed in place; sections: setup, baselines reproduced, advanced tuning, comparison + bootstrap, fairness audit, mitigation + threshold optimisation, Pareto, export) |
| Modular scripts | `src/` (see module map); `src/run_all.py` reproduces everything |
| Manuscript enhancement | `report/Milestone04_Report.docx` = Sections 5-9 to follow the Milestone 3 Section 4 draft (export to PDF from Word / Pages) |
| Tables and figures | `outputs/tables/t00..t26*.csv`, `outputs/figures/fig1..fig15*.png`, `outputs/results.json` |

## Reproduce

```bash
brew install libomp                      # macOS: XGBoost's wheel needs OpenMP
cd milestone4
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/run_all.py          # ~15 min: baselines re-created, XGBoost + MLP random search, bootstrap, OOF thresholds, audit, mitigation, Pareto, figures
.venv/bin/python src/make_report.py      # rebuild report/Milestone04_Report.docx from outputs/results.json
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Milestone04_AdvancedModels.ipynb
.venv/bin/python tools/build_notebook.py # only if you edit the notebook cell definitions
M4_QUICK=1 .venv/bin/python src/run_all.py   # 4-minute smoke test (2 search draws, 50 bootstrap draws)
```

Random state 42 throughout. The hold-out is the identical 29 districts used in Milestone 3 (asserted in code); the full
district-to-fold assignment is written to `outputs/tables/t00_fold_assignment.csv` so every teammate gets the same split.

## Inputs (`data/`, read only)

Same seven Milestone 2 files as Milestone 3 (`aser2025_training.csv`, `feature_manifest.json`, `data_dictionary.csv`,
`encoding_log.txt`, `subgroup_counts.csv`, `attrition.csv`, codebook) plus `m3_results.json`, the frozen Milestone 3
`results.json` from which the baseline hyperparameters, cross-validation scores and subgroup recalls are read.

## Module map (`src/`)

| Module | Role |
|---|---|
| `config.py` | Milestone 3 constants plus: `M3_BEST_PARAMS` (from `data/m3_results.json`), `PROTECTED` attributes with privileged / unprivileged levels, province handling (ICT excluded), gap grid for the Pareto sweep, recommended TPR-gap tolerance 0.05, group FN-cost multipliers |
| `data.py`, `evaluate.py`, `importance.py` | Unchanged from Milestone 3 (`evaluate.expected_cost_rows` added for per-child FN costs) |
| `cv.py` | Milestone 3 grouped stratified CV plus `tune_random()` (RandomizedSearchCV on the same folds) and `fold_assignment()` |
| `models.py` | Milestone 3 builders plus `XGBEarlyStop` (XGBoost whose early-stopping slice is carved from its own training fold), `build_xgb` / `XGB_DIST`, `build_mlp` / `MLP_DIST`, `build_from_m3()` |
| `fairness.py` | Hand-coded audit: `group_rates`, `disparity` (DI ratio, DP difference, equal-opportunity gap, predictive-equality gap, equalized-odds difference), `bootstrap_disparity` (district-cluster CIs), `audit`; mitigation: `group_thresholds` (vectorised cost-minimising cut-offs under a TPR-gap constraint, binary or multi-level), `pareto_frontier`, `reweighing_weights` (Kamiran-Calders) |
| `plots.py` | Milestone 3 figures plus benchmark bars, fairness panels, disparity forest plots, Pareto plots, before/after bars, XGBoost importance |
| `run_all.py` | Pipeline in nine sections (split, baselines reproduced, random search, hold-out + bootstrap, OOF thresholds, audit, mitigation, importance, figures) -> `outputs/` |
| `make_report.py` | Builds the Word manuscript section from `outputs/results.json` |

## Where each rubric requirement is met

**A. Advanced architecture and optimisation (30)** - `models.py`; `cv.tune_random`; search spaces `results.json -> search_space` and report Table 12; grids `t03c`, `t03d`; baselines reproduced `t02`; CV summary `t05`; hold-out at both cut-offs `t07`, `t07b`; bootstrap gains `t06b`; figures `fig14`, `fig1`, `fig3`, `fig15`. Leakage: district-grouped split asserted, scaler and early-stopping slice inside folds, cut-offs on training OOF (`t09` with oracle columns).

**B. Bias and disparity audit (35)** - `fairness.py`; attributes `results.json -> protected_attributes`; audit at cost-tuned and 0.5 cut-offs with district-bootstrap CIs `t22`; group rates `t22b`; change vs Milestone 3 gaps `t23`; figures `fig11_*`, `fig11b`, `fig11c`; Pareto frontier `t25`, `fig12_*`.

**C. Mitigation and asymmetric thresholds (20)** - post-processing group cut-offs `t24` (+ province `t24c`), pre-processing reweighing `t26`, group FN-cost knob `t24b`; `fig13`.

**D. Report and reproducibility (15)** - report sections 8 (FATEN scorecard, living lab) and 9; executed notebook; this README.

## Headline results

See `HEADLINE.md` (generated by `tools/make_headline.py`) or report Tables 14, 16, 19 and 21.

## Notes and caveats

- Recommended-by-cost model and mitigation-focus model can differ: group cut-offs need a smooth score, so the pipeline takes the best ranker among models within 1% of the lowest hold-out cost (`results.json -> mitigation_focus`, with `n_distinct_scores` as evidence; a decision tree gives one score per leaf).
- A positive prediction is a flag for outreach (a benefit); the harm is a missed child. Equal opportunity (TPR gap) is the primary fairness metric; demographic parity / disparate impact are reported in the conventional direction too.
- Islamabad Capital Territory has no hold-out rows and is excluded from the province audit; groups with fewer than 100 positives are flagged.
- Cut-offs chosen on training OOF transfer imperfectly: hold-out gaps are larger than the training gaps the constraint was set on; both are reported. An earlier XGBoost configuration (learning rate 0.01, 800-tree cap) under-trained and compressed its scores around 0.5, which made per-group cut-offs meaningless; the search floor was raised to 0.02 with a 1,500-tree cap and 50-round early stopping.
- In-processing mitigation (fairlearn ExponentiatedGradient) and calibration were not implemented.
- The teammate plan's manuscript cross-reference fixes (Table 3 / Figure 16 numbering, 86.8% full-data enrolled share vs 87.5% hold-out majority accuracy) concern the team's merged draft, not this builder; the report states both percentages explicitly.
