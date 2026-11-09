# Headline results

Hold-out: 32,516 children in 29 districts (identical to Milestone 3). Cut-offs chosen on training out-of-fold predictions, cost ratio 5:1.

| Model | Source | CV ROC-AUC (mean +/- sd) | Hold-out ROC-AUC | Hold-out PR-AUC | Cost-tuned cut-off | Recall / precision there | Flagged per 1,000 | Cost per 1,000 |
|---|---|---|---|---|---|---|---|---|
| Majority class | M3 | - | 0.500 | 0.125 | - | - | - | 626 |
| Logistic regression | M3 | 0.736 +/- 0.047 | 0.674 | 0.214 | 0.54 | 0.57 / 0.20 | 353 | 552 |
| Decision tree | M3 | 0.744 +/- 0.051 | 0.711 | 0.236 | 0.16 | 0.62 / 0.24 | 328 | 487 |
| Random forest | M3 | 0.755 +/- 0.045 | 0.725 | 0.254 | 0.13 | 0.71 / 0.22 | 403 | 496 |
| XGBoost | M4 | 0.749 +/- 0.051 | 0.732 | 0.266 | 0.53 | 0.64 / 0.24 | 343 | 485 |
| Neural network (MLP) | M4 | 0.746 +/- 0.048 | 0.721 | 0.257 | 0.13 | 0.69 / 0.22 | 396 | 504 |

## Bootstrap AUC gains (1,000 district re-samples)

| Model | vs | Gain | 95% CI |
|---|---|---|---|
| XGBoost | Random forest | +0.008 | [-0.002, +0.017] |
| XGBoost | Logistic regression | +0.058 | [+0.033, +0.081] |
| XGBoost | Decision tree | +0.021 | [+0.010, +0.034] |
| Neural network (MLP) | Random forest | -0.004 | [-0.014, +0.004] |
| Neural network (MLP) | Logistic regression | +0.046 | [+0.021, +0.070] |
| Neural network (MLP) | XGBoost | -0.012 | [-0.021, -0.003] |

## Fairness audit, XGBoost at cut-off 0.53 (unprivileged vs privileged)

| Attribute | DI ratio | TPR priv. | TPR unpriv. | Equal-opp. gap [95% CI] | Pred.-equality gap |
|---|---|---|---|---|---|
| Gender | 1.44 | 0.61 | 0.69 | +0.07 [-0.02, +0.17] | +0.13 |
| Area | 1.95 | 0.48 | 0.71 | +0.23 [+0.11, +0.37] | +0.19 |
| Disability | 0.75 | 0.66 | 0.54 | -0.12 [-0.27, +0.04] | -0.08 |
| Legal identity | 1.90 | 0.52 | 0.73 | +0.21 [+0.09, +0.32] | +0.20 |
| Poverty | 2.65 | 0.49 | 0.76 | +0.27 [+0.14, +0.43] | +0.32 |
| Province: Balochistan vs Punjab | 2.96 | 0.52 | 0.88 | +0.36 [+0.25, +0.55] | +0.48 |
| Province: Khyber Pakhtuknkhwa vs Punjab | 1.33 | 0.52 | 0.68 | +0.16 [-0.08, +0.40] | +0.05 |
| Province: Sindh vs Punjab | 0.52 | 0.52 | 0.45 | -0.07 [-0.45, +0.15] | -0.15 |

## Post-processing: group-specific cut-offs, XGBoost (allowed TPR gap 0.05)

| Attribute | Cut-offs after {unpriv/priv} | EO gap before -> after | Cost/1,000 before -> after | Flagged/1,000 before -> after |
|---|---|---|---|---|
| Gender | {"1": 0.595, "0": 0.522} | +0.07 -> -0.03 | 485 -> 485 | 343 -> 323 |
| Area | {"0": 0.522, "1": 0.452} | +0.23 -> +0.13 | 485 -> 486 | 343 -> 375 |
| Disability | {"0": 0.522, "1": 0.567} | -0.12 -> -0.18 | 485 -> 486 | 343 -> 348 |
| Legal identity | {"0": 0.537, "1": 0.455} | +0.21 -> +0.09 | 485 -> 497 | 343 -> 390 |
| Poverty | {"0": 0.467, "1": 0.57} | +0.27 -> +0.13 | 485 -> 493 | 343 -> 361 |

## Pre-processing: reweighing

| Model | Attribute | EO gap before -> after | Cost/1,000 before -> after |
|---|---|---|---|
| Logistic regression | Area | +0.35 -> +0.12 | 552 -> 551 |
| XGBoost | Area | +0.23 -> +0.08 | 485 -> 494 |
| Logistic regression | Gender | +0.07 -> -0.01 | 552 -> 554 |
| XGBoost | Gender | +0.07 -> +0.01 | 485 -> 483 |
| Logistic regression | Legal identity | +0.24 -> -0.12 | 552 -> 608 |
| XGBoost | Legal identity | +0.21 -> +0.00 | 485 -> 515 |

Mitigation focus model: XGBoost (distinct hold-out scores: Logistic regression 13,283, Decision tree 221, Random forest 12,974, XGBoost 13,193, Neural network (MLP) 12,949). Runtime 14 min.
