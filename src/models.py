"""Baseline architectures and their hyperparameter grids."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE


def build_naive() -> DummyClassifier:
    """Majority-class benchmark: predicts 'enrolled' for every child."""
    return DummyClassifier(strategy="most_frequent")


def build_logreg() -> Pipeline:
    """Standardised logistic regression (scaler is fitted inside each CV fold -> no leakage)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE)),
    ])


LOGREG_GRID = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0],
    "clf__class_weight": [None, "balanced"],
}


def build_tree() -> Pipeline:
    return Pipeline([
        ("clf", DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)),
    ])


TREE_GRID = {
    "clf__max_depth": [3, 5, 7, 10, 15, None],
    "clf__min_samples_leaf": [1, 50, 200, 1000],
    "clf__class_weight": [None, "balanced"],
}

def build_rf() -> Pipeline:
    """Random forest: many de-correlated trees averaged; each tree sees a bootstrap sample and sqrt(13) features per split."""
    return Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=1, random_state=RANDOM_STATE)),
    ])


RF_GRID = {
    "clf__max_depth": [8, 12, None],
    "clf__min_samples_leaf": [20, 100, 300],
    "clf__class_weight": [None, "balanced_subsample"],
}

MODELS = ("logreg", "tree", "rf")          # the tuned learners, in reporting order

MODEL_NAMES = {
    "naive": "Majority-class baseline",
    "logreg": "Logistic regression",
    "tree": "Decision tree",
    "rf": "Random forest",
    "logreg_demo": "Logistic regression (age, female, urban)",
}


# ----------------------------------------------------------------------------- Milestone 4 additions
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


class XGBEarlyStop(ClassifierMixin, BaseEstimator):
    """XGBoost with early stopping on a stratified 10% slice carved from ITS OWN training data.

    Inside cross-validation each fold's training part is split again, so the validation slice used to stop boosting
    never comes from the CV test fold and never from the hold-out. n_estimators is only an upper bound.
    """

    def __init__(self, n_estimators=1500, max_depth=5, learning_rate=0.05, min_child_weight=50, subsample=1.0,
                 colsample_bytree=1.0, reg_lambda=1.0, scale_pos_weight=1.0, early_stopping_rounds=50,
                 val_fraction=0.1, random_state=RANDOM_STATE):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float); y = np.asarray(y).astype(int)
        idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=self.val_fraction, stratify=y,
                                          random_state=self.random_state)
        w_tr = None if sample_weight is None else np.asarray(sample_weight)[idx_tr]
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight, subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda, scale_pos_weight=self.scale_pos_weight,
            tree_method="hist", objective="binary:logistic", eval_metric="aucpr",
            early_stopping_rounds=self.early_stopping_rounds, n_jobs=1, random_state=self.random_state, verbosity=0,
        )
        self.model_.fit(X[idx_tr], y[idx_tr], sample_weight=w_tr, eval_set=[(X[idx_va], y[idx_va])], verbose=False)
        self.best_iteration_ = int(self.model_.best_iteration)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


def build_xgb() -> Pipeline:
    return Pipeline([("clf", XGBEarlyStop())])


XGB_DIST = {
    "clf__max_depth": randint(3, 9),                 # 3..8
    "clf__learning_rate": loguniform(0.02, 0.15),   # floor raised from 0.01: with 0.01 the booster was still improving at the tree cap and its scores were compressed around 0.5
    "clf__min_child_weight": randint(20, 201),       # 20..200 (hessian weight ~ child count for logistic)
    "clf__subsample": uniform(0.6, 0.4),             # 0.6..1.0
    "clf__colsample_bytree": uniform(0.6, 0.4),
    "clf__reg_lambda": uniform(0.0, 10.0),
    "clf__scale_pos_weight": [1.0, 6.6],             # 6.6 ~ negatives / positives
}
XGB_N_ITER = 30


def build_mlp() -> Pipeline:
    """Regularised feed-forward network: L2 (alpha) + early stopping on a 10% slice of its own training data."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, max_iter=300,
                              random_state=RANDOM_STATE)),
    ])


MLP_DIST = {
    "clf__hidden_layer_sizes": [(32,), (64, 32)],
    "clf__alpha": loguniform(1e-4, 1e-1),
    "clf__learning_rate_init": [1e-3, 3e-3],
}
MLP_N_ITER = 12


def build_from_m3(key: str, params: dict) -> Pipeline:
    """Re-create a Milestone 3 baseline with its tuned parameters (no re-tuning), for a like-for-like comparison."""
    pipe = {"logreg": build_logreg, "tree": build_tree, "rf": build_rf}[key]()
    pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
    return pipe


BASELINES = ("logreg", "tree", "rf")                 # Milestone 3 models, reproduced
ADVANCED = ("xgb", "mlp")                            # Milestone 4 models, tuned here
ALL_MODELS = (*BASELINES, *ADVANCED)
MODEL_NAMES.update({"xgb": "XGBoost (gradient boosting)", "mlp": "Neural network (MLP)"})
