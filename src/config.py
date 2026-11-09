"""Paths, column roles and global constants for Milestone 3.

Column roles are read from feature_manifest.json (produced in Milestone 2) so
nothing is hard-coded here except the demographic subset used for the H5 test.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # milestone3/
DATA_DIR = ROOT / "data"                             # Milestone 2 input files
OUT_DIR = ROOT / "outputs"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"
REPORT_DIR = ROOT / "report"

TRAINING_CSV = DATA_DIR / "aser2025_training.csv"
MANIFEST_JSON = DATA_DIR / "feature_manifest.json"
SUBGROUP_CSV = DATA_DIR / "subgroup_counts.csv"
DICTIONARY_CSV = DATA_DIR / "data_dictionary.csv"
ATTRITION_CSV = DATA_DIR / "attrition.csv"

RANDOM_STATE = 42
N_OUTER_SPLITS = 5     # one fold held out -> ~80/20 split, grouped by district
N_INNER_SPLITS = 5     # grouped stratified CV used for tuning on the training part
THRESHOLD_GRID = [round(x, 2) for x in [i / 100 for i in range(5, 96)]]
COST_RATIOS = [1, 3, 5, 10]   # cost of a false negative relative to a false positive
RECOMMENDED_COST_RATIO = 5

with open(MANIFEST_JSON) as fh:
    MANIFEST = json.load(fh)

TARGET: str = MANIFEST["target"]
FEATURES: list[str] = MANIFEST["features"]
ID_COL: str = MANIFEST["id"]
GROUP_COL: str = MANIFEST["group_for_cv"]
WEIGHT_COL: str = MANIFEST["weight"]
PROVINCE_REF: str = MANIFEST["province_reference_category"]

# Bare demographic model for hypothesis H5 (instructions: age, female, urban only)
DEMO_FEATURES: list[str] = ["age", "female", "urban"]
HYPOTHESIS_FEATURES = {
    "H1 Poverty": "assets_n",
    "H2 Legal identity": "bform",
    "H3 Disability": "any_disability",
    "H4 Gender x area": "female_x_urban",
}
PROVINCE_COLS = [c for c in FEATURES if c.startswith("province_")]
CONTROL_FEATURES = ["age", "house", "digital_n", *PROVINCE_COLS]

# Human-readable labels for figures/tables
LABELS = {
    "age": "Age (years)",
    "female": "Female",
    "urban": "Urban",
    "female_x_urban": "Female x Urban",
    "bform": "B-form registered",
    "any_disability": "Any disability",
    "assets_n": "Household assets (0-8)",
    "digital_n": "Digital use (0-5)",
    "house": "Housing (1 kachha - 3 pacca)",
    "province_Sindh": "Province: Sindh",
    "province_Khyber Pakhtuknkhwa": "Province: KP",
    "province_Balochistan": "Province: Balochistan",
    "province_Islamabad Capital Territory": "Province: ICT",
}

for _d in (TABLES_DIR, FIGURES_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- Milestone 4 additions
M3_RESULTS_JSON = DATA_DIR / "m3_results.json"       # frozen copy of Milestone 3 outputs/results.json (baseline params + numbers)
with open(M3_RESULTS_JSON) as fh:
    M3 = json.load(fh)
M3_BEST_PARAMS: dict = M3["best_params"]              # {"logreg": {...}, "tree": {...}, "rf": {...}}

# Protected / sensitive attributes for the fairness audit. "unpriv" is the group SDG 4.5 is concerned about.
# A positive prediction here = being flagged for outreach (a benefit), so the harm of concern is UNDER-flagging the unprivileged group.
PROTECTED = {
    "female":         {"label": "Gender",          "priv": 0, "unpriv": 1, "names": {0: "Boys", 1: "Girls"}},
    "urban":          {"label": "Area",            "priv": 1, "unpriv": 0, "names": {1: "Urban", 0: "Rural"}},
    "any_disability": {"label": "Disability",      "priv": 0, "unpriv": 1, "names": {0: "No disability", 1: "Disability"}},
    "bform":          {"label": "Legal identity",  "priv": 1, "unpriv": 0, "names": {1: "B-form registered", 0: "Unregistered"}},
    "poor":           {"label": "Poverty",         "priv": 0, "unpriv": 1, "names": {0: "Not bottom tercile", 1: "Bottom asset tercile"}},
}
PROVINCE_ATTR = {"label": "Province", "priv": "Punjab", "exclude": ["Islamabad Capital Territory"]}   # ICT has no hold-out rows
PARETO_ATTRS = ["female", "urban", "any_disability"]           # one Pareto frontier per attribute
TPR_GAP_GRID = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]   # max allowed TPR gap; dense near 0
RECOMMENDED_TPR_GAP = 0.05                                      # tolerance used for the headline mitigated operating point
SMALL_GROUP_POSITIVES = 100                                     # grey out groups with fewer positives
GROUP_FN_MULTIPLIER = {"female": 1.5, "any_disability": 1.5}    # policy knob: a missed girl / disabled child costs 1.5x
DI_BAND = (0.8, 1.25)                                           # four-fifths rule band
