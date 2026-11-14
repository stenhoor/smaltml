#!/usr/bin/env python3
"""Runs smaltml's compare cases through scikit-learn and writes the results to fixtures/compare.json.

ADR 0024, section 7: each case is run once in Python; the results become the tests (test/compare.test.ts), which CI
runs without Python. Rerun this to add cases or to check a new scikit-learn:

    pip install -r compare/requirements.txt
    npm run compare

The data is made here, with NumPy's seeded generator, and written into the fixture, so smaltml reads exactly the
numbers scikit-learn did. Each case states its tolerance: how far smaltml's results may be from scikit-learn's.
"""
import json
import math
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(20261114)


def tolist(a):
    """Arrays as lists, NaN as null (JSON has no NaN)."""
    a = np.asarray(a)
    if a.dtype.kind in "US":
        return a.tolist()
    return [None if (isinstance(v, float) and math.isnan(v)) else v for v in a.ravel().tolist()] if a.ndim == 1 else [tolist(r) for r in a]


def features(n, p, scale=None, shift=None):
    X = rng.normal(size=(n, p))
    if scale is not None:
        X = X * np.asarray(scale) + np.asarray(shift)
    return X


cases = []


def case(name, kind, inputs, expected, tol, note=""):
    if "params" in inputs:  # JSON has no infinity: C = inf is written "Infinity"
        inputs["params"] = {k: ("Infinity" if v == math.inf else v) for k, v in inputs["params"].items()}
    cases.append({"name": name, "kind": kind, "input": inputs, "expected": expected, "tol": tol, **({"note": note} if note else {})})


# StandardScaler -------------------------------------------------------------------------------------------------
for name, X in [
    ("scaler: columns of different sizes", features(60, 4, [1, 10, 1000, 0.01], [0, -5, 1e4, 3])),
    ("scaler: a constant column", np.column_stack([features(30, 2), np.full(30, 7.0)])),
    ("scaler: missing values", np.where(rng.random((40, 3)) < 0.15, np.nan, features(40, 3))),
]:
    for params in [{}, {"with_mean": False}, {"with_std": False}]:
        s = StandardScaler(**params).fit(X)
        case(f"{name} {params or ''}".strip(), "StandardScaler",
             {"X": tolist(X), "params": params},
             {"mean": tolist(s.mean_) if s.mean_ is not None else None, "var": tolist(s.var_) if s.var_ is not None else None,
              "scale": tolist(s.scale_) if s.scale_ is not None else None, "transform": tolist(s.transform(X))},
             {"rtol": 1e-12, "atol": 1e-12})

# LinearRegression -----------------------------------------------------------------------------------------------
for name, n, p, intercept in [("linear: three columns", 100, 3, True), ("linear: no intercept", 80, 2, False),
                              ("linear: ten columns", 300, 10, True), ("linear: as many rows as columns", 4, 4, True)]:
    X = features(n, p, rng.uniform(0.5, 50, p), rng.uniform(-20, 20, p))
    y = X @ rng.normal(size=p) * 3 + 5 + rng.normal(scale=0.5, size=n)
    m = LinearRegression(fit_intercept=intercept).fit(X, y)
    Xn = features(10, p)
    case(name, "LinearRegression", {"X": tolist(X), "y": tolist(y), "Xnew": tolist(Xn), "params": {"fit_intercept": intercept}},
         {"coef": tolist(m.coef_), "intercept": float(m.intercept_), "predict": tolist(m.predict(Xn))},
         {"rtol": 1e-8, "atol": 1e-8})

# LogisticRegression ---------------------------------------------------------------------------------------------
def classification(n, p, k, sep=1.0):
    centres = rng.normal(scale=sep, size=(k, p))
    y = rng.integers(0, k, size=n)
    return centres[y] + rng.normal(size=(n, p)), y


for name, (X, y), params, tol in [
    # Converged tightly, both sides find the same optimum.
    ("logistic: two classes", classification(200, 3, 2), {"tol": 1e-10, "max_iter": 10000}, {"rtol": 1e-5, "atol": 1e-6}),
    ("logistic: three classes", classification(240, 4, 3), {"tol": 1e-10, "max_iter": 10000}, {"rtol": 1e-5, "atol": 1e-6}),
    ("logistic: C = 0.05", classification(150, 2, 2, 2.0), {"C": 0.05, "tol": 1e-10, "max_iter": 10000}, {"rtol": 1e-5, "atol": 1e-6}),
    ("logistic: no intercept", classification(150, 3, 3), {"fit_intercept": False, "tol": 1e-10, "max_iter": 10000}, {"rtol": 1e-5, "atol": 1e-6}),
    ("logistic: no penalty", classification(200, 2, 2, 0.5), {"C": math.inf, "tol": 1e-10, "max_iter": 10000}, {"rtol": 1e-5, "atol": 1e-6}),
    # The defaults (tol 1e-4, 100 iterations): stopped early, the two optimizers land a little apart.
    ("logistic: defaults", classification(300, 5, 3), {}, {"rtol": 2e-3, "atol": 2e-3}),
]:
    m = LogisticRegression(**params).fit(X, y)
    Xn = rng.normal(size=(12, X.shape[1]))
    case(name, "LogisticRegression", {"X": tolist(X), "y": tolist(y), "Xnew": tolist(Xn), "params": params},
         {"classes": tolist(m.classes_), "coef": tolist(m.coef_), "intercept": tolist(m.intercept_),
          "predict": tolist(m.predict(Xn)), "predictProba": tolist(m.predict_proba(Xn))}, tol)

X, yi = classification(120, 2, 2)
ys = np.array(["no", "yes"])[yi]
m = LogisticRegression(tol=1e-10, max_iter=10000).fit(X, ys)
case("logistic: string labels", "LogisticRegression", {"X": tolist(X), "y": tolist(ys), "Xnew": tolist(X[:8]), "params": {"tol": 1e-10, "max_iter": 10000}},
     {"classes": tolist(m.classes_), "coef": tolist(m.coef_), "intercept": tolist(m.intercept_),
      "predict": tolist(m.predict(X[:8])), "predictProba": tolist(m.predict_proba(X[:8]))}, {"rtol": 1e-5, "atol": 1e-6})

# Pipeline -------------------------------------------------------------------------------------------------------
X, y = classification(200, 3, 3)
X = X * [1, 100, 0.01] + [0, 50, 1]
pipe = make_pipeline(StandardScaler(), LogisticRegression(tol=1e-10, max_iter=10000)).fit(X, y)
case("pipeline: scaler then logistic", "Pipeline", {"X": tolist(X), "y": tolist(y), "Xnew": tolist(X[:10]), "params": {"tol": 1e-10, "max_iter": 10000}},
     {"predict": tolist(pipe.predict(X[:10])), "predictProba": tolist(pipe.predict_proba(X[:10]))}, {"rtol": 1e-5, "atol": 1e-6})

# Metrics --------------------------------------------------------------------------------------------------------
t = rng.normal(size=50) * 4 + 2
p_ = t + rng.normal(size=50)
case("metrics: regression", "regression-metrics", {"yTrue": tolist(t), "yPred": tolist(p_)},
     {"meanSquaredError": mean_squared_error(t, p_), "r2Score": r2_score(t, p_)}, {"rtol": 1e-12, "atol": 1e-12})
case("metrics: r2 of a constant y", "regression-metrics", {"yTrue": [3.0, 3.0, 3.0], "yPred": [3.0, 3.0, 2.0]},
     {"meanSquaredError": mean_squared_error([3, 3, 3], [3, 3, 2]), "r2Score": r2_score([3, 3, 3], [3, 3, 2])}, {"rtol": 0, "atol": 0})
yt = rng.integers(0, 3, size=40)
proba = rng.dirichlet([1, 1, 1], size=40)
proba[0] = [1.0, 0.0, 0.0]  # a zero probability for a true label is clipped, not infinite
yt[0] = 1
case("metrics: classification", "classification-metrics",
     {"yTrue": tolist(yt), "yPred": tolist(proba.argmax(axis=1)), "yProba": tolist(proba)},
     {"accuracyScore": accuracy_score(yt, proba.argmax(axis=1)), "logLoss": log_loss(yt, proba)}, {"rtol": 1e-12, "atol": 1e-12})
yb = rng.integers(0, 2, size=30)
pb = rng.uniform(0.05, 0.95, size=30)
case("metrics: binary log loss from one column", "classification-metrics",
     {"yTrue": tolist(yb), "yPred": tolist((pb > 0.5).astype(int)), "yProba1": tolist(pb)},
     {"accuracyScore": accuracy_score(yb, pb > 0.5), "logLoss": log_loss(yb, pb)}, {"rtol": 1e-12, "atol": 1e-12})

# train_test_split: sizes, and the order without shuffling ----------------------------------------------------------
for n, opts in [(10, {}), (7, {}), (100, {"test_size": 0.2}), (9, {"test_size": 0.33}), (20, {"train_size": 0.6}),
                (20, {"test_size": 5}), (20, {"train_size": 0.5, "test_size": 0.25})]:
    X = np.arange(n * 2, dtype=float).reshape(n, 2)
    Xtr, Xte = train_test_split(X, shuffle=False, **opts)
    case(f"split: {n} rows {opts or ''}".strip(), "trainTestSplit", {"n": n, "params": opts},
         {"nTrain": len(Xtr), "nTest": len(Xte), "XTrain": tolist(Xtr), "XTest": tolist(Xte)}, {"rtol": 0, "atol": 0})

# MinMaxScaler, RobustScaler, SimpleImputer ---------------------------------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

Xs = features(50, 3, [1, 100, 0.1], [0, 50, -2])
Xs = np.column_stack([Xs, np.full(50, 4.0)])  # a constant column
Xm = np.where(rng.random((50, 4)) < 0.1, np.nan, Xs)
Xnew = features(8, 4, [3, 300, 0.3, 1], [0, 50, -2, 4])
for params in [{}, {"feature_range": (-1, 1)}, {"feature_range": (0, 10), "clip": True}]:
    for name, X in [("minmax", Xs), ("minmax: missing values", Xm)]:
        s = MinMaxScaler(**params).fit(X)
        case(f"{name} {params or ''}".strip(), "MinMaxScaler", {"X": tolist(X), "Xnew": tolist(Xnew), "params": {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}},
             {"dataMin": tolist(s.data_min_), "dataMax": tolist(s.data_max_), "scale": tolist(s.scale_), "min": tolist(s.min_),
              "transform": tolist(s.transform(Xnew))}, {"rtol": 1e-12, "atol": 1e-12})
for params in [{}, {"unit_variance": True}, {"quantile_range": (10.0, 90.0)}, {"with_centering": False}, {"with_scaling": False}]:
    for name, X in [("robust", Xs), ("robust: missing values", Xm)]:
        s = RobustScaler(**params).fit(X)
        case(f"{name} {params or ''}".strip(), "RobustScaler", {"X": tolist(X), "Xnew": tolist(Xnew), "params": {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}},
             {"center": tolist(s.center_) if s.center_ is not None else None, "scale": tolist(s.scale_) if s.scale_ is not None else None,
              "transform": tolist(s.transform(Xnew))}, {"rtol": 1e-12, "atol": 1e-12})
Xi = np.round(features(30, 4, [2, 10, 1, 1], [5, 0, 1, 0]), 0)
Xi[rng.random((30, 4)) < 0.2] = np.nan
Xi[:, 3] = np.nan  # a column with no values
for params in [{}, {"strategy": "median"}, {"strategy": "most_frequent"}, {"strategy": "constant", "fill_value": -1},
               {"keep_empty_features": True}, {"strategy": "median", "keep_empty_features": True}]:
    s = SimpleImputer(**params).fit(Xi)
    case(f"imputer {params or ''}".strip(), "SimpleImputer", {"X": tolist(Xi), "params": params},
         {"statistics": tolist(s.statistics_), "transform": tolist(s.transform(Xi))}, {"rtol": 1e-12, "atol": 1e-12})

# Encoders --------------------------------------------------------------------------------------------------------
colours = rng.choice(["red", "green", "blue", None], size=40, p=[0.4, 0.3, 0.2, 0.1])
sizes = rng.choice([1.0, 3.0, 5.0, np.nan], size=40, p=[0.3, 0.3, 0.3, 0.1])
Xe = np.empty((40, 2), dtype=object)
Xe[:, 0] = colours
Xe[:, 1] = sizes
Xc = Xe[:, :1]  # text only, for comparing names
for params in [{}, {"handle_unknown": "use_encoded_value", "unknown_value": -1}]:
    e = OrdinalEncoder(**params).fit(Xe)
    unseen = np.array([["purple", 3.0], ["red", 7.0]], dtype=object) if params else Xe[:5]
    case(f"ordinal {params or ''}".strip(), "OrdinalEncoder", {"X": tolist(Xe), "Xnew": tolist(unseen), "params": params},
         {"transform": tolist(e.transform(unseen).astype(float))}, {"rtol": 0, "atol": 0})
for params in [{}, {"drop": "first"}, {"drop": "if_binary"}, {"handle_unknown": "ignore"}]:
    e = OneHotEncoder(sparse_output=False, **params).fit(Xe)
    unseen = np.array([["purple", 3.0], ["red", np.nan]], dtype=object) if params.get("handle_unknown") else Xe[:6]
    ec = OneHotEncoder(sparse_output=False, **params).fit(Xc)
    case(f"one-hot {params or ''}".strip(), "OneHotEncoder", {"X": tolist(Xe), "Xnew": tolist(unseen), "params": params},
         {"transform": tolist(e.transform(unseen)), "textNames": ec.get_feature_names_out().tolist()}, {"rtol": 0, "atol": 0})
binary = np.array([["yes"], ["no"], ["yes"]], dtype=object)
e = OneHotEncoder(sparse_output=False, drop="if_binary").fit(binary)
case("one-hot: drop if binary on two categories", "OneHotEncoder", {"X": tolist(binary), "Xnew": tolist(binary), "params": {"drop": "if_binary"}},
     {"transform": tolist(e.transform(binary)), "textNames": e.get_feature_names_out().tolist()}, {"rtol": 0, "atol": 0})

# ColumnTransformer, and a pipeline that starts with one --------------------------------------------------------------
num = features(40, 2, [10, 2], [50, 0])
Xt = np.empty((40, 4), dtype=object)
Xt[:, 0] = num[:, 0]
Xt[:, 1] = colours
Xt[:, 2] = num[:, 1]
Xt[:, 3] = rng.integers(0, 5, 40).astype(float)
for remainder in ["drop", "passthrough"]:
    ct = ColumnTransformer([("scale", StandardScaler(), [0, 2]), ("onehot", OneHotEncoder(sparse_output=False), [1])], remainder=remainder).fit(Xt)
    case(f"column transformer, remainder {remainder}", "ColumnTransformer", {"X": tolist(Xt), "params": {"remainder": remainder}},
         {"transform": tolist(ct.transform(Xt).astype(float)), "names": ct.get_feature_names_out().tolist()}, {"rtol": 1e-12, "atol": 1e-12})
yt = (num[:, 0] > 50).astype(int) ^ (colours == "red").astype(int)
pipe = make_pipeline(ColumnTransformer([("scale", StandardScaler(), [0, 2]), ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1])]),
                     LogisticRegression(tol=1e-10, max_iter=10000)).fit(Xt, yt)
case("pipeline: column transformer then logistic", "ColumnPipeline", {"X": tolist(Xt), "y": tolist(yt), "params": {}},
     {"predictProba": tolist(pipe.predict_proba(Xt))}, {"rtol": 1e-5, "atol": 1e-6})

out = {
    "comment": "Written by compare/run.py (npm run compare); don't edit by hand.",
    "reference": {"scikit-learn": sklearn.__version__, "numpy": np.__version__},
    "cases": cases,
}
path = ROOT / "fixtures" / "compare.json"
path.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False) + "\n")
print(f"{len(cases)} cases written to {path.relative_to(ROOT)} (scikit-learn {sklearn.__version__})")
