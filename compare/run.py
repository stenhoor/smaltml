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
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
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


def fit_converged(model, X, y):
    """Fits the model; returns it and whether scikit-learn fitted it without a ConvergenceWarning. Other warnings pass."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    for w in caught:
        if not issubclass(w.category, ConvergenceWarning):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return model, not any(issubclass(w.category, ConvergenceWarning) for w in caught)


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
    m, converged = fit_converged(LogisticRegression(**params), X, y)
    Xn = rng.normal(size=(12, X.shape[1]))
    case(name, "LogisticRegression", {"X": tolist(X), "y": tolist(y), "Xnew": tolist(Xn), "params": params},
         {"classes": tolist(m.classes_), "coef": tolist(m.coef_), "intercept": tolist(m.intercept_),
          "predict": tolist(m.predict(Xn)), "predictProba": tolist(m.predict_proba(Xn)), "converged": converged}, tol)

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

# Ridge, Lasso, ElasticNet ----------------------------------------------------------------------------------------
from sklearn.linear_model import ElasticNet, Lasso, Ridge

def regression_data(n, p, informative):
    X = features(n, p, rng.uniform(0.5, 20, p), rng.uniform(-10, 10, p))
    w = np.zeros(p)
    w[:informative] = rng.normal(size=informative) * 3
    return X, X @ w + 7 + rng.normal(scale=1.0, size=n)

Xr, yr = regression_data(120, 8, 3)
Xrn = features(10, 8, rng.uniform(0.5, 20, 8), rng.uniform(-10, 10, 8))
for params in [{}, {"alpha": 0.01}, {"alpha": 50.0}, {"fit_intercept": False}]:
    m = Ridge(**params).fit(Xr, yr)
    case(f"ridge {params or ''}".strip(), "Ridge", {"X": tolist(Xr), "y": tolist(yr), "Xnew": tolist(Xrn), "params": params},
         {"coef": tolist(m.coef_), "intercept": float(m.intercept_), "predict": tolist(m.predict(Xrn))}, {"rtol": 1e-9, "atol": 1e-9})
for cls, kind, grid in [(Lasso, "Lasso", [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 5.0}, {"alpha": 0.5, "fit_intercept": False}]),
                        (ElasticNet, "ElasticNet", [{"alpha": 0.1}, {"alpha": 1.0, "l1_ratio": 0.2}, {"alpha": 0.3, "l1_ratio": 0.9}, {"alpha": 2.0, "l1_ratio": 0.0}])]:
    for params in grid:
        # Converged tightly, both find the one optimum.
        tight = {**params, "tol": 1e-12, "max_iter": 100000}
        m, converged = fit_converged(cls(**tight), Xr, yr)
        case(f"{kind.lower()} {params}", kind, {"X": tolist(Xr), "y": tolist(yr), "Xnew": tolist(Xrn), "params": tight},
             {"coef": tolist(m.coef_), "intercept": float(m.intercept_), "predict": tolist(m.predict(Xrn)),
              "zero": [bool(v == 0) for v in m.coef_], "converged": converged}, {"rtol": 1e-7, "atol": 1e-7})
    # The defaults: each stops by its own iterations, close to the optimum.
    m, converged = fit_converged(cls(), Xr, yr)
    case(f"{kind.lower()}: defaults", kind, {"X": tolist(Xr), "y": tolist(yr), "Xnew": tolist(Xrn), "params": {}},
         {"coef": tolist(m.coef_), "intercept": float(m.intercept_), "predict": tolist(m.predict(Xrn)),
          "zero": [bool(v == 0) for v in m.coef_], "converged": converged}, {"rtol": 1e-3, "atol": 1e-3})
big, converged = fit_converged(Lasso(alpha=1000.0), Xr, yr)
case("lasso: a penalty that zeroes every coefficient", "Lasso", {"X": tolist(Xr), "y": tolist(yr), "Xnew": tolist(Xrn), "params": {"alpha": 1000.0}},
     {"coef": tolist(big.coef_), "intercept": float(big.intercept_), "predict": tolist(big.predict(Xrn)), "zero": [bool(v == 0) for v in big.coef_],
      "converged": converged},
     {"rtol": 1e-12, "atol": 1e-12})

# Metrics: classification and absolute error ------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score

t = rng.normal(size=40) * 5
case("metrics: mean absolute error", "mae", {"yTrue": tolist(t), "yPred": tolist(t + rng.normal(size=40))},
     {"value": mean_absolute_error(t, t + 0)}, {"rtol": 1e-12, "atol": 1e-12})
cases[-1]["expected"]["value"] = mean_absolute_error(cases[-1]["input"]["yTrue"], cases[-1]["input"]["yPred"])
for name, yt, yp in [
    ("binary", rng.integers(0, 2, 60), rng.integers(0, 2, 60)),
    ("three classes", rng.integers(0, 3, 80), rng.integers(0, 3, 80)),
    ("strings", rng.choice(["cat", "dog", "fox"], 50), rng.choice(["cat", "dog", "fox", "owl"], 50)),
]:
    averages = ["binary", "micro", "macro", "weighted", None] if name == "binary" else ["micro", "macro", "weighted", None]
    for avg in averages:
        out = {}
        for fn, key in [(precision_score, "precision"), (recall_score, "recall"), (f1_score, "f1")]:
            v = fn(yt, yp, average=avg, zero_division=0)
            out[key] = tolist(v) if avg is None else float(v)
        case(f"metrics: precision, recall, F1, {name}, average {avg}", "prf", {"yTrue": tolist(yt), "yPred": tolist(yp), "params": {"average": avg}}, out, {"rtol": 1e-12, "atol": 1e-12})
    labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    case(f"metrics: confusion matrix, {name}", "confusion", {"yTrue": tolist(yt), "yPred": tolist(yp)},
         {"matrix": tolist(confusion_matrix(yt, yp).astype(float)), "labels": labels}, {"rtol": 0, "atol": 0})
# No predictions of a class: the zero-division value.
case("metrics: precision with a class never predicted", "prf", {"yTrue": [0, 1, 2, 2], "yPred": [0, 0, 0, 2], "params": {"average": "macro"}},
     {"precision": precision_score([0, 1, 2, 2], [0, 0, 0, 2], average="macro", zero_division=0),
      "recall": recall_score([0, 1, 2, 2], [0, 0, 0, 2], average="macro", zero_division=0),
      "f1": f1_score([0, 1, 2, 2], [0, 0, 0, 2], average="macro", zero_division=0)}, {"rtol": 1e-12, "atol": 1e-12})
yb = rng.integers(0, 2, 70)
sb = np.round(rng.normal(size=70) + yb, 1)  # rounded, so there are ties
case("metrics: ROC AUC, binary with ties", "auc", {"yTrue": tolist(yb), "yScore": tolist(sb), "params": {}},
     {"value": roc_auc_score(yb, sb)}, {"rtol": 1e-12, "atol": 1e-12})
y3 = rng.integers(0, 3, 90)
s3 = rng.dirichlet([1, 1, 1], size=90)
for avg in ["macro", "weighted"]:
    case(f"metrics: ROC AUC, three classes, one against the rest, {avg}", "auc", {"yTrue": tolist(y3), "yScore": tolist(s3), "params": {"multi_class": "ovr", "average": avg}},
         {"value": roc_auc_score(y3, s3, multi_class="ovr", average=avg)}, {"rtol": 1e-12, "atol": 1e-12})

# Folds -------------------------------------------------------------------------------------------------------------
for n, k in [(10, 5), (11, 3), (7, 2), (100, 10)]:
    folds = [tolist(test) for _, test in KFold(n_splits=k).split(np.zeros(n))]
    case(f"folds: KFold, {n} rows in {k}", "kfold", {"n": n, "params": {"n_splits": k}}, {"tests": folds}, {"rtol": 0, "atol": 0})
for n, k in [(30, 5), (31, 4)]:
    ys = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    sizes = sorted(len(test) for _, test in StratifiedKFold(n_splits=k).split(np.zeros(n), ys))
    case(f"folds: StratifiedKFold, {n} rows in {k}", "stratified", {"y": tolist(ys), "params": {"n_splits": k}}, {"testSizes": sizes}, {"rtol": 0, "atol": 0})

# Cross-validation and grid search, on KFold's folds -----------------------------------------------------------------
Xg, yg = regression_data(80, 6, 4)
cv = KFold(n_splits=5)
for scoring in [None, "neg_mean_squared_error", "neg_mean_absolute_error"]:
    s = cross_val_score(make_pipeline(StandardScaler(), Ridge(alpha=2.0)), Xg, yg, cv=cv, scoring=scoring)
    case(f"cross-validation: scaled ridge, {scoring or 'default (R2)'}", "cvscore", {"X": tolist(Xg), "y": tolist(yg), "params": {"scoring": scoring}, "model": "scaled-ridge"},
         {"scores": tolist(s)}, {"rtol": 1e-9, "atol": 1e-9})
Xc, yc = classification(150, 4, 3)
for scoring in [None, "neg_log_loss", "f1_macro"]:
    s = cross_val_score(LogisticRegression(tol=1e-10, max_iter=10000), Xc, yc, cv=cv, scoring=scoring)
    case(f"cross-validation: logistic, {scoring or 'default (accuracy)'}", "cvscore", {"X": tolist(Xc), "y": tolist(yc), "params": {"scoring": scoring}, "model": "logistic"},
         {"scores": tolist(s)}, {"rtol": 1e-5, "atol": 1e-6})
ybin = (yc == 0).astype(int)
s = cross_val_score(LogisticRegression(tol=1e-10, max_iter=10000), Xc, ybin, cv=cv, scoring="roc_auc")
case("cross-validation: logistic, roc_auc", "cvscore", {"X": tolist(Xc), "y": tolist(ybin), "params": {"scoring": "roc_auc"}, "model": "logistic"},
     {"scores": tolist(s)}, {"rtol": 1e-6, "atol": 1e-6})
g = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.01, 1.0, 30.0, 300.0]}, cv=cv).fit(Xg, yg)
case("grid search: scaled ridge over alpha", "grid", {"X": tolist(Xg), "y": tolist(yg), "params": {}, "model": "scaled-ridge", "grid": {"ridge__alpha": [0.01, 1.0, 30.0, 300.0]}},
     {"params": g.cv_results_["params"], "mean": tolist(g.cv_results_["mean_test_score"]), "std": tolist(g.cv_results_["std_test_score"]),
      "rank": tolist(g.cv_results_["rank_test_score"]), "best": g.best_params_}, {"rtol": 1e-9, "atol": 1e-9})
grid = {"C": [0.01, 0.1, 1.0, 10.0], "fit_intercept": [True, False]}
g = GridSearchCV(LogisticRegression(tol=1e-10, max_iter=10000), grid, cv=cv, scoring="f1_macro").fit(Xc, yc)
case("grid search: logistic over C and fit_intercept, f1_macro", "grid", {"X": tolist(Xc), "y": tolist(yc), "params": {"scoring": "f1_macro"}, "model": "logistic", "grid": grid},
     {"params": g.cv_results_["params"], "mean": tolist(g.cv_results_["mean_test_score"]), "std": tolist(g.cv_results_["std_test_score"]),
      "rank": tolist(g.cv_results_["rank_test_score"]), "best": g.best_params_}, {"rtol": 1e-5, "atol": 1e-6})

# LogisticRegression with solver newton-cholesky (appended, so the cases above keep their data) ------------------
for name, (X, y), params, tol in [
    ("logistic newton-cholesky: two classes", classification(300, 4, 2), {"solver": "newton-cholesky", "tol": 1e-10, "max_iter": 100}, {"rtol": 1e-6, "atol": 1e-7}),
    ("logistic newton-cholesky: C = 0.05", classification(200, 3, 2, 2.0), {"solver": "newton-cholesky", "C": 0.05, "tol": 1e-10}, {"rtol": 1e-6, "atol": 1e-7}),
    ("logistic newton-cholesky: no intercept", classification(150, 3, 2), {"solver": "newton-cholesky", "fit_intercept": False, "tol": 1e-10}, {"rtol": 1e-6, "atol": 1e-7}),
    ("logistic newton-cholesky: no penalty", classification(250, 2, 2, 0.5), {"solver": "newton-cholesky", "C": math.inf, "tol": 1e-10}, {"rtol": 1e-6, "atol": 1e-7}),
    ("logistic newton-cholesky: defaults", classification(400, 6, 2), {"solver": "newton-cholesky"}, {"rtol": 1e-4, "atol": 1e-5}),
]:
    m, converged = fit_converged(LogisticRegression(**params), X, y)
    Xn = rng.normal(size=(12, X.shape[1]))
    case(name, "LogisticRegression", {"X": tolist(X), "y": tolist(y), "Xnew": tolist(Xn), "params": params},
         {"classes": tolist(m.classes_), "coef": tolist(m.coef_), "intercept": tolist(m.intercept_),
          "predict": tolist(m.predict(Xn)), "predictProba": tolist(m.predict_proba(Xn)), "converged": converged}, tol)

# Fits stopped before they converge (appended, so the cases above keep their data) ---------------------------------
# Only whether each converged, and its iterations: two solvers stopped early land at different points, so the
# coefficients aren't compared. scikit-learn raises a ConvergenceWarning for each; smaltml warns ML_CONVERGENCE.
Xu, yu = classification(300, 4, 3)
Xu = Xu * rng.uniform(1, 100, 4) + rng.uniform(-50, 50, 4)  # unscaled columns
for name, model, params, X, y in [
    ("unfinished: logistic lbfgs, 5 iterations", "LogisticRegression", {"max_iter": 5}, Xu, yu),
    ("unfinished: logistic lbfgs, two classes, 3 iterations", "LogisticRegression", {"max_iter": 3}, Xu, (yu == 0).astype(int)),
    ("unfinished: logistic newton-cholesky, 1 iteration", "LogisticRegression", {"solver": "newton-cholesky", "max_iter": 1, "tol": 1e-12}, Xu, (yu == 0).astype(int)),
    ("unfinished: lasso, 2 iterations", "Lasso", {"alpha": 0.1, "max_iter": 2, "tol": 1e-12}, Xr, yr),
    ("unfinished: elasticnet, 3 iterations", "ElasticNet", {"alpha": 0.1, "max_iter": 3, "tol": 1e-12}, Xr, yr),
    ("finished: logistic lbfgs, the same data scaled", "LogisticRegression", {}, StandardScaler().fit_transform(Xu), yu),
]:
    cls = {"LogisticRegression": LogisticRegression, "Lasso": Lasso, "ElasticNet": ElasticNet}[model]
    m, converged = fit_converged(cls(**params), X, y)
    case(name, "convergence", {"model": model, "X": tolist(X), "y": tolist(y), "params": params},
         {"converged": converged, "nIter": int(np.max(m.n_iter_))}, {"rtol": 0, "atol": 0})

# Sample weights (appended, so the cases above keep their data) ---------------------------------------------------
Xw, yw = regression_data(150, 6, 4)
Xwn = features(10, 6, rng.uniform(0.5, 20, 6), rng.uniform(-10, 10, 6))
sw = rng.uniform(0.0, 3.0, size=150)
sw[:10] = 0.0  # rows that don't count at all
sw_int = rng.integers(1, 4, size=150).astype(float)  # whole numbers: the same as repeating rows
for name, kind, cls, params, weights, tol in [
    ("linear: sample weights", "LinearRegression", LinearRegression, {}, sw, {"rtol": 1e-9, "atol": 1e-9}),
    ("linear: sample weights, no intercept", "LinearRegression", LinearRegression, {"fit_intercept": False}, sw, {"rtol": 1e-9, "atol": 1e-9}),
    ("linear: whole-number sample weights", "LinearRegression", LinearRegression, {}, sw_int, {"rtol": 1e-9, "atol": 1e-9}),
    ("ridge: sample weights", "Ridge", Ridge, {"alpha": 3.0}, sw, {"rtol": 1e-9, "atol": 1e-9}),
    ("ridge: sample weights, no intercept", "Ridge", Ridge, {"alpha": 3.0, "fit_intercept": False}, sw, {"rtol": 1e-9, "atol": 1e-9}),
    ("lasso: sample weights", "Lasso", Lasso, {"alpha": 0.2, "tol": 1e-12, "max_iter": 100000}, sw, {"rtol": 1e-7, "atol": 1e-7}),
    ("lasso: sample weights, no intercept", "Lasso", Lasso, {"alpha": 0.2, "fit_intercept": False, "tol": 1e-12, "max_iter": 100000}, sw, {"rtol": 1e-7, "atol": 1e-7}),
    ("elasticnet: sample weights", "ElasticNet", ElasticNet, {"alpha": 0.3, "l1_ratio": 0.4, "tol": 1e-12, "max_iter": 100000}, sw, {"rtol": 1e-7, "atol": 1e-7}),
]:
    m = cls(**params).fit(Xw, yw, sample_weight=weights)
    expected = {"coef": tolist(m.coef_), "intercept": float(m.intercept_), "predict": tolist(m.predict(Xwn))}
    if kind in ("Lasso", "ElasticNet"):
        expected["zero"] = [bool(v == 0) for v in m.coef_]
    case(name, kind, {"X": tolist(Xw), "y": tolist(yw), "Xnew": tolist(Xwn), "params": params, "sampleWeight": tolist(weights)}, expected, tol)

# LogisticRegression with sample and class weights (appended) ---------------------------------------------------
Xl2, yl2 = classification(240, 3, 2)
Xl3, yl3 = classification(300, 4, 3)
Xln = rng.normal(size=(12, 4))
yl2[:180] = np.where(rng.random(180) < 0.8, 0, yl2[:180])  # imbalanced, so class weights matter
swl2 = rng.uniform(0.0, 3.0, size=240)
swl3 = rng.uniform(0.0, 3.0, size=300)
tight = {"tol": 1e-10, "max_iter": 10000}
for name, X_, y_, params, weights in [
    ("logistic: sample weights, 2 classes", Xl2, yl2, tight, swl2),
    ("logistic: sample weights, 3 classes", Xl3, yl3, tight, swl3),
    ("logistic: class_weight balanced, 2 classes", Xl2, yl2, {**tight, "class_weight": "balanced"}, None),
    ("logistic: class_weight balanced, 3 classes", Xl3, yl3, {**tight, "class_weight": "balanced"}, None),
    ("logistic: class_weight by class, 2 classes", Xl2, yl2, {**tight, "class_weight": {1: 4.0}}, None),
    ("logistic: sample and class weights, 3 classes", Xl3, yl3, {**tight, "C": 0.3, "class_weight": "balanced"}, swl3),
    ("logistic newton-cholesky: sample and class weights", Xl2, yl2, {"solver": "newton-cholesky", "tol": 1e-10, "class_weight": "balanced"}, swl2),
]:
    m = LogisticRegression(**params).fit(X_, y_, sample_weight=weights)
    Xn_ = Xln[:, : X_.shape[1]]
    inputs = {"X": tolist(X_), "y": tolist(y_), "Xnew": tolist(Xn_), "params": params}
    if weights is not None:
        inputs["sampleWeight"] = tolist(weights)
    case(name, "LogisticRegression", inputs,
         {"classes": tolist(m.classes_), "coef": tolist(m.coef_), "intercept": tolist(m.intercept_),
          "predict": tolist(m.predict(Xn_)), "predictProba": tolist(m.predict_proba(Xn_))}, {"rtol": 1e-5, "atol": 1e-6})

# train_test_split with stratify: the classes' counts in each set (appended) -------------------------------------
from sklearn.model_selection import train_test_split
for name, counts, kw in [
    ("split: stratify, three classes", [53, 31, 16], {"test_size": 0.3}),
    ("split: stratify, a rare class", [90, 7, 3], {"test_size": 0.2}),
    ("split: stratify, train_size given", [40, 25, 20, 15], {"train_size": 0.6}),
    ("split: stratify, sizes in rows", [33, 21, 11], {"train_size": 40, "test_size": 17}),
    ("split: stratify, text labels", [12, 9, 6], {"test_size": 0.34}),
]:
    labels = np.repeat(np.arange(len(counts)), counts)
    if "text" in name:
        labels = np.array(["cat", "dog", "eel"])[labels]
    order = rng.permutation(len(labels))
    labels = labels[order]
    X_ = np.arange(len(labels), dtype=float).reshape(-1, 1)
    per = []
    for seed in (0, 1, 2):  # the counts must not depend on the seed: no ties, so smaltml's rule gives the same
        _, _, ytr, yte = train_test_split(X_, labels, stratify=labels, random_state=seed, **kw)
        classes = np.unique(labels)
        per.append(([int((ytr == c).sum()) for c in classes], [int((yte == c).sum()) for c in classes]))
    assert per[0] == per[1] == per[2], name
    params = {("testSize" if k == "test_size" else "trainSize"): v for k, v in kw.items()}
    case(name, "split-stratify", {"y": tolist(labels), "params": params},
         {"classes": tolist(np.unique(labels)), "train": per[0][0], "test": per[0][1]}, {"rtol": 0, "atol": 0})

# LogisticRegression newton-cholesky with more than two classes (appended) --------------------------------------
for name, (X_, y_), params, tol in [
    ("logistic newton-cholesky: three classes", classification(300, 4, 3), {"solver": "newton-cholesky", "tol": 1e-10}, {"rtol": 1e-6, "atol": 1e-7}),
    ("logistic newton-cholesky: five classes", classification(500, 3, 5, 1.5), {"solver": "newton-cholesky", "tol": 1e-10}, {"rtol": 1e-6, "atol": 1e-7}),
    ("logistic newton-cholesky: three classes, C = 0.05", classification(200, 3, 3), {"solver": "newton-cholesky", "C": 0.05, "tol": 1e-10}, {"rtol": 1e-6, "atol": 1e-7}),
    ("logistic newton-cholesky: three classes, no intercept", classification(200, 3, 3), {"solver": "newton-cholesky", "fit_intercept": False, "tol": 1e-10}, {"rtol": 1e-6, "atol": 1e-7}),
    ("logistic newton-cholesky: three classes, defaults", classification(400, 5, 3), {"solver": "newton-cholesky"}, {"rtol": 1e-4, "atol": 1e-5}),
]:
    m = LogisticRegression(**params).fit(X_, y_)
    Xn_ = rng.normal(size=(12, X_.shape[1]))
    case(name, "LogisticRegression", {"X": tolist(X_), "y": tolist(y_), "Xnew": tolist(Xn_), "params": params},
         {"classes": tolist(m.classes_), "coef": tolist(m.coef_), "intercept": tolist(m.intercept_),
          "predict": tolist(m.predict(Xn_)), "predictProba": tolist(m.predict_proba(Xn_)), "nIter": int(m.n_iter_[0])}, tol)

# LogisticRegression with solver saga: L1 and elastic-net penalties (appended) -----------------------------------
from sklearn.preprocessing import StandardScaler as _Scaler
def scaled(X):
    return _Scaler().fit_transform(X)
Xs2, ys2 = classification(200, 4, 2)
Xs3, ys3 = classification(240, 4, 3)
Xs2, Xs3 = scaled(Xs2), scaled(Xs3)
sws = rng.uniform(0.2, 2.0, size=200)
saga_tight = {"solver": "saga", "tol": 1e-10, "max_iter": 20000, "random_state": 0}
for name, X_, y_, params, weights in [
    ("logistic saga: L1, two classes", Xs2, ys2, {**saga_tight, "l1_ratio": 1.0, "C": 0.5}, None),
    ("logistic saga: strong L1 zeroes coefficients", Xs2, ys2, {**saga_tight, "l1_ratio": 1.0, "C": 0.03}, None),
    ("logistic saga: elastic net, two classes", Xs2, ys2, {**saga_tight, "l1_ratio": 0.5, "C": 0.2}, None),
    ("logistic saga: L2 only (l1_ratio 0)", Xs2, ys2, {**saga_tight, "l1_ratio": 0.0}, None),
    ("logistic saga: L1, three classes", Xs3, ys3, {**saga_tight, "l1_ratio": 1.0, "C": 0.3}, None),
    ("logistic saga: elastic net, three classes, no intercept", Xs3, ys3, {**saga_tight, "l1_ratio": 0.3, "C": 0.5, "fit_intercept": False}, None),
    ("logistic saga: L1 with sample weights", Xs2, ys2, {**saga_tight, "l1_ratio": 1.0, "C": 0.5}, sws),
]:
    m = LogisticRegression(**params).fit(X_, y_, sample_weight=weights)
    Xn_ = scaled(rng.normal(size=(12, X_.shape[1])))
    ours = {k: v for k, v in params.items() if k != "random_state"}
    inputs = {"X": tolist(X_), "y": tolist(y_), "Xnew": tolist(Xn_), "params": {**ours, "seed": 0}}
    if weights is not None:
        inputs["sampleWeight"] = tolist(weights)
    case(name, "LogisticRegression", inputs,
         {"classes": tolist(m.classes_), "coef": tolist(m.coef_), "intercept": tolist(m.intercept_),
          "predict": tolist(m.predict(Xn_)), "predictProba": tolist(m.predict_proba(Xn_)),
          "zero": [[bool(v == 0) for v in row] for row in m.coef_]}, {"rtol": 1e-5, "atol": 1e-6})

# StandardScaler partial_fit over batches (appended) ------------------------------------------------------------
for name, X_, sizes in [
    ("scaler partial_fit: three batches", features(90, 3, [1, 50, 0.1], [2, -40, 7]), [30, 45, 15]),
    ("scaler partial_fit: batches of one row", features(12, 2, [3, 1], [1, 9]), [1] * 12),
    ("scaler partial_fit: missing values", np.where(rng.random((60, 3)) < 0.2, np.nan, features(60, 3)), [20, 25, 15]),
    ("scaler partial_fit: a column missing in the first batch",
     np.column_stack([features(40, 2), np.concatenate([np.full(15, np.nan), rng.normal(size=25)])]), [15, 25]),
]:
    for params in [{}, {"with_mean": False}, {"with_std": False}]:
        s = StandardScaler(**params)
        batches = np.split(X_, np.cumsum(sizes)[:-1])
        for b in batches:
            s.partial_fit(b)
        seen = s.n_samples_seen_
        note = ""
        if "first batch" in name:
            # A column with no value in the first batch stays NaN in scikit-learn's partial_fit for good (its update
            # multiplies the NaN mean by a count of 0). smaltml gives the column's statistics over the rows that have
            # them: the same as fit() on every row, which is what this case expects.
            assert np.isnan(s.mean_[2]), "scikit-learn's partial_fit no longer gives NaN here: drop the divergence"
            s = StandardScaler(**params).fit(X_)
            note = "deliberate: scikit-learn's partial_fit gives NaN for this column; expected is fit() on every row"
        case(f"{name} {params or ''}".strip(), "StandardScaler-partial",
             {"batches": [tolist(b) for b in batches], "params": params},
             {"mean": tolist(s.mean_) if s.mean_ is not None else None, "var": tolist(s.var_) if s.var_ is not None else None,
              "scale": tolist(s.scale_) if s.scale_ is not None else None,
              "nSamplesSeen": tolist(seen) if isinstance(seen, np.ndarray) else int(seen), "transform": tolist(s.transform(X_))},
             {"rtol": 1e-10, "atol": 1e-12}, note)

# positive: coefficients held at 0 or more (appended) -------------------------------------------------------------
Xp = features(150, 6, rng.uniform(0.5, 5, 6), rng.uniform(-3, 3, 6))
yp = Xp @ np.array([2.0, -1.5, 0.7, -0.3, 1.2, 0.0]) + 4 + rng.normal(scale=0.5, size=150)
Xpn = features(10, 6, rng.uniform(0.5, 5, 6), rng.uniform(-3, 3, 6))
swp = rng.uniform(0.2, 2.0, size=150)
for name, cls, kind, params, ours, weights, tol in [
    ("linear positive", LinearRegression, "LinearRegression", {"positive": True}, None, None, {"rtol": 1e-9, "atol": 1e-9}),
    ("linear positive, no intercept", LinearRegression, "LinearRegression", {"positive": True, "fit_intercept": False}, None, None, {"rtol": 1e-9, "atol": 1e-9}),
    ("linear positive, sample weights", LinearRegression, "LinearRegression", {"positive": True}, None, swp, {"rtol": 1e-9, "atol": 1e-9}),
    # scikit-learn's Ridge solves positive by L-BFGS-B, to its tol; ours exactly, so the agreement is lbfgs's.
    ("ridge positive", Ridge, "Ridge", {"positive": True, "alpha": 10.0, "tol": 1e-12, "max_iter": 100000}, {"positive": True, "alpha": 10.0}, None, {"rtol": 1e-6, "atol": 1e-6}),
    ("ridge positive, no intercept", Ridge, "Ridge", {"positive": True, "alpha": 1.0, "fit_intercept": False, "tol": 1e-12, "max_iter": 100000},
     {"positive": True, "alpha": 1.0, "fit_intercept": False}, None, {"rtol": 1e-6, "atol": 1e-6}),
    ("lasso positive", Lasso, "Lasso", {"positive": True, "alpha": 0.1, "tol": 1e-12, "max_iter": 100000}, None, None, {"rtol": 1e-8, "atol": 1e-8}),
    ("lasso positive, sample weights", Lasso, "Lasso", {"positive": True, "alpha": 0.1, "tol": 1e-12, "max_iter": 100000}, None, swp, {"rtol": 1e-8, "atol": 1e-8}),
    ("elasticnet positive", ElasticNet, "ElasticNet", {"positive": True, "alpha": 0.2, "l1_ratio": 0.4, "tol": 1e-12, "max_iter": 100000}, None, None, {"rtol": 1e-8, "atol": 1e-8}),
    ("elasticnet positive, l1_ratio 0", ElasticNet, "ElasticNet", {"positive": True, "alpha": 0.2, "l1_ratio": 0.0, "tol": 1e-12, "max_iter": 100000}, None, None, {"rtol": 1e-8, "atol": 1e-8}),
]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # ElasticNet with l1_ratio 0 warns that it isn't the lasso
        m = cls(**params).fit(Xp, yp, sample_weight=weights)
    inputs = {"X": tolist(Xp), "y": tolist(yp), "Xnew": tolist(Xpn), "params": ours if ours is not None else params}
    if weights is not None:
        inputs["sampleWeight"] = tolist(weights)
    assert (m.coef_ >= 0).all() and (m.coef_ == 0).any(), name  # the constraint binds somewhere
    case(name, kind, inputs, {"coef": tolist(m.coef_), "intercept": float(m.intercept_), "predict": tolist(m.predict(Xpn)),
                              **({"zero": [bool(v == 0) for v in m.coef_]} if kind in ("Lasso", "ElasticNet") else {})}, tol)

# Two identical columns: many answers; scikit-learn's nnls (Lawson and Hanson's too) gives all to the first.
xt = rng.uniform(0, 3, 50)
Xt = np.column_stack([xt, xt, rng.normal(size=50)])
yt = 2 * xt - 0.5 * Xt[:, 2] + 1 + rng.normal(scale=0.1, size=50)
m = LinearRegression(positive=True).fit(Xt, yt)
assert m.coef_[1] == 0 and m.coef_[0] > 0
case("linear positive, two identical columns", "LinearRegression", {"X": tolist(Xt), "y": tolist(yt), "Xnew": tolist(Xt[:5]), "params": {"positive": True}},
     {"coef": tolist(m.coef_), "intercept": float(m.intercept_), "predict": tolist(m.predict(Xt[:5]))}, {"rtol": 1e-9, "atol": 1e-9})

# MultiOutputRegressor: a regressor per target (appended) --------------------------------------------------------
from sklearn.multioutput import MultiOutputRegressor
Xm = features(80, 4, rng.uniform(0.5, 5, 4), rng.uniform(-3, 3, 4))
Ym = np.column_stack([Xm @ rng.normal(size=4) + 2, Xm @ rng.normal(size=4) - 1, Xm @ rng.normal(size=4)]) + rng.normal(scale=0.3, size=(80, 3))
Xmn = features(8, 4, rng.uniform(0.5, 5, 4), rng.uniform(-3, 3, 4))
swm = rng.uniform(0.2, 2.0, size=80)
for name, cls, kind, params, weights, tol in [
    ("multioutput: linear", LinearRegression, "LinearRegression", {}, None, {"rtol": 1e-9, "atol": 1e-9}),
    ("multioutput: ridge, sample weights", Ridge, "Ridge", {"alpha": 3.0}, swm, {"rtol": 1e-9, "atol": 1e-9}),
    ("multioutput: lasso", Lasso, "Lasso", {"alpha": 0.2, "tol": 1e-12, "max_iter": 100000}, None, {"rtol": 1e-8, "atol": 1e-8}),
    ("multioutput: elasticnet, positive", ElasticNet, "ElasticNet", {"alpha": 0.1, "l1_ratio": 0.5, "positive": True, "tol": 1e-12, "max_iter": 100000}, None, {"rtol": 1e-8, "atol": 1e-8}),
]:
    m = MultiOutputRegressor(cls(**params)).fit(Xm, Ym, sample_weight=weights)
    # For these models a two-dimensional y fits the same: the targets don't interact.
    native = cls(**params).fit(Xm, Ym, sample_weight=weights)
    assert np.allclose(native.coef_, [e.coef_ for e in m.estimators_], rtol=1e-8, atol=1e-10), name
    inputs = {"X": tolist(Xm), "Y": tolist(Ym), "Xnew": tolist(Xmn), "model": kind, "params": params}
    if weights is not None:
        inputs["sampleWeight"] = tolist(weights)
    case(name, "MultiOutputRegressor", inputs,
         {"coef": [tolist(e.coef_) for e in m.estimators_], "intercept": [float(e.intercept_) for e in m.estimators_],
          "predict": tolist(m.predict(Xmn))}, tol)

# A two-dimensional y: a column per target, on the models themselves (appended) ----------------------------------
for name, cls, kind, params, weights, Y_, tol in [
    ("2-D y: linear", LinearRegression, "LinearRegression", {}, None, Ym, {"rtol": 1e-9, "atol": 1e-9}),
    ("2-D y: linear, one column", LinearRegression, "LinearRegression", {"fit_intercept": False}, None, Ym[:, :1], {"rtol": 1e-9, "atol": 1e-9}),
    ("2-D y: ridge, sample weights", Ridge, "Ridge", {"alpha": 3.0}, swm, Ym, {"rtol": 1e-9, "atol": 1e-9}),
    ("2-D y: linear positive", LinearRegression, "LinearRegression", {"positive": True}, None, Ym, {"rtol": 1e-9, "atol": 1e-9}),
    ("2-D y: lasso", Lasso, "Lasso", {"alpha": 0.2, "tol": 1e-12, "max_iter": 100000}, None, Ym, {"rtol": 1e-8, "atol": 1e-8}),
    ("2-D y: elasticnet, sample weights", ElasticNet, "ElasticNet", {"alpha": 0.1, "l1_ratio": 0.3, "tol": 1e-12, "max_iter": 100000}, swm, Ym, {"rtol": 1e-8, "atol": 1e-8}),
]:
    m = cls(**params).fit(Xm, Y_, sample_weight=weights)
    assert m.coef_.ndim == 2 and m.coef_.shape[0] == Y_.shape[1], name
    inputs = {"X": tolist(Xm), "Y": tolist(Y_), "Xnew": tolist(Xmn), "model": kind, "params": params}
    if weights is not None:
        inputs["sampleWeight"] = tolist(weights)
    case(name, "2-D y", inputs, {"coefs": tolist(m.coef_), "intercepts": tolist(np.atleast_1d(m.intercept_)), "predict": tolist(m.predict(Xmn))}, tol)

# Infrequent categories in the encoders (appended) -----------------------------------------------------------------
pets = rng.choice(["cat", "dog", "eel", "emu", "fox", "gnu", "owl", None], size=60, p=[0.3, 0.25, 0.05, 0.05, 0.15, 0.05, 0.1, 0.05])
legs = rng.choice([0.0, 2.0, 4.0, 6.0, 8.0], size=60, p=[0.1, 0.3, 0.45, 0.1, 0.05])
Xf = np.empty((60, 2), dtype=object)
Xf[:, 0] = pets
Xf[:, 1] = legs
# OrdinalEncoder keeps a missing value NaN (smaltml's text null): written as NaN, scikit-learn treats it the same.
Xfo = Xf.copy()
Xfo[:, 0] = [np.nan if v is None else v for v in Xf[:, 0]]
seen = np.array([["cat", 4.0], ["eel", 8.0], ["owl", 0.0], [None, 2.0], ["gnu", 6.0]], dtype=object)
unseen = np.array([["yak", 4.0], ["cat", 3.0], ["emu", 8.0]], dtype=object)
def infrequent_list(e):
    return [None if c is None else [None if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in c.tolist()] for c in e.infrequent_categories_]
for params, Xnew in [
    ({"min_frequency": 5}, seen),
    ({"min_frequency": 0.1}, seen),
    ({"max_categories": 3}, seen),
    ({"max_categories": 1}, seen),
    ({"min_frequency": 2, "max_categories": 4}, seen),
    ({"min_frequency": 5, "handle_unknown": "infrequent_if_exist"}, unseen),
    ({"max_categories": 3, "handle_unknown": "ignore"}, unseen),
    ({"min_frequency": 5, "drop": "first"}, seen),
    ({"max_categories": 2, "drop": "if_binary"}, seen),
]:
    e = OneHotEncoder(sparse_output=False, **params).fit(Xf)
    case(f"one-hot infrequent {params}", "encoder-infrequent", {"encoder": "OneHotEncoder", "X": tolist(Xf), "Xnew": tolist(Xnew), "params": params},
         {"transform": tolist(e.transform(Xnew)), "names": [n.replace(".0_", "_").replace(".0", "") for n in e.get_feature_names_out().tolist()],
          "infrequent": infrequent_list(e), **({"dropIdx": [int(d) for d in e.drop_idx_]} if params.get("drop") else {})}, {"rtol": 0, "atol": 0})
seen_o = np.array([["cat", 4.0], ["eel", 8.0], ["owl", 0.0], [np.nan, 2.0], ["gnu", 6.0]], dtype=object)
for params, Xnew in [
    ({"min_frequency": 5}, seen_o),
    ({"max_categories": 3}, seen_o),
    ({"min_frequency": 0.1, "handle_unknown": "use_encoded_value", "unknown_value": -1}, np.array([["yak", 4.0], ["cat", 3.0]], dtype=object)),
]:
    e = OrdinalEncoder(**params).fit(Xfo)
    case(f"ordinal infrequent {params}", "encoder-infrequent", {"encoder": "OrdinalEncoder", "X": tolist(Xfo), "Xnew": tolist(Xnew), "params": params},
         {"transform": tolist(e.transform(Xnew).astype(float)), "infrequent": infrequent_list(e)}, {"rtol": 0, "atol": 0})

# GroupKFold and groups in cross-validation and grid search (appended) -------------------------------------------
from sklearn.model_selection import GroupKFold
for name, groups, k in [
    ("folds: GroupKFold, ties", [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4], 3),
    ("folds: GroupKFold, text groups", ["b", "b", "a", "a", "c", "c", "d", "d", "e"], 3),
    ("folds: GroupKFold, 40 groups", rng.integers(0, 40, size=200).tolist(), 5),
    ("folds: GroupKFold, 2 folds", rng.integers(0, 7, size=50).tolist(), 2),
]:
    folds = [tolist(test) for _, test in GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups)]
    case(name, "groupkfold", {"groups": groups, "params": {"n_splits": k}}, {"tests": folds}, {"rtol": 0, "atol": 0})
gg = rng.integers(0, 12, size=80).tolist()
s = cross_val_score(make_pipeline(StandardScaler(), Ridge(alpha=2.0)), Xg, yg, cv=GroupKFold(n_splits=4), groups=gg)
case("cross-validation: scaled ridge, GroupKFold", "cvscore-groups", {"X": tolist(Xg), "y": tolist(yg), "groups": gg, "model": "scaled-ridge"},
     {"scores": tolist(s)}, {"rtol": 1e-9, "atol": 1e-9})
g = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.01, 1.0, 30.0]}, cv=GroupKFold(n_splits=4)).fit(Xg, yg, groups=gg)
case("grid search: scaled ridge, GroupKFold", "grid-groups", {"X": tolist(Xg), "y": tolist(yg), "groups": gg, "model": "scaled-ridge", "grid": {"ridge__alpha": [0.01, 1.0, 30.0]}},
     {"mean": tolist(g.cv_results_["mean_test_score"]), "rank": tolist(g.cv_results_["rank_test_score"]), "best": g.best_params_}, {"rtol": 1e-9, "atol": 1e-9})

# Several scorers, train scores and refit by name (appended) --------------------------------------------------------
from sklearn.model_selection import cross_validate
for name, model, X_, y_, scoring in [
    ("cross_validate: scaled ridge, two scorers, train scores", "scaled-ridge", Xg, yg, ["r2", "neg_mean_absolute_error"]),
    ("cross_validate: logistic, named scorers, train scores", "logistic", Xc, yc, {"acc": "accuracy", "loss": "neg_log_loss"}),
]:
    est = make_pipeline(StandardScaler(), Ridge(alpha=2.0)) if model == "scaled-ridge" else LogisticRegression(tol=1e-10, max_iter=10000)
    r = cross_validate(est, X_, y_, cv=KFold(n_splits=5), scoring=scoring, return_train_score=True)
    keys = list(scoring) if isinstance(scoring, list) else list(scoring.keys())
    case(name, "cross-validate", {"X": tolist(X_), "y": tolist(y_), "model": model, "scoring": scoring},
         {"test": {k: tolist(r[f"test_{k}"]) for k in keys}, "train": {k: tolist(r[f"train_{k}"]) for k in keys}},
         {"rtol": 1e-9, "atol": 1e-9} if model == "scaled-ridge" else {"rtol": 1e-5, "atol": 1e-6})
grid = {"ridge__alpha": [0.01, 1.0, 30.0, 300.0]}
g = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), grid, cv=KFold(n_splits=5), scoring=["r2", "neg_mean_squared_error"],
                 refit="neg_mean_squared_error", return_train_score=True).fit(Xg, yg)
res = {k: {"meanTest": tolist(g.cv_results_[f"mean_test_{k}"]), "rankTest": tolist(g.cv_results_[f"rank_test_{k}"]),
           "meanTrain": tolist(g.cv_results_[f"mean_train_{k}"]), "stdTrain": tolist(g.cv_results_[f"std_train_{k}"])} for k in ["r2", "neg_mean_squared_error"]}
case("grid search: two scorers, refit by name, train scores", "grid-multi", {"X": tolist(Xg), "y": tolist(yg), "model": "scaled-ridge", "grid": grid,
     "scoring": ["r2", "neg_mean_squared_error"], "refit": "neg_mean_squared_error"},
     {"scores": res, "best": g.best_params_, "bestScore": float(g.best_score_), "bestIndex": int(g.best_index_)}, {"rtol": 1e-9, "atol": 1e-9})
g = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), grid, cv=KFold(n_splits=5), return_train_score=True).fit(Xg, yg)
case("grid search: one scorer, train scores", "grid-train", {"X": tolist(Xg), "y": tolist(yg), "model": "scaled-ridge", "grid": grid},
     {"meanTrain": tolist(g.cv_results_["mean_train_score"]), "stdTrain": tolist(g.cv_results_["std_train_score"]), "best": g.best_params_}, {"rtol": 1e-9, "atol": 1e-9})

# Decision trees (appended): the whole tree, node for node, on data without tied splits ------------------------------
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
def tree_expected(m, Xnew, classifier):
    t = m.tree_
    out = {"left": tolist(t.children_left), "right": tolist(t.children_right), "feature": tolist(t.feature),
           "threshold": tolist(t.threshold), "impurity": tolist(t.impurity), "nSamples": tolist(t.n_node_samples),
           "weightedNSamples": tolist(t.weighted_n_node_samples), "value": tolist(t.value[:, 0, :].ravel()),
           "importances": tolist(m.feature_importances_), "depth": int(m.get_depth()), "leaves": int(m.get_n_leaves()),
           "predict": tolist(m.predict(Xnew)), "apply": tolist(m.apply(Xnew))}
    if classifier:
        out["proba"] = tolist(m.predict_proba(Xnew))
        out["classes"] = tolist(m.classes_)
    return out
Xt = rng.normal(size=(80, 4)) * [1.0, 3.0, 0.5, 2.0]
yt = 2 * Xt[:, 0] - Xt[:, 1] + np.sin(3 * Xt[:, 2]) + rng.normal(scale=0.3, size=80)
Xtn = rng.normal(size=(15, 4)) * [1.0, 3.0, 0.5, 2.0]
swt = rng.uniform(0.3, 2.0, size=80)
for name, params, weights in [
    ("tree regressor: leaves of 3 or more", {"min_samples_leaf": 3}, None),
    ("tree regressor: depth 3", {"max_depth": 3, "min_samples_leaf": 2}, None),
    ("tree regressor: min_samples_split a share", {"min_samples_split": 0.2, "min_samples_leaf": 3}, None),
    ("tree regressor: 6 leaves, best first", {"max_leaf_nodes": 6}, None),
    ("tree regressor: min_impurity_decrease", {"min_impurity_decrease": 0.05, "min_samples_leaf": 3}, None),
    ("tree regressor: friedman_mse", {"criterion": "friedman_mse", "max_depth": 4, "min_samples_leaf": 3}, None),
    ("tree regressor: friedman_mse, min_impurity_decrease", {"criterion": "friedman_mse", "min_impurity_decrease": 0.1, "min_samples_leaf": 3}, None),
    ("tree regressor: sample weights", {"max_depth": 4, "min_samples_leaf": 3}, swt),
    ("tree regressor: min_weight_fraction_leaf", {"min_weight_fraction_leaf": 0.08}, swt),
]:
    m = DecisionTreeRegressor(random_state=0, **params).fit(Xt, yt, sample_weight=weights)
    inputs = {"X": tolist(Xt), "y": tolist(yt), "Xnew": tolist(Xtn), "params": params}
    if weights is not None:
        inputs["sampleWeight"] = tolist(weights)
    case(name, "DecisionTreeRegressor", inputs, tree_expected(m, Xtn, False), {"rtol": 1e-10, "atol": 1e-10})
yc3 = np.where(Xt[:, 0] + 0.5 * Xt[:, 1] + rng.normal(scale=0.7, size=80) > 0.8, "high", np.where(Xt[:, 2] > 0, "mid", "low"))
for name, params, weights, labels in [
    # Small nodes can tie: two columns that split the same rows the same way. Larger leaves keep these cases clear.
    ("tree classifier: gini, three classes", {"max_depth": 3, "min_samples_leaf": 6}, None, yc3),
    ("tree classifier: entropy", {"criterion": "entropy", "max_depth": 3, "min_samples_leaf": 6}, None, yc3),
    ("tree classifier: log_loss, 5 leaves best first", {"criterion": "log_loss", "max_leaf_nodes": 5, "min_samples_leaf": 3}, None, yc3),
    ("tree classifier: balanced, with sample weights", {"class_weight": "balanced", "max_depth": 3, "min_samples_leaf": 4}, swt, yc3),
    ("tree classifier: class weights by label", {"class_weight": {"high": 3.0}, "max_depth": 3, "min_samples_leaf": 4}, None, yc3),
    ("tree classifier: two classes, numbers", {"max_depth": 3, "min_samples_leaf": 5}, None, (Xt[:, 0] > 0).astype(int)),
]:
    m = DecisionTreeClassifier(random_state=0, **params).fit(Xt, labels, sample_weight=weights)
    inputs = {"X": tolist(Xt), "y": tolist(labels), "Xnew": tolist(Xtn), "params": params}
    if weights is not None:
        inputs["sampleWeight"] = tolist(weights)
    case(name, "DecisionTreeClassifier", inputs, tree_expected(m, Xtn, True), {"rtol": 1e-10, "atol": 1e-10})

# Rows of weight 0 take no part in a tree: not in a node's sample count, not a threshold's neighbour.
swz = np.where(np.arange(80) % 4 == 0, 0.0, swt)
for name, params in [
    ("tree regressor: rows of weight 0, leaves of 5 or more", {"min_samples_leaf": 5}),
    ("tree regressor: rows of weight 0, min_samples_split a share", {"min_samples_split": 0.25, "min_samples_leaf": 2}),
]:
    m = DecisionTreeRegressor(random_state=0, **params).fit(Xt, yt, sample_weight=swz)
    case(name, "DecisionTreeRegressor", {"X": tolist(Xt), "y": tolist(yt), "Xnew": tolist(Xtn), "params": params, "sampleWeight": tolist(swz)},
         tree_expected(m, Xtn, False), {"rtol": 1e-10, "atol": 1e-10})
params = {"max_depth": 3, "min_samples_leaf": 5}
m = DecisionTreeClassifier(random_state=0, **params).fit(Xt, yc3, sample_weight=swz)
case("tree classifier: rows of weight 0", "DecisionTreeClassifier", {"X": tolist(Xt), "y": tolist(yc3), "Xnew": tolist(Xtn), "params": params, "sampleWeight": tolist(swz)},
     tree_expected(m, Xtn, True), {"rtol": 1e-10, "atol": 1e-10})

# Random forests (appended, ADR 0039). Exact where the forest is determined: no bootstrap and every column, so each
# tree is the single tree. Otherwise smaltml draws its own samples, so the cases compare the mean score over five seeds
# on held-out rows, and the out-of-bag score, within a margin.
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
# Leaves large enough that no two columns split a node's rows the same way: with such a tie, which column wins
# depends on the seed, in scikit-learn as in smaltml.
fixed = {"bootstrap": False, "max_features": None, "n_estimators": 4}
for name, params in [
    ("forest regressor: no bootstrap, every column", {**fixed, "min_samples_leaf": 5}),
    ("forest regressor: no bootstrap, sample weights", {**fixed, "max_depth": 4, "min_samples_leaf": 3}),
]:
    weights = swt if "weights" in name else None
    m = RandomForestRegressor(random_state=0, **params).fit(Xt, yt, sample_weight=weights)
    inputs = {"X": tolist(Xt), "y": tolist(yt), "Xnew": tolist(Xtn), "params": params}
    if weights is not None:
        inputs["sampleWeight"] = tolist(weights)
    case(name, "forest-exact", inputs, {"predict": tolist(m.predict(Xtn)), "importances": tolist(m.feature_importances_)}, {"rtol": 1e-10, "atol": 1e-10})
for name, params in [
    ("forest classifier: no bootstrap, every column", {**fixed, "max_depth": 3, "min_samples_leaf": 6}),
    ("forest classifier: no bootstrap, balanced", {**fixed, "class_weight": "balanced", "max_depth": 3, "min_samples_leaf": 6}),
]:
    m = RandomForestClassifier(random_state=0, **params).fit(Xt, yc3)
    case(name, "forest-exact", {"X": tolist(Xt), "y": tolist(yc3), "Xnew": tolist(Xtn), "params": params, "classifier": True},
         {"predict": tolist(m.predict(Xtn)), "proba": tolist(m.predict_proba(Xtn)), "classes": tolist(m.classes_),
          "importances": tolist(m.feature_importances_)}, {"rtol": 1e-10, "atol": 1e-10})

def forest_data(n, seed):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 6)) * [1.0, 2.0, 0.5, 1.5, 1.0, 1.0]
    f = 2 * X[:, 0] - X[:, 1] + np.sin(3 * X[:, 2]) + X[:, 3] * X[:, 4]
    y = f + r.normal(scale=0.5, size=n)
    labels = np.where(f > 1.0, "high", np.where(f > -1.0, "mid", "low"))
    return X, y, labels
# The margins: 0.02, and 0.005 for the 2000-row regressor, whose scores vary least. Each is at least three standard
# errors of the difference of the two means, from scikit-learn's seed-to-seed spread on this data (measured over 30
# seeds, 2026-11-18), with 20 seeds on the small data, whose scores vary most, and 5 on the large.
X, y, lab = forest_data(120 + 500, 11)
sw_forest = np.random.default_rng(13).uniform(0.2, 3.0, size=120)
small = (X[:120], X[120:], y[:120], y[120:], lab[:120], lab[120:])
X, y, lab = forest_data(2000 + 4000, 12)
large = (X[:2000], X[2000:], y[:2000], y[2000:], lab[:2000], lab[2000:])
for name, data, params, classifier, weights, margin, seeds in [
    ("forest regressor, small: defaults, scored", small, {"n_estimators": 50, "oob_score": True}, False, None, 0.02, list(range(20))),
    ("forest regressor, small: sample weights", small, {"n_estimators": 50, "oob_score": True}, False, sw_forest, 0.02, list(range(20))),
    ("forest classifier, small: defaults, scored", small, {"n_estimators": 50, "oob_score": True}, True, None, 0.02, list(range(20))),
    ("forest classifier, small: entropy, balanced, depth 6", small, {"n_estimators": 50, "criterion": "entropy", "class_weight": "balanced", "max_depth": 6, "oob_score": True}, True, None, 0.02, list(range(20))),
    ("forest classifier, small: class weights by label, sample weights", small, {"n_estimators": 50, "class_weight": {"mid": 3.0}, "max_samples": 80, "oob_score": True}, True, sw_forest, 0.02, list(range(20))),
    ("forest regressor, 2000 rows: defaults, scored", large, {"oob_score": True}, False, None, 0.005, [0, 1, 2, 3, 4]),
    ("forest regressor, 2000 rows: a third of the columns, half the rows", large, {"n_estimators": 50, "max_features": 0.34, "max_samples": 0.5, "min_samples_leaf": 2, "oob_score": True}, False, None, 0.02, [0, 1, 2, 3, 4]),
    ("forest classifier, 2000 rows: defaults, scored", large, {"n_estimators": 50, "oob_score": True}, True, None, 0.02, [0, 1, 2, 3, 4]),
]:
    Xa, Xb, ya, yb, la, lb = data
    test_scores, oob_scores = [], []
    for s in seeds:
        if classifier:
            m = RandomForestClassifier(random_state=s, **params).fit(Xa, la, sample_weight=weights)
            test_scores.append(accuracy_score(lb, m.predict(Xb)))
        else:
            m = RandomForestRegressor(random_state=s, **params).fit(Xa, ya, sample_weight=weights)
            test_scores.append(r2_score(yb, m.predict(Xb)))
        oob_scores.append(m.oob_score_)
    inputs = {"X": tolist(Xa), "y": tolist(la if classifier else ya), "Xnew": tolist(Xb), "ynew": tolist(lb if classifier else yb),
              "params": params, "classifier": classifier, "seeds": seeds}
    if weights is not None:
        inputs["sampleWeight"] = tolist(weights)
    case(name, "forest-score", inputs, {"test": float(np.mean(test_scores)), "oob": float(np.mean(oob_scores))}, {"rtol": 0, "atol": margin})

# Extra trees, and single trees with splitter="random" (appended, ADR 0039): random thresholds, so only the scores
# compare, as the random forests' do above: the mean over the seeds, on held-out rows, within a margin. Each margin is
# at least three standard errors of the difference of the two means, from scikit-learn's seed-to-seed spread on this
# data (measured over 30 seeds, 2026-11-19): 40 seeds where an out-of-bag score varies most, 50 for a single tree.
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
for name, model, data, params, classifier, weights, margin, seeds in [
    ("extra trees regressor, small: defaults", "extra", small, {"n_estimators": 50}, False, None, 0.02, list(range(20))),
    ("extra trees regressor, small: sample weights", "extra", small, {"n_estimators": 50}, False, sw_forest, 0.02, list(range(20))),
    ("extra trees regressor, small: a third of the columns", "extra", small, {"n_estimators": 50, "max_features": 0.34}, False, None, 0.02, list(range(20))),
    ("extra trees regressor, small: bootstrap, scored", "extra", small, {"n_estimators": 50, "bootstrap": True, "oob_score": True}, False, None, 0.02, list(range(40))),
    ("extra trees classifier, small: defaults", "extra", small, {"n_estimators": 50}, True, None, 0.02, list(range(20))),
    ("extra trees classifier, small: entropy, balanced, depth 6", "extra", small, {"n_estimators": 50, "criterion": "entropy", "class_weight": "balanced", "max_depth": 6}, True, None, 0.02, list(range(20))),
    ("extra trees classifier, small: bootstrap, scored, sample weights", "extra", small, {"n_estimators": 50, "bootstrap": True, "oob_score": True}, True, sw_forest, 0.02, list(range(40))),
    ("extra trees regressor, 2000 rows: defaults", "extra", large, {}, False, None, 0.005, [0, 1, 2, 3, 4]),
    ("extra trees classifier, 2000 rows: defaults", "extra", large, {"n_estimators": 50}, True, None, 0.02, [0, 1, 2, 3, 4]),
    ("tree regressor, 2000 rows: random splitter, depth 8", "tree", large, {"splitter": "random", "max_depth": 8}, False, None, 0.02, list(range(50))),
    ("tree classifier, small: random splitter, depth 5", "tree", small, {"splitter": "random", "max_depth": 5}, True, None, 0.02, list(range(50))),
]:
    Xa, Xb, ya, yb, la, lb = data
    test_scores, oob_scores = [], []
    for s in seeds:
        if classifier:
            M = ExtraTreesClassifier if model == "extra" else DecisionTreeClassifier
            m = M(random_state=s, **params).fit(Xa, la, sample_weight=weights)
            test_scores.append(accuracy_score(lb, m.predict(Xb)))
        else:
            M = ExtraTreesRegressor if model == "extra" else DecisionTreeRegressor
            m = M(random_state=s, **params).fit(Xa, ya, sample_weight=weights)
            test_scores.append(r2_score(yb, m.predict(Xb)))
        if params.get("oob_score"):
            oob_scores.append(m.oob_score_)
    inputs = {"X": tolist(Xa), "y": tolist(la if classifier else ya), "Xnew": tolist(Xb), "ynew": tolist(lb if classifier else yb),
              "params": params, "classifier": classifier, "seeds": seeds, "model": model}
    if weights is not None:
        inputs["sampleWeight"] = tolist(weights)
    case(name, "forest-score", inputs, {"test": float(np.mean(test_scores)), "oob": float(np.mean(oob_scores)) if oob_scores else None}, {"rtol": 0, "atol": margin})

out = {
    "comment": "Written by compare/run.py (npm run compare); don't edit by hand.",
    "reference": {"scikit-learn": sklearn.__version__, "numpy": np.__version__},
    "cases": cases,
}
path = ROOT / "fixtures" / "compare.json"
path.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False) + "\n")
print(f"{len(cases)} cases written to {path.relative_to(ROOT)} (scikit-learn {sklearn.__version__})")
