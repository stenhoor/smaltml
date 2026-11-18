#!/usr/bin/env python3
"""Times scikit-learn on the benchmark data that bench/smaltml.mjs makes (run it first): the same cases, the same
parameters, the median of several runs after a warm-up. Writes bench/results/scikit-learn-<threads>.json.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 bench/reference.py   # one thread
    python3 bench/reference.py                                            # every core
"""
import json, os, platform, time, warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from threadpoolctl import threadpool_info

warnings.simplefilter("ignore", ConvergenceWarning)  # both libraries stop at the same max_iter on these data
here = Path(__file__).resolve().parent
data = here / "data"

def median(f, reps):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))

results = {}
for s in json.loads((data / "meta.json").read_text()):
    n, p, key = s["n"], s["p"], s["key"]
    X = np.fromfile(data / f"{key}.X").reshape((n, p), order="F")
    y, y2, y3 = (np.fromfile(data / f"{key}.{k}") for k in ("y", "y2", "y3"))
    reps = 7 if n < 100_000 else 3
    lr = LinearRegression().fit(X, y)
    l3 = LogisticRegression().fit(X, y3)
    r = {
        "StandardScaler fit + transform": median(lambda: StandardScaler().fit_transform(X), reps),
        "LinearRegression fit": median(lambda: LinearRegression().fit(X, y), reps),
        "LinearRegression predict": median(lambda: lr.predict(X), reps),
        "Ridge fit": median(lambda: Ridge(alpha=1.0).fit(X, y), reps),
        "Lasso fit (alpha 0.1)": median(lambda: Lasso(alpha=0.1).fit(X, y), reps),
        "LogisticRegression fit, 2 classes": median(lambda: LogisticRegression().fit(X, y2), reps),
        "LogisticRegression fit, 2 classes, newton-cholesky": median(lambda: LogisticRegression(solver="newton-cholesky").fit(X, y2), reps),
        "LogisticRegression fit, 3 classes": median(lambda: LogisticRegression().fit(X, y3), reps),
        "LogisticRegression predictProba, 3 classes": median(lambda: l3.predict_proba(X), reps),
        # Trees are single-threaded in scikit-learn; their fits don't use BLAS.
        "DecisionTreeRegressor fit, depth 10": median(lambda: DecisionTreeRegressor(max_depth=10, random_state=0).fit(X, y), reps),
        "DecisionTreeClassifier fit, 3 classes, depth 10": median(lambda: DecisionTreeClassifier(max_depth=10, random_state=0).fit(X, y3), reps),
        # Forests on one thread: n_jobs's default.
        "RandomForestRegressor fit, 10 trees, depth 10": median(lambda: RandomForestRegressor(n_estimators=10, max_depth=10, random_state=0).fit(X, y), reps),
        "RandomForestClassifier fit, 3 classes, 10 trees, depth 10": median(lambda: RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0).fit(X, y3), reps),
    }
    r["iterations"] = {"lasso": int(Lasso(alpha=0.1).fit(X, y).n_iter_), "logistic2": int(LogisticRegression().fit(X, y2).n_iter_[0]), "logistic3": int(l3.n_iter_[0])}
    results[s["name"]] = r

threads = max(i["num_threads"] for i in threadpool_info()) if threadpool_info() else 1
out = {
    "library": f"scikit-learn {sklearn.__version__}, NumPy {np.__version__}",
    "runtime": f"Python {platform.python_version()}",
    "threads": threads,
    "blas": ", ".join(sorted({f'{i["internal_api"]} {i["version"]}' if i.get("version") else i["internal_api"] for i in threadpool_info()})),
    "cpu": f"{os.cpu_count()} cores",
    "date": time.strftime("%Y-%m-%d"),
    "milliseconds": results,
}
(here / "results").mkdir(exist_ok=True)
(here / "results" / f"scikit-learn-{threads}.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
