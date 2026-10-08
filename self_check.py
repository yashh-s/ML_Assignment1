"""
self_check.py - quick sanity test you can run yourself (takes a few seconds).

The real test labels are hidden, so we imitate the grading:
hide 20% of the TRAINING rows, fit on the other 80%, and score on the hidden 20%
with the same metrics the grader uses (MSE and R^2).

It also scores two simpler models so you can see why the final choices were made.
"""
import warnings

import numpy as np

from cv_ridge import load
from final_model import CONFIG, PolynomialLasso, make
from polyreg import PolynomialRegressor, mse, r2

warnings.filterwarnings("ignore")


def holdout(var, seed=42, frac=0.2):
    X, y, _, _ = load(var)
    idx = np.random.default_rng(seed).permutation(len(y))
    n_val = int(frac * len(y))
    va, tr = idx[:n_val], idx[n_val:]
    return X[tr], y[tr], X[va], y[va]


def score(model, Xtr, ytr, Xva, yva):
    p = model.fit(Xtr, ytr).predict(Xva)
    return mse(yva, p), r2(yva, p)


if __name__ == "__main__":
    for var in (1, 2):
        cfg = CONFIG[var]
        Xtr, ytr, Xva, yva = holdout(var)
        D = cfg["degree"]
        rows = [
            ("degree 1 (straight line)", PolynomialRegressor(1, 0.0)),
            (f"degree {D}, plain least squares", PolynomialRegressor(D, 0.0)),
            (f"degree {D}, {cfg['method']}  <- FINAL", make(cfg)),
        ]
        print(f"\n=== var{var}: train on {len(ytr)} rows, score on {len(yva)} hidden rows ===")
        print(f"{'model':40s} {'MSE':>10s} {'R^2':>8s}")
        for name, m in rows:
            e, r = score(m, Xtr, ytr, Xva, yva)
            print(f"{name:40s} {e:10.3f} {r:8.4f}")
