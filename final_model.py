"""
final_model.py - train the selected model for each problem on ALL training data
and write the submission files  outputs/BT2024013_pred_var1.csv / _var2.csv.

Hyper-parameters below were chosen by cross-validation (see cv_ridge.py,
cv_lasso.py and the tables they write into outputs/):

  var1 : degree 5, Lasso (L1),  alpha = 10^-2.5   -> CV MSE ~0.31 (ridge best was ~0.43)
  var2 : degree 11, Ridge (L2), lambda = 0.1      -> CV MSE ~0.25 (lasso ~0.26, same noise floor)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from cv_ridge import load, shift_weights
from polyreg import (MinMaxScaler11, PolynomialRegressor, design_matrix, exponents,
                     kfold_indices, mse, n_terms, r2)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
ROLL = "BT2024013"

CONFIG = {
    1: dict(method="lasso", degree=5, alpha=10 ** -2.5),
    2: dict(method="ridge", degree=11, lam=0.1),
}


class PolynomialLasso:
    """Monomial features of total degree <= D on inputs scaled to [-1, 1],
    weights from  min (1/2n)||y - Phi w - b||^2 + alpha * ||w||_1 ."""

    def __init__(self, degree, alpha):
        self.degree, self.alpha = degree, alpha

    def fit(self, X, y):
        self.scaler = MinMaxScaler11().fit(X)
        self.exps = exponents(X.shape[1], self.degree)
        Phi = design_matrix(self.scaler.transform(X), self.exps, "monomial")
        self.model = Lasso(alpha=self.alpha, max_iter=50000, tol=1e-6).fit(Phi, y)
        self.w, self.b = self.model.coef_, self.model.intercept_
        return self

    def predict(self, X):
        return self.model.predict(design_matrix(self.scaler.transform(X), self.exps, "monomial"))


def make(cfg):
    if cfg["method"] == "lasso":
        return PolynomialLasso(cfg["degree"], cfg["alpha"])
    return PolynomialRegressor(cfg["degree"], cfg["lam"], "monomial")


def term_name(e, feats):
    parts = [f"{f}^{p}" if p > 1 else f for f, p in zip(feats, e) if p > 0]
    return "*".join(parts) if parts else "1"


def main():
    out = ROOT / "outputs"
    for var, cfg in CONFIG.items():
        X, y, Xte, feats = load(var)
        wts, _, _ = shift_weights(X, Xte)

        # honest out-of-fold estimate (5-fold, 3 repeats) of the final configuration
        cv, cvw, oof0 = [], [], np.empty_like(y)
        for seed in (0, 1, 2):
            for va in kfold_indices(len(y), 5, seed):
                tr = np.setdiff1d(np.arange(len(y)), va)
                p = make(cfg).fit(X[tr], y[tr]).predict(X[va])
                cv.append(mse(y[va], p))
                cvw.append(float(np.average((y[va] - p) ** 2, weights=wts[va])))
                if seed == 0:
                    oof0[va] = p
        np.save(out / f"oof_final_var{var}.npy", oof0)

        model = make(cfg).fit(X, y)
        p_tr, p_te = model.predict(X), model.predict(Xte)
        sub = pd.DataFrame({"y": p_te})
        sub.to_csv(out / f"{ROLL}_pred_var{var}.csv", index=False)

        nnz = int(np.sum(np.abs(model.w) > 0))
        top = np.argsort(-np.abs(model.w))[:8]
        info = dict(var=var, **cfg, n_terms=n_terms(X.shape[1], cfg["degree"]),
                    nonzero_terms=nnz + 1,
                    cv_mse=float(np.mean(cv)), cv_se=float(np.std(cv, ddof=1) / np.sqrt(len(cv))),
                    cv_wmse=float(np.mean(cvw)), cv_wse=float(np.std(cvw, ddof=1) / np.sqrt(len(cvw))),
                    cv_r2=float(1 - np.mean(cv) / y.var()),
                    train_mse=mse(y, p_tr), train_r2=r2(y, p_tr),
                    y_train_mean=float(y.mean()), y_train_std=float(y.std()),
                    pred_test_mean=float(p_te.mean()), pred_test_std=float(p_te.std()),
                    pred_test_min=float(p_te.min()), pred_test_max=float(p_te.max()),
                    largest_terms=[(term_name(model.exps[i], feats), round(float(model.w[i]), 4)) for i in top])
        (out / f"summary_final_var{var}.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
