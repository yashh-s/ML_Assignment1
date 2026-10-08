"""
lasso_cv.py - sparse polynomial regression (L1-regularised least squares on the
same polynomial features), plus "relaxed lasso": use the lasso only to pick WHICH
polynomial terms matter, then refit those terms by (almost) plain least squares.

usage: python cv_lasso.py --var 1 --degrees 3-10
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path

from polyreg import MinMaxScaler11, design_matrix, exponents, kfold_indices, mse
from cv_ridge import load, shift_weights

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
ALPHAS = np.logspace(-1.5, -4.5, 13)       # lasso strengths (decreasing for warm starts)
RELAX_LAM = 1e-6                           # tiny ridge for the relaxed refit (numerical safety)


def fit_lasso_path(P, y, alphas):
    """Returns intercepts (n_alpha,) and coefs (n_alpha, p) on uncentred P."""
    mu_x, mu_y = P.mean(0), y.mean()
    _, coefs, _ = lasso_path(P - mu_x, y - mu_y, alphas=alphas, max_iter=5000, tol=1e-4)
    coefs = coefs.T
    b = mu_y - coefs @ mu_x
    return b, coefs


def relaxed_refit(P, y, support, lam=RELAX_LAM):
    if support.sum() == 0:
        return y.mean(), np.zeros(P.shape[1])
    S = P[:, support]
    mu_x, mu_y = S.mean(0), y.mean()
    A = S - mu_x
    w_s = np.linalg.solve(A.T @ A + lam * len(y) * np.eye(A.shape[1]), A.T @ (y - mu_y))
    w = np.zeros(P.shape[1]); w[support] = w_s
    return mu_y - w_s @ mu_x, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", type=int, required=True)
    ap.add_argument("--degrees", required=True)          # e.g. 3-10
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alphas", default="", help="lo,hi,n  (log10 range), e.g. -3,-1.7,14")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    global ALPHAS
    if args.alphas:
        a_lo, a_hi, a_n = args.alphas.split(",")
        ALPHAS = np.logspace(float(a_hi), float(a_lo), int(a_n))
    lo, hi = map(int, args.degrees.split("-"))
    X, y, Xte, _ = load(args.var)
    wts, _, _ = shift_weights(X, Xte)
    n = len(y)
    rows = []
    t0 = time.time()
    for seed in map(int, args.seeds.split(",")):
        for fi, va in enumerate(kfold_indices(n, args.folds, seed)):
            tr = np.setdiff1d(np.arange(n), va)
            sc = MinMaxScaler11().fit(X[tr])
            zt, zv = sc.transform(X[tr]), sc.transform(X[va])
            for D in range(lo, hi + 1):
                E = exponents(X.shape[1], D)
                Pt, Pv = design_matrix(zt, E, "monomial"), design_matrix(zv, E, "monomial")
                b, W = fit_lasso_path(Pt, y[tr], ALPHAS)
                for ai, a in enumerate(ALPHAS):
                    for kind in ("lasso", "relaxed"):
                        if kind == "lasso":
                            bb, ww = b[ai], W[ai]
                        else:
                            sup = W[ai] != 0
                            if sup.sum() >= 0.8 * len(tr):   # refit would be ill-posed
                                continue
                            bb, ww = relaxed_refit(Pt, y[tr], sup)
                        r = y[va] - (bb + Pv @ ww)
                        rows.append(dict(seed=seed, fold=fi, degree=D, alpha=a, kind=kind,
                                         nnz=int((W[ai] != 0).sum()),
                                         val_mse=float(np.mean(r ** 2)),
                                         val_wmse=float(np.average(r ** 2, weights=wts[va]))))
            print(f"seed {seed} fold {fi} done  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    s = df.groupby(["kind", "degree", "alpha"]).agg(
        val_mse=("val_mse", "mean"), val_wmse=("val_wmse", "mean"),
        val_wstd=("val_wmse", "std"), nnz=("nnz", "mean"), n=("val_mse", "size")).reset_index()
    s["val_wse"] = s.val_wstd / np.sqrt(s.n)
    s = s[s.n == s.n.max()]
    s.to_csv(ROOT / "outputs" / f"lasso_cv_var{args.var}{args.tag}.csv", index=False)
    print(s.sort_values("val_wmse").head(15).to_string())


if __name__ == "__main__":
    main()
