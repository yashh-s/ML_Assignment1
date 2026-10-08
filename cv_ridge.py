"""
cv_ridge.py - model selection for polynomial + ridge (L2) regression.

usage:  python cv_ridge.py --var 1 --max-degree 10
        python cv_ridge.py --var 2 --max-degree 20

Steps
  1. load  data/<ROLL>_train_var<k>.csv  and  data/<ROLL>_test_var<k>.csv
  2. K-fold cross-validation over   basis x degree x ridge-lambda
  3. pick the ridge configuration with the lowest CV MSE
  4. save the full CV table (outputs/cv_ridge_var<k>.csv) and a summary json
The FINAL predictions are produced by final_model.py.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from polyreg import (PolynomialRegressor, MinMaxScaler11, design_matrix, exponents,
                     kfold_indices, mse, n_terms, r2, ridge_path)

ROLL = "BT2024013"
ROOT = Path(__file__).parent
LAMBDAS = [0.0] + list(np.logspace(-12, 2, 29))


def load(var):
    tr = pd.read_csv(ROOT / "data" / f"{ROLL}_train_var{var}.csv")
    te = pd.read_csv(ROOT / "data" / f"{ROLL}_test_var{var}.csv")
    feats = [c for c in tr.columns if c.lower() != "y"]
    ycol = [c for c in tr.columns if c.lower() == "y"][0]
    feats_te = [c for c in feats if c in te.columns]
    assert feats_te == feats, f"test file columns differ: {te.columns.tolist()}"
    return tr[feats].to_numpy(float), tr[ycol].to_numpy(float), te[feats].to_numpy(float), feats


def shift_weights(X, Xte):
    """Importance weights for covariate shift.
    Inputs are clipped to [-1, 1]; we bucket every point by HOW MANY of its
    coordinates sit on the boundary (|x| == 1) and weight training points by
    p_test(bucket) / p_train(bucket), so the CV score mimics the test mix."""
    d = X.shape[1]
    cap = min(d, 4)
    k_tr = np.minimum((np.abs(X) >= 1 - 1e-12).sum(1), cap)
    k_te = np.minimum((np.abs(Xte) >= 1 - 1e-12).sum(1), cap)
    p_tr = np.bincount(k_tr, minlength=cap + 1) / len(k_tr)
    p_te = np.bincount(k_te, minlength=cap + 1) / len(k_te)
    w = (p_te / np.where(p_tr > 0, p_tr, 1))[k_tr]
    return w / w.mean(), p_tr, p_te


def cross_validate(X, y, degrees, bases, k=5, seeds=(0, 1), max_terms=12000, wts=None):
    if wts is None:
        wts = np.ones(len(y))
    rows = []
    n, d = X.shape
    for seed in seeds:
        folds = kfold_indices(n, k, seed)
        for fi, val_idx in enumerate(folds):
            tr_idx = np.setdiff1d(np.arange(n), val_idx)
            sc = MinMaxScaler11().fit(X[tr_idx])
            z_tr, z_va = sc.transform(X[tr_idx]), sc.transform(X[val_idx])
            for basis in bases:
                for D in degrees:
                    if n_terms(d, D) > max_terms:
                        continue
                    E = exponents(d, D)
                    P_tr, P_va = design_matrix(z_tr, E, basis), design_matrix(z_va, E, basis)
                    pv = ridge_path(P_tr, y[tr_idx], P_va, LAMBDAS)
                    pt = ridge_path(P_tr, y[tr_idx], P_tr, LAMBDAS)
                    for li, lam in enumerate(LAMBDAS):
                        rows.append(dict(seed=seed, fold=fi, basis=basis, degree=D, lam=lam,
                                         val_mse=mse(y[val_idx], pv[li]),
                                         val_wmse=float(np.average((y[val_idx] - pv[li]) ** 2, weights=wts[val_idx])),
                                         train_mse=mse(y[tr_idx], pt[li])))
            print(f"  seed {seed} fold {fi} done", flush=True)
    df = pd.DataFrame(rows)
    g = df.groupby(["basis", "degree", "lam"])
    summ = g.agg(val_mse=("val_mse", "mean"), val_std=("val_mse", "std"),
                 val_wmse=("val_wmse", "mean"), val_wstd=("val_wmse", "std"),
                 train_mse=("train_mse", "mean"), n=("val_mse", "size")).reset_index()
    summ["val_se"] = summ["val_std"] / np.sqrt(summ["n"])
    summ["val_wse"] = summ["val_wstd"] / np.sqrt(summ["n"])
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", type=int, required=True)
    ap.add_argument("--max-degree", type=int, required=True)
    ap.add_argument("--bases", default="monomial,legendre")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()

    t0 = time.time()
    X, y, Xte, feats = load(args.var)
    print(f"var{args.var}: train {X.shape}, test {Xte.shape}, features {feats}")
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)

    wts, bp_tr, bp_te = shift_weights(X, Xte)
    print("boundary-count mix  train:", bp_tr.round(3), " test:", bp_te.round(3))
    summ = cross_validate(X, y, range(1, args.max_degree + 1), args.bases.split(","),
                          k=args.folds, seeds=tuple(int(s) for s in args.seeds.split(",")), wts=wts)
    summ.to_csv(out / f"cv_ridge_var{args.var}.csv", index=False)

    # selection criterion: plain CV MSE (the edge-weighted CV MSE is kept as a cross-check)
    best = summ.loc[summ.val_mse.idxmin()]
    print("\nbest config:", best.to_dict())

    model = PolynomialRegressor(int(best.degree), float(best.lam), best.basis).fit(X, y)
    p_tr = model.predict(X)
    p_te = model.predict(Xte)
    

    # out-of-fold predictions of the chosen config (for honest parity plot)
    oof = np.empty_like(y)
    for val_idx in kfold_indices(len(y), args.folds, 0):
        tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        m = PolynomialRegressor(int(best.degree), float(best.lam), best.basis).fit(X[tr_idx], y[tr_idx])
        oof[val_idx] = m.predict(X[val_idx])
    np.save(out / f"oof_ridge_var{args.var}.npy", oof)

    info = dict(var=args.var, n_train=len(y), n_test=len(Xte), features=feats,
                basis=best.basis, degree=int(best.degree), lam=float(best.lam),
                n_terms=n_terms(X.shape[1], int(best.degree)),
                cv_mse=float(best.val_mse), cv_se=float(best.val_se),
                cv_wmse=float(best.val_wmse), cv_wse=float(best.val_wse),
                bucket_p_train=bp_tr.tolist(), bucket_p_test=bp_te.tolist(),
                oof_wmse=float(np.average((y - oof) ** 2, weights=wts)),
                oof_mse=mse(y, oof), oof_r2=r2(y, oof),
                train_mse=mse(y, p_tr), train_r2=r2(y, p_tr),
                y_mean=float(y.mean()), y_var=float(y.var()),
                test_pred_mean=float(p_te.mean()), test_pred_std=float(p_te.std()),
                train_range=[X.min(0).tolist(), X.max(0).tolist()],
                test_range=[Xte.min(0).tolist(), Xte.max(0).tolist()],
                seconds=round(time.time() - t0, 1))
    (out / f"summary_ridge_var{args.var}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
