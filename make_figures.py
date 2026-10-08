"""make_figures.py - figures for the report (run after cv_ridge.py, cv_lasso.py, final_model.py)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT, FIG = ROOT / "outputs", ROOT / "figures"
FIG.mkdir(exist_ok=True)
ROLL = "BT2024013"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.8, "legend.frameon": False, "axes.titlesize": 9,
    "axes.titlecolor": INK, "savefig.dpi": 220, "savefig.bbox": "tight",
})


def load(var):
    tr = pd.read_csv(ROOT / "data" / f"{ROLL}_train_var{var}.csv")
    te = pd.read_csv(ROOT / "data" / f"{ROLL}_test_var{var}.csv")
    return tr, te


def n_clipped(X, cap):
    return np.minimum((np.abs(X) >= 1).sum(1), cap)


def fig_eda():
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.5))
    for ax, var in zip(axes[:2], [1, 2]):
        tr, te = load(var)
        bins = np.linspace(-1, 1, 22)
        ax.hist(tr.drop(columns="y").to_numpy().ravel(), bins, density=True, color=BLUE, alpha=.55,
                label="train", edgecolor="white", linewidth=.5)
        ax.hist(te.to_numpy().ravel(), bins, density=True, color=ORANGE, alpha=.55,
                label="test", edgecolor="white", linewidth=.5)
        ax.set_title(f"var{var}: all input values pooled")
        ax.set_xlabel("feature value"); ax.set_ylabel("density"); ax.legend()
    ax = axes[2]
    w = 0.2
    for i, (var, cap) in enumerate([(1, 4), (2, 3)]):
        tr, te = load(var)
        ptr = np.bincount(n_clipped(tr.drop(columns="y").to_numpy(), cap), minlength=cap + 1) / len(tr)
        pte = np.bincount(n_clipped(te.to_numpy(), cap), minlength=cap + 1) / len(te)
        k = np.arange(cap + 1)
        off = (i * 2 - 1.5) * w
        ax.bar(k + off, ptr, w * .9, color=BLUE if var == 1 else AQUA, label=f"var{var} train")
        ax.bar(k + off + w, pte, w * .9, color=BLUE if var == 1 else AQUA, alpha=.45,
               label=f"var{var} test", hatch="///", edgecolor="white")
    ax.set_xticks(range(5)); ax.set_xticklabels(["0", "1", "2", "3", "≥4"])
    ax.set_xlabel("# coordinates of a row sitting exactly at ±1")
    ax.set_ylabel("fraction of rows"); ax.set_title("Boundary mix: train vs test"); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIG / "eda.png"); plt.close(fig)


def best_per_degree(df, kind_col=None):
    return df.loc[df.groupby("degree").val_mse.idxmin()].sort_values("degree")


def fig_cv(var, sweep):
    ridge = pd.read_csv(OUT / f"cv_ridge_var{var}.csv")
    ridge = ridge[ridge.basis == "monomial"]
    lasso = pd.read_csv(OUT / f"lasso_cv_var{var}.csv")
    lasso = lasso[lasso.kind == "lasso"]
    s = json.loads((OUT / f"summary_final_var{var}.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))

    ax = axes[0]
    ols = ridge[ridge.lam == 0].sort_values("degree")
    ax.semilogy(ols.degree, np.maximum(ols.train_mse, 1e-3), "-o", color=BLUE, ms=3, label="training error")
    if (ols.train_mse < 1e-3).any():
        d0 = ols.degree[ols.train_mse < 1e-3].min()
        ax.annotate("train error ≈ 0\n(more terms than points:\nexact interpolation)", (d0, 1e-3),
                    textcoords="offset points", xytext=(4, 12), fontsize=6.5, color=INK2)
    ax.semilogy(ols.degree, ols.val_mse, "-o", color=ORANGE, ms=3, label="validation error (5-fold CV)")
    ax.set_title("No regularisation (plain least squares)"); ax.set_xlabel("degree D"); ax.set_ylabel("MSE (log scale)")
    ax.legend(fontsize=7)

    ax = axes[1]
    br, bl = best_per_degree(ridge), best_per_degree(lasso)
    ax.plot(br.degree, br.val_mse, "-o", color=BLUE, ms=3, label="ridge (best λ per D)")
    ax.plot(bl.degree, bl.val_mse, "-o", color=AQUA, ms=3, label="lasso (best α per D)")
    ax.axvline(s["degree"], color=MUTED, lw=1, ls="--")
    ax.set_yscale("log"); ax.set_title(f"Regularised: CV MSE vs degree (chosen D={s['degree']})")
    ax.set_xlabel("degree D"); ax.set_ylabel("CV MSE (log)"); ax.legend(fontsize=7)

    ax = axes[2]
    if sweep == "ridge":
        sub = ridge[(ridge.degree == s["degree"]) & (ridge.lam > 0)].sort_values("lam")
        x, xl, chosen = sub.lam, "ridge strength λ", s["lam"]
    else:
        f = pd.read_csv(OUT / f"lasso_cv_var{var}_fine.csv")
        sub = f[(f.kind == "lasso") & (f.degree == s["degree"])].sort_values("alpha")
        x, xl, chosen = sub.alpha, "lasso strength α", s["alpha"]
    ax.semilogx(x, sub.val_mse, "-o", color=AQUA, ms=3, label="validation (CV)")
    if "train_mse" in sub:
        ax.semilogx(x, sub.train_mse, ":", color=BLUE, label="training")
    ax.axvline(chosen, color=MUTED, lw=1, ls="--")
    if sweep == "lasso":
        for xi, yi, nz in list(zip(x, sub.val_mse, sub.nnz))[::3]:
            ax.annotate(f"{nz:.0f} terms", (xi, yi), textcoords="offset points", xytext=(0, 6),
                        fontsize=6.5, color=INK2, ha="center")
    if sweep == "ridge":
        ax.set_yscale("log")
    ax.set_title(f"Tuning the penalty at D={s['degree']}")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=6))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(xl); ax.set_ylabel("MSE" + (" (log)" if sweep == "ridge" else "")); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / f"cv_var{var}.png"); plt.close(fig)


def fig_resid():
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.4))
    for i, (var, cap) in enumerate([(1, 4), (2, 3)]):
        tr, _ = load(var)
        y, X = tr.y.to_numpy(), tr.drop(columns="y").to_numpy()
        oof = np.load(OUT / f"oof_final_var{var}.npy")
        s = json.loads((OUT / f"summary_final_var{var}.json").read_text())
        ax = axes[2 * i]
        ax.scatter(y, oof, s=5, color=BLUE, alpha=.5, linewidths=0)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], color=INK2, lw=1)
        ax.set_title(f"var{var}: held-out predictions (R²={s['cv_r2']:.3f})")
        ax.set_xlabel("true y"); ax.set_ylabel("predicted y")
        ax = axes[2 * i + 1]
        k = n_clipped(X, cap)
        m = [np.mean((y[k == j] - oof[k == j]) ** 2) for j in range(cap + 1)]
        ax.bar(range(cap + 1), m, color=ORANGE, width=.6)
        ax.set_xticks(range(cap + 1)); ax.set_xticklabels([str(j) for j in range(cap)] + [f"≥{cap}"])
        ax.set_title(f"var{var}: held-out MSE by boundary count"); ax.set_xlabel("# coords at ±1"); ax.set_ylabel("MSE")
    fig.tight_layout(); fig.savefig(FIG / "residuals.png"); plt.close(fig)


if __name__ == "__main__":
    fig_eda()
    fig_cv(1, "lasso")
    fig_cv(2, "ridge")
    fig_resid()
    print("figures written to", FIG)
