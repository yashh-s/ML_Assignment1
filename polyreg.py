"""
polyreg.py - Polynomial regression from scratch (NumPy only).

Model:   y  ~  w0 + sum_k  w_k * phi_k(x)
where phi_k ranges over every multivariate polynomial term whose total
degree (sum of exponents) is <= D, exactly as defined in the assignment.

Two equivalent polynomial bases are supported:
  * "monomial" : phi(x) = x1^a1 * x2^a2 * ... * xd^ad
  * "legendre" : phi(x) = P_a1(x1) * P_a2(x2) * ... * P_ad(xd)
Both span the SAME space of polynomials of total degree <= D, so the
Legendre model is still plain polynomial regression - it is just a
better-conditioned set of coordinates for that space (important for D=20).

Inputs are first min-max scaled to [-1, 1] using the TRAINING data only.
Weights are found by (ridge-regularised) least squares, solved via SVD:
    w = V diag( s / (s^2 + lambda) ) U^T y        (lambda = 0 -> OLS / pinv)
"""
from itertools import combinations_with_replacement
from math import comb

import numpy as np


# ------------------------------------------------------------------ basis
def exponents(n_features: int, degree: int) -> np.ndarray:
    """All exponent tuples (a1..ad) with 1 <= sum(a) <= degree (intercept excluded)."""
    rows = []
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), deg):
            e = np.zeros(n_features, dtype=np.int16)
            for j in combo:
                e[j] += 1
            rows.append(e)
    return np.array(rows, dtype=np.int16).reshape(-1, n_features)


def n_terms(n_features: int, degree: int) -> int:
    """Number of terms incl. intercept = C(d + D, D)."""
    return comb(n_features + degree, degree)


def _univariate_table(z: np.ndarray, degree: int, basis: str) -> np.ndarray:
    """T[j, k, i] = basis polynomial of order k evaluated at z[i, j]."""
    n, d = z.shape
    T = np.empty((d, degree + 1, n))
    T[:, 0, :] = 1.0
    if degree >= 1:
        T[:, 1, :] = z.T
    for k in range(1, degree):
        if basis == "monomial":
            T[:, k + 1, :] = T[:, k, :] * z.T
        elif basis == "legendre":  # Bonnet recursion
            T[:, k + 1, :] = ((2 * k + 1) * z.T * T[:, k, :] - k * T[:, k - 1, :]) / (k + 1)
        else:
            raise ValueError(basis)
    return T


def design_matrix(z: np.ndarray, exps: np.ndarray, basis: str) -> np.ndarray:
    degree = int(exps.sum(1).max()) if len(exps) else 0
    T = _univariate_table(z, degree, basis)
    Phi = np.ones((z.shape[0], len(exps)))
    for j in range(z.shape[1]):
        Phi *= T[j, exps[:, j], :].T
    return Phi


# ------------------------------------------------------------------ scaling
class MinMaxScaler11:
    def fit(self, X):
        self.lo, self.hi = X.min(0), X.max(0)
        return self

    def transform(self, X):
        return 2 * (X - self.lo) / (self.hi - self.lo) - 1


# ------------------------------------------------------------------ solver
def ridge_path(Phi_tr, y_tr, Phi_ev, lambdas):
    """Fit centred ridge for every lambda with ONE SVD; return predictions on Phi_ev.
    The intercept is never penalised (handled by centring)."""
    mu_x, mu_y = Phi_tr.mean(0), y_tr.mean()
    A = Phi_tr - mu_x
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Uty = U.T @ (y_tr - mu_y)
    B = Phi_ev - mu_x
    BV = B @ Vt.T
    tol = s.max() * max(A.shape) * np.finfo(float).eps
    preds = []
    for lam in lambdas:
        if lam == 0:
            f = np.where(s > tol, 1.0 / np.where(s > tol, s, 1.0), 0.0)
        else:
            f = s / (s ** 2 + lam)
        preds.append(mu_y + BV @ (f * Uty))
    return np.array(preds)  # (n_lambdas, n_eval)


class PolynomialRegressor:
    def __init__(self, degree, lam=0.0, basis="legendre"):
        self.degree, self.lam, self.basis = degree, lam, basis

    def fit(self, X, y):
        self.scaler = MinMaxScaler11().fit(X)
        z = self.scaler.transform(X)
        self.exps = exponents(X.shape[1], self.degree)
        Phi = design_matrix(z, self.exps, self.basis)
        self.mu_x, self.mu_y = Phi.mean(0), y.mean()
        A = Phi - self.mu_x
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        if self.lam == 0:
            tol = s.max() * max(A.shape) * np.finfo(float).eps
            f = np.where(s > tol, 1.0 / np.where(s > tol, s, 1.0), 0.0)
        else:
            f = s / (s ** 2 + self.lam)
        self.w = Vt.T @ (f * (U.T @ (y - self.mu_y)))
        self.cond = s.max() / s.min() if s.min() > 0 else np.inf
        return self

    def predict(self, X):
        Phi = design_matrix(self.scaler.transform(X), self.exps, self.basis)
        return self.mu_y + (Phi - self.mu_x) @ self.w


# ------------------------------------------------------------------ metrics
def mse(y, p):
    return float(np.mean((y - p) ** 2))


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def kfold_indices(n, k, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, k)
