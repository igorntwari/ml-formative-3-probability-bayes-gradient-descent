"""Expectation Maximization (EM) for a two component 1D Gaussian mixture.

Implemented from scratch with NumPy only (no scikit-learn, no scipy.stats).
Used in Part 1 of Formative 3 on the Galton parents vs children heights data.

Notation (matches the tracking table in the brief):
    mu[k]   mean of component k
    var[k]  variance (sigma squared) of component k
    pi[k]   mixing coefficient (prior weight) of component k
Component 0 is the shorter group ("Child"), component 1 the taller ("Father").

Complexity: every EM iteration is O(N * K) time and memory, with K = 2 here,
so a full fit is O(T * N) for T iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CM_PER_INCH = 2.54
VARIANCE_FLOOR = 1e-6  # stops a component collapsing onto one point (var -> 0, LL -> +inf)
DEFAULT_TOLERANCE = 1e-6
DEFAULT_MAX_ITER = 1000
VALID_PARENTS = ("father", "mother")
VALID_UNITS = ("in", "cm")


@dataclass(frozen=True)
class GMMParams:
    """Parameters of a K component 1D Gaussian mixture (arrays of shape (K,))."""

    mu: np.ndarray
    var: np.ndarray
    pi: np.ndarray


@dataclass(frozen=True)
class EMStep:
    """One row of the tracking table: parameters theta_t and log likelihood L(theta_t)."""

    iteration: int
    params: GMMParams
    log_likelihood: float


# ---------------------------------------------------------------- data loading


def load_heights(csv_path: str | Path, parent: str = "father") -> tuple[np.ndarray, np.ndarray]:
    """Return (heights, labels) for children pooled with one parent type.

    The CSV has one row per CHILD, so each parent is repeated once per child.
    Parents are de duplicated by family; otherwise a father with 10 children
    would be counted 10 times and drag the "parent" Gaussian toward him.

    labels: 0 = child, 1 = parent. EM never sees them; they are only used
    afterwards to check what the unsupervised model actually learned.
    """
    if parent not in VALID_PARENTS:
        raise ValueError(f"parent must be one of {VALID_PARENTS}, got {parent!r}")

    df = pd.read_csv(csv_path, dtype={"family": str})  # ids like "001" and "136A"
    required = {"family", parent, "childHeight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    children = df["childHeight"].to_numpy(dtype=float)
    parents = df.drop_duplicates("family")[parent].to_numpy(dtype=float)
    heights = np.concatenate([children, parents])
    if not np.all(np.isfinite(heights)):
        raise ValueError("heights contain NaN or infinite values")

    labels = np.concatenate([np.zeros(children.size, dtype=int), np.ones(parents.size, dtype=int)])
    return heights, labels


# ------------------------------------------------------------- core maths


def gaussian_log_pdf(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """log N(x | mu, var) for every point and component. Shape (N, K).

    Logs instead of raw densities so tiny probabilities do not underflow to 0.
    """
    x = np.asarray(x, dtype=float)[:, None]
    return -0.5 * np.log(2.0 * np.pi * var) - (x - mu) ** 2 / (2.0 * var)


def _log_sum_exp(a: np.ndarray) -> np.ndarray:
    """Stable log(sum(exp(a))) along axis 1 (subtract the row max first)."""
    row_max = a.max(axis=1, keepdims=True)
    return row_max[:, 0] + np.log(np.exp(a - row_max).sum(axis=1))


def e_step(x: np.ndarray, params: GMMParams) -> tuple[np.ndarray, float]:
    """Expectation step.

    Responsibility r[i, k] = P(component k | x_i)
                           = pi_k N(x_i | mu_k, var_k) / sum_j pi_j N(x_i | mu_j, var_j)
    Also returns the log likelihood L = sum_i log sum_k pi_k N(x_i | mu_k, var_k),
    which falls out of the same numbers (the denominator), so it costs nothing extra.
    """
    weighted_log = np.log(params.pi) + gaussian_log_pdf(x, params.mu, params.var)  # (N, K)
    log_denominator = _log_sum_exp(weighted_log)  # (N,)
    responsibilities = np.exp(weighted_log - log_denominator[:, None])
    return responsibilities, float(log_denominator.sum())


def m_step(x: np.ndarray, responsibilities: np.ndarray) -> GMMParams:
    """Maximization step: weighted maximum likelihood estimates.

    N_k    = sum_i r[i, k]                       (effective number of points in k)
    mu_k   = sum_i r[i, k] x_i / N_k
    var_k  = sum_i r[i, k] (x_i - mu_k)^2 / N_k
    pi_k   = N_k / N
    """
    x = np.asarray(x, dtype=float)
    n_k = responsibilities.sum(axis=0)
    mu = (responsibilities * x[:, None]).sum(axis=0) / n_k
    var = (responsibilities * (x[:, None] - mu) ** 2).sum(axis=0) / n_k
    return GMMParams(mu=mu, var=np.maximum(var, VARIANCE_FLOOR), pi=n_k / x.size)


# ------------------------------------------------------ initialization / fit


def mean_split_estimate(x: np.ndarray) -> GMMParams:
    """The naive approach: cut at the global mean, then take each pile's stats.

    Used as EM's iteration 0, so the tracking table shows directly how much EM
    improves on it. Equivalent to one M step with hard 0/1 responsibilities.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2 or np.ptp(x) == 0:
        raise ValueError("need at least two distinct heights to split")
    is_tall = x > x.mean()
    hard_responsibilities = np.column_stack([~is_tall, is_tall]).astype(float)
    return m_step(x, hard_responsibilities)


def fit_em(
    x: np.ndarray,
    init: GMMParams,
    tol: float = DEFAULT_TOLERANCE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> list[EMStep]:
    """Run EM until the log likelihood improves by less than `tol`.

    Returns the full history: history[t] holds theta_t and L(theta_t), so
    history[0] is the initialization. EM guarantees L never decreases.
    """
    x = np.asarray(x, dtype=float)
    params = init
    history: list[EMStep] = []
    for iteration in range(max_iter + 1):
        responsibilities, log_likelihood = e_step(x, params)
        history.append(EMStep(iteration, params, log_likelihood))
        if iteration > 0 and log_likelihood - history[-2].log_likelihood < tol:
            break
        params = m_step(x, responsibilities)
    return history


# ---------------------------------------------------------------- reporting


def posterior(height: float, params: GMMParams, unit: str = "in") -> np.ndarray:
    """P(component k | height) for one new height. Accepts inches or centimetres."""
    if unit not in VALID_UNITS:
        raise ValueError(f"unit must be one of {VALID_UNITS}, got {unit!r}")
    height = float(height)
    if not np.isfinite(height) or height <= 0:
        raise ValueError(f"height must be a positive finite number, got {height}")
    height_in = height / CM_PER_INCH if unit == "cm" else height
    responsibilities, _ = e_step(np.array([height_in]), params)
    return responsibilities[0]


def tracking_table(history: list[EMStep], names: tuple[str, str] = ("Child", "Father")) -> pd.DataFrame:
    """The brief's tracking table: one row per iteration."""
    rows = [
        {
            "Iteration": step.iteration,
            f"mu1 ({names[0]})": step.params.mu[0],
            f"mu2 ({names[1]})": step.params.mu[1],
            "sigma1^2": step.params.var[0],
            "sigma2^2": step.params.var[1],
            "pi1": step.params.pi[0],
            "pi2": step.params.pi[1],
            "Log Likelihood": step.log_likelihood,
        }
        for step in history
    ]
    return pd.DataFrame(rows).set_index("Iteration")


def decision_boundary(params: GMMParams, low: float = 0.0, high: float = 120.0) -> list[float]:
    """Heights in [low, high] where both components are equally likely.

    Solves pi_1 N(x|mu_1,var_1) = pi_2 N(x|mu_2,var_2), a quadratic in x
    because the variances differ (two roots; usually only one is a real height).
    """
    (m1, m2), (v1, v2), (p1, p2) = params.mu, params.var, params.pi
    a = 1 / (2 * v2) - 1 / (2 * v1)
    b = m1 / v1 - m2 / v2
    c = m2**2 / (2 * v2) - m1**2 / (2 * v1) + np.log(p1 / p2) - 0.5 * np.log(v1 / v2)
    roots = np.roots([a, b, c]) if abs(a) > 1e-12 else np.array([-c / b])
    return sorted(float(r.real) for r in roots if abs(r.imag) < 1e-9 and low <= r.real <= high)


def score_against_labels(responsibilities: np.ndarray, labels: np.ndarray) -> float:
    """Accuracy of the hard assignment (argmax responsibility) vs hidden labels."""
    return float((responsibilities.argmax(axis=1) == labels).mean())
