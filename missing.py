r"""
Missing data mechanisms (Appendix D of MissScore).

Produces a binary mask m \in {0, 1}^{n x d} where 1 = missing, 0 = observed.

  - MCAR : each entry independently Bernoulli(alpha).
  - MAR  : a random subset of the variables is kept fully observed; the rest
           have missingness driven by a logistic model in the observed subset.
  - MNAR : logistic model whose inputs are the *full* variables (variant 1),
           or a two-set construction where the input set is masked MCAR and
           drives the second set (variant 2).

These follow the protocol of Muzellec et al. 2020 (NeurIPS) which the paper
adopts; we re-implement here without any external dependency.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def _pick_coeffs(rng: np.random.Generator, X: np.ndarray, idxs_obs, idxs_nas, self_mask: bool = False):
    """Return random logistic coefficients of the right shape."""
    n, d = X.shape
    if self_mask:
        coeffs = rng.standard_normal(d)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = rng.standard_normal((d_obs, d_na))
    return coeffs


def _fit_intercepts(X_obs, coeffs, p, self_mask: bool = False):
    """Choose intercepts so the average missingness probability equals p."""
    from scipy import optimize

    if self_mask:
        d = X_obs.shape[1]
        intercepts = np.zeros(d)
        for j in range(d):
            def f(b, j=j):
                return _sigmoid(X_obs[:, j] * coeffs[j] + b).mean() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_na = coeffs.shape[1]
        intercepts = np.zeros(d_na)
        for j in range(d_na):
            def f(b, j=j):
                return _sigmoid(X_obs @ coeffs[:, j] + b).mean() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def produce_NA(X: np.ndarray, p_miss: float, mecha: str = "MCAR",
               opt: str = "logistic", p_obs: float = 0.5, q: float = 0.3,
               seed: int = 0):
    """
    Generate a missingness mask for X according to `mecha`.

    Args:
        X       : (n, d) clean data.
        p_miss  : target missing ratio in [0, 1).
        mecha   : 'MCAR' | 'MAR' | 'MNAR'.
        opt     : MNAR variant ('logistic' or 'selfmasked').
        p_obs   : proportion of variables kept fully observed under MAR.
        q       : controls the second mechanism for MNAR-logistic.
        seed    : RNG seed.

    Returns:
        dict with keys:
            X_init : original X
            X_incomp : X with NaNs at missing positions
            mask : (n, d) boolean tensor, True = missing
    """
    rng = np.random.default_rng(seed)
    X = X.astype(np.float64)
    n, d = X.shape
    mask = np.zeros((n, d), dtype=bool)

    if mecha == "MCAR":
        mask = rng.random((n, d)) < p_miss

    elif mecha == "MAR":
        # Random subset of variables fully observed
        d_obs = max(1, int(p_obs * d))
        idxs_obs = rng.choice(d, d_obs, replace=False)
        idxs_nas = np.array([j for j in range(d) if j not in idxs_obs])

        coeffs = _pick_coeffs(rng, X, idxs_obs, idxs_nas)
        # Rescale coefficients so logits have unit variance per column
        scale = np.std(X[:, idxs_obs] @ coeffs, axis=0) + 1e-12
        coeffs = coeffs / scale

        intercepts = _fit_intercepts(X[:, idxs_obs], coeffs, p_miss)
        probs = _sigmoid(X[:, idxs_obs] @ coeffs + intercepts)
        mask[:, idxs_nas] = rng.random((n, len(idxs_nas))) < probs

    elif mecha == "MNAR":
        if opt == "selfmasked":
            coeffs = _pick_coeffs(rng, X, None, None, self_mask=True)
            scale = np.std(X * coeffs, axis=0) + 1e-12
            coeffs = coeffs / scale
            intercepts = _fit_intercepts(X, coeffs, p_miss, self_mask=True)
            probs = _sigmoid(X * coeffs + intercepts)
            mask = rng.random((n, d)) < probs
        else:  # 'logistic' variant
            d_obs = max(1, int(p_obs * d))
            idxs_obs = rng.choice(d, d_obs, replace=False)
            idxs_nas = np.array([j for j in range(d) if j not in idxs_obs])

            # First mask the 'observed' inputs MCAR-style
            mask[:, idxs_obs] = rng.random((n, len(idxs_obs))) < q

            X_masked = X.copy()
            X_masked[mask] = 0.0  # zero-fill for the logistic regression
            coeffs = _pick_coeffs(rng, X_masked, idxs_obs, idxs_nas)
            scale = np.std(X_masked[:, idxs_obs] @ coeffs, axis=0) + 1e-12
            coeffs = coeffs / scale
            intercepts = _fit_intercepts(X_masked[:, idxs_obs], coeffs, p_miss)
            probs = _sigmoid(X_masked[:, idxs_obs] @ coeffs + intercepts)
            mask[:, idxs_nas] = rng.random((n, len(idxs_nas))) < probs
    else:
        raise ValueError(f"Unknown mechanism: {mecha}")

    X_incomp = X.copy()
    X_incomp[mask] = np.nan
    return {
        "X_init": X,
        "X_incomp": X_incomp,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# Logistic-regression-based estimation of P[m_i=0|x] and P[m_i m_j=0|x]
# (used by the MAR objective; Section 3 of the paper)
# ---------------------------------------------------------------------------
def estimate_mar_probabilities(X_filled: np.ndarray, mask: np.ndarray):
    """
    Fit per-variable logistic regressions that predict whether each entry is
    observed, given the *filled* data. Returns:
        p_obs      : (n, d) estimated P[m_i = 0 | x_obs]
        p_pair_obs : (n, d, d) estimated P[m_i m_j = 0 | x_obs]
                     -- using independence approximation off-diagonal,
                     which the paper also uses since fitting d^2 logistic
                     regressions is expensive (and degenerate when a pair
                     is never simultaneously observed).
    """
    n, d = X_filled.shape
    p_obs = np.ones((n, d))

    for j in range(d):
        y = (~mask[:, j]).astype(int)  # 1 = observed
        if y.sum() == 0 or y.sum() == n or d == 1:
            # Variable always missing / always observed, or only one variable
            # left (no predictors): fall back to the marginal observed rate.
            p_obs[:, j] = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
            continue
        # Predict observedness from the other variables (the "observed" set).
        feat_idx = [k for k in range(d) if k != j]
        Xf = X_filled[:, feat_idx]
        lr = LogisticRegression(max_iter=500)
        lr.fit(Xf, y)
        p_obs[:, j] = lr.predict_proba(Xf)[:, 1].clip(1e-3, 1 - 1e-3)

    # Pairwise approximation: P[m_i m_j = 0] approx P[m_i = 0] * P[m_j = 0]
    # (this is the cheap-and-stable choice; replace if you want to fit the
    # full d*(d-1)/2 joint logistic models).
    p_pair_obs = p_obs[:, :, None] * p_obs[:, None, :]
    # Ensure the diagonal equals p_obs (avoid double-shrinkage from squaring).
    for j in range(d):
        p_pair_obs[:, j, j] = p_obs[:, j]

    return p_obs, p_pair_obs
