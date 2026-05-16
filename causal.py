"""
Causal discovery with missing data (Algorithm 3 of MissScore).

Intuition (Rolland et al. 2022): for an additive-noise model
    X_i = f_i(X_PA(i)) + Z_i,  Z_i Gaussian,
the diagonal Hessian entry H_jj(log p) is constant in X_j  <=>  j is a leaf.
Equivalently  Var_X[ H_jj(log p) ] = 0  iff  j is a leaf node.

Algorithm 3 (strict):
    for k = 1..d:
        jointly train s1, s2 on the current x_obs (Algorithm 1)
        generate n new samples with Algorithm 2 + bootstrap
        estimate s2 on those samples
        V_j = Var_X[ diag(s2)_j ]
        leaf = argmin_j V_j ; prepend to order ; drop that column
    prune with a CAM-style step using the recovered order.

We implement both:
    - retrain_each_step=True  : the strict paper algorithm (slow, accurate)
    - retrain_each_step=False : train once, reuse s2 (fast DiffAN-style shortcut)
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from sklearn.linear_model import LassoCV

from .models import ScoreNetLarge, ScoreNetSmall
from .train import train_missscore
from .sampling import ozaki_sample, langevin_sample


def _leaf_by_hessian_variance(
    score_net: torch.nn.Module,
    eval_points: torch.Tensor,
    active_cols: List[int],
) -> int:
    """Return the active column with the smallest Var[diag(s2)]."""
    score_net.eval()
    with torch.no_grad():
        _, s2_diag = score_net(eval_points)
    var = s2_diag.var(dim=0).cpu().numpy()  # (d_current,)
    local = int(np.argmin([var[i] for i in range(len(active_cols))]))
    return active_cols[local]


def _bootstrap_eval_points(
    score_net: torch.nn.Module,
    X_filled_t: torch.Tensor,
    d_cur: int,
    device: str,
    use_ozaki: bool = True,
) -> torch.Tensor:
    """
    Generate n new samples (Algorithm 2) with bootstrapped initialisation,
    used to estimate Var_X[diag(s2)] more robustly than reusing training data.
    """
    n = X_filled_t.shape[0]
    boot_idx = torch.randint(0, n, (n,), device=device)
    init = X_filled_t[boot_idx].clone()
    if use_ozaki:
        return ozaki_sample(score_net, n, d_cur, eps=5e-3, n_steps=40,
                            init=init, device=device).to(device)
    return langevin_sample(score_net, n, d_cur, eps=5e-3, n_steps=80,
                           init=init, device=device).to(device)


def _cam_prune(X: np.ndarray, order: List[int], threshold: float = 0.05) -> np.ndarray:
    """
    CAM-style pruning: regress each node on its predecessors in `order` with
    LassoCV; keep edge k->j iff |coef_k| > threshold. Returns adjacency A
    with A[k, j] = 1 meaning k -> j.
    """
    d = X.shape[1]
    A = np.zeros((d, d), dtype=int)
    pos = {v: i for i, v in enumerate(order)}
    for j in order:
        preds = [k for k in order if pos[k] < pos[j]]
        if not preds:
            continue
        Xp, y = X[:, preds], X[:, j]
        try:
            lasso = LassoCV(cv=3, max_iter=3000, n_alphas=12).fit(Xp, y)
            coefs = lasso.coef_
        except Exception:
            coefs = np.corrcoef(np.c_[Xp, y].T)[-1, :-1]
        for k, c in zip(preds, coefs):
            if abs(c) > threshold:
                A[k, j] = 1
    return A


def missscore_causal_discovery(
    X_incomp: np.ndarray,
    mask: np.ndarray,
    mechanism: str = "mcar",
    sigma: float = 0.1,
    omega: float = 1.0,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 1e-3,
    prune_threshold: float = 0.05,
    retrain_each_step: bool = True,
    small_net: bool = True,
    device: str = "cpu",
    verbose: bool = False,
):
    """
    Run Algorithm 3.

    Args:
        retrain_each_step : True  -> strict paper algorithm (retrain after
                                     removing each leaf; uses bootstrapped
                                     Algorithm-2 samples to estimate variance).
                            False -> train once and reuse (fast shortcut).
        small_net : use the 3-layer Softplus net (good for low-d synthetic
                    chains); set False to use the 5-layer LeakyReLU net.

    Returns dict with 'order' (topological order, sources first) and
    'adjacency' (pruned DAG).
    """
    n, d = X_incomp.shape
    Net = ScoreNetSmall if small_net else ScoreNetLarge

    full_order: List[int] = []         # leaves get prepended -> sources end up first
    active = list(range(d))            # original column indices still in play

    # Work on a mutable copy whose columns we drop as we identify leaves.
    Xi_cur = X_incomp.copy()
    m_cur = mask.copy()

    if not retrain_each_step:
        # ---- fast path: train once on full data, reuse s2 ----
        net = Net(d)
        net, _ = train_missscore(
            net, X_incomp, mask, mechanism=mechanism, sigma=sigma, omega=omega,
            use_vr=True, epochs=epochs, batch_size=batch_size, lr=lr,
            device=device, verbose=verbose, log_every=max(epochs // 4, 1),
        )
        Xf = X_incomp.copy(); Xf[mask] = 0.0
        Xf_t = torch.tensor(Xf, dtype=torch.float32, device=device)
        nodes = list(range(d))
        order: List[int] = []
        with torch.no_grad():
            _, s2d = net(Xf_t)
        var_all = s2d.var(dim=0).cpu().numpy()
        for _ in range(d):
            leaf = nodes[int(np.argmin([var_all[i] for i in nodes]))]
            order = [leaf] + order
            nodes.remove(leaf)
        A = _cam_prune(Xf, order, threshold=prune_threshold)
        return {"order": order, "adjacency": A}

    # ---- strict path: retrain after each leaf removal ----
    for step in range(d):
        d_cur = Xi_cur.shape[1]
        net = Net(d_cur)
        net, _ = train_missscore(
            net, Xi_cur, m_cur, mechanism=mechanism, sigma=sigma, omega=omega,
            use_vr=True, epochs=epochs, batch_size=batch_size, lr=lr,
            device=device, verbose=verbose, log_every=max(epochs, 1),
        )

        Xf_cur = Xi_cur.copy(); Xf_cur[m_cur] = 0.0
        Xf_cur_t = torch.tensor(Xf_cur, dtype=torch.float32, device=device)

        if d_cur > 1:
            eval_pts = _bootstrap_eval_points(net, Xf_cur_t, d_cur, device)
        else:
            eval_pts = Xf_cur_t

        # leaf among the *current* columns (local indices 0..d_cur-1)
        local_cols = list(range(d_cur))
        leaf_local = _leaf_by_hessian_variance(net, eval_pts, local_cols)
        leaf_global = active[leaf_local]

        full_order = [leaf_global] + full_order
        del active[leaf_local]

        # drop the leaf column from the working data
        keep = [c for c in range(d_cur) if c != leaf_local]
        Xi_cur = Xi_cur[:, keep]
        m_cur = m_cur[:, keep]

        if verbose:
            print(f"step {step}: removed global node {leaf_global}, "
                  f"order so far = {full_order}")

    Xf_all = X_incomp.copy(); Xf_all[mask] = 0.0
    A = _cam_prune(Xf_all, full_order, threshold=prune_threshold)
    return {"order": full_order, "adjacency": A}
