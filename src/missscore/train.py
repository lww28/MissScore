"""
Training loop for MissScore (Algorithm 1).

  Input : observed data x_obs (with NaNs), score model, noise level sigma,
          weight omega for the second-order term.
  repeat:
      Sample noise z ~ N(0, I)
      Compute perturbed data x~_obs = x_obs + sigma z
      If MAR: refit logistic regressions for P[m=0|x] and P[mm^T=0|x]
      Update theta by gradient descent on  L_DSM + omega L_D2SM (Eq. 7)
  until convergence
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import dsm_loss
from .missing import estimate_mar_probabilities


def fill_missing(X_incomp: np.ndarray, mask: np.ndarray, fill_value: float = 0.0):
    """Replace NaNs with a constant (paper uses 0 for continuous variables)."""
    X_filled = X_incomp.copy()
    X_filled[mask] = fill_value
    return X_filled


def train_missscore(
    score_net: torch.nn.Module,
    X_incomp: np.ndarray,
    mask: np.ndarray,
    mechanism: str = "mcar",
    sigma: float = 0.1,
    omega: float = 1.0,
    use_vr: bool = True,
    s2_anchor: float = 0.05,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
    log_every: int = 10,
):
    """
    Train s1 and s2 jointly with Eq. (7) on incomplete data.

    Args mirror the paper's defaults: sigma=0.1, lr=1e-3, omega=1.
    Set use_vr=True for sigma -> 0 stability (essential per Section 3.1).

    Returns the trained score_net.
    """
    score_net = score_net.to(device)
    opt = torch.optim.Adam(score_net.parameters(), lr=lr)

    # Fill NaNs with 0 (paper's choice; the mask zeros them out in the loss).
    X_filled = fill_missing(X_incomp, mask, fill_value=0.0)

    # For MAR we estimate P[m=0|x] once on the filled data
    # (the paper refits inside the loop if missingness depends on the perturbed
    # data; for tabular MAR the static fit is what they actually use).
    p_obs_full = None
    p_pair_obs_full = None
    if mechanism == "mar":
        p_obs_full, p_pair_obs_full = estimate_mar_probabilities(X_filled, mask)
        p_obs_full = torch.tensor(p_obs_full, dtype=torch.float32, device=device)
        p_pair_obs_full = torch.tensor(p_pair_obs_full, dtype=torch.float32, device=device)

    X_t = torch.tensor(X_filled, dtype=torch.float32, device=device)
    m_t = torch.tensor(mask.astype(np.float32), device=device)

    n = X_t.shape[0]
    idx_t = torch.arange(n, device=device)

    if mechanism == "mar":
        ds = TensorDataset(X_t, m_t, p_obs_full, p_pair_obs_full, idx_t)
    else:
        ds = TensorDataset(X_t, m_t, idx_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    score_net.train()
    history = []
    for ep in range(epochs):
        running = 0.0
        n_batches = 0
        for batch in loader:
            if mechanism == "mar":
                xb, mb, pb, ppb, _ = batch
            else:
                xb, mb, _ = batch
                pb = ppb = None

            opt.zero_grad()
            loss = dsm_loss(
                score_net=score_net,
                x_filled=xb,
                mask=mb,
                sigma=sigma,
                mechanism=mechanism,
                p_obs=pb,
                p_pair_obs=ppb,
                use_vr=use_vr,
                omega=omega,
                s2_anchor=s2_anchor,
            )
            loss.backward()
            # gentle gradient clipping for stability at small sigma
            torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=10.0)
            opt.step()
            running += loss.item()
            n_batches += 1
        avg = running / max(n_batches, 1)
        history.append(avg)
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"epoch {ep:4d} | L_joint = {avg:.4f}")

    return score_net, history
