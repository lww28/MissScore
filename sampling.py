"""
Sampling from a trained MissScore model (Algorithm 2).

  - Langevin dynamics (Eq. 10):
        x_{t+1} = x_t + (eps/2) s1(x_t) + sqrt(eps) z_t

  - Ozaki discretisation (Eq. 11), using only diag(s2) as the paper does:
        M_{t-1} = (exp(eps s2) - I) s2^{-1}
        Sigma_{t-1} = (exp(2 eps s2) - I) s2^{-1}     # diagonal version
        x_{t+1} = x_t + M_{t-1} s1(x_t) + Sigma^{1/2} z_t

Working with the diagonal of s2 avoids inverting / exponentiating a d x d
matrix, which is exactly the simplification the paper makes in Appendix E.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


@torch.no_grad()
def langevin_sample(
    score_net: torch.nn.Module,
    n_samples: int,
    d: int,
    eps: float = 5e-3,
    n_steps: int = 300,
    init: Optional[torch.Tensor] = None,
    device: str = "cpu",
):
    """Plain Langevin dynamics using the first-order score (Eq. 10)."""
    score_net.eval()
    if init is None:
        x = torch.randn(n_samples, d, device=device)
    else:
        x = init.to(device).clone()
    for _ in range(n_steps):
        s1, _ = score_net(x)
        z = torch.randn_like(x)
        x = x + 0.5 * eps * s1 + math.sqrt(eps) * z
    return x.cpu()


@torch.no_grad()
def ozaki_sample(
    score_net: torch.nn.Module,
    n_samples: int,
    d: int,
    eps: float = 5e-3,
    n_steps: int = 100,
    init: Optional[torch.Tensor] = None,
    device: str = "cpu",
    s2_floor: float = -50.0,
    s2_ceil: float = -1e-3,
):
    """
    Ozaki sampling with the diagonal of s2.

    For the diagonal of a *log-concave* density, s2_ii is negative. To keep
    the matrix exponential and the square root well-defined we clamp s2 into
    [s2_floor, s2_ceil] (the paper notes the same numerical issue in low-noise
    regimes; clamping is the standard fix in their reference implementation).
    """
    score_net.eval()
    if init is None:
        x = torch.randn(n_samples, d, device=device)
    else:
        x = init.to(device).clone()

    for _ in range(n_steps):
        s1, s2_diag = score_net(x)
        # Negative-definite enforcement on the diagonal
        s2_diag = s2_diag.clamp(min=s2_floor, max=s2_ceil)

        # M = (exp(eps * s2) - I) * s2^{-1}, all element-wise on the diagonal
        exp_eps_s2 = torch.exp(eps * s2_diag)
        M_diag = (exp_eps_s2 - 1.0) / s2_diag

        # Sigma = (exp(2 eps s2) - I) * s2^{-1}  (must be >= 0 -- it is, since
        # both factors are negative when s2 < 0)
        exp_2eps_s2 = torch.exp(2.0 * eps * s2_diag)
        Sigma_diag = (exp_2eps_s2 - 1.0) / s2_diag
        Sigma_diag = Sigma_diag.clamp_min(1e-12)
        Sigma_sqrt = torch.sqrt(Sigma_diag)

        z = torch.randn_like(x)
        x = x + M_diag * s1 + Sigma_sqrt * z

    return x.cpu()
