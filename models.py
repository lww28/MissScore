"""
MissScore: High-Order Score Estimation in the Presence of Missing Data

Faithful implementation of the score networks and training objectives from the paper.

Key equations implemented here:
  - Eq. (5)  L_DSM      : first-order DSM loss with missing mask
  - Eq. (6)  L_D2SM     : second-order DSM loss with missing mask
  - Eq. (7)  L_joint    : multi-task joint training objective
  - Eq. (8)  L_DSM-VR   : variance-reduced first-order DSM
  - Eq. (9)  L_D2SM-VR  : variance-reduced second-order DSM (antithetic sampling)

Weights:
  - MCAR : g1 = (1 - m),                 g2 = (1 - m)(1 - m)^T
  - MAR  : g1 = (1 - m)/sqrt(P[m=0|x]),  g2 = (1 - m)(1 - m)^T / sqrt(P[mm^T=0|x])
  - MNAR : we fall back to the MCAR objective (paper choice; may be biased)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Score networks
# ---------------------------------------------------------------------------
class ScoreNetSmall(nn.Module):
    """
    Lightweight 3-layer MLP with Softplus activation, used in the Swiss-roll /
    low-d synthetic experiments of Section 3 / Section E (Appendix).
    Outputs concatenated first-order score (d,) and the diagonal of the
    second-order score (d,) -- we only model the diagonal of s2, as the paper
    does in the Ozaki sampling experiments.
    """

    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.d = d
        self.backbone = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
        )
        self.head_s1 = nn.Linear(hidden, d)
        # Output the diagonal of the Hessian (matches Appendix E "we only use
        # the diagonal of s2 ... to avoid inversion / exponentiation costs").
        self.head_s2_diag = nn.Linear(hidden, d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        s1 = self.head_s1(h)
        s2_diag = self.head_s2_diag(h)
        return s1, s2_diag


class ScoreNetLarge(nn.Module):
    """
    5-layer MLP with LeakyReLU + LayerNorm + Dropout(0.2) on the first layer,
    as described in Appendix E.3 and F.3.

      - first two layers : hidden = max(128, 3 * d)
      - last three layers: hidden = max(1024, 5 * d)
    """

    def __init__(self, d: int):
        super().__init__()
        h_small = max(128, 3 * d)
        h_large = max(1024, 5 * d)

        self.l1 = nn.Linear(d, h_small)
        self.ln1 = nn.LayerNorm(h_small)
        self.drop1 = nn.Dropout(0.2)

        self.l2 = nn.Linear(h_small, h_small)
        self.ln2 = nn.LayerNorm(h_small)

        self.l3 = nn.Linear(h_small, h_large)
        self.ln3 = nn.LayerNorm(h_large)

        self.l4 = nn.Linear(h_large, h_large)
        self.ln4 = nn.LayerNorm(h_large)

        self.act = nn.LeakyReLU(0.2)

        # Output heads: s1 (d,) and diagonal of s2 (d,)
        self.head_s1 = nn.Linear(h_large, d)
        self.head_s2_diag = nn.Linear(h_large, d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.drop1(self.act(self.ln1(self.l1(x))))
        h = self.act(self.ln2(self.l2(h)))
        h = self.act(self.ln3(self.l3(h)))
        h = self.act(self.ln4(self.l4(h)))
        return self.head_s1(h), self.head_s2_diag(h)


# ---------------------------------------------------------------------------
# Weight construction for the three missing mechanisms
# ---------------------------------------------------------------------------
def compute_weights(
    mask: torch.Tensor,
    mechanism: str,
    p_obs: Optional[torch.Tensor] = None,
    p_pair_obs: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the per-element weights for the DSM and D2SM objectives.

    Args:
        mask     : (B, d) binary mask -- 1 = missing, 0 = observed.
        mechanism: 'mcar' | 'mar' | 'mnar'.
        p_obs    : (B, d) estimated P[m_i = 0 | x_obs] -- only for MAR.
        p_pair_obs: (B, d, d) estimated P[m_i m_j = 0 | x_obs] -- only for MAR.

    Returns:
        w1 : (B, d)     for first-order objective
        w2 : (B, d, d)  for second-order objective (we keep the full d x d weight
                       even though our s2 head only models the diagonal -- this
                       lets you slot in a full-Hessian head later without changes)

    Notes on the paper:
      - Eq. (5):  w1 = (1 - m) under MCAR
                  w1 = (1 - m) / sqrt(P[m=0|x_obs]) under MAR
      - Eq. (6):  w2 = (1 - m)(1 - m)^T under MCAR
                  w2 = (1 - m)(1 - m)^T / sqrt(P[m m^T = 0|x_obs]) under MAR
      - MNAR:     paper uses the MCAR objective as a (biased) approximation.
    """
    obs = (1.0 - mask).float()  # 1 = observed

    if mechanism == "mcar" or mechanism == "mnar":
        w1 = obs
        w2 = obs.unsqueeze(2) * obs.unsqueeze(1)  # (B, d, d)
        return w1, w2

    if mechanism == "mar":
        if p_obs is None or p_pair_obs is None:
            raise ValueError("MAR requires p_obs and p_pair_obs (estimated via logistic regression).")
        w1 = obs / torch.sqrt(p_obs.clamp_min(eps))
        w2 = (obs.unsqueeze(2) * obs.unsqueeze(1)) / torch.sqrt(p_pair_obs.clamp_min(eps))
        return w1, w2

    raise ValueError(f"Unknown mechanism: {mechanism}")


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def dsm_loss(
    score_net: nn.Module,
    x_filled: torch.Tensor,
    mask: torch.Tensor,
    sigma: float,
    mechanism: str = "mcar",
    p_obs: Optional[torch.Tensor] = None,
    p_pair_obs: Optional[torch.Tensor] = None,
    use_vr: bool = True,
    omega: float = 1.0,
    s2_anchor: float = 0.0,
) -> torch.Tensor:
    """
    Joint training objective L_joint = L_DSM + omega * L_D2SM (Eq. 7).

    When use_vr=True we use the variance-reduced versions Eq. (8) and Eq. (9),
    which the paper shows are essential as sigma -> 0.

    Args:
        score_net : returns (s1, diag(s2)) given the perturbed observation.
        x_filled  : (B, d) observed data with NaNs filled in (e.g. 0 for
                    continuous, a new category for discrete).
        mask      : (B, d) 1 = missing, 0 = observed.
        sigma     : noise level for the Gaussian perturbation x~|x ~ N(x, sigma^2 I).
        mechanism : 'mcar' | 'mar' | 'mnar'.
        p_obs, p_pair_obs : only used for MAR (see compute_weights).
        use_vr    : whether to use the variance-reduced version.
        omega     : weight on the second-order loss (Eq. 7).
        s2_anchor : when use_vr=True, optionally mix in a small-weight,
                    *non-VR* D2SM residual term (Eq. 6). The VR objective
                    Eq. (9) is a control-variate functional that shares the
                    same global optimum as Eq. (6) but is slow to drive s2 on
                    its own (its value is not a plain score-matching residual).
                    For sigma not extremely small, adding a small s2_anchor *
                    L_D2SM (Eq. 6) keeps the same minimiser while greatly
                    speeding up / stabilising s2 convergence. Set to 0.0 to
                    use the strict paper objective only.
    """
    obs = (1.0 - mask).float()  # observed indicator

    # Antithetic noise sample z used by both s1 and s2 (matches Eq. 9).
    z = torch.randn_like(x_filled)
    x_plus = x_filled + sigma * z   # x~+
    x_minus = x_filled - sigma * z  # x~-

    # Weights for the (non-VR) objectives
    w1, w2 = compute_weights(mask, mechanism, p_obs, p_pair_obs)
    w2_diag = torch.diagonal(w2, dim1=-2, dim2=-1)  # (B, d) -- we model diag(s2)

    # ----- first-order loss -----
    # Eq. (5): || (s1(x~) + (x~ - x) / sigma^2) * w1 ||^2
    # Note we mask multiplicatively so that "filled" missing entries (set to 0)
    # do not contaminate the residual: (x~ - x_filled) is well-defined because
    # the residual at missing positions is killed by (1 - m) anyway.
    s1_plus, s2_diag_plus = score_net(x_plus)
    s1_minus, s2_diag_minus = score_net(x_minus)
    s1_center, s2_diag_center = score_net(x_filled)

    # For the non-VR first-order term we just use x_plus.
    residual_s1 = s1_plus + (x_plus - x_filled) / (sigma ** 2)
    l_dsm_base = ((residual_s1 * w1) ** 2).sum(dim=1).mean()

    # ----- second-order loss (diagonal only) -----
    # psi_ii(x~) = s2_ii(x~) + s1_i(x~)^2   (from psi = s2 + s1 s1^T)
    # phi_ii    = I_ii - z_i z_i = 1 - z_i^2
    # Eq. (6): || (psi_ii + phi_ii / sigma^2) * w2_ii ||^2
    psi_plus = s2_diag_plus + s1_plus ** 2
    phi_diag_plus = 1.0 - z ** 2  # since z is the *same* z used for x_plus
    residual_s2 = psi_plus + phi_diag_plus / (sigma ** 2)
    l_d2sm_base = ((residual_s2 * w2_diag) ** 2).sum(dim=1).mean()

    if not use_vr:
        return l_dsm_base + omega * l_d2sm_base

    # ============================================================
    # Variance-reduced versions
    # ============================================================
    # Eq. (8): L_DSM-VR = L_DSM - E[ (2/sigma) s1(x_obs;theta)^T z * g1  +
    #                                ||z * g1||^2 / sigma^2 ]
    # where g1 = (1 - m) (MCAR) or (1 - m)/sqrt(P[m=0|x]) (MAR).
    # We compute s1 at x_filled (the "x_obs" in the paper) for the control variate,
    # as the derivation is around x (not x~).
    g1 = w1  # same vector
    cv1 = (2.0 / sigma) * (s1_center * z * g1).sum(dim=1) \
          + ((z * g1) ** 2).sum(dim=1) / (sigma ** 2)
    l_dsm_vr = l_dsm_base - cv1.mean()

    # Eq. (9): L_D2SM-VR = E[ ( psi(x~+)^2 + psi(x~-)^2
    #                          + 2 (I - z z^T)/sigma * Psi ) * g2 ]
    # where Psi = psi(x~+) + psi(x~-) - 2 psi(x).
    # We implement the *diagonal* version (matches our s2 head).
    psi_diag_plus = s2_diag_plus + s1_plus ** 2
    psi_diag_minus = s2_diag_minus + s1_minus ** 2
    psi_diag_center = s2_diag_center + s1_center ** 2
    Psi_diag = psi_diag_plus + psi_diag_minus - 2.0 * psi_diag_center

    # diag(I - z z^T) = 1 - z^2
    phi_diag = 1.0 - z ** 2
    # NB: in the paper the cross-term is (I - z z^T)/sigma, *not* /sigma^2,
    # because antithetic sampling cancels the 1/sigma^2 term that motivated VR.
    integrand = (
        psi_diag_plus ** 2
        + psi_diag_minus ** 2
        + 2.0 * (phi_diag / sigma) * Psi_diag
    )
    l_d2sm_vr = (integrand * w2_diag).sum(dim=1).mean()

    total = l_dsm_vr + omega * l_d2sm_vr
    if s2_anchor > 0.0:
        # Optional non-VR D2SM anchor (Eq. 6). Same global optimum as Eq. (9);
        # it just gives s2 a well-conditioned gradient signal when sigma is not
        # tiny. CRUCIAL: the non-VR residual Eq. (6) itself has the 1/sigma^2
        # variance blow-up that motivated VR in the first place, so we must
        # damp the anchor as sigma -> 0. We scale it by sigma^2 (the inverse
        # of the blow-up factor) and hard-disable it below sigma = 0.05.
        if sigma >= 0.05:
            anchor_scaled = s2_anchor * (sigma ** 2) / (0.1 ** 2)
            total = total + anchor_scaled * l_d2sm_base
    return total
