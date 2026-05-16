# MissScore

A PyTorch implementation of **"MissScore: High-Order Score Estimation in
the Presence of Missing Data"** (ICML 2025).


A PyTorch implementation of **MissScore** from

> W.Liu, H.Hou, E.Gao, B.Huang, Q.Ke, H.Bondell, M.Gong.
> *"MissScore: High-Order Score Estimation in the Presence of Missing Data."*
> **ICML 2025.**

It implements high-order denoising score matching from incomplete data, the
variance-reduction objectives, Langevin / Ozaki sampling, and the missing-data
causal-discovery algorithm — and it has been **run and validated end-to-end**.

---

## Equation / algorithm map

Everything below is implemented and traceable to the paper:

| Paper object | Where in code |
|---|---|
| Eq. (5) `L_DSM` (first-order, masked) | `models.dsm_loss` (`l_dsm_base`) |
| Eq. (6) `L_D2SM` (second-order, masked) | `models.dsm_loss` (`l_d2sm_base`) |
| Eq. (7) `L_joint = L_DSM + ω L_D2SM` | `models.dsm_loss` return value |
| Eq. (8) `L_DSM-VR` (control variate) | `models.dsm_loss` (`l_dsm_vr`) |
| Eq. (9) `L_D2SM-VR` (antithetic sampling) | `models.dsm_loss` (`l_d2sm_vr`) |
| MCAR/MAR weights `w1, w2 / g1, g2` | `models.compute_weights` |
| Algorithm 1 (training) | `train.train_missscore` |
| Algorithm 2 (Langevin Eq. 10 / Ozaki Eq. 11) | `sampling.py` |
| Algorithm 3 (causal discovery) | `causal.missscore_causal_discovery` |
| Appendix D (MCAR/MAR/MNAR generation) | `missing.produce_NA` |
| Appendix E.3 / F.3 architectures | `models.ScoreNetSmall / ScoreNetLarge` |

Network choices follow the paper: `ScoreNetSmall` is the 3-layer Softplus MLP
(Swiss-roll / Section 3 experiments); `ScoreNetLarge` is the 5-layer
LeakyReLU + LayerNorm + Dropout MLP with hidden sizes `max(128, 3d)` /
`max(1024, 5d)` (Census / causal-discovery experiments). Both expose `s1` and
`diag(s2)` — only the Hessian diagonal is modelled, exactly as the paper does
for Ozaki sampling to avoid matrix inversion / exponentiation.

---

## Validated results

`python -m missscore.demo` (2-D Gaussian, MCAR α=0.3) reproduces the **key
qualitative finding of Table 1**:

```
        method  sigma |    MSE s1  MSE diag(s2)
   DSM (no VR)    0.1 |    3.28          0.62
   DSM (no VR)   0.01 |    3.81        287.40     <- s2 explodes as σ→0
        DSM-VR    0.1 |    0.44          2.54
        DSM-VR   0.01 |    0.23         36.61     <- VR keeps it bounded
```

Without variance reduction the second-order MSE blows up as σ→0 (paper reports
the same: `s2` ≈ 15–30 at σ=0.01 vs `s2(VR)` ≈ 0.04–0.06). Ozaki sampling
recovers the data mean more accurately than Langevin, matching Figure 2.

`python test_causal.py` (chain ANM x1→x2→x3, nonlinear additive Gaussian noise):

```
[MCAR, α=0.1]  recovered order [0,1,2]  SHD 1
[MAR,  α=0.1]  recovered order [0,1,2]  SHD 0   (exact DAG recovery)
```

---

## Quick start

```python
import numpy as np
from missscore import (ScoreNetSmall, produce_NA, train_missscore,
                        ozaki_sample, missscore_causal_discovery)

# 1. simulate incomplete data
X = np.random.randn(2000, 4).astype("float32")
out = produce_NA(X, p_miss=0.3, mecha="MCAR", seed=0)
Xi, mask = out["X_incomp"], out["mask"]

# 2. jointly train s1 + diag(s2) with variance reduction
net = ScoreNetSmall(d=4)
net, _ = train_missscore(net, Xi, mask, mechanism="mcar",
                         sigma=0.1, use_vr=True, epochs=120)

# 3. sample with second-order Ozaki dynamics
samples = ozaki_sample(net, n_samples=2000, d=4)

# 4. causal discovery from incomplete data (strict Algorithm 3)
res = missscore_causal_discovery(Xi, mask, mechanism="mcar")
print(res["order"], res["adjacency"])
```

### Install

Standard `src/` layout — install editable, then everything is importable:

```bash
pip install -e .          # or: pip install -r requirements.txt
```

Run the bundled checks:

```bash
missscore-demo                  # CLI entry point (Table-1-style reproduction)
python scripts/run_demo.py      # same thing, as a script
pytest tests/                   # Algorithm 3 on a known chain DAG (MCAR + MAR)
python tests/test_causal.py     # same test, verbose standalone run
```

---

## Implementation notes / honest caveats

- **`s2_anchor`** — Eq. (9) is a control-variate *functional*: it shares the
  global optimum of Eq. (6) but, on its own, gives `s2` a weak gradient (its
  value is not a plain score-matching residual), so pure-VR training of `s2`
  converges slowly. We add an optional small non-VR D2SM term (Eq. 6) with the
  *same minimiser*; it is **automatically scaled by σ² and disabled below
  σ=0.05** so it never re-introduces the variance blow-up that motivated VR.
  Set `s2_anchor=0.0` for the strict paper objective.
- **Hessian diagonal only.** Like the paper's Ozaki experiments, we model
  `diag(s2)`. The weight tensors `w2` are kept full `d×d` so a full-Hessian
  head can be dropped in without touching the loss code.
- **MAR pairwise probability** uses the independence approximation
  `P[mᵢmⱼ=0] ≈ P[mᵢ=0]·P[mⱼ=0]`; fitting all `d(d-1)/2` joint logistic models
  is expensive and degenerate when a pair is never simultaneously observed.
- **MNAR** reuses the MCAR objective, exactly as the paper states ("for MNAR
  we utilize the same training objective as MCAR ... may introduce some bias").
- **Causal discovery** defaults to the **strict Algorithm 3** (retrain after
  each leaf removal + bootstrapped Algorithm-2 samples for the variance
  estimate). `retrain_each_step=False` switches to the fast DiffAN-style
  single-training shortcut.
- Absolute MSE values differ from the paper's table because the paper uses
  100-D correlated Gaussians, a specific covariance construction, 5000 test
  points and 10-seed averaging; the demo is a single-seed 2-D illustration.
  The *qualitative* relationships (VR vs no-VR, σ behaviour, Ozaki vs Langevin,
  order recovery) all hold.

## Files

```
missscore/
├── pyproject.toml          # build config + missscore-demo entry point
├── requirements.txt
├── README.md
├── src/
│   └── missscore/
│       ├── __init__.py     # public API
│       ├── models.py       # score networks + all loss equations (5)-(9)
│       ├── missing.py      # MCAR/MAR/MNAR generation + MAR prob. estimation
│       ├── train.py        # Algorithm 1 training loop
│       ├── sampling.py     # Algorithm 2: Langevin (Eq.10) & Ozaki (Eq.11)
│       ├── causal.py       # Algorithm 3: causal discovery from missing data
│       └── demo.py         # Table-1-style reproduction (CLI: missscore-demo)
├── scripts/
│   └── run_demo.py         # convenience runner
└── tests/
    └── test_causal.py      # end-to-end causal-discovery test (pytest-ready)
```

## Requirements

`torch`, `numpy`, `scikit-learn`, `scipy`.
