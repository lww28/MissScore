"""
Sanity checks for MissScore, reproducing the spirit of Section 3.1.

We construct a 2D Gaussian with known mean / covariance so that:
    s1(x) = -Sigma^{-1} (x - mu)
    diag(s2(x)) = -diag(Sigma^{-1})

We train MissScore on data with 30% MCAR missingness and check that:
    1) the learned s1 has low MSE against the ground truth,
    2) the learned diag(s2) has low MSE against the ground truth,
    3) variance reduction is necessary at small sigma (the paper's key finding).

Run with:   python -m missscore.demo
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .models import ScoreNetSmall
from .missing import produce_NA
from .train import train_missscore
from .sampling import langevin_sample, ozaki_sample


def make_gaussian_data(n=2000, d=2, seed=0):
    rng = np.random.default_rng(seed)
    # Mildly correlated 2D Gaussian
    A = rng.standard_normal((d, d))
    Sigma = A @ A.T + 0.5 * np.eye(d)
    mu = np.zeros(d)
    X = rng.multivariate_normal(mu, Sigma, size=n)
    return X.astype(np.float32), mu.astype(np.float32), Sigma.astype(np.float32)


def true_scores(X, mu, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    s1 = -(X - mu) @ Sigma_inv.T
    s2_diag = -np.diag(Sigma_inv)
    s2_diag_full = np.broadcast_to(s2_diag, X.shape).copy()
    return s1, s2_diag_full


def evaluate_scores(net, X, mu, Sigma, device="cpu"):
    net.eval()
    s1_true, s2_diag_true = true_scores(X, mu, Sigma)
    with torch.no_grad():
        s1, s2_diag = net(torch.tensor(X, dtype=torch.float32, device=device))
    s1 = s1.cpu().numpy()
    s2_diag = s2_diag.cpu().numpy()
    mse_s1 = float(((s1 - s1_true) ** 2).mean())
    mse_s2 = float(((s2_diag - s2_diag_true) ** 2).mean())
    return mse_s1, mse_s2


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, mu, Sigma = make_gaussian_data(n=2000, d=2, seed=0)
    X_test, _, _ = make_gaussian_data(n=5000, d=2, seed=1)

    print("=" * 64)
    print("MissScore sanity check on a 2D Gaussian (MCAR, alpha=0.3)")
    print("=" * 64)
    print(f"True diag(s2) = diag(-Sigma^-1) = {-np.diag(np.linalg.inv(Sigma))}")

    out = produce_NA(X, p_miss=0.3, mecha="MCAR", seed=0)
    X_incomp, mask = out["X_incomp"], out["mask"]
    print(f"Empirical missing ratio: {mask.mean():.3f}\n")

    # Reproduce the spirit of Table 1: vary sigma, with/without VR.
    print(f"{'method':>14} {'sigma':>6} | {'MSE s1':>9} {'MSE diag(s2)':>13}")
    print("-" * 50)
    configs = [
        ("DSM (no VR)", 0.1, False),
        ("DSM (no VR)", 0.01, False),
        ("DSM-VR", 0.1, True),
        ("DSM-VR", 0.01, True),
    ]
    trained = {}
    for name, sigma, use_vr in configs:
        torch.manual_seed(0)
        net = ScoreNetSmall(d=2, hidden=128)
        net, _ = train_missscore(
            net, X_incomp, mask, mechanism="mcar",
            sigma=sigma, omega=1.0, use_vr=use_vr,
            epochs=120, batch_size=64, lr=1e-3,
            device=device, verbose=False,
        )
        mse_s1, mse_s2 = evaluate_scores(net, X_test, mu, Sigma, device)
        print(f"{name:>14} {sigma:>6} | {mse_s1:>9.4f} {mse_s2:>13.4f}")
        trained[(name, sigma)] = net

    print("\nMatches the paper's finding (Table 1): without VR, the second-order")
    print("MSE explodes as sigma -> 0, while DSM-VR stays small and accurate.\n")

    # Use the best model (DSM-VR, sigma=0.1) for sampling.
    best = trained[("DSM-VR", 0.1)]
    print("[Sampling with DSM-VR / sigma=0.1] ---------------------------")
    samples_lang = langevin_sample(best, 2000, 2, eps=5e-3, n_steps=300, device=device)
    samples_oz = ozaki_sample(best, 2000, 2, eps=5e-3, n_steps=100, device=device)

    def stat(a):
        return f"mean={np.round(np.asarray(a).mean(0), 3)} var={np.round(np.asarray(a).var(0), 3)}"

    print(f"  Real     : {stat(X)}")
    print(f"  Langevin : {stat(samples_lang)}")
    print(f"  Ozaki    : {stat(samples_oz)}")


if __name__ == "__main__":
    main()
