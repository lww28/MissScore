"""End-to-end test of MissScore causal discovery on a known chain ANM."""
import numpy as np
from missscore.missing import produce_NA
from missscore.causal import missscore_causal_discovery


def make_chain_anm(n=1000, seed=0):
    """x1 -> x2 -> x3, nonlinear additive Gaussian noise. True order: x1,x2,x3."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3)) * 0.5
    x1 = z[:, 0]
    x2 = np.sin(2.0 * x1) + z[:, 1]
    x3 = 0.5 * x2 ** 2 - x2 + z[:, 2]
    X = np.stack([x1, x2, x3], axis=1).astype(np.float32)
    # True adjacency: 0->1, 1->2
    A_true = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 0]])
    return X, A_true


def shd(A_pred, A_true):
    return int(np.sum(A_pred != A_true))


if __name__ == "__main__":
    np.random.seed(0)
    X, A_true = make_chain_anm(1000, seed=0)
    print("True DAG (chain x1->x2->x3). Correct topological order = [0, 1, 2]")
    print("True adjacency:\n", A_true)

    for mecha in ("MCAR", "MAR"):
        out = produce_NA(X, p_miss=0.1, mecha=mecha, seed=1)
        Xi, m = out["X_incomp"], out["mask"]
        res = missscore_causal_discovery(
            Xi, m,
            mechanism=mecha.lower(),
            sigma=0.1, omega=1.0,
            epochs=80, batch_size=128, lr=1e-3,
            prune_threshold=0.05,
            verbose=False,
        )
        print(f"\n[{mecha}, alpha=0.1]  (missing ratio {m.mean():.2f})")
        print("  Recovered order    :", res["order"])
        print("  Recovered adjacency:\n", res["adjacency"])
        print("  SHD vs true        :", shd(res["adjacency"], A_true))
