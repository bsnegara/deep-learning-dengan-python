import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import (average_precision_score, precision_score,
                             roc_auc_score)

rng = np.random.default_rng(0)

def data(n_pos, n_neg):  #<1>
    """Skor "model" yang sama: logit positif ~ N(2,1), negatif ~ N(0,1)."""
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
    logit = np.r_[rng.normal(2, 1, n_pos), rng.normal(0, 1, n_neg)]
    return y, logit

for prev in [0.5, 0.1, 0.01]:
    n_pos = int(20000 * prev)
    y, z = data(n_pos, 20000 - n_pos)
    p = 1 / (1 + np.exp(-(z - 1)))       # ambang di tengah dua mode
    print(f"prevalensi {prev:>4}: AUROC {roc_auc_score(y, p):.3f}  "
          f"AUPRC {average_precision_score(y, p):.3f}  "
          f"presisi@0.5 {precision_score(y, p > 0.5):.3f}")

def ece(y, p, m=10):  #<2>
    keyakinan = np.maximum(p, 1 - p)
    benar = (p > 0.5) == y
    kelompok = np.minimum((keyakinan - 0.5) * 2 * m, m - 1).astype(int)
    total = 0.0
    for b in range(m):
        k = kelompok == b
        if k.any():
            total += k.mean() * abs(benar[k].mean() - keyakinan[k].mean())
    return total

y, z = data(5000, 5000)
z_yakin = 3 * (z - 1)                    # model "terlalu yakin": logit x3
idx = rng.permutation(len(y))
val, uji = idx[:5000], idx[5000:]

def nll(T, i):  #<3>
    p = 1 / (1 + np.exp(-z_yakin[i] / T))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y[i] * np.log(p) + (1 - y[i]) * np.log(1 - p))

T = minimize_scalar(lambda t: nll(t, val), bounds=(0.1, 10),
                    method="bounded").x
for nama, suhu in [("sebelum", 1.0), ("sesudah", T)]:
    p = 1 / (1 + np.exp(-z_yakin[uji] / suhu))
    print(f"{nama}: T={suhu:.2f}  ECE {ece(y[uji], p):.3f}  "
          f"Brier {np.mean((p - y[uji]) ** 2):.3f}  "
          f"AUROC {roc_auc_score(y[uji], p):.3f}")
