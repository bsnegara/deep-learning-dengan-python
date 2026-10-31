import numpy as np

T = 1000
beta = np.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)
s = 0.008                                          # jadwal kosinus
f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2) ** 2
ab_kosinus = (f / f[0])[1:]
for t in [250, 500, 750]:  #<1>
    print(f"t={t}: sisa sinyal linear {alpha_bar[t - 1]:.3f}, "
          f"kosinus {ab_kosinus[t - 1]:.3f}")

# Data 1D: campuran dua Gaussian di -2 dan +2 (simpangan baku 0.3)
mu, s0 = np.array([-2.0, 2.0]), 0.3

def eps_optimal(x, t):  #<2>
    """Penghapus derau "sempurna": -sqrt(1-ab) * skor p_t(x)."""
    ab = alpha_bar[t]
    m, v = np.sqrt(ab) * mu, ab * s0 ** 2 + (1 - ab)
    log_w = -0.5 * (x[:, None] - m) ** 2 / v
    w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    skor = (w * (m - x[:, None]) / v).sum(axis=1)
    return -np.sqrt(1 - ab) * skor

def ddpm(n, rng):  #<3>
    x = rng.normal(size=n)
    for t in reversed(range(T)):
        eps = eps_optimal(x, t)
        koef = beta[t] / np.sqrt(1 - alpha_bar[t])
        x = (x - koef * eps) / np.sqrt(alpha[t])
        if t > 0:
            x += np.sqrt(beta[t]) * rng.normal(size=n)
    return x

def ddim(n, rng, langkah=50):  #<4>
    x = rng.normal(size=n)
    jadwal = np.linspace(T - 1, 0, langkah).astype(int)
    for i, t in enumerate(jadwal):
        eps = eps_optimal(x, t)
        x0 = (x - np.sqrt(1 - alpha_bar[t]) * eps) / np.sqrt(alpha_bar[t])
        if i == len(jadwal) - 1:
            return x0
        ab = alpha_bar[jadwal[i + 1]]
        x = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps

rng = np.random.default_rng(0)
for nama, x in [("DDPM 1000 langkah", ddpm(20000, rng)),
                ("DDIM   50 langkah", ddim(20000, rng))]:
    kanan = x[x > 0]
    print(f"{nama}: {np.mean(x > 0):.1%} di mode kanan, "
          f"rerata {kanan.mean():.2f}, simpangan {kanan.std():.2f}")
