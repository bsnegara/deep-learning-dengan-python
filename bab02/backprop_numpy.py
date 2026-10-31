import numpy as np
from sklearn.datasets import load_digits

# Data: 1.797 citra digit 8x8 (versi mini MNIST yang ikut scikit-learn)
X, y = load_digits(return_X_y=True)
X = X / 16.0                                    # piksel 0..16 -> 0..1
rng = np.random.default_rng(0)
urutan = rng.permutation(len(X))
X, y = X[urutan], y[urutan]
X_latih, y_latih, X_uji, y_uji = X[:1400], y[:1400], X[1400:], y[1400:]

def inisialisasi(masuk=64, sembunyi=32, keluar=10):  #<1>
    return {"W1": rng.normal(0, np.sqrt(2 / masuk), (masuk, sembunyi)),
            "b1": np.zeros(sembunyi),
            "W2": rng.normal(0, np.sqrt(2 / sembunyi),
                             (sembunyi, keluar)),
            "b2": np.zeros(keluar)}

def maju(p, X):  #<2>
    z1 = X @ p["W1"] + p["b1"]
    h = np.maximum(z1, 0)                       # relu
    z2 = h @ p["W2"] + p["b2"]
    z2 = z2 - z2.max(axis=1, keepdims=True)     # stabilitas numerik
    prob = np.exp(z2) / np.exp(z2).sum(axis=1, keepdims=True)
    return prob, (X, z1, h)

def loss_ce(prob, y):
    return -np.mean(np.log(prob[np.arange(len(y)), y] + 1e-12))

def mundur(p, prob, cache, y):  #<3>
    X, z1, h = cache
    n = len(y)
    d_z2 = prob.copy()
    d_z2[np.arange(n), y] -= 1       # softmax+CE: prob - one_hot
    d_z2 /= n
    g = {"W2": h.T @ d_z2, "b2": d_z2.sum(axis=0)}
    d_h = d_z2 @ p["W2"].T
    d_z1 = d_h * (z1 > 0)            # turunan relu
    g["W1"] = X.T @ d_z1
    g["b1"] = d_z1.sum(axis=0)
    return g

# --- Uji gradien dengan beda hingga terpusat  #<4>
p = inisialisasi()
prob, cache = maju(p, X_latih[:20])
g = mundur(p, prob, cache, y_latih[:20])
eps, i, j = 1e-5, 3, 7
p["W1"][i, j] += eps
r_plus = loss_ce(maju(p, X_latih[:20])[0], y_latih[:20])
p["W1"][i, j] -= 2 * eps
r_minus = loss_ce(maju(p, X_latih[:20])[0], y_latih[:20])
p["W1"][i, j] += eps
print(f"gradien analitis: {g['W1'][i, j]:.8f}")
print(f"gradien numerik : {(r_plus - r_minus) / (2 * eps):.8f}")

# --- Pelatihan: SGD mini-batch  #<5>
p = inisialisasi()
for epoch in range(1, 31):
    urutan = rng.permutation(len(X_latih))
    for k in range(0, len(X_latih), 32):
        b = urutan[k:k + 32]
        prob, cache = maju(p, X_latih[b])
        g = mundur(p, prob, cache, y_latih[b])
        for nama in p:
            p[nama] -= 0.1 * g[nama]
    if epoch % 10 == 0:
        r_latih = loss_ce(maju(p, X_latih)[0], y_latih)
        akurasi = (maju(p, X_uji)[0].argmax(axis=1) == y_uji).mean()
        print(f"epoch {epoch:2d}: loss latih {r_latih:.4f}, "
              f"akurasi uji {akurasi:.3f}")
