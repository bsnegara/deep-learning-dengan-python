import numpy as np

rng = np.random.default_rng(1)
f = lambda x: np.sin(2 * np.pi * x)                  # fungsi sebenarnya
x_latih = np.sort(rng.uniform(0, 1, 15))
y_latih = f(x_latih) + rng.normal(0, 0.25, 15)  # berderau  #<1>
x_uji = rng.uniform(0, 1, 500)
y_uji = f(x_uji) + rng.normal(0, 0.25, 500)

def fitur(x, derajat):
    return np.vander(x, derajat + 1, increasing=True)  # 1, x, x^2, ...

def latih(derajat, lam=0.0):  #<2>
    X = fitur(x_latih, derajat)
    I = np.eye(X.shape[1])
    I[0, 0] = 0                           # bias tidak dipenalti
    return np.linalg.solve(X.T @ X + lam * I, X.T @ y_latih)

def mse(w, x, y, derajat):
    return np.mean((fitur(x, derajat) @ w - y) ** 2)

print("derajat  lambda   MSE latih  MSE uji   |w| maks")
konfigurasi = [(1, 0), (3, 0), (12, 0), (12, 1e-4), (12, 1e-2)]  #<3>
for derajat, lam in konfigurasi:
    w = latih(derajat, lam)
    m_latih = mse(w, x_latih, y_latih, derajat)
    m_uji = mse(w, x_uji, y_uji, derajat)
    print(f"{derajat:7d}  {lam:6.0e}  {m_latih:9.4f}  {m_uji:8.4f}"
          f"  {np.abs(w).max():9.1f}")
