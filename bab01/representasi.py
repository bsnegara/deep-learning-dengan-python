import numpy as np

rng = np.random.default_rng(0)

# --- Kasus 1: dua kelas dipisahkan garis diagonal x + y = 0
X = rng.uniform(-2, 2, size=(200, 2))  #<1>
y = (X[:, 0] + X[:, 1] > 0).astype(int)

sudut = np.pi / 4  #<2>
R = np.array([[np.cos(sudut), np.sin(sudut)],
              [-np.sin(sudut), np.cos(sudut)]])
X_baru = X @ R.T  # koordinat setelah sumbu diputar 45 derajat

prediksi = (X_baru[:, 0] > 0).astype(int)  #<3>
print("Kasus 1, akurasi aturan x_baru > 0:", (prediksi == y).mean())

# --- Kasus 2: dua lingkaran konsentris
n = 200
y2 = rng.integers(0, 2, n)
r_dalam, r_luar = rng.uniform(0.0, 1.0, n), rng.uniform(1.5, 2.5, n)
r = np.where(y2 == 1, r_dalam, r_luar)
theta = rng.uniform(0, 2 * np.pi, n)
X2 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

jari_jari = np.sqrt((X2 ** 2).sum(axis=1))  #<4>
prediksi2 = (jari_jari < 1.25).astype(int)
print("Kasus 2, akurasi aturan r < 1.25 :", (prediksi2 == y2).mean())
