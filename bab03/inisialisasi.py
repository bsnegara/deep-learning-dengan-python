import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=(1000, 512))          # batch masukan acak  #<1>

def simpangan_per_lapisan(skala, n_lapisan=10, lebar=512):
    h, hasil = x, []
    for _ in range(n_lapisan):
        W = rng.normal(0, skala(lebar), size=(lebar, lebar))  #<2>
        h = np.maximum(h @ W, 0)                              # relu
        hasil.append(h.std())
    return hasil

skema = {
    "terlalu kecil (0,01)": lambda n: 0.01,
    "Glorot sqrt(1/n)": lambda n: np.sqrt(1 / n),  #<3>
    "He sqrt(2/n)": lambda n: np.sqrt(2 / n),
    "terlalu besar (0,1)": lambda n: 0.1,
}
for nama, skala in skema.items():
    s = simpangan_per_lapisan(skala)
    print(f"{nama:22s}", " ".join(f"{v:8.1e}" for v in s[::3]))
