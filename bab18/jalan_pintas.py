import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)

def buat(n, korelasi):  #<1>
    """Fitur "isi" bermakna tetapi berderau; fitur "latar" = label
    dengan peluang "korelasi" (mis. sapi hampir selalu di padang rumput)."""
    y = rng.integers(0, 2, n)
    isi = y + rng.normal(0, 1.0, n)            # sinyal asli, berderau
    latar = np.where(rng.random(n) < korelasi, y, 1 - y)
    latar = latar + rng.normal(0, 0.1, n)      # sinyal palsu, bersih
    return np.c_[isi, latar], y

X, y = buat(5000, korelasi=0.95)
model = LogisticRegression().fit(X, y)
print("bobot [isi, latar]:", np.round(model.coef_[0], 2))  #<2>
for nama, k in [("uji sumber sama (95%)", 0.95),
                ("uji tempat baru (50%)", 0.50),
                ("uji terbalik (5%)", 0.05)]:
    Xu, yu = buat(5000, k)
    print(f"{nama}: akurasi {model.score(Xu, yu):.3f}")
hanya_isi = LogisticRegression().fit(X[:, :1], y)  #<3>
Xu, yu = buat(5000, 0.05)
print(f"model tanpa fitur latar, uji terbalik: "
      f"{hanya_isi.score(Xu[:, :1], yu):.3f}")
