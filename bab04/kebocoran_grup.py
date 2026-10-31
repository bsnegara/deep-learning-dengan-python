import numpy as np
from sklearn.model_selection import (
    train_test_split, GroupShuffleSplit, StratifiedShuffleSplit)
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(0)
# 60 orang, masing-masing 20 foto. Label ditetapkan ACAK per orang,
# jadi sebenarnya tidak ada pola yang dapat digeneralisasi.  #<1>
n_orang, per_orang = 60, 20
gaya = rng.normal(size=(n_orang, 10))        # ciri khas tiap orang
label_orang = rng.integers(0, 2, n_orang)
grup = np.repeat(np.arange(n_orang), per_orang)
X = gaya[grup] + 0.1 * rng.normal(size=(len(grup), 10))
y = label_orang[grup]

def skor(idx_latih, idx_uji):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X[idx_latih], y[idx_latih])
    return model.score(X[idx_uji], y[idx_uji])

idx = np.arange(len(y))
a, b = train_test_split(idx, test_size=0.25, random_state=0)  #<2>
print(f"Pembagian per foto  : akurasi uji {skor(a, b):.2f}")

g = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)  #<3>
a, b = next(g.split(X, y, groups=grup))
print(f"Pembagian per orang : akurasi uji {skor(a, b):.2f}")

# Pembagian berstrata menjaga proporsi kelas di setiap bagian  #<4>
y_timpang = (rng.random(1000) < 0.05).astype(int)   # 5% positif
s = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
a, b = next(s.split(np.zeros(1000), y_timpang))
print(f"Proporsi positif    : latih {y_timpang[a].mean():.3f},"
      f" uji {y_timpang[b].mean():.3f}")
