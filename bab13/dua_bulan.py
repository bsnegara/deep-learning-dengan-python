import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelSpreading

X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
X_uji, y_uji = make_moons(n_samples=1000, noise=0.1, random_state=1)
rng = np.random.default_rng(5)
berlabel = np.concatenate([rng.choice(np.where(y == c)[0], 2,
                                      replace=False) for c in (0, 1)])
print("jumlah label:", len(berlabel))  #<1>

# 1. Terawasi: hanya 4 titik berlabel
svm = SVC(kernel="rbf", gamma=1.0).fit(X[berlabel], y[berlabel])
print(f"hanya berlabel  : {svm.score(X_uji, y_uji):.3f}")

# 2. Pseudo-labeling berulang: tambahkan prediksi paling yakin  #<2>
idx_l, y_l = list(berlabel), list(y[berlabel])
for putaran in range(10):
    model = SVC(kernel="rbf", gamma=1.0).fit(X[idx_l], y_l)
    sisa = np.setdiff1d(np.arange(len(X)), idx_l)
    skor = model.decision_function(X[sisa])
    yakin = sisa[np.argsort(-np.abs(skor))[:80]]      # 80 paling yakin
    idx_l += list(yakin)
    y_l += list(model.predict(X[yakin]))
print(f"pseudo-labeling : {model.score(X_uji, y_uji):.3f}")

# 3. Propagasi label pada graf kemiripan (asumsi klaster)  #<3>
y_sebagian = np.full(len(X), -1)                     # -1 = tanpa label
y_sebagian[berlabel] = y[berlabel]
lp = LabelSpreading(kernel="rbf", gamma=20, alpha=0.2, max_iter=200)
lp.fit(X, y_sebagian)
print(f"propagasi label : {lp.score(X_uji, y_uji):.3f}")
