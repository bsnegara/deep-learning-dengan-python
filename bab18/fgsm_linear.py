import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

X, y = load_digits(return_X_y=True)
X = X / 16.0                                   # piksel dalam [0, 1]
model = LogisticRegression(max_iter=2000).fit(X[:1500], y[:1500])
Xu, yu = X[1500:], y[1500:]
W, b = model.coef_, model.intercept_

def fgsm(x, y, eps):  #<1>
    z = x @ W.T + b
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    p[np.arange(len(y)), y] -= 1               # dLoss/dz = p - y
    grad_x = p @ W                             # dLoss/dx
    return np.clip(x + eps * np.sign(grad_x), 0, 1)  #<2>

rng = np.random.default_rng(0)
for eps in [0.0, 0.05, 0.1, 0.2]:
    acak = np.clip(Xu + eps * rng.choice([-1, 1], Xu.shape), 0, 1)  #<3>
    print(f"eps={eps:.2f}: akurasi derau acak "
          f"{model.score(acak, yu):.3f}, FGSM "
          f"{model.score(fgsm(Xu, yu, eps), yu):.3f}")
