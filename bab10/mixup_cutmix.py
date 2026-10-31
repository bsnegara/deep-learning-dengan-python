import numpy as np

rng = np.random.default_rng(0)
x1, x2 = np.zeros((32, 32, 3)), np.ones((32, 32, 3))   # dua citra mainan
y1, y2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])    # label one-hot

def mixup(x1, y1, x2, y2, alpha=0.2):  #<1>
    lam = rng.beta(alpha, alpha)
    return lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2

def cutmix(x1, y1, x2, y2, alpha=1.0):  #<2>
    lam = rng.beta(alpha, alpha)
    H, W = x1.shape[:2]
    h, w = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
    cy, cx = rng.integers(H), rng.integers(W)
    a, b = np.clip(cy - h // 2, 0, H), np.clip(cy + h // 2, 0, H)
    c, d = np.clip(cx - w // 2, 0, W), np.clip(cx + w // 2, 0, W)
    x = x1.copy()
    x[a:b, c:d] = x2[a:b, c:d]                          # tempel potongan
    lam_nyata = 1 - (b - a) * (d - c) / (H * W)  #<3>
    return x, lam_nyata * y1 + (1 - lam_nyata) * y2

x, y = mixup(x1, y1, x2, y2)
print("Mixup : rerata piksel", round(x.mean(), 3), "label", y.round(3))
x, y = cutmix(x1, y1, x2, y2)
print("CutMix: rerata piksel", round(x.mean(), 3), "label", y.round(3))
