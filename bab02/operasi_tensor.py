import time
import numpy as np

def relu_naif(x):  #<1>
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0.0)
    return x

def dot_naif(x, y):  #<2>
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            for k in range(x.shape[1]):
                z[i, j] += x[i, k] * y[k, j]
    return z

def ukur(f, *args, ulang=3):
    t0 = time.perf_counter()
    for _ in range(ulang):
        f(*args)
    return (time.perf_counter() - t0) / ulang * 1000  # milidetik

rng = np.random.default_rng(0)
x = rng.normal(size=(128, 512))
w = rng.normal(size=(512, 64))

print(f"relu  naif : {ukur(relu_naif, x):8.2f} ms")
print(f"relu  NumPy: {ukur(np.maximum, x, 0.0):8.2f} ms")  #<3>
print(f"dot   naif : {ukur(dot_naif, x[:32], w[:, :16]):8.2f} ms"
      "  (hanya 32x512 @ 512x16)")
print(f"dot   NumPy: {ukur(np.dot, x, w):8.2f} ms  (128x512 @ 512x64)")
print("hasil sama :", np.allclose(dot_naif(x[:4], w), x[:4] @ w))
