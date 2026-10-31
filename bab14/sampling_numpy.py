import numpy as np

logit = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -0.5])
token = ["kucing", "anjing", "burung", "ikan", "meja", "awan"]

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

for suhu in [0.5, 1.0, 2.0]:  #<1>
    print(f"suhu {suhu}:", np.round(softmax(logit / suhu), 3))

def top_k(p, k):  #<2>
    q = np.where(p >= np.sort(p)[-k], p, 0.0)
    return q / q.sum()

def top_p(p, ambang):  #<3>
    urutan = np.argsort(p)[::-1]
    kumulatif = np.cumsum(p[urutan])
    jumlah = np.searchsorted(kumulatif, ambang) + 1   # token terkecil
    q = np.zeros_like(p)
    q[urutan[:jumlah]] = p[urutan[:jumlah]]
    return q / q.sum()

p = softmax(logit)
print("top-k=3  :", np.round(top_k(p, 3), 3))
print("top-p=0.6:", np.round(top_p(p, 0.6), 3))
rng = np.random.default_rng(0)
print("sampel   :", [token[rng.choice(6, p=top_p(p, 0.6))]
                     for _ in range(6)])
