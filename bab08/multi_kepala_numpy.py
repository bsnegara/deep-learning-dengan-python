import numpy as np
from atensi_numpy import softmax

rng = np.random.default_rng(0)
n, d_model, h = 5, 16, 4                  # 5 token, dimensi 16, 4 head
d_k = d_model // h
X = rng.normal(size=(n, d_model))
Wq, Wk, Wv, Wo = (rng.normal(0, d_model ** -0.5, (d_model, d_model))
                  for _ in range(4))  #<1>

def pisah_kepala(M):  #<2>
    return M.reshape(n, h, d_k).transpose(1, 0, 2)   # (h, n, d_k)

Q, K, V = (pisah_kepala(X @ W) for W in (Wq, Wk, Wv))
skor = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)       # (h, n, n)  #<3>
bobot = softmax(skor, axis=-1)
kepala = bobot @ V                                    # (h, n, d_k)
gabung = kepala.transpose(1, 0, 2).reshape(n, d_model)  #<4>
keluaran = gabung @ Wo

print("skor per head :", skor.shape)
print("keluaran        :", keluaran.shape)
print("parameter MHA   :", 4 * d_model * d_model, "(tanpa bias)")
