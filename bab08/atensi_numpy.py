import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  #<1>
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def atensi(Q, K, V):
    d_k = Q.shape[-1]
    skor = Q @ K.T / np.sqrt(d_k)  #<2>
    bobot = softmax(skor, axis=-1)  #<3>
    return bobot @ V, bobot  #<4>


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(4, 8))       # 4 kueri berdimensi 8
    K = rng.normal(size=(6, 8))       # 6 kunci
    V = rng.normal(size=(6, 16))      # 6 nilai berdimensi 16
    keluaran, bobot = atensi(Q, K, V)
    print(keluaran.shape, bobot.sum(axis=-1))  # (4, 16) [1. 1. 1. 1.]
