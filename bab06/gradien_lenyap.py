import numpy as np

rng = np.random.default_rng(0)
d = 32
U = rng.normal(0, 1 / np.sqrt(d), (d, d))         # bobot rekuren acak
W = rng.normal(0, 1 / np.sqrt(d), (d, d))

def norma_gradien(langkah, aditif=False, f=0.97):  #<1>
    """Norma dh_T/dh_0 untuk masukan acak sepanjang "langkah"."""
    h = np.zeros(d)
    J = np.eye(d)                                  # jakobian kumulatif
    for _ in range(langkah):
        x = rng.normal(size=d)
        h_baru = np.tanh(W @ x + U @ h)
        J_lokal = (1 - h_baru ** 2)[:, None] * U  #<2>
        if aditif:                                 # jalur ala LSTM  #<3>
            h_baru = f * h + (1 - f) * h_baru
            J_lokal = f * np.eye(d) + (1 - f) * J_lokal
        J = J_lokal @ J
        h = h_baru
    return np.linalg.norm(J, 2)

print("langkah   RNN sederhana   jalur aditif (f=0.97)")
for T in [1, 10, 30, 100]:
    print(f"{T:7d}   {norma_gradien(T):13.2e}   "
          f"{norma_gradien(T, aditif=True):13.2e}")
