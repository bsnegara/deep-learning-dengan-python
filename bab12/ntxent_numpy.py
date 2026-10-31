import numpy as np

def nt_xent(z1, z2, suhu=0.5):
    z = np.concatenate([z1, z2])
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    n = len(z1)
    sim = z @ z.T / suhu
    np.fill_diagonal(sim, -np.inf)
    target = np.concatenate([np.arange(n, 2 * n), np.arange(n)])
    log_p = sim - np.log(np.exp(sim).sum(axis=1, keepdims=True))  #<1>
    return -log_p[np.arange(2 * n), target].mean()

rng = np.random.default_rng(0)
N, d = 256, 128
dasar = rng.normal(size=(N, d))
skenario = {
    "kolaps total": (np.ones((N, d)), np.ones((N, d))),  #<2>
    "acak": (rng.normal(size=(N, d)), rng.normal(size=(N, d))),
    "baik (positif mirip)": (dasar + 0.1 * rng.normal(size=(N, d)),
                             dasar + 0.1 * rng.normal(size=(N, d))),
}
print(f"log(2N-1) = {np.log(2 * N - 1):.3f}")
for nama, (z1, z2) in skenario.items():
    std = np.concatenate([z1, z2]).std(axis=0).mean()  #<3>
    print(f"{nama:21s} loss = {nt_xent(z1, z2):6.3f}"
          f"   simpangan baku fitur = {std:.3f}")
