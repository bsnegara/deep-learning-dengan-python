import numpy as np

rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, size=(512, 512)).astype("float32")
W[:, :8] *= 20                       # beberapa kanal bernilai besar
x = rng.normal(size=(64, 512)).astype("float32")

def kuantisasi(w, bit=8, sumbu=None):  #<1>
    """Kuantisasi simetris: q = round(w / s), s = maks|w| / (2^(b-1)-1)."""
    qmaks = 2 ** (bit - 1) - 1
    s = np.abs(w).max(axis=sumbu, keepdims=True) / qmaks
    q = np.clip(np.round(w / s), -qmaks, qmaks).astype(np.int8)
    return q, s

y_asli = x @ W
for bit in [8, 4]:
    for nama, sumbu in [("per tensor", None), ("per kanal ", 0)]:  #<2>
        q, s = kuantisasi(W, bit, sumbu)
        y = x @ (q.astype("float32") * s)          # dekuantisasi
        galat = np.linalg.norm(y - y_asli) / np.linalg.norm(y_asli)
        print(f"int{bit} {nama}: galat relatif keluaran {galat:.4f}")

# int4 per grup: setiap 64 bobot berurutan punya skala sendiri
Wg = W.T.reshape(512, 8, 64)                         #<3>
q, s = kuantisasi(Wg, 4, sumbu=2)
W4 = (q.astype("float32") * s).reshape(512, 512).T
galat = np.linalg.norm(x @ W4 - y_asli) / np.linalg.norm(y_asli)
print(f"int4 per grup 64: galat relatif keluaran {galat:.4f}")

print(f"ukuran float32 {W.nbytes / 1024:.0f} KB -> int8 "
      f"{W.size / 1024:.0f} KB -> int4 {W.size / 2048:.0f} KB")  #<4>
