import numpy as np

def sgd(w, g, keadaan, laju):
    return w - laju * g

def momentum(w, g, keadaan, laju, beta=0.8):  #<1>
    keadaan["v"] = beta * keadaan.get("v", 0) - laju * g
    return w + keadaan["v"]

def adam(w, g, keadaan, laju, b1=0.9, b2=0.999, eps=1e-8):  #<2>
    t = keadaan["t"] = keadaan.get("t", 0) + 1
    keadaan["m"] = b1 * keadaan.get("m", 0) + (1 - b1) * g
    keadaan["s"] = b2 * keadaan.get("s", 0) + (1 - b2) * g ** 2
    m_hat = keadaan["m"] / (1 - b1 ** t)
    s_hat = keadaan["s"] / (1 - b2 ** t)
    return w - laju * m_hat / (np.sqrt(s_hat) + eps)

def jalankan(langkah, A, laju, maks=20000):
    """Minimalkan 0.5 * sum(A * w**2) dari w0 = (-9, 1.5)."""
    w, keadaan = np.array([-9.0, 1.5]), {}
    for t in range(1, maks + 1):
        w = langkah(w, A * w, keadaan, laju)  #<3>
        if not np.all(np.abs(w) < 1e6):
            return "divergen"
        if 0.5 * np.sum(A * w ** 2) < 1e-6:
            return t
    return f">{maks}"

for A in [np.array([1.0, 25.0]), np.array([1.0, 2500.0])]:  #<4>
    laju_aman = 1.9 / A.max()   # SGD stabil bila laju < 2/kelengkungan
    print(f"kelengkungan {A}:")
    print(f"  SGD      laju=0.07   : {jalankan(sgd, A, 0.07)}")
    hasil = jalankan(sgd, A, laju_aman)
    print(f"  SGD      laju={laju_aman:.5f}: {hasil}")
    hasil = jalankan(momentum, A, laju_aman)
    print(f"  Momentum laju={laju_aman:.5f}: {hasil}")
    print(f"  Adam     laju=0.3    : {jalankan(adam, A, 0.3)}")
