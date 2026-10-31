import numpy as np
from scipy.stats import ks_2samp

rng = np.random.default_rng(0)
latih = rng.normal(0, 1, 50000)          # mis. kecerahan citra saat latih

def psi(acuan, baru, k=10):  #<1>
    tepi = np.quantile(acuan, np.linspace(0, 1, k + 1))
    tepi[0], tepi[-1] = -np.inf, np.inf
    q = np.histogram(acuan, tepi)[0] / len(acuan)
    p = np.histogram(baru, tepi)[0] / len(baru)
    q, p = np.clip(q, 1e-6, None), np.clip(p, 1e-6, None)
    return np.sum((p - q) * np.log(p / q))

for nama, geser, skala in [("sama", 0.0, 1.0), ("geser 0.05", 0.05, 1.0),
                           ("geser 0.5", 0.5, 1.0), ("melebar", 0.0, 1.5)]:
    produksi = rng.normal(geser, skala, 20000)  #<2>
    ks = ks_2samp(latih, produksi)
    print(f"{nama:<11} PSI {psi(latih, produksi):.3f}  "
          f"KS {ks.statistic:.3f}  p-value {ks.pvalue:.1e}")
