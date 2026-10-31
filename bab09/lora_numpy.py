import numpy as np

rng = np.random.default_rng(0)
d, k, r, alpha = 768, 768, 8, 16
W = rng.normal(0, 0.02, (d, k))           # bobot pralatih (beku)  #<1>
A = rng.normal(0, 0.02, (r, k))           # dilatih
B = np.zeros((d, r))                      # dilatih, mulai dari nol  #<2>

def maju_lora(x):
    return x @ W.T + (alpha / r) * (x @ A.T) @ B.T

x = rng.normal(size=(4, k))
print("awal identik dengan model pralatih:",
      np.allclose(maju_lora(x), x @ W.T))

B = rng.normal(0, 0.02, (d, r))           # misalkan B sudah dilatih
W_gabung = W + (alpha / r) * B @ A        # gabungkan untuk inferensi  #<3>
print("setelah digabung, hasil sama     :",
      np.allclose(maju_lora(x), x @ W_gabung.T))
print(f"parameter penuh: {d * k:,} | parameter LoRA: {r * (d + k):,}"
      f" ({r * (d + k) / (d * k):.1%})")
