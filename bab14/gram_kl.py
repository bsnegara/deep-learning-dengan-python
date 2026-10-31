import numpy as np

rng = np.random.default_rng(0)
# --- Matriks Gram tidak peka posisi  #<1>
F = rng.normal(size=(4, 100))          # 4 kanal x 100 posisi spasial
acak = rng.permutation(100)            # acak urutan posisi
print("Gram sama setelah posisi diacak:",
      np.allclose(F @ F.T, F[:, acak] @ F[:, acak].T))

# --- KL( N(mu, sigma^2) || N(0, 1) ): rumus tertutup vs Monte Carlo  #<2>
mu, log_var = 1.0, np.log(0.25)        # sigma = 0.5
kl_rumus = -0.5 * (1 + log_var - mu ** 2 - np.exp(log_var))
sigma = np.exp(0.5 * log_var)          # simpangan baku: faktor 0.5!
z = mu + sigma * rng.normal(size=1_000_000)   # trik reparametrisasi
log_q = -0.5 * ((z - mu) / sigma) ** 2 - np.log(sigma)
log_p = -0.5 * z ** 2
print(f"KL rumus tertutup: {kl_rumus:.4f}")
print(f"KL Monte Carlo   : {np.mean(log_q - log_p):.4f}")
