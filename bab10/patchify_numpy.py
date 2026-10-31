import numpy as np

rng = np.random.default_rng(0)
H = W = 8; C = 3; P = 4; D = 5            # citra 8x8 RGB, patch 4x4
citra = rng.normal(size=(H, W, C))
W_proj = rng.normal(size=(P * P * C, D))   # proyeksi linear patch -> D

# Cara 1: potong patch, ratakan, kalikan matriks  #<1>
patch = citra.reshape(H // P, P, W // P, P, C).transpose(0, 2, 1, 3, 4)
patch = patch.reshape(-1, P * P * C)                  # (4 patch, 48)
token_1 = patch @ W_proj                               # (4, D)

# Cara 2: konvolusi dengan kernel PxP dan stride P  #<2>
kernel = W_proj.reshape(P, P, C, D)
token_2 = np.zeros((H // P, W // P, D))
for i in range(H // P):
    for j in range(W // P):
        jendela = citra[i * P:(i + 1) * P, j * P:(j + 1) * P]
        token_2[i, j] = np.einsum("abc,abcd->d", jendela, kernel)

print("jumlah token :", token_1.shape[0])
print("hasil identik:", np.allclose(token_1, token_2.reshape(-1, D)))
