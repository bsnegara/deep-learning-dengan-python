import numpy as np
from atensi_numpy import atensi

# Tiga token dengan embedding 2 dimensi (dibuat tangan agar mudah dilacak)
X = np.array([[1.0, 0.0],      # "kucing"
              [0.0, 1.0],      # "duduk"
              [1.0, 1.0]])     # "tikar"
_, bobot = atensi(X, X, X)  #<1>
print(np.round(bobot, 3))

# Ekuivarian permutasi: tukar urutan token, keluaran ikut tertukar  #<2>
P = np.array([2, 0, 1])
y, _ = atensi(X, X, X)
y_acak, _ = atensi(X[P], X[P], X[P])
print("ekuivarian permutasi:", np.allclose(y[P], y_acak))
