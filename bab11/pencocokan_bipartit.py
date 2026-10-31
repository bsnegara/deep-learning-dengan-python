import numpy as np
from scipy.optimize import linear_sum_assignment

# 2 objek sebenarnya dan 4 prediksi (object query) dari DETR
kotak_gt = np.array([[0.10, 0.10, 0.40, 0.50],      # (x1, y1, x2, y2)
                     [0.55, 0.20, 0.90, 0.80]])
kelas_gt = np.array([0, 1])                          # 0 = kucing, 1 = anjing
kotak_pred = np.array([[0.52, 0.22, 0.88, 0.79],
                       [0.12, 0.08, 0.41, 0.52],
                       [0.30, 0.30, 0.60, 0.60],
                       [0.00, 0.70, 0.20, 0.95]])
# peluang kelas [kucing, anjing, tidak-ada-objek] per prediksi
p_kelas = np.array([[0.10, 0.80, 0.10],
                    [0.70, 0.20, 0.10],
                    [0.30, 0.30, 0.40],
                    [0.05, 0.05, 0.90]])

biaya_kelas = -p_kelas[:, kelas_gt]                  # (4 pred, 2 gt)  #<1>
biaya_kotak = np.abs(kotak_pred[:, None] - kotak_gt[None]).sum(-1)  #<2>
biaya = biaya_kelas + 5.0 * biaya_kotak

pred, gt = linear_sum_assignment(biaya)  #<3>
for p_i, g_i in zip(pred, gt):
    print(f"prediksi {p_i} <-> objek {g_i}  (biaya {biaya[p_i, g_i]:.2f})")
sisa = sorted(set(range(4)) - set(pred))
print("prediksi tanpa pasangan (dilatih menjadi 'tidak ada objek'):", sisa)
