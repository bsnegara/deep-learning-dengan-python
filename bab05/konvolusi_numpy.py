import numpy as np
from scipy.signal import correlate2d

def konvolusi2d(x, k, stride=1, padding=0):  #<1>
    if padding:
        x = np.pad(x, padding)
    kh, kw = k.shape
    tinggi = (x.shape[0] - kh) // stride + 1
    lebar = (x.shape[1] - kw) // stride + 1
    y = np.zeros((tinggi, lebar))
    for i in range(tinggi):
        for j in range(lebar):
            a, b = i * stride, j * stride
            jendela = x[a:a + kh, b:b + kw]
            y[i, j] = np.sum(jendela * k)  #<2>
    return y

# Citra 8x8: persegi terang di tengah
citra = np.zeros((8, 8))
citra[2:6, 2:6] = 1.0

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  #<3>
tepi = konvolusi2d(citra, sobel_x)
print("bentuk keluaran:", tepi.shape)
print(tepi.astype(int))
print("sama dengan SciPy:",
      np.allclose(tepi, correlate2d(citra, sobel_x, mode="valid")))
print("padding=1, stride=2 ->",
      konvolusi2d(citra, sobel_x, stride=2, padding=1).shape)  #<4>
