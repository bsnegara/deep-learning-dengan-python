import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])
y = 2.0 * x  #<1>

w, laju = 0.0, 0.05  #<2>
for langkah in range(6):
    prediksi = w * x
    loss = np.mean((prediksi - y) ** 2)  #<3>
    gradien = np.mean(2 * (prediksi - y) * x)  #<4>
    print(f"langkah {langkah}: w = {w:.4f}  loss = {loss:7.4f}  "
          f"gradien = {gradien:7.3f}")
    w = w - laju * gradien  #<5>
