import math
import numpy as np
import tensorflow as tf
import keras

class DenseNaif:                                             #<1>
    def __init__(self, masuk, keluar, aktivasi):
        self.aktivasi = aktivasi
        awal = tf.random.uniform((masuk, keluar), 0, 1e-1)
        self.W = tf.Variable(awal)
        self.b = tf.Variable(tf.zeros((keluar,)))

    def __call__(self, x):
        return self.aktivasi(tf.matmul(x, self.W) + self.b)

    @property
    def bobot(self):
        return [self.W, self.b]

class SequentialNaif:                                        #<2>
    def __init__(self, lapisan):
        self.lapisan = lapisan

    def __call__(self, x):
        for l in self.lapisan:
            x = l(x)
        return x

    @property
    def bobot(self):
        return [w for l in self.lapisan for w in l.bobot]

model = SequentialNaif([DenseNaif(784, 512, tf.nn.relu),
                        DenseNaif(512, 10, tf.nn.softmax)])

def satu_langkah(x, y, laju=1e-3):                           #<3>
    with tf.GradientTape() as tape:
        prediksi = model(x)
        loss = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(y, prediksi))
    gradien = tape.gradient(loss, model.bobot)               #<4>
    for g, w in zip(gradien, model.bobot):
        w.assign_sub(laju * g)                               #<5>
    return loss

(x_latih, y_latih), (x_uji, y_uji) = keras.datasets.mnist.load_data()
x_latih = x_latih.reshape(-1, 784).astype("float32") / 255
x_uji = x_uji.reshape(-1, 784).astype("float32") / 255

for epoch in range(10):                                      #<6>
    urutan = np.random.permutation(len(x_latih))
    for i in range(math.ceil(len(x_latih) / 128)):
        j = urutan[i * 128:(i + 1) * 128]
        loss = satu_langkah(x_latih[j], y_latih[j])
    print(f"epoch {epoch}: loss batch terakhir {float(loss):.3f}")

prediksi = model(x_uji).numpy().argmax(axis=1)
print(f"Akurasi uji: {(prediksi == y_uji).mean():.3f}")
