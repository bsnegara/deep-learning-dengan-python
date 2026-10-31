"""Verifikasi klaim Bab 4 (K07, K08): MNIST + RMSprop dengan laju 1.0 vs 1e-2."""
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras
from keras import layers

(x_latih, y_latih), (x_uji, y_uji) = keras.datasets.mnist.load_data()
x_latih = x_latih.reshape(-1, 784).astype("float32") / 255
x_uji = x_uji.reshape(-1, 784).astype("float32") / 255

for lr in [1.0, 0.01]:
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input(shape=(784,)),
                              layers.Dense(512, activation="relu"),
                              layers.Dense(10, activation="softmax")])
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_latih, y_latih, epochs=10, batch_size=128, verbose=0)
    akurasi = model.evaluate(x_uji, y_uji, verbose=0)[1]
    print(f"lr={lr} : akurasi uji {akurasi:.2%}")
