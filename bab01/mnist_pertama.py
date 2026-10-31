import keras
from keras import layers

(x_latih, y_latih), (x_uji, y_uji) = keras.datasets.mnist.load_data()  #<1>
x_latih = x_latih.reshape((60000, 28 * 28)).astype("float32") / 255
x_uji = x_uji.reshape((10000, 28 * 28)).astype("float32") / 255

model = keras.Sequential([  #<2>
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="rmsprop",  #<3>
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_latih, y_latih, epochs=5, batch_size=128)  #<4>

loss_uji, akurasi_uji = model.evaluate(x_uji, y_uji)
print(f"Akurasi uji: {akurasi_uji:.3f}")
