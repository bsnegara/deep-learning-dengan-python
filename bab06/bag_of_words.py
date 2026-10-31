import keras
from keras import layers

# teks_latih, y_latih, teks_val, y_val: daftar string dan label 0/1
vektorisasi = layers.TextVectorization(
    max_tokens=20000,
    ngrams=2,                      # unigram + bigram  #<1>
    output_mode="tf_idf")          # bobot TF-IDF      #<2>
vektorisasi.adapt(teks_latih)      # kosakata & IDF dari data LATIH

x_latih = vektorisasi(teks_latih)  # (sampel, 20000), padat
x_val = vektorisasi(teks_val)

model = keras.Sequential([  #<3>
    keras.Input(shape=(20000,)),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="rmsprop", loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(x_latih, y_latih, validation_data=(x_val, y_val), epochs=10)

# Model inferensi yang menerima STRING mentah  #<4>
masukan = keras.Input(shape=(1,), dtype="string")
model_teks = keras.Model(masukan, model(vektorisasi(masukan)))
model_teks.predict(keras.ops.convert_to_tensor(
    [["Film yang luar biasa, aktingnya memukau."]]))
