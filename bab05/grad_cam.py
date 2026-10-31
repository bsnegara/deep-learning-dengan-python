import numpy as np
import tensorflow as tf
import keras
from keras.applications import xception

model = xception.Xception(weights="imagenet")  #<1>
lapisan_konv = model.get_layer("block14_sepconv2_act")
model_konv = keras.Model(model.inputs, lapisan_konv.output)

# Bagian atas model: dari peta fitur terakhir hingga skor kelas
masukan_atas = keras.Input(shape=lapisan_konv.output.shape[1:])
x = model.get_layer("avg_pool")(masukan_atas)
x = model.get_layer("predictions")(x)
model_atas = keras.Model(masukan_atas, x)

def grad_cam(citra):  #<2>
    with tf.GradientTape() as tape:
        peta = model_konv(citra)
        tape.watch(peta)
        prediksi = model_atas(peta)
        kelas = tf.argmax(prediksi[0])
        skor = prediksi[:, kelas]
    grad = tape.gradient(skor, peta)  #<3>
    bobot = tf.reduce_mean(grad, axis=(0, 1, 2))   # alpha_k
    cam = tf.reduce_sum(peta[0] * bobot, axis=-1)  #<4>
    cam = tf.nn.relu(cam) / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), int(kelas)

jalur = keras.utils.get_file(
    "gajah.jpg", "https://img-datasets.s3.amazonaws.com/elephant.jpg")
citra = keras.utils.img_to_array(
    keras.utils.load_img(jalur, target_size=(299, 299)))
citra = xception.preprocess_input(citra[None])
cam, kelas = grad_cam(citra)
nama = xception.decode_predictions(np.eye(1000)[None, kelas], top=1)
print(cam.shape, nama[0][0][1])        # (10, 10) African_elephant
