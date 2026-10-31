import keras
from keras import ops

class DenseSederhana(keras.layers.Layer):  #<1>
    def __init__(self, unit, aktivasi=None, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.aktivasi = keras.activations.get(aktivasi)

    def build(self, bentuk_masukan):  #<2>
        dim_masuk = bentuk_masukan[-1]
        self.W = self.add_weight(shape=(dim_masuk, self.unit),
                                 initializer="glorot_uniform", name="W")
        self.b = self.add_weight(shape=(self.unit,),
                                 initializer="zeros", name="b")

    def call(self, x):  #<3>
        return self.aktivasi(ops.matmul(x, self.W) + self.b)

    def get_config(self):  #<4>
        return {**super().get_config(), "unit": self.unit,
                "aktivasi": keras.activations.serialize(self.aktivasi)}

lapisan = DenseSederhana(32, aktivasi="relu")
y = lapisan(ops.ones((2, 784)))     # build() dipanggil otomatis di sini
print(y.shape, [w.shape for w in lapisan.weights])
