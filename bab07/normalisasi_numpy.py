import numpy as np

rng = np.random.default_rng(0)
# batch 4 sampel x 3 fitur, dengan skala fitur yang sangat berbeda
x = rng.normal(loc=[0.0, 50.0, -3.0], scale=[1.0, 20.0, 0.1],
               size=(4, 3))

def batch_norm(x, eps=1e-5):  #<1>
    mu = x.mean(axis=0, keepdims=True)        # per fitur, lintas batch
    var = x.var(axis=0, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def layer_norm(x, eps=1e-5):  #<2>
    mu = x.mean(axis=1, keepdims=True)        # per sampel, lintas fitur
    var = x.var(axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def rms_norm(x, eps=1e-5):  #<3>
    return x / np.sqrt((x ** 2).mean(axis=1, keepdims=True) + eps)

np.set_printoptions(precision=2, suppress=True)
print("BN: rerata per fitur   ", batch_norm(x).mean(axis=0))
print("BN: simpangan per fitur", batch_norm(x).std(axis=0))
print("LN: rerata per sampel  ", layer_norm(x).mean(axis=1))
print("LN: simpangan/sampel   ", layer_norm(x).std(axis=1))
print("RMS: akar rerata kuadrat",
      np.sqrt((rms_norm(x) ** 2).mean(axis=1)))
