"""Verifikasi Tabel bag-of-words IMDB di Bab 6 (K15-K17)."""
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import pathlib, tarfile
import numpy as np
import keras
from keras import layers

def siapkan_aclimdb(folder="data"):
    """Siapkan data/aclImdb/{train,test}/{neg,pos,unsup}/*.txt.
    Sumber utama: server Stanford. Bila tidak dapat dihubungi, unduh salinan
    resmi yang sama dari Hugging Face (stanfordnlp/imdb)."""
    akar = pathlib.Path(folder) / "aclImdb"
    if (akar / "train" / "pos").exists():
        return
    try:
        jalur = keras.utils.get_file(
            "aclImdb_v1.tar.gz",
            origin="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
        with tarfile.open(jalur) as berkas_tar:
            berkas_tar.extractall(folder)
    except Exception as galat:
        print("Server Stanford tidak dapat dihubungi:", galat)
        print("Memakai salinan Hugging Face (stanfordnlp/imdb) ...")
        from datasets import load_dataset
        ds = load_dataset("stanfordnlp/imdb")
        for bagian, nama in [("train", "train"), ("test", "test"), ("unsupervised", "train")]:
            for i, contoh in enumerate(ds[bagian]):
                jenis = "unsup" if bagian == "unsupervised" else ("pos" if contoh["label"] == 1 else "neg")
                tujuan = akar / nama / jenis
                tujuan.mkdir(parents=True, exist_ok=True)
                (tujuan / f"{bagian}_{i}.txt").write_text(contoh["text"], encoding="utf-8")

siapkan_aclimdb()

def baca_folder(folder):
    teks, label = [], []
    for jenis, nilai in [("neg", 0), ("pos", 1)]:
        for p in sorted((pathlib.Path(folder) / jenis).glob("*.txt")):
            teks.append(p.read_text(encoding="utf-8"))
            label.append(nilai)
    return np.array(teks), np.array(label)

teks_latih, y_latih = baca_folder("data/aclImdb/train")
teks_uji, y_uji = baca_folder("data/aclImdb/test")
acak = np.random.default_rng(0).permutation(len(teks_latih))
teks_latih, y_latih = teks_latih[acak], y_latih[acak]
teks_val, y_val = teks_latih[-5000:], y_latih[-5000:]
teks_latih, y_latih = teks_latih[:-5000], y_latih[:-5000]

VARIAN = [("unigram multi-hot", 1, "multi_hot"),
          ("unigram + bigram multi-hot", 2, "multi_hot"),
          ("unigram + bigram TF-IDF", 2, "tf_idf")]
for nama, ngram, mode in VARIAN:
    keras.utils.set_random_seed(0)
    vek = layers.TextVectorization(max_tokens=20000, ngrams=ngram,
                                   output_mode=mode)
    vek.adapt(teks_latih)
    x_l, x_v, x_u = vek(teks_latih), vek(teks_val), vek(teks_uji)
    model = keras.Sequential([keras.Input(shape=(20000,)),
                              layers.Dense(16, activation="relu"),
                              layers.Dropout(0.5),
                              layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer="rmsprop", loss="binary_crossentropy",
                  metrics=["accuracy"])
    cek = keras.callbacks.ModelCheckpoint("bow_terbaik.keras",
                                          save_best_only=True)
    model.fit(x_l, y_latih, validation_data=(x_v, y_val), epochs=10,
              batch_size=32, callbacks=[cek], verbose=0)
    terbaik = keras.models.load_model("bow_terbaik.keras")
    akurasi = terbaik.evaluate(x_u, y_uji, verbose=0)[1]
    print(f"{nama:28s}: akurasi uji {akurasi:.2%}")
