# Catatan Perubahan: Notebook Asli (Keras 2.0.8) → buku Deep Learning dengan Python (Keras 3)

Notebook asli adalah notebook pendamping *Deep Learning with Python* edisi 1 (F. Chollet, kode berlisensi MIT).
Semuanya ditulis untuk Keras 2.0.8 + TensorFlow 1.x (2017), dan **sebagian besar tidak lagi berjalan** di Colab saat ini.
Semua notebook sudah ditulis ulang ke Keras 3 dengan penjelasan berbahasa Indonesia.

## 1. Pemetaan berkas

| Asli | Baru | Bab buku | Status asli di Keras 3 |
|---|---|---|---|
| 2.1-a-first-look-at-a-neural-network | bab02_01_mnist_pertama | 2 | jalan dengan peringatan |
| 3.5-classifying-movie-reviews | bab03_01_klasifikasi_biner_imdb | 3 | gagal (`lr=`) |
| 3.6-classifying-newswires | bab03_02_klasifikasi_multikelas_reuters | 3 | jalan; label grafik keliru |
| 3.7-predicting-house-prices | bab03_03_regresi_harga_rumah | 3 | gagal (`val_mean_absolute_error`) |
| 4.4-overfitting-and-underfitting | bab04_01_overfitting_underfitting | 4 | gagal (`np.randint`) |
| 5.1-introduction-to-convnets | bab05_01_pengantar_convnet | 5 | jalan dengan peringatan |
| 5.2-using-convnets-with-small-datasets | bab05_02_convnet_data_kecil | 5 | gagal (`ImageDataGenerator`, `fit_generator`, path lokal) |
| 5.3-using-a-pretrained-convnet | bab05_03_convnet_pralatih | 5 | gagal + prapemrosesan VGG16 salah |
| 5.4-visualizing-what-convnets-learn | bab05_04_visualisasi_convnet | 5 | gagal (`K.gradients`, `K.function`) |
| 6.1-one-hot-encoding-of-words-or-characters | bab06_01_one_hot_encoding | 6 | gagal (`Tokenizer`); hasil `hash()` tidak deterministik |
| 6.1-using-word-embeddings | bab06_02_word_embedding | 6 | gagal (`Tokenizer`, `input_length`, path lokal) |
| 6.2-understanding-recurrent-neural-networks | bab06_03_rnn_lstm | 6 | gagal (`keras.preprocessing.sequence`) |
| 6.3-advanced-usage-of-recurrent-neural-networks | bab06_04_rnn_lanjut_jena | 6 | gagal (`fit_generator`, path lokal) |
| 6.4-sequence-processing-with-convnets | bab06_05_konvolusi_1d | 6 | gagal + bug sigmoid |
| 8.1-text-generation-with-lstm | bab14_01_generasi_teks_lstm | 14 | gagal (`np.bool`) |
| 8.2-deep-dream | bab14_02_deepdream | 14 | gagal (`K.gradients`, `scipy.misc`) |
| 8.3-neural-style-transfer | bab14_03_transfer_gaya | 14 | gagal (`K.placeholder`, `scipy.misc`) |
| 8.4-generating-images-with-vaes | bab14_04_vae | 14 | gagal + bug sampling |
| 8.5-introduction-to-gans | bab14_05_gan | 14 | gagal (`decay=`); perilaku `trainable` berubah |

Bab 7–13 dan 15–17 adalah materi baru, sehingga notebooknya (17 berkas) ditulis dari nol dan tidak tercantum di tabel ini. Lihat `README.md`.

## 2. Bug pada kode asli (bukan sekadar API usang)

| Notebook | Masalah | Perbaikan |
|---|---|---|
| 4.4 | `np.randint` tidak ada (seharusnya `np.random.randint`). Data **uji** dipakai sebagai validasi. | `rng.random`; validasi diambil dari data latih |
| 3.5 / 3.6 | Judul/label grafik akurasi dan loss tertukar atau keliru | Fungsi plot bersama dengan label yang benar |
| 3.7 | Kunci `val_mean_absolute_error` tidak ada lagi | `val_mae`; epoch terbaik dipilih otomatis; `clear_session()` per fold |
| 5.3 | Gambar hanya dibagi 255, padahal VGG16 butuh `preprocess_input` (BGR + kurangi rerata) | `vgg16.preprocess_input` |
| 6.1 (one-hot) | `hash()` Python diacak per proses, jadi indeks berubah setiap kali dijalankan | `hashlib.md5` / lapisan `Hashing` |
| 6.4 | `Dense(1)` tanpa sigmoid dengan `binary_crossentropy` | `Dense(1, activation="sigmoid")` |
| 8.1 | `dtype=np.bool` dihapus sejak NumPy 1.24 | `dtype=bool` |
| 8.4 | `z = mean + exp(log_var) * eps` memakai varians sebagai simpangan baku | `exp(0.5 * log_var)` |
| 8.5 | Data diskalakan ke [0, 1] padahal generator memakai `tanh` ([-1, 1]) | Skala ke [-1, 1] |

## 3. Penggantian API

| Keras 2 / TF1 (asli) | Keras 3 (baru) |
|---|---|
| `from keras import ...` dengan TF1 | `import keras` + `os.environ["KERAS_BACKEND"]` |
| `input_shape=` di lapisan pertama | `keras.Input(shape=...)` |
| `optimizers.RMSprop(lr=..., decay=...)` | `RMSprop(learning_rate=...)` (+ `schedules` bila perlu) |
| `ImageDataGenerator`, `flow_from_directory` | `image_dataset_from_directory` + lapisan augmentasi (`RandomFlip`, `RandomRotation`, `RandomZoom`) + `Rescaling` |
| `fit_generator`, `evaluate_generator` | `fit`, `evaluate` (menerima `tf.data`) |
| `keras.preprocessing.text.Tokenizer` | `layers.TextVectorization` |
| `keras.preprocessing.sequence.pad_sequences` | `keras.utils.pad_sequences` |
| Generator Python manual (Jena) | `keras.utils.timeseries_dataset_from_array` |
| `Embedding(..., input_length=n)` | `keras.Input(shape=(n,))` |
| `K.gradients`, `K.function`, `K.placeholder` | `tf.GradientTape` (+ `tf.function`) |
| `scipy.optimize.fmin_l_bfgs_b` (gaya) | `tf.Variable` + SGD dengan `ExponentialDecay` |
| `scipy.misc.imsave/imread/imresize` | `keras.utils.save_img/load_img`, `tf.image.resize` |
| Lapisan loss kustom + `add_loss` (VAE) | Subclass `keras.Model` dengan `train_step` |
| `discriminator.trainable = False` + model gabungan (GAN) | Subclass `keras.Model` dengan `train_step` dan dua optimizer |
| `model.save("x.h5")` | `model.save("x.keras")` / `ModelCheckpoint(..., save_best_only=True)` |
| `history["acc"]` | `history["accuracy"]` |
| Path lokal (`/Users/fchollet/...`) | `keras.utils.get_file` dari URL publik |

## 4. Penyesuaian lain

- Seluruh teks penjelasan, nama variabel, dan komentar berbahasa Indonesia; istilah teknis tetap dalam bahasa Inggris bila lazim.
- Setiap notebook dibuka dengan atribusi, lingkungan yang dibutuhkan, dan daftar perubahan dari versi asli.
- Notebook bersifat *backend-agnostik* kecuali yang membutuhkan gradien terhadap masukan atau `train_step` kustom (5.4, Bab 14 bagian 2–5): notebook tersebut menetapkan backend TensorFlow.
- Beberapa ukuran dikecilkan agar nyaman di Colab gratis: jendela Jena 5 hari, generasi teks 30 epoch, dan teks dibangkitkan setiap 5 epoch.
- Dataset Boston Housing diberi catatan etika (fitur `B` bermasalah); tetap dipakai karena bawaan Keras dan bersifat historis.
