# Notebook Pendamping *Deep Learning dengan Python* (Benny Sukma Negara)

36 notebook Jupyter untuk seluruh bab praktik buku:
- **19 notebook adaptasi** (Bab 2–6, 14) dari notebook pendamping *Deep Learning with Python* (F. Chollet, lisensi MIT), diperbarui ke Keras 3. Perubahan dan bug yang diperbaiki dicatat di [`CATATAN-PERUBAHAN.md`](CATATAN-PERUBAHAN.md).
- **17 notebook baru** (Bab 7–13, 15–17) untuk topik yang tidak ada di buku asli.

Kerangka kode: **Keras 3** sebagai default, **PyTorch** untuk Bab 12–13 (serta Faster R-CNN di `bab11_02`).

## Menjalankan
1. Buka notebook di Google Colab (**Runtime → Change runtime type → GPU**).
2. Jalankan sel dari atas ke bawah. Dataset dan bobot pralatih diunduh otomatis.
3. `bab05_04` memakai model dari `bab05_02`, jadi jalankan keduanya di sesi yang sama.

Kebutuhan: Python ≥ 3.10, `keras>=3.3`, `tensorflow>=2.16`, `torch>=2.3`, `torchvision`, `scikit-learn`, `matplotlib`, `scipy`, `pandas`.
Semuanya sudah tersedia di Colab. Paket tambahan (`keras-tuner`, `keras-hub`, `fastapi`) dipasang oleh sel `!pip` di notebook yang membutuhkannya.

## Uji asap semua notebook
```bash
python uji_notebook.py                 # semua notebook (epoch/langkah dipangkas)
python uji_notebook.py bab12 bab13     # hanya notebook yang namanya mengandung pola ini
```
Skrip ini mengubah setiap notebook menjadi skrip Python, memangkas epoch dan langkah, menjalankannya berurutan,
lalu mencetak ringkasan OK/GAGAL. Log per notebook disimpan di `_uji/`.
Unduhan data cukup besar (±3 GB total), jadi sebaiknya uji per kelompok bab.

## Daftar notebook

| Bab | Notebook | Kerangka | Waktu penuh (T4) |
|---|---|---|---|
| 2 | `bab02_01_mnist_pertama` | Keras | 1 mnt |
| 3 | `bab03_01_klasifikasi_biner_imdb`, `bab03_02_klasifikasi_multikelas_reuters`, `bab03_03_regresi_harga_rumah` | Keras | 2–5 mnt |
| 4 | `bab04_01_overfitting_underfitting` | Keras | 5 mnt |
| 5 | `bab05_01_pengantar_convnet`, `bab05_02_convnet_data_kecil`, `bab05_03_convnet_pralatih`, `bab05_04_visualisasi_convnet` | Keras | 5–30 mnt |
| 6 | `bab06_01_one_hot_encoding`, `bab06_02_word_embedding`, `bab06_03_rnn_lstm`, `bab06_04_rnn_lanjut_jena`, `bab06_05_konvolusi_1d` | Keras | 2–40 mnt |
| 7 | `bab07_01_api_fungsional_callback`, `bab07_02_convnet_modern_tuning` | Keras | 5–20 mnt |
| 8 | `bab08_01_atensi_transformer_imdb`, `bab08_02_transformer_terjemahan` | Keras | 10–30 mnt |
| 9 | `bab09_01_pretraining_mlm_mini`, `bab09_02_model_pralatih_keras_hub` | Keras, KerasHub | 15–30 mnt |
| 10 | `bab10_01_vit_dari_nol_cifar10`, `bab10_02_finetune_convnext` | Keras | 15–25 mnt |
| 11 | `bab11_01_segmentasi_unet`, `bab11_02_deteksi_objek` | Keras (+ torchvision) | 10–20 mnt |
| 12 | `bab12_01_simclr_cifar10` | PyTorch | 60 mnt |
| 13 | `bab13_01_fixmatch`, `bab13_02_mixmatch` | PyTorch | 45–90 mnt |
| 14 | `bab14_01_generasi_teks_lstm`, `bab14_02_deepdream`, `bab14_03_transfer_gaya`, `bab14_04_vae`, `bab14_05_gan` | Keras | 5–40 mnt |
| 15 | `bab15_01_ddpm_fashion_mnist` | Keras | 25–40 mnt |
| 16 | `bab16_01_evaluasi_data_tidak_seimbang`, `bab16_02_evaluasi_multilabel` | Keras + scikit-learn | 5–10 mnt |
| 17 | `bab17_01_ekspor_kuantisasi_layanan` | Keras/TF, TFLite, FastAPI | 10 mnt |
