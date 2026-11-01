# Deep Learning dengan Python — Kode Pendamping

Kode, notebook, dan perangkat verifikasi untuk buku
**_Deep Learning dengan Python: Konsep, Kode, dan Penerapan Modern_**
(Benny Sukma Negara, Teknik Informatika, UIN Sultan Syarif Kasim Riau).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23831607.svg)](https://doi.org/10.5281/zenodo.23831607)

> *Companion code for an Indonesian-language deep learning textbook: runnable
> NumPy listings, 36 Keras 3 / PyTorch notebooks for Google Colab, and a
> verification kit that checks every number quoted in the book.*

## Isi

| Folder | Isi |
|---|---|
| `bab01/` … `bab18/` | 35 skrip listing buku. Sebagian besar hanya butuh NumPy dan scikit-learn; keluarannya dicetak apa adanya di buku. |
| `notebook/` | 36 notebook Jupyter (Keras 3; PyTorch untuk Bab 12–13). |
| `verifikasi/` | Skrip untuk menjalankan notebook secara penuh dan mencocokkan angka di naskah (`PANDUAN.md`). |
| `jalankan_semua.py` | Menjalankan semua skrip listing dan melaporkan OK/GAGAL. |

## Mulai cepat

**Di komputer sendiri** (Python 3.11+):
```bash
git clone https://github.com/bsnegara/deep-learning-dengan-python.git
cd deep-learning-dengan-python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python jalankan_semua.py
```

**Di Google Colab:** klik tautan *Buka di Colab* pada tabel di bawah, lalu pilih
**Runtime → Change runtime type → T4 GPU**.

## Status verifikasi

Ke-32 angka di naskah yang berasal dari notebook sudah dicocokkan dengan hasil
eksekusi penuh di Kaggle (GPU T4, 1 November 2026): **32/32 sesuai** setelah
empat angka naskah diperbarui. Laporan: [`verifikasi/laporan_2026-11-01.md`](verifikasi/laporan_2026-11-01.md).

## Daftar notebook

| Bab | Notebook | Colab |
|---|---|---|
| 2 | `bab02_01_mnist_pertama` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab02_01_mnist_pertama.ipynb) |
| 3 | `bab03_01_klasifikasi_biner_imdb` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab03_01_klasifikasi_biner_imdb.ipynb) |
| 3 | `bab03_02_klasifikasi_multikelas_reuters` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab03_02_klasifikasi_multikelas_reuters.ipynb) |
| 3 | `bab03_03_regresi_harga_rumah` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab03_03_regresi_harga_rumah.ipynb) |
| 4 | `bab04_01_overfitting_underfitting` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab04_01_overfitting_underfitting.ipynb) |
| 5 | `bab05_01_pengantar_convnet` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab05_01_pengantar_convnet.ipynb) |
| 5 | `bab05_02_convnet_data_kecil` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab05_02_convnet_data_kecil.ipynb) |
| 5 | `bab05_03_convnet_pralatih` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab05_03_convnet_pralatih.ipynb) |
| 5 | `bab05_04_visualisasi_convnet` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab05_04_visualisasi_convnet.ipynb) |
| 6 | `bab06_01_one_hot_encoding` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab06_01_one_hot_encoding.ipynb) |
| 6 | `bab06_02_word_embedding` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab06_02_word_embedding.ipynb) |
| 6 | `bab06_03_rnn_lstm` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab06_03_rnn_lstm.ipynb) |
| 6 | `bab06_04_rnn_lanjut_jena` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab06_04_rnn_lanjut_jena.ipynb) |
| 6 | `bab06_05_konvolusi_1d` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab06_05_konvolusi_1d.ipynb) |
| 7 | `bab07_01_api_fungsional_callback` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab07_01_api_fungsional_callback.ipynb) |
| 7 | `bab07_02_convnet_modern_tuning` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab07_02_convnet_modern_tuning.ipynb) |
| 8 | `bab08_01_atensi_transformer_imdb` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab08_01_atensi_transformer_imdb.ipynb) |
| 8 | `bab08_02_transformer_terjemahan` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab08_02_transformer_terjemahan.ipynb) |
| 9 | `bab09_01_pretraining_mlm_mini` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab09_01_pretraining_mlm_mini.ipynb) |
| 9 | `bab09_02_model_pralatih_keras_hub` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab09_02_model_pralatih_keras_hub.ipynb) |
| 10 | `bab10_01_vit_dari_nol_cifar10` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab10_01_vit_dari_nol_cifar10.ipynb) |
| 10 | `bab10_02_finetune_convnext` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab10_02_finetune_convnext.ipynb) |
| 11 | `bab11_01_segmentasi_unet` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab11_01_segmentasi_unet.ipynb) |
| 11 | `bab11_02_deteksi_objek` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab11_02_deteksi_objek.ipynb) |
| 12 | `bab12_01_simclr_cifar10` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab12_01_simclr_cifar10.ipynb) |
| 13 | `bab13_01_fixmatch` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab13_01_fixmatch.ipynb) |
| 13 | `bab13_02_mixmatch` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab13_02_mixmatch.ipynb) |
| 14 | `bab14_01_generasi_teks_lstm` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab14_01_generasi_teks_lstm.ipynb) |
| 14 | `bab14_02_deepdream` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab14_02_deepdream.ipynb) |
| 14 | `bab14_03_transfer_gaya` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab14_03_transfer_gaya.ipynb) |
| 14 | `bab14_04_vae` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab14_04_vae.ipynb) |
| 14 | `bab14_05_gan` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab14_05_gan.ipynb) |
| 15 | `bab15_01_ddpm_fashion_mnist` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab15_01_ddpm_fashion_mnist.ipynb) |
| 16 | `bab16_01_evaluasi_data_tidak_seimbang` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab16_01_evaluasi_data_tidak_seimbang.ipynb) |
| 16 | `bab16_02_evaluasi_multilabel` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab16_02_evaluasi_multilabel.ipynb) |
| 17 | `bab17_01_ekspor_kuantisasi_layanan` | [Buka di Colab](https://colab.research.google.com/github/bsnegara/deep-learning-dengan-python/blob/main/notebook/bab17_01_ekspor_kuantisasi_layanan.ipynb) |

## Lisensi dan atribusi

- Kode di repositori ini: **MIT License** (lihat `LICENSE`).
- Sembilan belas notebook diadaptasi dari notebook pendamping *Deep Learning with
  Python* edisi pertama karya François Chollet (MIT License). Atribusi lengkap
  dan teks lisensi aslinya ada di `ATRIBUSI.md`; rincian perubahan di
  `notebook/CATATAN-PERUBAHAN.md`.
- Dataset dan model pralatih diunduh dari sumber aslinya dan tunduk pada
  lisensinya masing-masing (daftar di Lampiran A buku).
- Naskah buku tidak termasuk dalam repositori ini.

## Sitasi

Bila memakai kode ini, silakan sitasi melalui berkas `CITATION.cff`
(tombol **Cite this repository** di GitHub) atau DOI Zenodo:
[10.5281/zenodo.23831607](https://doi.org/10.5281/zenodo.23831607).
