# Atribusi dan lisensi pihak ketiga

Kode buku *Deep Learning dengan Python: Konsep, Kode, dan Penerapan Modern*
(Benny Sukma Negara, 2026) dirilis dengan lisensi MIT (lihat `LICENSE`).

## 1. Notebook adaptasi dari *Deep Learning with Python* (François Chollet)

Sembilan belas notebook di `notebook/` merupakan adaptasi dari notebook
pendamping buku *Deep Learning with Python* (edisi pertama) karya François Chollet:

- Sumber: https://github.com/fchollet/deep-learning-with-python-notebooks
- Lisensi: MIT License

Notebook asli berasal dari edisi pertama (2017, Keras 2.0.8). Pemetaan
lengkap berkas asli ke berkas baru ada di `notebook/CATATAN-PERUBAHAN.md`
(bagian 1). Ringkasnya:

| Notebook di repositori ini | Notebook asli |
|---|---|
| `bab02_01` | 2.1-a-first-look-at-a-neural-network |
| `bab03_01`–`bab03_03` | 3.5, 3.6, 3.7 |
| `bab04_01` | 4.4-overfitting-and-underfitting |
| `bab05_01`–`bab05_04` | 5.1, 5.2, 5.3, 5.4 |
| `bab06_01`–`bab06_05` | 6.1 (dua notebook), 6.2, 6.3, 6.4 |
| `bab14_01`–`bab14_05` | 8.1, 8.2, 8.3, 8.4, 8.5 |

Perubahan utama: diperbarui ke Keras 3 (backend-agnostik), komentar dan
penjelasan dalam bahasa Indonesia, perbaikan bug, serta sel verifikasi angka.
Rinciannya ada di `notebook/CATATAN-PERUBAHAN.md`.

Sesuai lisensi MIT, pemberitahuan hak cipta dan izin dari repositori asli
disertakan di bawah ini:

```
MIT License

Copyright (c) 2017-present François Chollet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. Dataset dan model pralatih

Notebook mengunduh data dan bobot pralatih langsung dari sumber aslinya;
repositori ini tidak mendistribusikan ulang berkas-berkas tersebut. Daftar
lisensinya ada di Lampiran A buku (Tabel "Dataset dan model pralatih yang
dipakai"). Periksa ulang ketentuan sumber sebelum penggunaan komersial.
