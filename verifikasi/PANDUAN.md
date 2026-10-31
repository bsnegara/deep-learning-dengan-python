# Panduan verifikasi kode dan angka naskah (Google Colab)

Tujuannya ada dua:
1. **Uji asap:** memastikan semua notebook dan skrip berjalan tanpa error.
2. **Verifikasi angka:** memastikan 32 angka di naskah (penanda merah `K01`–`K32`
   di PDF draf) sesuai dengan hasil eksekusi sungguhan.

Perkiraan waktu GPU (T4): uji asap ±1,5 jam, verifikasi penuh ±6 jam.
Karena sesi Colab gratis dibatasi, kerjakan per kelompok (langkah 6).
Semua hasil disimpan di Google Drive, jadi aman bila sesi terputus.

---

## Langkah 1 — Unggah proyek ke Google Drive
Unggah `dl-python-latex.zip` ke **Google Drive → My Drive** (folder utama).

## Langkah 2 — Buka Colab dan aktifkan GPU
Buka https://colab.research.google.com → **New notebook** →
**Runtime → Change runtime type → T4 GPU → Save**.

## Langkah 3 — Siapkan proyek (salin ke satu sel, jalankan)
Sel ini mencari ZIP di seluruh Drive, jadi nama dan lokasinya tidak masalah.
```python
from google.colab import drive
drive.mount("/content/drive")

import glob, os, subprocess
kandidat = sorted(glob.glob("/content/drive/MyDrive/**/dl-python-latex*.zip", recursive=True),
                  key=os.path.getmtime)
if not kandidat:
    raise FileNotFoundError("ZIP belum ada di Google Drive. Unggah dl-python-latex.zip ke My Drive, "
                            "tunggu sampai selesai, lalu jalankan sel ini lagi.")
print("Memakai:", kandidat[-1])
subprocess.run(["rm", "-rf", "/content/dl-python"])
subprocess.run(["unzip", "-q", "-o", kandidat[-1], "-d", "/content"], check=True)

!pip -q install nbclient nbformat keras-hub keras-tuner fastapi uvicorn httpx datasets

# folder hasil disimpan permanen di Drive
!mkdir -p /content/drive/MyDrive/dl-python-hasil
!rm -rf /content/dl-python/kode/verifikasi/hasil
!ln -s /content/drive/MyDrive/dl-python-hasil /content/dl-python/kode/verifikasi/hasil
%cd /content/dl-python/kode
!cat VERSI.txt
```
Baris terakhir keluarannya menunjukkan versi ZIP yang sedang dipakai.
Ulangi langkah 2–3 setiap kali membuka sesi baru. Bila ada ZIP versi baru,
hapus ZIP lama dari Drive agar yang terbaru yang dipakai.

**Alternatif tanpa ZIP (dari GitHub):** ganti sel di atas dengan
```python
from google.colab import drive
drive.mount("/content/drive")
!rm -rf /content/dl-python && mkdir -p /content/dl-python
!git clone -q https://github.com/bsnegara/deep-learning-dengan-python /content/dl-python/kode
!pip -q install nbclient nbformat keras-hub keras-tuner fastapi uvicorn httpx datasets
!mkdir -p /content/drive/MyDrive/dl-python-hasil
!ln -s /content/drive/MyDrive/dl-python-hasil /content/dl-python/kode/verifikasi/hasil
%cd /content/dl-python/kode
!cat VERSI.txt
```

## Langkah 4 — Uji asap (cepat, epoch dipangkas)
```python
!python jalankan_semua.py
%cd /content/dl-python/kode/notebook
!python uji_notebook.py bab02 bab03 bab04 bab05 bab06
!python uji_notebook.py bab07 bab08 bab09 bab10 bab11
!python uji_notebook.py bab12 bab13 bab14 bab15 bab16 bab17
%cd /content/dl-python/kode
```
Setiap baris harus berakhir dengan **OK**. Bila ada **GAGAL**, buka log-nya di
`notebook/_uji/` lalu kirim pesan error-nya untuk diperbaiki.

## Langkah 5 — Verifikasi penuh, kelompok 1 (±2 jam)
```python
%cd /content/dl-python/kode/verifikasi
!python jalankan_penuh.py bab02 bab03 lr_mnist bow_imdb bab05_01 bab06
```

## Langkah 6 — Kelompok 2 dan 3 (boleh di sesi berbeda)
```python
%cd /content/dl-python/kode/verifikasi
!python jalankan_penuh.py bab05_02 bab05_03 bab09 bab10      # ±1,5 jam
```
```python
%cd /content/dl-python/kode/verifikasi
!python jalankan_penuh.py bab12 bab15 bab16                  # ±2 jam
```
Yang sudah selesai otomatis dilewati. Untuk mengulang satu notebook:
`!python jalankan_penuh.py --ulang bab05_03`.

## Langkah 7 — Buat laporan
```python
%cd /content/dl-python/kode/verifikasi
!python cocokkan.py
!cp laporan_verifikasi.md /content/drive/MyDrive/dl-python-hasil/
```
Hasilnya per angka:
- **OK**: angka di naskah sesuai.
- **BEDA**: hasil di luar toleransi; angka di naskah perlu diperbarui.
- **BELUM**: notebook belum dijalankan atau gagal.

Kirim berkas `laporan_verifikasi.md` (ada di Drive, folder `dl-python-hasil`).
Angka berstatus BEDA akan diperbarui di naskah. Setelah semuanya OK, tanda
merah dapat dihilangkan dengan mode final (`\draffalse`).
