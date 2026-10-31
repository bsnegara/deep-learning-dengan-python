"""Jalankan semua skrip listing; laporkan yang berhasil, gagal, atau
dilewati karena pustaka deep learning belum terpasang."""
import importlib.util
import os
import pathlib
import subprocess
import sys

akar = pathlib.Path(__file__).parent
# Potongan listing yang tidak berdiri sendiri; diuji lewat skrip lain
POTONGAN = {"bab06/bag_of_words.py": "verifikasi/bow_imdb.py"}
# Skrip yang memakai tf.GradientTape atau tf.data butuh backend TensorFlow
lingkungan = dict(os.environ, KERAS_BACKEND=os.environ.get("KERAS_BACKEND", "tensorflow"))
punya = {m: importlib.util.find_spec(m) is not None
         for m in ["keras", "torch", "tensorflow"]}
for berkas in sorted(akar.glob("bab*/*.py")):
    relatif = berkas.relative_to(akar).as_posix()
    if relatif in POTONGAN:
        print(f"POTONG {relatif} (diuji lewat {POTONGAN[relatif]})")
        continue
    teks = berkas.read_text()
    kurang = [m for m in punya if f"import {m}" in teks and not punya[m]]
    if kurang:
        print(f"LEWATI {berkas.relative_to(akar)} (butuh {', '.join(kurang)})")
        continue
    hasil = subprocess.run([sys.executable, berkas.name], cwd=berkas.parent,
                           capture_output=True, timeout=1800,
                           env=lingkungan)
    status = "OK    " if hasil.returncode == 0 else "GAGAL "
    print(f"{status}{berkas.relative_to(akar)}")
