"""Jalankan notebook (penuh, bukan versi dipangkas) dan skrip verifikasi.

Hasil disimpan di folder hasil/ :
  hasil/<notebook>.ipynb   notebook lengkap dengan keluarannya
  hasil/<skrip>.txt        keluaran skrip verifikasi (lr_mnist.py, bow_imdb.py)
  hasil/waktu.json         lama eksekusi tiap notebook/skrip (menit)

Pemakaian:
  python jalankan_penuh.py                  # semua yang dibutuhkan klaim.py
  python jalankan_penuh.py bab05 bab06      # hanya yang namanya mengandung pola
  python jalankan_penuh.py --ulang bab05_03 # paksa ulang walau hasil sudah ada
"""
import json, pathlib, subprocess, sys, time

import nbformat
from nbclient import NotebookClient

from klaim import KLAIM

DI_SINI = pathlib.Path(__file__).resolve().parent
FOLDER_NB = DI_SINI.parent / "notebook"
HASIL = DI_SINI / "hasil"
HASIL.mkdir(exist_ok=True)
WAKTU = HASIL / "waktu.json"

# urutan penting: bab05_04 memakai model hasil bab05_02
SUMBER = sorted({k["sumber"] for k in KLAIM}, key=lambda s: (s.endswith(".py"), s))


def simpan_waktu(nama, menit):
    data = json.loads(WAKTU.read_text()) if WAKTU.exists() else {}
    data[nama] = round(menit, 1)
    WAKTU.write_text(json.dumps(data, indent=1))


def jalankan_notebook(nama):
    nb = nbformat.read(FOLDER_NB / f"{nama}.ipynb", as_version=4)
    klien = NotebookClient(nb, timeout=None, kernel_name="python3",
                           resources={"metadata": {"path": str(FOLDER_NB)}})
    mulai = time.time()
    try:
        klien.execute()
        status = "OK"
    except Exception as e:                      # tetap simpan keluaran parsial
        status = f"GAGAL: {type(e).__name__}: {str(e).splitlines()[-1][:200]}"
    nbformat.write(nb, HASIL / f"{nama}.ipynb")
    return status, (time.time() - mulai) / 60


def jalankan_skrip(nama):
    mulai = time.time()
    p = subprocess.run([sys.executable, nama], cwd=DI_SINI,
                       capture_output=True, text=True)
    (HASIL / f"{nama[:-3]}.txt").write_text(p.stdout + "\n" + p.stderr[-3000:])
    status = "OK" if p.returncode == 0 else f"GAGAL (kode {p.returncode})"
    return status, (time.time() - mulai) / 60


def main():
    argumen = [a for a in sys.argv[1:] if a != "--ulang"]
    ulang = "--ulang" in sys.argv
    daftar = [s for s in SUMBER if not argumen or any(a in s for a in argumen)]
    for nama in daftar:
        sudah = (HASIL / (nama[:-3] + ".txt" if nama.endswith(".py")
                          else nama + ".ipynb")).exists()
        if sudah and not ulang:
            print(f"LEWATI  {nama} (hasil sudah ada; pakai --ulang untuk mengulang)")
            continue
        print(f"MULAI   {nama} ...", flush=True)
        if nama.endswith(".py"):
            status, menit = jalankan_skrip(nama)
        else:
            status, menit = jalankan_notebook(nama)
        simpan_waktu(nama, menit)
        print(f"{status:7s} {nama} ({menit:.1f} menit)", flush=True)


if __name__ == "__main__":
    main()
