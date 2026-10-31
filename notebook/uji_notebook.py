"""Uji asap (smoke test) semua notebook buku Deep Learning dengan Python.

Menjalankan setiap notebook secara berurutan dalam folder ini dengan epoch dan
iterasi yang dipangkas, lalu melaporkan OK/GAGAL. Tujuannya memastikan kode
berjalan dari awal sampai akhir, bukan mereproduksi akurasi.

Pemakaian (Colab, runtime GPU):
    !git clone ... atau unggah folder notebook/, lalu
    %cd notebook
    !python uji_notebook.py                 # semua notebook
    !python uji_notebook.py bab06 bab14_04  # hanya yang namanya mengandung pola ini
"""
import json, os, re, subprocess, sys, time, glob

# Penggantian umum: semua fit() cukup 1 epoch, loop manual dipangkas
UMUM = [
    (r"epochs=\d+", "epochs=1"),
    (r"^(EPOCH\w*) = \d+", r"\1 = 1"),
    (r"^(LANGKAH\w*) = [\d_]+", r"\1 = 20"),
    (r"^EVAL_SETIAP = \d+", "EVAL_SETIAP = 10"),
    (r"^MAKS_PERCOBAAN = \d+", "MAKS_PERCOBAAN = 2"),
]

# Penggantian khusus per notebook (pola regex -> pengganti)
KHUSUS = {
    "bab03_03": [(r"^EPOCH = \d+", "EPOCH = 15")],
    "bab09_02": [(r"^N_LATIH = \d+", "N_LATIH = 256")],
    "bab05_04": [(r"langkah=40", "langkah=2")],
    "bab14_01": [(r"^EPOCH_TOTAL = \d+", "EPOCH_TOTAL = 1"), (r"n=400", "n=20")],
    "bab14_02": [(r"^ITERASI = \d+", "ITERASI = 2")],
    "bab14_03": [(r"^ITERASI = \d+", "ITERASI = 3"), (r"^SIMPAN_SETIAP = \d+", "SIMPAN_SETIAP = 1")],
}

def ke_skrip(jalur):
    nb = json.load(open(jalur, encoding="utf-8"))
    kunci = os.path.basename(jalur)[:8]
    bagian = []
    for i, sel in enumerate(nb["cells"]):
        if sel["cell_type"] != "code":
            continue
        kode = "".join(sel["source"])
        baris = []
        for b in kode.split("\n"):
            if b.lstrip().startswith("!pip"):          # !pip install ... -> subprocess
                argumen = b.lstrip()[1:].split()[1:]
                baris.append(f"import subprocess, sys; subprocess.run([sys.executable, '-m', 'pip', *{argumen!r}], check=True)")
            elif not b.lstrip().startswith(("!", "%")):
                baris.append(b)
        kode = "\n".join(baris)
        for pola, ganti in UMUM + KHUSUS.get(kunci, []):
            kode = re.sub(pola, ganti, kode, flags=re.M)
        bagian.append(f"# ---- sel {i} ----\n{kode}\n")
    return "\n".join(bagian)

def main():
    filter_ = sys.argv[1:]
    daftar = sorted(glob.glob("bab*.ipynb"))
    if filter_:
        daftar = [p for p in daftar if any(f in p for f in filter_)]
    os.makedirs("_uji", exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    hasil = []
    for p in daftar:
        skrip = os.path.join("_uji", p.replace(".ipynb", ".py"))
        open(skrip, "w", encoding="utf-8").write(ke_skrip(p))
        print(f"\n>>> {p}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, skrip], env=env, capture_output=True, text=True)
        durasi = time.time() - t0
        open(skrip.replace(".py", ".log"), "w", encoding="utf-8").write(r.stdout + "\n" + r.stderr)
        status = "OK" if r.returncode == 0 else "GAGAL"
        hasil.append((p, status, durasi))
        print(f"    {status} ({durasi:.0f} dtk)")
        if status == "GAGAL":
            print("    " + "\n    ".join(r.stderr.strip().split("\n")[-8:]))
    print("\n================ RINGKASAN ================")
    for p, s, d in hasil:
        print(f"{s:6s} {d:6.0f} dtk  {p}")
    gagal = sum(s == "GAGAL" for _, s, _ in hasil)
    print(f"\n{len(hasil) - gagal}/{len(hasil)} notebook lolos. Log lengkap ada di folder _uji/.")
    sys.exit(1 if gagal else 0)

if __name__ == "__main__":
    main()
