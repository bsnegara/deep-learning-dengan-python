import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

dokumen = [  #<1>
    "Mahasiswa wajib hadir minimal 75 persen pertemuan untuk ikut ujian.",
    "Cuti akademik dapat diajukan paling lama dua semester.",
    "Nilai minimum kelulusan mata kuliah adalah C.",
    "Batas waktu pengisian KRS adalah minggu pertama semester.",
    "Skripsi dapat diajukan setelah menempuh minimal 110 SKS.",
]
vektorisasi = TfidfVectorizer().fit(dokumen)
indeks = vektorisasi.transform(dokumen)          # matriks (5, kosakata)

def ambil(pertanyaan, k=2):  #<2>
    q = vektorisasi.transform([pertanyaan])
    skor = (indeks @ q.T).toarray().ravel()       # kemiripan kosinus
    teratas = np.argsort(skor)[::-1][:k]
    return [(dokumen[i], skor[i]) for i in teratas]

pertanyaan = "Berapa persen kehadiran minimal agar boleh ikut ujian?"
konteks = ambil(pertanyaan)
for teks, skor in konteks:
    print(f"{skor:.2f}  {teks}")

prompt = ("Jawab hanya berdasarkan konteks. Jika tidak ada, katakan "  #<3>
          "tidak tahu.\n\nKonteks:\n"
          + "\n".join(f"- {t}" for t, _ in konteks)
          + f"\n\nPertanyaan: {pertanyaan}\nJawaban:")
print(prompt)
