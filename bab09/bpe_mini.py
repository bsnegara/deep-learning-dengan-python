from collections import Counter

korpus = ("makan makanan dimakan pemakan makan "
          "minum minuman diminum peminum").split()
# Setiap kata dipecah menjadi karakter + penanda akhir kata "_"  #<1>
kata = Counter(tuple(k) + ("_",) for k in korpus)

def hitung_pasangan(kata):
    pasangan = Counter()
    for simbol, frek in kata.items():
        for a, b in zip(simbol, simbol[1:]):
            pasangan[a, b] += frek  #<2>
    return pasangan

def gabung(kata, pasangan_terpilih):
    a, b = pasangan_terpilih
    hasil = Counter()
    for simbol, frek in kata.items():
        baru, i = [], 0
        while i < len(simbol):
            if i < len(simbol) - 1 and (simbol[i], simbol[i + 1]) == (a, b):
                baru.append(a + b)  #<3>
                i += 2
            else:
                baru.append(simbol[i])
                i += 1
        hasil[tuple(baru)] += frek
    return hasil

for langkah in range(1, 9):
    terbaik, frek = hitung_pasangan(kata).most_common(1)[0]
    kata = gabung(kata, terbaik)
    print(f"gabungan {langkah}: {terbaik[0]!r} + {terbaik[1]!r}"
          f" -> {''.join(terbaik)!r} (frekuensi {frek})")
print("hasil akhir:", [" ".join(s) for s in kata])
