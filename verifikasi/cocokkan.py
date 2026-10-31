"""Bandingkan angka di naskah dengan hasil eksekusi di folder hasil/.

Menulis laporan_verifikasi.md dan mencetak ringkasannya.
Status: OK (sesuai), BEDA (perlu ubah naskah atau cek ulang),
        BELUM (hasil belum ada / pola tidak ditemukan).
"""
import json, pathlib, re

from klaim import KLAIM

DI_SINI = pathlib.Path(__file__).resolve().parent
HASIL = DI_SINI / "hasil"


def teks_keluaran(sumber):
    if sumber.endswith(".py"):
        p = HASIL / f"{sumber[:-3]}.txt"
        return p.read_text() if p.exists() else None
    p = HASIL / f"{sumber}.ipynb"
    if not p.exists():
        return None
    nb = json.loads(p.read_text(encoding="utf-8"))
    gabung = lambda x: "".join(x) if isinstance(x, list) else x
    potongan = []
    for sel in nb["cells"]:
        for out in sel.get("outputs", []):
            if out.get("output_type") == "stream":
                potongan.append(gabung(out.get("text", "")))
            elif "data" in out and "text/plain" in out["data"]:
                potongan.append(gabung(out["data"]["text/plain"]))
    return "\n".join(potongan)


def periksa(k, waktu):
    jenis = k["cek"][0]
    if jenis == "menit_maks":
        if k["sumber"] not in waktu:
            return "BELUM", "-", "notebook belum dijalankan"
        m = waktu[k["sumber"]]
        return ("OK" if m <= k["cek"][1] else "BEDA"), f"{m} menit", f"<= {k['cek'][1]} menit"
    teks = teks_keluaran(k["sumber"])
    if teks is None:
        return "BELUM", "-", "hasil belum ada"
    cocok = re.search(k["pola"], teks, flags=re.M)
    if not cocok:
        return "BELUM", "-", "pola tidak ditemukan di keluaran"
    g = cocok.groups()
    if jenis == "nilai":
        _, target, tol = k["cek"]
        v = float(g[0])
        return ("OK" if abs(v - target) <= tol else "BEDA"), f"{v:g}", f"{target} ± {tol}"
    if jenis == "rentang":
        _, lo, hi = k["cek"]
        v = float(g[0])
        return ("OK" if lo <= v <= hi else "BEDA"), f"{v:g}", f"{lo}–{hi}"
    if jenis == "lebih_besar":
        a, b = float(g[0]), float(g[1])
        return ("OK" if a > b else "BEDA"), f"{a:g} vs {b:g}", "pertama > kedua"
    if jenis == "lebih_kecil":
        a, b = float(g[0]), float(g[1])
        return ("OK" if a < b else "BEDA"), f"{a:g} -> {b:g}", "naik"
    if jenis == "tepat":
        ok = list(g) == list(k["cek"][1:])
        return ("OK" if ok else "BEDA"), " / ".join(g), " / ".join(k["cek"][1:])
    return "BELUM", "-", f"jenis cek tidak dikenal: {jenis}"


def main():
    waktu = json.loads((HASIL / "waktu.json").read_text()) \
        if (HASIL / "waktu.json").exists() else {}
    baris = ["# Laporan verifikasi angka naskah", "",
             "| ID | Bab | Status | Hasil | Harapan | Naskah | Sumber |",
             "|---|---|---|---|---|---|---|"]
    hitung = {"OK": 0, "BEDA": 0, "BELUM": 0}
    for k in KLAIM:
        status, hasil, harapan = periksa(k, waktu)
        hitung[status] += 1
        baris.append(f"| {k['id']} | {k['bab']} | **{status}** | {hasil} | "
                     f"{harapan} | {k['teks']} (`{k['tex']}`) | `{k['sumber']}` |")
        print(f"{k['id']}  {status:5s}  hasil={hasil:<16s} harapan={harapan:<14s} {k['teks'][:50]}")
    baris += ["", f"OK: {hitung['OK']} | BEDA: {hitung['BEDA']} | BELUM: {hitung['BELUM']}",
              "", "Untuk status BEDA: perbarui angka di naskah (cari ID-nya di berkas .tex,"
              " misalnya `\\cekangka{...}{K05}`), lalu jalankan ulang skrip ini."]
    (DI_SINI / "laporan_verifikasi.md").write_text("\n".join(baris) + "\n")
    print(f"\nOK: {hitung['OK']} | BEDA: {hitung['BEDA']} | BELUM: {hitung['BELUM']}")
    print("Laporan lengkap: laporan_verifikasi.md")


if __name__ == "__main__":
    main()
