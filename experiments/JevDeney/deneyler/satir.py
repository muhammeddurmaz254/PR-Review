"""Deney-3b: Jev'in kusurlu dediği dosyada sorunlu hunk'ı (değişiklik bloğunu) bulmak.

TypeSafe "line-by-line search" tarifine dayanır: her hunk ayrı bir Noul, hepsi tek
istekte. En yüksek olasılıklı hunk tahmindir. Tek hunk varsa çağrı yapılmaz.

Kullanım (proje kökünden):
  py JevDeney/deneyler/satir.py --kaynak JevDeney/sonuclar/jev_gercek_hepsi_<tarih>.json --istemci gercek
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

DENEY = Path(__file__).resolve().parent
sys.path.insert(0, str(DENEY.parent))

from istemci import HamYanit, JevIstemci  # noqa: E402
from kos import _anahtar, _istemci_kur, _onbellek_oku  # noqa: E402
from veri import VARSAYILAN_REPOS, Vaka, diffi_bol, vakalari_yukle  # noqa: E402

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    metin: str
    yeni_bas: int
    yeni_son: int


def hunklari_bol(parca: str) -> list[Hunk]:
    """Dosya diff'ini hunk'lara böler. Aralık yeni (head) taraftaki satırlardır;
    yalnızca silme içeren hunk, silmenin olduğu yerdeki tek satır olarak alınır."""
    hunklar: list[Hunk] = []
    satirlar: list[str] = []
    bas = say = None
    for satir in parca.replace("\r\n", "\n").split("\n"):
        e = _HUNK.match(satir)
        if e:
            if satirlar:
                hunklar.append(Hunk("\n".join(satirlar), bas, bas + max(say, 1) - 1))
            bas, say = int(e.group(1)), int(e.group(2) or 1)
            satirlar = [satir]
        elif satirlar:
            satirlar.append(satir)
    if satirlar:
        hunklar.append(Hunk("\n".join(satirlar).rstrip("\n"), bas, bas + max(say, 1) - 1))
    return hunklar


BAGLAM_SATIRI = 2   # blok metnine eklenen önceki/sonraki değişmeyen satır sayısı


def bloklari_bol(parca: str) -> list[Hunk]:
    """Her hunk'ı ardışık değişen satır dizilerine (bloklara) böler.

    Aralık, bloğun eklediği satırların yeni taraftaki numaralarıdır. Yalnızca
    silme içeren blok, silmenin olduğu yerdeki yeni satır olarak alınır."""
    bloklar: list[Hunk] = []
    for hunk in hunklari_bol(parca):
        satirlar = hunk.metin.split("\n")[1:]          # ilk satır @@ başlığı
        yeni_no = hunk.yeni_bas
        numaralar = []                                  # (satır, yeni no ya da None)
        for satir in satirlar:
            if satir.startswith("-"):
                numaralar.append((satir, None, yeni_no))
            else:
                numaralar.append((satir, yeni_no, yeni_no))
                yeni_no += 1
        i = 0
        while i < len(numaralar):
            if numaralar[i][0][:1] not in "+-" or numaralar[i][0] == "":
                i += 1
                continue
            j = i
            while j < len(numaralar) and numaralar[j][0][:1] in ("+", "-") and numaralar[j][0]:
                j += 1
            eklenen = [n for satir, n, _ in numaralar[i:j] if satir.startswith("+")]
            bas, son = (min(eklenen), max(eklenen)) if eklenen else (numaralar[i][2],) * 2
            metin = [x[0] for x in numaralar[max(0, i - BAGLAM_SATIRI):j + BAGLAM_SATIRI]]
            bloklar.append(Hunk("\n".join(metin), bas, son))
            i = j
    return bloklar


_BIRIM = {
    "hunk": {"anahtar": "hunks", "on_ek": "h", "ad": "hunk", "bol": hunklari_bol},
    "blok": {"anahtar": "blocks", "on_ek": "b", "ad": "block", "bol": bloklari_bol},
}


def satir_govde(dosya: str, hunklar: list[Hunk], model: str, birim: str = "hunk",
                tam_diff: str | None = None) -> dict:
    b = _BIRIM[birim]
    sorular = {}
    for i in range(len(hunklar)):
        yol = f"{b['anahtar']}.{b['on_ek']}{i}"
        odak = (f"The file is known to contain one problem introduced by this change. "
                f"The other entries in `{b['anahtar']}` are the rest of the same file diff.")
        if tam_diff is not None:
            odak += " The full `diff` is given for context."
        sorular[f"{b['on_ek']}{i}"] = {
            "type": "noul",
            "instructions": {
                "question": f"Does the {b['ad']} `{yol}` contain the change that makes "
                            f"`file_path` behave incorrectly or insecurely?",
                "focus": odak,
            },
            "criteria": {
                "true": {"what": f"The lines this {b['ad']} adds or changes are where the problem is"},
                "false": {"what": f"The lines this {b['ad']} adds or changes are correct; the "
                                  f"problem is in another {b['ad']}"},
            },
        }
    durum = {"file_path": dosya}
    if tam_diff is not None:
        durum["diff"] = tam_diff
    durum[b["anahtar"]] = {f"{b['on_ek']}{i}": h.metin for i, h in enumerate(hunklar)}
    return {"model": model, "state": durum, "questions": sorular}


def satir_kos(vakalar: list[Vaka], kaynak: dict, istemci: JevIstemci, onbellek_yolu: Path,
              parcalar: dict[str, dict[str, str]] | None = None, birim: str = "hunk") -> dict:
    """Kaynak sonuçta kusurlu_mu=True olan her dosya için hunk tahmini üretir."""
    isaretli = {(v["vaka_kimligi"], d["dosya"]) for v in kaynak["vakalar"]
                for d in v["dosyalar"] if d.get("kusurlu_mu")}
    onbellek = _onbellek_oku(onbellek_yolu)
    onbellek_yolu.parent.mkdir(parents=True, exist_ok=True)
    sonuc_vakalar = []
    with onbellek_yolu.open("a", encoding="utf-8") as ek:
        for vaka in vakalar:
            dosyalar = [d for d in vaka.degisen_dosyalar if (vaka.vaka_kimligi, d) in isaretli]
            if not dosyalar:
                continue
            vp = (parcalar or {}).get(vaka.vaka_kimligi) or diffi_bol(
                vaka.diff_yolu.read_text(encoding="utf-8"))
            kayitlar = []
            for dosya in dosyalar:
                hunklar = _BIRIM[birim]["bol"](vp[dosya])
                kayit = {"dosya": dosya, "tek_hunk": len(hunklar) == 1,
                         "hunk_araliklari": [[h.yeni_bas, h.yeni_son] for h in hunklar],
                         "olasiliklar": {}, "sure_ms": None, "giris_token": None}
                if len(hunklar) == 1:
                    secilen = 0
                else:
                    govde = satir_govde(dosya, hunklar, istemci.model, birim,
                                        vp[dosya] if birim == "blok" else None)
                    anahtar = _anahtar(govde)
                    if anahtar in onbellek:
                        k = onbellek[anahtar]
                        ham = HamYanit(k["cevaplar"], k["sure_ms"], k["giris_token"], k["model_yanit"])
                    else:
                        try:
                            ham = istemci.gonder(govde)
                        except Exception as hata:
                            ham = HamYanit(None, None, None, None, hata=f"{type(hata).__name__}: {hata}")
                        if ham.hata is None:
                            k = {"anahtar": anahtar, "istek": govde, "cevaplar": ham.cevaplar,
                                 "sure_ms": ham.sure_ms, "giris_token": ham.giris_token,
                                 "model_yanit": ham.model_yanit}
                            ek.write(json.dumps(k, ensure_ascii=False) + "\n")
                            ek.flush()
                            onbellek[anahtar] = k
                    kayit["sure_ms"], kayit["giris_token"] = ham.sure_ms, ham.giris_token
                    if ham.hata:
                        kayit["hata"] = ham.hata
                        kayitlar.append(kayit)
                        continue
                    kayit["olasiliklar"] = ham.cevaplar
                    on_ek = _BIRIM[birim]["on_ek"]
                    secilen = max(range(len(hunklar)), key=lambda i: ham.cevaplar[f"{on_ek}{i}"])
                kayit["tahmin_bas"] = hunklar[secilen].yeni_bas
                kayit["tahmin_son"] = hunklar[secilen].yeni_son
                kayitlar.append(kayit)
            sonuc_vakalar.append({"veri_kumesi": vaka.veri_kumesi,
                                  "vaka_kimligi": vaka.vaka_kimligi, "dosyalar": kayitlar})
    ek_ad = "" if istemci.ad == "gercek" else f"_{istemci.ad}"
    arac = "jev_satir" if birim == "hunk" else "jev_blok"
    return {"arac": f"{arac}{ek_ad}", "birim": birim, "model": istemci.model,
            "kaynak_arac": kaynak.get("arac"), "tarih": dt.date.today().isoformat(),
            "birim_fiyat_usd_1m": getattr(istemci, "birim_fiyat_usd_1m", None),
            "vakalar": sonuc_vakalar}


def konumlari_yukle(repos: Path = VARSAYILAN_REPOS) -> dict[str, list[dict]]:
    """vaka → [{dosya, bas, son}] : kusur konumları ve eşdeğer konumları."""
    sonuc: dict[str, list[dict]] = {}
    for bench in ("halka_bench", "zincir_bench"):
        for yol in sorted((Path(repos) / bench / "labels").glob("*.json")):
            etiket = json.loads(yol.read_text(encoding="utf-8"))
            liste = []
            for k in etiket["kusurlar"]:
                for konum in [k["konum"], *k.get("esdeger_konumlar", [])]:
                    if "baslangic_satiri" in konum:
                        liste.append({"dosya": konum["dosya"], "bas": konum["baslangic_satiri"],
                                      "son": konum.get("bitis_satiri", konum["baslangic_satiri"])})
            sonuc[etiket["vaka_kimligi"]] = liste
    return sonuc


def _cakisir(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def satir_degerlendir(vakalar: list[Vaka], sonuc: dict, konumlar: dict[str, list[dict]]) -> dict:
    """Yalnızca doğru işaretlenmiş (pozitif) dosyalarda hunk isabetini ölçer."""
    pozitif = {(v.vaka_kimligi, d) for v in vakalar for d in v.pozitif}
    toplam_pozitif = len(pozitif)
    dogru, isabet, hata, cok_hunk, cok_isabet = 0, 0, 0, 0, 0
    rastgele, rastgele_cok = [], []
    buyuk_isabet = 0
    for v in sonuc["vakalar"]:
        for d in v["dosyalar"]:
            if (v["vaka_kimligi"], d["dosya"]) not in pozitif:
                continue
            if d.get("hata"):
                hata += 1
                continue
            dogru += 1
            hedef = [(k["bas"], k["son"]) for k in konumlar.get(v["vaka_kimligi"], [])
                     if k["dosya"] == d["dosya"]]
            vurdu = any(_cakisir((d["tahmin_bas"], d["tahmin_son"]), h) for h in hedef)
            isabet += vurdu
            araliklar = [tuple(x) for x in d["hunk_araliklari"]]
            beklenen = sum(any(_cakisir(a, h) for h in hedef) for a in araliklar) / len(araliklar)
            rastgele.append(beklenen)
            if not d["tek_hunk"]:
                rastgele_cok.append(beklenen)
                cok_hunk += 1
                buyuk = max(araliklar, key=lambda x: x[1] - x[0])
                buyuk_isabet += any(_cakisir(buyuk, h) for h in hedef)
                cok_isabet += vurdu
    return {"dogru_dosya": dogru, "isabet": isabet, "hata": hata,
            "isabet_orani": isabet / dogru if dogru else 0.0,
            "cok_hunk_dosya": cok_hunk, "cok_hunk_isabet": cok_isabet,
            "cok_hunk_orani": cok_isabet / cok_hunk if cok_hunk else 0.0,
            "rastgele_beklenen": statistics.mean(rastgele) if rastgele else 0.0,
            "rastgele_cok_hunk": statistics.mean(rastgele_cok) if rastgele_cok else 0.0,
            "en_buyuk_cok_hunk_isabet": buyuk_isabet,
            "uctan_uca_satir_r": isabet / toplam_pozitif if toplam_pozitif else 0.0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Jev Deney-3b: hunk düzeyinde satır bulma")
    ap.add_argument("--kaynak", type=Path, required=True, help="dosya düzeyi sonuç (ortak biçim)")
    ap.add_argument("--istemci", choices=("sahte", "gercek"), default="sahte")
    ap.add_argument("--repos", type=Path, default=VARSAYILAN_REPOS)
    ap.add_argument("--birim", choices=tuple(_BIRIM), default="hunk")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")

    kaynak = json.loads(a.kaynak.read_text(encoding="utf-8"))
    istemci = _istemci_kur(a.istemci)
    vakalar = vakalari_yukle(a.repos)
    sonuc = satir_kos(vakalar, kaynak, istemci,
                      DENEY / "onbellek" / f"{a.birim}_satir_{istemci.ad}.jsonl"
                      if a.birim != "hunk" else DENEY / "onbellek" / f"satir_{istemci.ad}.jsonl",
                      birim=a.birim)
    sonuc["degerlendirme"] = satir_degerlendir(vakalar, sonuc, konumlari_yukle(a.repos))
    cikti = DENEY / "sonuclar" / f"{sonuc['arac']}_{kaynak.get('arac')}_{sonuc['tarih']}.json"
    cikti.parent.mkdir(exist_ok=True)
    cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(sonuc["degerlendirme"], ensure_ascii=False, indent=1))
    print(f"-> {cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
