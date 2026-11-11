"""Deney-1 koşucusu: varyant (pr_baglam | atomik) için ortak biçimde sonuç üretir.

Kullanım (proje kökünden):
  py JevDeney/deneyler/deney_kos.py --varyant pr_baglam --istemci gercek
  py JevDeney/deneyler/deney_kos.py --varyant atomik --istemci sahte --vaka authz-01-kusurlu
Çıktı: JevDeney/deneyler/sonuclar/jev_<varyant>[_<istemci>]_<küme|vaka>_<tarih>.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

DENEY = Path(__file__).resolve().parent
sys.path.insert(0, str(DENEY.parent))
sys.path.insert(0, str(DENEY))

from istemci import HamYanit, JevIstemci, SORULAR, istek_govdesi  # noqa: E402
from kos import ESIK, _anahtar, _istemci_kur, _onbellek_oku  # noqa: E402
from varyantlar import (ATOMIK_SORULAR, ATOMIK_V2_SORULAR, atomik_govde, atomik_v2_govde,
                        pr_baglam_govde)  # noqa: E402
from veri import KUMELER, VARSAYILAN_REPOS, Vaka, diffi_bol, vakalari_yukle  # noqa: E402

VARYANTLAR = ("pr_baglam", "atomik", "atomik_v2", "ana")


def birimler(varyant: str, vaka: Vaka, parcalar: dict[str, str],
             model: str) -> list[tuple[dict, dict[str, str]]]:
    """Bir vakanın istek birimleri: (gövde, soru adı → dosya)."""
    if varyant == "pr_baglam":
        return [pr_baglam_govde({d: parcalar[d] for d in vaka.degisen_dosyalar}, model)]
    if varyant == "atomik":
        return [(atomik_govde(d, parcalar[d], model), {q: d for q in ATOMIK_SORULAR})
                for d in vaka.degisen_dosyalar]
    if varyant == "atomik_v2":
        return [(atomik_v2_govde(d, parcalar[d], model), {q: d for q in ATOMIK_V2_SORULAR})
                for d in vaka.degisen_dosyalar]
    if varyant == "ana":
        return [(istek_govdesi(d, parcalar[d], model), {q: d for q in SORULAR})
                for d in vaka.degisen_dosyalar]
    raise ValueError(varyant)


def _gonder(istemci: JevIstemci, govde: dict) -> HamYanit:
    try:
        return istemci.gonder(govde)
    except Exception as hata:  # tek isteğin hatası koşuyu durdurmaz
        return HamYanit(None, None, None, None, hata=f"{type(hata).__name__}: {hata}")


def deney_kos(vakalar: list[Vaka], istemci: JevIstemci, varyant: str, onbellek_yolu: Path,
              esik: float = ESIK) -> dict:
    onbellek = _onbellek_oku(onbellek_yolu)
    onbellek_yolu.parent.mkdir(parents=True, exist_ok=True)
    sonuc_vakalar = []
    with onbellek_yolu.open("a", encoding="utf-8") as ek:
        for vaka in vakalar:
            parcalar = diffi_bol(vaka.diff_yolu.read_text(encoding="utf-8"))
            kayitlar = {d: {"dosya": d, "olasiliklar": {}, "sure_ms": None, "giris_token": None}
                        for d in vaka.degisen_dosyalar}
            for govde, harita in birimler(varyant, vaka, parcalar, istemci.model):
                anahtar = _anahtar(govde)
                if anahtar in onbellek:
                    k = onbellek[anahtar]
                    ham = HamYanit(k["cevaplar"], k["sure_ms"], k["giris_token"], k["model_yanit"])
                else:
                    ham = _gonder(istemci, govde)
                    if ham.hata is None:
                        k = {"anahtar": anahtar, "istek": govde, "cevaplar": ham.cevaplar,
                             "sure_ms": ham.sure_ms, "giris_token": ham.giris_token,
                             "model_yanit": ham.model_yanit}
                        ek.write(json.dumps(k, ensure_ascii=False) + "\n")
                        ek.flush()
                        onbellek[anahtar] = k
                # Süre ve token isteğe aittir: birimdeki ilk dosyaya yazılır (ORTAK_FORMAT).
                ilk = kayitlar[next(iter(harita.values()))]
                ilk["sure_ms"], ilk["giris_token"] = ham.sure_ms, ham.giris_token
                for soru, dosya in harita.items():
                    if ham.hata:
                        kayitlar[dosya]["hata"] = ham.hata
                    else:
                        kayitlar[dosya]["olasiliklar"][soru] = ham.cevaplar[soru]
            for k in kayitlar.values():
                olas = k["olasiliklar"]
                k["skor"] = max(olas.values()) if olas and "hata" not in k else None
                k["kusurlu_mu"] = k["skor"] is not None and k["skor"] >= esik
            sonuc_vakalar.append({"veri_kumesi": vaka.veri_kumesi,
                                  "vaka_kimligi": vaka.vaka_kimligi,
                                  "dosyalar": list(kayitlar.values())})
    ek_ad = "" if istemci.ad == "gercek" else f"_{istemci.ad}"
    return {"arac": f"jev_{varyant}{ek_ad}", "surum": istemci.model, "model": istemci.model,
            "varyant": varyant, "tarih": dt.date.today().isoformat(), "esik": esik,
            "karar_kurali": "kusurlu_mu = max(dosyanin olasiliklari) >= esik",
            "birim_fiyat_usd_1m": getattr(istemci, "birim_fiyat_usd_1m", None),
            "vakalar": sonuc_vakalar}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Jev Deney-1 varyant koşusu")
    ap.add_argument("--varyant", choices=VARYANTLAR, required=True)
    ap.add_argument("--istemci", choices=("sahte", "gercek"), default="sahte")
    ap.add_argument("--kume", choices=KUMELER + ("hepsi",), default="hepsi")
    ap.add_argument("--vaka")
    ap.add_argument("--repos", type=Path, default=VARSAYILAN_REPOS)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")

    istemci = _istemci_kur(a.istemci)
    vakalar = [v for v in vakalari_yukle(a.repos)
               if (a.kume == "hepsi" or v.veri_kumesi == a.kume)
               and (a.vaka is None or v.vaka_kimligi == a.vaka)]
    if not vakalar:
        raise SystemExit("Seçime uyan vaka yok.")
    sonuc = deney_kos(vakalar, istemci, a.varyant,
                      DENEY / "onbellek" / f"{a.varyant}_{istemci.ad}.jsonl")
    cikti = DENEY / "sonuclar" / f"{sonuc['arac']}_{a.vaka or a.kume}_{sonuc['tarih']}.json"
    cikti.parent.mkdir(exist_ok=True)
    cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    dosyalar = [d for v in sonuc["vakalar"] for d in v["dosyalar"]]
    print(f"{len(vakalar)} vaka, {len(dosyalar)} dosya, "
          f"{sum(1 for d in dosyalar if d.get('hata'))} hata -> {cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
