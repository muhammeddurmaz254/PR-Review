"""Karar eşiği seçimi: yalnızca geliştirme kümelerinde, önceden sabit eşik listesinden.

Ayrılmış test kümesi (zincir_holdout) seçimde kullanılmaz; seçilen eşik ona bir kez
uygulanır.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from degerlendir import puanla  # noqa: E402
from veri import Vaka  # noqa: E402

ESIKLER = (0.5, 0.6, 0.7)
GELISTIRME = ("halka", "zincir_dev")


def esik_uygula(sonuc: dict, esik: float) -> dict:
    """Kararları `max(olasiliklar) >= esik` ile yeniden yazar; kaynak bozulmaz."""
    yeni = copy.deepcopy(sonuc)
    for v in yeni["vakalar"]:
        for d in v["dosyalar"]:
            olas = d.get("olasiliklar") or {}
            d["kusurlu_mu"] = bool(olas) and not d.get("hata") and max(olas.values()) >= esik
    yeni["esik"] = esik
    yeni["arac"] = f"{sonuc['arac']}_e{round(esik * 100):03d}"
    return yeni


def esik_sec(vakalar: list[Vaka], sonuc: dict, esikler=ESIKLER, gelistirme=GELISTIRME) -> dict:
    """Geliştirme kümelerinin birleşiminde F1'i en yüksek eşik; eşitlikte küçük olan."""
    dev = [v for v in vakalar if v.veri_kumesi in gelistirme]
    dev_kimlik = {v.vaka_kimligi for v in dev}
    tarama = {}
    for e in esikler:
        aday = esik_uygula(sonuc, e)
        aday["vakalar"] = [v for v in aday["vakalar"] if v["vaka_kimligi"] in dev_kimlik]
        m = puanla(dev, aday)["toplam"]
        tarama[e] = {"p": m.p, "r": m.r, "f1": m.f1}
    secilen = max(esikler, key=lambda e: (round(tarama[e]["f1"], 9), -e))
    return {"esik": secilen, "tarama": tarama, "gelistirme": list(gelistirme)}


def havuz_olustur(kaynaklar: dict[str, dict]) -> dict:
    """Aynı dosyalar için farklı koşuların olasılıklarını ön ekle birleştirir
    ('v1:wrong_logic'). Süre ve token toplanır (birlikte sorma maliyetinin üst sınırı)."""
    havuz: dict[tuple[str, str], dict] = {}
    sira: list[tuple[str, str, str]] = []
    for on_ek, sonuc in kaynaklar.items():
        for v in sonuc["vakalar"]:
            for d in v["dosyalar"]:
                k = (v["vaka_kimligi"], d["dosya"])
                if k not in havuz:
                    havuz[k] = {"dosya": d["dosya"], "olasiliklar": {}, "sure_ms": 0, "giris_token": 0}
                    sira.append((v["veri_kumesi"], v["vaka_kimligi"], d["dosya"]))
                h = havuz[k]
                for soru, p in (d.get("olasiliklar") or {}).items():
                    h["olasiliklar"][f"{on_ek}:{soru}"] = p
                h["sure_ms"] += d.get("sure_ms") or 0
                h["giris_token"] += d.get("giris_token") or 0
                if d.get("hata"):
                    h["hata"] = d["hata"]
    vakalar: dict[str, dict] = {}
    for kume, vaka, dosya in sira:
        vakalar.setdefault(vaka, {"veri_kumesi": kume, "vaka_kimligi": vaka, "dosyalar": []})
        vakalar[vaka]["dosyalar"].append(havuz[(vaka, dosya)])
    ilk = next(iter(kaynaklar.values()))
    return {"arac": "havuz", "model": ilk.get("model"),
            "birim_fiyat_usd_1m": ilk.get("birim_fiyat_usd_1m"), "vakalar": list(vakalar.values())}


def siniflari_uygula(sonuc: dict, siniflar: list[str], esik: float, arac: str) -> dict:
    """Karar: seçilen sınıfların maks. olasılığı >= eşik."""
    yeni = copy.deepcopy(sonuc)
    for v in yeni["vakalar"]:
        for d in v["dosyalar"]:
            olas = d.get("olasiliklar") or {}
            secili = [olas[s] for s in siniflar if s in olas]
            d["skor"] = max(secili) if secili and not d.get("hata") else None
            d["kusurlu_mu"] = d["skor"] is not None and d["skor"] >= esik
    yeni.update({"arac": arac, "esik": esik, "siniflar": list(siniflar),
                 "karar_kurali": "kusurlu_mu = max(secili siniflarin olasiliklari) >= esik"})
    return yeni


def _dev_f1(vakalar, dev_kimlik, sonuc, siniflar, esik):
    aday = siniflari_uygula(sonuc, siniflar, esik, "aday")
    aday["vakalar"] = [v for v in aday["vakalar"] if v["vaka_kimligi"] in dev_kimlik]
    return puanla([v for v in vakalar if v.vaka_kimligi in dev_kimlik], aday)["toplam"]


def sinif_sec(vakalar: list[Vaka], sonuc: dict, esikler=ESIKLER, gelistirme=GELISTIRME) -> dict:
    """Her eşik için sınıflar dev kesinliğine göre sıralanır, bütün önekler denenir;
    dev F1'i en yüksek (eşitlikte daha az sınıflı, sonra küçük eşikli) seçim döner.
    Ayrılmış test kümesi hiçbir adımda kullanılmaz."""
    dev_kimlik = {v.vaka_kimligi for v in vakalar if v.veri_kumesi in gelistirme}
    siniflar = sorted({s for v in sonuc["vakalar"] for d in v["dosyalar"]
                       for s in (d.get("olasiliklar") or {})})
    en_iyi = None
    for e in esikler:
        tek = {s: _dev_f1(vakalar, dev_kimlik, sonuc, [s], e) for s in siniflar}
        sira = sorted(siniflar, key=lambda s: (-tek[s].p, -tek[s].tp, s))
        for n in range(1, len(sira) + 1):
            m = _dev_f1(vakalar, dev_kimlik, sonuc, sira[:n], e)
            aday = (round(m.f1, 9), -n, -e)
            if en_iyi is None or aday > en_iyi[0]:
                en_iyi = (aday, {"siniflar": sira[:n], "esik": e, "dev_f1": m.f1,
                                 "dev_p": m.p, "dev_r": m.r})
    sonuc_secim = en_iyi[1]
    sonuc_secim["gelistirme"] = list(gelistirme)
    return sonuc_secim
