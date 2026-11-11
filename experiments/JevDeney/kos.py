"""Jev koşucusu: her değişen dosyanın diff parçası için Jev'e SORU'yu sorar,
sonucu ortak biçimde (ORTAK_FORMAT.md) `sonuclar/` klasörüne yazar.

Kullanım:
  py JevDeney/kos.py --istemci sahte --kume hepsi
  py JevDeney/kos.py --istemci gercek --kume halka      (anahtar gerekir)

Önbellek: `onbellek/<istemci>.jsonl`. Başarılı yanıtlar saklanır, aynı soru
tekrar sorulmaz. Hatalı yanıtlar saklanmaz, sonraki koşuda yeniden sorulur.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

from istemci import (SORULAR, GercekIstemci, JevIstemci, SahteIstemci, Yanit,
                     anahtar_oku, istek_govdesi)
from veri import KUMELER, VARSAYILAN_REPOS, Vaka, diffi_bol, vakalari_yukle

ESIK = 0.5
KLASOR = Path(__file__).parent


def _anahtar(govde: dict) -> str:
    """Önbellek anahtarı: modeli, soruları ve state'i içeren tam istek gövdesi."""
    return hashlib.sha256(json.dumps(govde, sort_keys=True).encode("utf-8")).hexdigest()


def _onbellek_oku(yol: Path) -> dict[str, dict]:
    if not yol.exists():
        return {}
    kayitlar = {}
    for satir in yol.read_text(encoding="utf-8").splitlines():
        if satir.strip():
            k = json.loads(satir)
            kayitlar[k["anahtar"]] = k
    return kayitlar


def _sor(istemci: JevIstemci, dosya: str, parca: str) -> Yanit:
    baslangic = time.perf_counter()
    try:
        yanit = istemci.sor(dosya, parca)
    except Exception as hata:  # tek dosyanın hatası koşuyu durdurmaz
        return Yanit(None, round((time.perf_counter() - baslangic) * 1000), None,
                     hata=f"{type(hata).__name__}: {hata}")
    if yanit.olasilik is None and yanit.hata is None:
        yanit.hata = "yanitta olasilik yok"
    return yanit


def kos(vakalar: list[Vaka], istemci: JevIstemci, onbellek_yolu: Path,
        esik: float = ESIK) -> dict:
    """Vakaları koşar ve ortak biçimde sonuç sözlüğü döndürür.

    Hata veren dosya "hayır" sayılmaz: `hata` alanıyla yazılır ve değerlendirici
    koşuyu geçersiz sayar. Tekrar koşulduğunda yalnızca hatalı dosyalar sorulur.
    """
    onbellek = _onbellek_oku(onbellek_yolu)
    onbellek_yolu.parent.mkdir(parents=True, exist_ok=True)
    sonuc_vakalar = []
    with onbellek_yolu.open("a", encoding="utf-8") as ek:
        for vaka in vakalar:
            parcalar = diffi_bol(vaka.diff_yolu.read_text(encoding="utf-8"))
            dosyalar = []
            for dosya in vaka.degisen_dosyalar:
                parca = parcalar[dosya]
                govde = istek_govdesi(dosya, parca, istemci.model)
                anahtar = _anahtar(govde)
                if anahtar in onbellek:
                    k = onbellek[anahtar]
                    yanit = Yanit(k["olasilik"], k["sure_ms"], k["giris_token"],
                                  olasiliklar=k["olasiliklar"], model_yanit=k.get("model_yanit"))
                else:
                    yanit = _sor(istemci, dosya, parca)
                    if yanit.hata is None:
                        k = {"anahtar": anahtar, "istek": govde,
                             "olasiliklar": yanit.olasiliklar, "olasilik": yanit.olasilik,
                             "model_yanit": yanit.model_yanit,
                             "sure_ms": yanit.sure_ms, "giris_token": yanit.giris_token}
                        ek.write(json.dumps(k, ensure_ascii=False) + "\n")
                        ek.flush()
                        onbellek[anahtar] = k
                kayit = {"dosya": dosya,
                         "kusurlu_mu": yanit.olasilik is not None and yanit.olasilik >= esik,
                         "skor": yanit.olasilik, "olasiliklar": yanit.olasiliklar,
                         "sure_ms": yanit.sure_ms, "giris_token": yanit.giris_token}
                if yanit.hata:
                    kayit["hata"] = yanit.hata
                dosyalar.append(kayit)
            sonuc_vakalar.append({"veri_kumesi": vaka.veri_kumesi,
                                  "vaka_kimligi": vaka.vaka_kimligi, "dosyalar": dosyalar})
    return {"arac": "jev" if istemci.ad == "gercek" else f"jev_{istemci.ad}",
            "surum": istemci.model, "model": istemci.model,
            "tarih": dt.date.today().isoformat(), "esik": esik,
            "karar_kurali": "kusurlu_mu = max(olasiliklar) >= esik",
            "sorular": SORULAR,
            "birim_fiyat_usd_1m": getattr(istemci, "birim_fiyat_usd_1m", None),
            "vakalar": sonuc_vakalar}


def _istemci_kur(ad: str) -> JevIstemci:
    if ad == "sahte":
        return SahteIstemci()
    anahtar = anahtar_oku(KLASOR)
    if not anahtar:
        ipucu = " (.env.txt var: uzantıyı .env yapın)" if (KLASOR / ".env.txt").exists() else ""
        raise SystemExit(f"TYPESAFE_API_KEY bulunamadı: ortam değişkeni ya da "
                         f"{KLASOR / '.env'} gerekli{ipucu}.")
    return GercekIstemci(anahtar)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Jev taban çizgisi koşusu")
    ap.add_argument("--istemci", choices=("sahte", "gercek"), default="sahte")
    ap.add_argument("--kume", choices=KUMELER + ("hepsi",), default="hepsi")
    ap.add_argument("--vaka", help="yalnızca bu vaka kimliği (duman testi için)")
    ap.add_argument("--repos", type=Path, default=VARSAYILAN_REPOS)
    ap.add_argument("--cikti", type=Path, default=None,
                    help="sonuç ve önbellek için kök klasör (varsayılan: JevDeney/)")
    a = ap.parse_args(argv)
    # Windows konsolu (cp1254) basamadığı karakterde çökmesin
    sys.stdout.reconfigure(errors="replace")

    istemci = _istemci_kur(a.istemci)
    vakalar = [v for v in vakalari_yukle(a.repos)
               if (a.kume == "hepsi" or v.veri_kumesi == a.kume)
               and (a.vaka is None or v.vaka_kimligi == a.vaka)]
    if not vakalar:
        raise SystemExit("Seçime uyan vaka yok.")
    kok = a.cikti or KLASOR
    sonuc = kos(vakalar, istemci, kok / "onbellek" / f"{istemci.ad}.jsonl")

    cikti = (a.cikti or KLASOR / "sonuclar") / f"jev_{istemci.ad}_{a.vaka or a.kume}_{sonuc['tarih']}.json"
    cikti.parent.mkdir(exist_ok=True)
    cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    dosyalar = [d for v in sonuc["vakalar"] for d in v["dosyalar"]]
    print(f"{len(vakalar)} vaka, {len(dosyalar)} dosya, "
          f"{sum(1 for d in dosyalar if d.get('hata'))} hata → {cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
