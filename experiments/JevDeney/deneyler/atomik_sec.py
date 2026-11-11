"""Atomik sınıf seçimi (Deney-1B geliştirme). Yeni Jev çağrısı yapmaz.

Havuz: ana koşu (ana:*), atomik v1 (v1:*), atomik v2 (v2:*) olasılıkları.
Seçim yalnızca halka + zincir_dev üzerinde yapılır (kurallar.sinif_sec); seçilen
kural sonra bütün kümelere bir kez uygulanır. zincir_holdout seçimde kullanılmaz.

Kullanım (proje kökünden):
  py JevDeney/deneyler/atomik_sec.py --ana A.json --v1 B.json --v2 C.json
Çıktı: deneyler/sonuclar/jev_atomik_secili_<tarih>.json ve secim_<tarih>.json
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

from kurallar import havuz_olustur, sinif_sec, siniflari_uygula  # noqa: E402
from veri import VARSAYILAN_REPOS, vakalari_yukle  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Atomik sınıf seçimi (yalnızca geliştirme kümeleri)")
    ap.add_argument("--ana", type=Path, required=True)
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--repos", type=Path, default=VARSAYILAN_REPOS)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")

    oku = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    havuz = havuz_olustur({"ana": oku(a.ana), "v1": oku(a.v1), "v2": oku(a.v2)})
    vakalar = vakalari_yukle(a.repos)
    secim = sinif_sec(vakalar, havuz)
    tarih = dt.date.today().isoformat()
    sonuc = siniflari_uygula(havuz, secim["siniflar"], secim["esik"], "jev_atomik_secili")
    sonuc["tarih"] = tarih
    sonuc["not"] = ("Süre ve token, havuzdaki üç koşunun toplamıdır (üst sınır). Yalnızca "
                    "seçili sınıflar sorulsaydı maliyet daha düşük olurdu.")
    cikti = DENEY / "sonuclar"
    (cikti / f"jev_atomik_secili_{tarih}.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    (cikti / f"secim_{tarih}.json").write_text(
        json.dumps(secim, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(secim, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
