"""Ana koşu ve yan deneylerin sonuçlarını tek bir Markdown raporunda karşılaştırır.

Kullanım (proje kökünden):
  py JevDeney/deneyler/karsilastir.py SONUC.json [SONUC2.json ...] [--satir SATIR.json ...]
Çıktı: JevDeney/deneyler/sonuclar/karsilastirma.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DENEY = Path(__file__).resolve().parent
sys.path.insert(0, str(DENEY.parent))
sys.path.insert(0, str(DENEY))

from degerlendir import TUM_KUMELER, puanla  # noqa: E402
from kirilim import cift_skoru, kirilim, temiz_yanlis_alarm  # noqa: E402
from veri import VARSAYILAN_REPOS, Vaka, vakalari_yukle  # noqa: E402


def _oran(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b:.0%})" if b else "—"


def _sayi(x, basamak=0):
    return "—" if x is None else f"{x:.{basamak}f}"


def rapor_md(vakalar: list[Vaka], sonuclar: list[dict], satirlar: list[dict],
             repos: Path = VARSAYILAN_REPOS) -> str:
    md = ["# Jev deneyleri — karşılaştırma", "",
          "Dosya düzeyi; eşik 0.5; birim (vaka, değişen dosya). Kural: `ORTAK_FORMAT.md`.", ""]
    metrikler = {s["arac"]: puanla(vakalar, s) for s in sonuclar}

    md += ["## Dosya düzeyi P / R / F1", "",
           "| Araç | " + " | ".join(TUM_KUMELER) + " | Hata | Medyan süre/PR (ms) | Maliyet (USD) |",
           "|---|" + "---|" * (len(TUM_KUMELER) + 3)]
    for arac, m in metrikler.items():
        hucre = [f"{m[k].p:.2f} / {m[k].r:.2f} / {m[k].f1:.2f}" for k in TUM_KUMELER]
        t = m["toplam"]
        md.append(f"| {arac} | " + " | ".join(hucre) +
                  f" | {t.hata_sayisi} | {_sayi(t.sure_medyan_ms)} | {_sayi(t.maliyet_usd, 4)} |")
    gecersiz = [a for a, m in metrikler.items() if m["toplam"].hata_sayisi]
    if gecersiz:
        md += ["", f"> **GEÇERSİZ:** {', '.join(gecersiz)} sonucunda hatalı dosya var; "
                   f"bu satırlar yorumlanmaz, koşu tekrarlanmalı."]

    md += ["", "## Çift skoru (halka) ve temiz vakalarda yanlış alarm", "",
           "| Araç | Çift skoru | Temiz vakalarda yanlış alarm (halka / zincir_dev / zincir_holdout) |",
           "|---|---|---|"]
    for s in sonuclar:
        b, t = cift_skoru(vakalar, s)
        ya = temiz_yanlis_alarm(vakalar, s)
        md.append(f"| {s['arac']} | {_oran(b, t)} | " +
                  " / ".join(_oran(*ya.get(k, (0, 0))) for k in TUM_KUMELER[:-1]) + " |")

    kirilimlar = {s["arac"]: kirilim(vakalar, s, repos) for s in sonuclar}
    eksenler = sorted({e for k in kirilimlar.values() for e in k})
    for eksen in eksenler:
        gruplar = sorted({g for k in kirilimlar.values() for g in k.get(eksen, {})})
        md += ["", f"## Kırılım: {eksen} (pozitif dosyalarda R = TP / pozitif)", "",
               "| Grup | " + " | ".join(kirilimlar) + " |", "|---|" + "---|" * len(kirilimlar)]
        for g in gruplar:
            hucre = []
            for k in kirilimlar.values():
                tp, fn = k.get(eksen, {}).get(g, (0, 0))
                hucre.append(_oran(tp, tp + fn))
            md.append(f"| {g} | " + " | ".join(hucre) + " |")

    if satirlar:
        md += ["", "## Satır bulma (Deney-3b: hunk ve blok)", "",
               "| Kaynak | Birim | Doğru dosya | İsabet (rastgele) | Birden çok birimli dosyada isabet (rastgele) | "
               "Aynı dosyalarda 'en büyük birim' kuralı | Uçtan uca satır R | Hata |",
               "|---|---|---|---|---|---|---|---|"]
        for s in satirlar:
            d = s["degerlendirme"]
            buyuk = d.get("en_buyuk_cok_hunk_isabet")
            md.append(f"| {s.get('kaynak_arac')} | {s.get('birim', 'hunk')} | {d['dogru_dosya']} | "
                      f"{_oran(d['isabet'], d['dogru_dosya'])} ({d['rastgele_beklenen']:.0%}) | "
                      f"{_oran(d['cok_hunk_isabet'], d['cok_hunk_dosya'])} "
                      f"({d.get('rastgele_cok_hunk', 0):.0%}) | "
                      f"{'—' if buyuk is None else _oran(buyuk, d['cok_hunk_dosya'])} | "
                      f"{d['uctan_uca_satir_r']:.2f} | {d['hata']} |")
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Jev sonuçlarını karşılaştırır")
    ap.add_argument("sonuclar", type=Path, nargs="+")
    ap.add_argument("--satir", type=Path, nargs="*", default=[])
    ap.add_argument("--repos", type=Path, default=VARSAYILAN_REPOS)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")
    oku = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    md = rapor_md(vakalari_yukle(a.repos), [oku(p) for p in a.sonuclar],
                  [oku(p) for p in a.satir], a.repos)
    cikti = DENEY / "sonuclar" / "karsilastirma.md"
    cikti.parent.mkdir(exist_ok=True)
    cikti.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
