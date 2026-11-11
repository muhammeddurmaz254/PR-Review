"""Ortak değerlendirici: ORTAK_FORMAT.md biçimindeki bir sonuç dosyasını puanlar.

Birim (vaka, değişen dosya) çiftidir. Kural:
- pozitif dosya işaretlenirse TP, işaretlenmezse FN
- pozitif olmayan, nötr olmayan dosya işaretlenirse FP
- nötr dosya (yalnızca eşdeğer konum) puana katılmaz
- sonuçta bulunmayan dosya "işaretlenmedi" sayılır

Kullanım: py JevDeney/degerlendir.py sonuclar/<dosya>.json [--repos YOL] [--cikti KLASOR]
"""
from __future__ import annotations

import argparse
import json
import sys
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from kirilim import cift_skoru, kirilim, temiz_yanlis_alarm
from veri import KUMELER, VARSAYILAN_REPOS, Vaka, vakalari_yukle

TUM_KUMELER = KUMELER + ("toplam",)


@dataclass
class Metrik:
    vaka: int = 0
    dosya: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    p: float = 0.0
    r: float = 0.0
    f1: float = 0.0
    hata_sayisi: int = 0
    sure_ort_ms: float | None = None
    sure_medyan_ms: float | None = None
    maliyet_usd: float | None = None
    uyarilar: list[str] = field(default_factory=list)


def _bol(a: float, b: float) -> float:
    return a / b if b else 0.0


def puanla(vakalar: list[Vaka], sonuc: dict) -> dict[str, Metrik]:
    """Küme adı → Metrik. 'toplam' bütün kümelerin birleşimidir."""
    fiyat = sonuc.get("birim_fiyat_usd_1m")
    tahmin = {v["vaka_kimligi"]: {d["dosya"]: d for d in v.get("dosyalar", [])}
              for v in sonuc.get("vakalar", [])}
    metrikler = {k: Metrik() for k in TUM_KUMELER}
    sureler: dict[str, list[float]] = {k: [] for k in TUM_KUMELER}
    tokenler: dict[str, int] = {k: 0 for k in TUM_KUMELER}

    for vaka in vakalar:
        hedefler = (metrikler[vaka.veri_kumesi], metrikler["toplam"])
        kumeler = (vaka.veri_kumesi, "toplam")
        dosya_tahmini = tahmin.get(vaka.vaka_kimligi, {})
        vaka_suresi = [d["sure_ms"] for d in dosya_tahmini.values()
                       if d.get("sure_ms") is not None]
        for m in hedefler:
            m.vaka += 1
            m.dosya += len(vaka.degisen_dosyalar)
        for k in kumeler:
            if vaka_suresi:
                sureler[k].append(sum(vaka_suresi))
            tokenler[k] += sum(d.get("giris_token") or 0 for d in dosya_tahmini.values())

        for dosya in vaka.degisen_dosyalar:
            kayit = dosya_tahmini.get(dosya, {})
            isaretli = bool(kayit.get("kusurlu_mu"))
            for m in hedefler:
                if kayit.get("hata"):
                    m.hata_sayisi += 1
                if dosya in vaka.pozitif:
                    if isaretli:
                        m.tp += 1
                    else:
                        m.fn += 1
                elif dosya not in vaka.notr and isaretli:
                    m.fp += 1
        for dosya in dosya_tahmini:
            if dosya not in vaka.degisen_dosyalar:
                for m in hedefler:
                    m.uyarilar.append(f"{vaka.vaka_kimligi}: etikette olmayan dosya {dosya}")

    bilinen = {v.vaka_kimligi for v in vakalar}
    for kimlik in tahmin:
        if kimlik not in bilinen:
            metrikler["toplam"].uyarilar.append(f"bilinmeyen vaka {kimlik}")

    for k, m in metrikler.items():
        m.p = _bol(m.tp, m.tp + m.fp)
        m.r = _bol(m.tp, m.tp + m.fn)
        m.f1 = _bol(2 * m.p * m.r, m.p + m.r)
        if sureler[k]:
            m.sure_ort_ms = statistics.mean(sureler[k])
            m.sure_medyan_ms = statistics.median(sureler[k])
        if fiyat is not None:
            m.maliyet_usd = tokenler[k] * fiyat / 1_000_000
    return metrikler


def _sayi(x: float | None, basamak: int = 0) -> str:
    return "—" if x is None else f"{x:.{basamak}f}"


def tablo_md(metrikler: dict[str, Metrik]) -> str:
    satirlar = [
        "| Küme | Vaka | Dosya | TP | FP | FN | P | R | F1 | Hata | Ort. süre/PR (ms) | Medyan (ms) | Maliyet (USD) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in TUM_KUMELER:
        m = metrikler[k]
        satirlar.append(
            f"| {k} | {m.vaka} | {m.dosya} | {m.tp} | {m.fp} | {m.fn} | {m.p:.3f} | {m.r:.3f} "
            f"| {m.f1:.3f} | {m.hata_sayisi} | {_sayi(m.sure_ort_ms)} | {_sayi(m.sure_medyan_ms)} "
            f"| {_sayi(m.maliyet_usd, 4)} |")
    return "\n".join(satirlar) + "\n"


_TEX_AD = {"halka": "halka", "zincir_dev": "zincir (geliştirme)",
           "zincir_holdout": "zincir (ayrılmış test)", "toplam": "Toplam"}


def tablo_tex(metrikler: dict[str, Metrik], arac_adi: str) -> str:
    """Rapor kuralı: tablo başlığı tablonun üstünde, booktabs."""
    satirlar = [
        "\\begin{table}[H]", "\\centering",
        f"\\caption{{{arac_adi} taban çizgisinin dosya düzeyi sonuçları}}",
        f"\\label{{tbl:{arac_adi.lower()}_dosya}}",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Veri kümesi & Dosya & P & R & F1 & Süre/PR (ms) & Maliyet (USD) \\\\", "\\midrule",
    ]
    for k in TUM_KUMELER:
        m = metrikler[k]
        if k == "toplam":
            satirlar.append("\\midrule")
        tire = lambda s: "--" if s == "—" else s  # noqa: E731
        satirlar.append(
            f"{_TEX_AD[k]} & {m.dosya} & {m.p:.3f} & {m.r:.3f} & {m.f1:.3f} & "
            f"{tire(_sayi(m.sure_medyan_ms))} & {tire(_sayi(m.maliyet_usd, 4))} \\\\")
    satirlar += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(satirlar) + "\n"


def _oran(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b:.0%})" if b else "—"


def kirilim_md(vakalar: list[Vaka], sonuc: dict, repos: Path = VARSAYILAN_REPOS) -> str:
    """Çift skoru, temiz vakalarda yanlış alarm ve pozitif dosyalarda R kırılımı."""
    b, t = cift_skoru(vakalar, sonuc)
    satirlar = ["", "## Çift skoru (halka)", "",
                f"Kusurlu varyantta pozitif dosya işaretli ve temiz ikizde hiç işaret yok: {_oran(b, t)}",
                "", "## Temiz vakalarda yanlış alarm", "",
                "| Küme | En az bir dosyası işaretlenen temiz vaka |", "|---|---|"]
    for kume, (a, n) in temiz_yanlis_alarm(vakalar, sonuc).items():
        satirlar.append(f"| {kume} | {_oran(a, n)} |")
    for eksen, gruplar in kirilim(vakalar, sonuc, repos).items():
        satirlar += ["", f"## Kırılım: {eksen} (pozitif dosyalarda R)", "",
                     "| Grup | TP / pozitif |", "|---|---|"]
        for grup, (tp, fn) in gruplar.items():
            satirlar.append(f"| {grup} | {_oran(tp, tp + fn)} |")
    return "\n".join(satirlar) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ortak biçimdeki sonuç dosyasını puanlar.")
    ap.add_argument("sonuc", type=Path)
    ap.add_argument("--repos", type=Path, default=VARSAYILAN_REPOS)
    ap.add_argument("--cikti", type=Path, default=Path(__file__).parent / "sonuclar")
    a = ap.parse_args(argv)
    # Windows konsolu (cp1254) basamadığı karakterde çökmesin
    sys.stdout.reconfigure(errors="replace")

    sonuc = json.loads(a.sonuc.read_text(encoding="utf-8"))
    vakalar = vakalari_yukle(a.repos)
    m = puanla(vakalar, sonuc)
    arac = sonuc.get("arac", "arac")
    baslik = (f"# {arac} — dosya düzeyi sonuçlar\n\n"
              f"Kaynak: `{a.sonuc.name}` · sürüm: {sonuc.get('surum', '?')} · "
              f"tarih: {sonuc.get('tarih', '?')}\n\n")
    notlar = [f"- {v.vaka_kimligi}: etikette olup diff'te olmayan {d} ölçüme katılmadı"
              for v in vakalar for d in v.diffte_olmayan]
    uyarilar = [f"- {u}" for u in m["toplam"].uyarilar]
    hatali = m["toplam"].hata_sayisi
    if hatali:
        # Hata veren çağrı "hayır" değildir: bu sonuçla puan raporlanmaz.
        baslik += (f"> **GEÇERSİZ:** {hatali} dosyada araç hata verdi. Bu tablo rapora "
                   f"konmaz. Koşuyu tekrarlayın (önbellek yalnızca hatalı dosyaları "
                   f"yeniden sorar).\n\n")
    govde = baslik + tablo_md(m) + kirilim_md(vakalar, sonuc, a.repos)
    if notlar:
        govde += "\n## Veri notları\n" + "\n".join(notlar) + "\n"
    if uyarilar:
        govde += "\n## Uyarılar\n" + "\n".join(uyarilar) + "\n"

    a.cikti.mkdir(parents=True, exist_ok=True)
    (a.cikti / f"ozet_{arac}.md").write_text(govde, encoding="utf-8")
    tex = a.cikti / f"tablo_{arac}.tex"
    if hatali:
        tex.unlink(missing_ok=True)   # eski geçerli tablo yanlışlıkla rapora girmesin
    else:
        tex.write_text(tablo_tex(m, arac.capitalize()), encoding="utf-8")
    print(govde)
    return 2 if hatali else 0


if __name__ == "__main__":
    raise SystemExit(main())
