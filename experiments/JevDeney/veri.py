"""halka_bench ve zincir_bench veri kümelerini salt okunur okur.

Her vaka için değişen dosyaları ve dosya düzeyi altın etiketleri üretir:
- pozitif: bir kusurun `konum.dosya` değeri
- notr: yalnızca `esdeger_konumlar` içinde geçen, pozitif olmayan dosya
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

VARSAYILAN_REPOS = Path(r"C:\Users\hvehb\Desktop\Repos")
KUMELER = ("halka", "zincir_dev", "zincir_holdout")


@dataclass
class Vaka:
    veri_kumesi: str
    vaka_kimligi: str
    degisen_dosyalar: list[str]
    pozitif: set[str] = field(default_factory=set)
    notr: set[str] = field(default_factory=set)
    diff_yolu: Path | None = None
    diffte_olmayan: list[str] = field(default_factory=list)


def _kume_adi(bench: str, etiket: dict) -> str:
    if bench == "halka_bench":
        return "halka"
    yaka = etiket.get("yaka")
    if yaka not in ("dev", "holdout"):
        raise ValueError(f"{etiket['vaka_kimligi']}: beklenmeyen yaka {yaka!r}")
    return f"zincir_{yaka}"


def vakalari_yukle(repos: Path = VARSAYILAN_REPOS) -> list[Vaka]:
    """İki veri kümesindeki bütün vakaları kimliğe göre sıralı döndürür."""
    vakalar: list[Vaka] = []
    for bench in ("halka_bench", "zincir_bench"):
        kok = Path(repos) / bench
        for etiket_yolu in sorted((kok / "labels").glob("*.json")):
            etiket = json.loads(etiket_yolu.read_text(encoding="utf-8"))
            pozitif = {k["konum"]["dosya"] for k in etiket["kusurlar"]}
            esdeger = {e["dosya"] for k in etiket["kusurlar"]
                       for e in k.get("esdeger_konumlar", [])}
            diff_yolu = kok / "diffs" / f"{etiket['vaka_kimligi']}.diff"
            diffteki = diffi_bol(diff_yolu.read_text(encoding="utf-8"))
            # Ölçüm birimi diff'te gerçekten bulunan dosyadır. Bazı temiz
            # ikizlerin etiketi değişmeyen bir dosyayı da listeler.
            vakalar.append(Vaka(
                veri_kumesi=_kume_adi(bench, etiket),
                vaka_kimligi=etiket["vaka_kimligi"],
                degisen_dosyalar=[d for d in etiket["degisen_dosyalar"] if d in diffteki],
                pozitif=pozitif,
                notr=esdeger - pozitif,
                diff_yolu=diff_yolu,
                diffte_olmayan=[d for d in etiket["degisen_dosyalar"] if d not in diffteki],
            ))
    return vakalar


_BASLIK = re.compile(r"^diff --git a/(.+?) b/(.+)$")


def diffi_bol(metin: str) -> dict[str, str]:
    """Birleşik diff'i dosya başına parçalara böler. Anahtar: dosya yolu.

    Yeni dosyada ad `b/`, silinen dosyada (`+++ /dev/null`) ad `a/` tarafından alınır.
    """
    metin = metin.replace("\r\n", "\n")
    parcalar: dict[str, str] = {}
    satirlar: list[str] = []
    a_adi = b_adi = None

    def kapat():
        if satirlar:
            silindi = any(s == "+++ /dev/null" for s in satirlar)
            parcalar[a_adi if silindi else b_adi] = "\n".join(satirlar) + "\n"

    for satir in metin.split("\n"):
        eslesme = _BASLIK.match(satir)
        if eslesme:
            kapat()
            a_adi, b_adi = eslesme.group(1), eslesme.group(2)
            satirlar = [satir]
        elif satirlar:
            satirlar.append(satir)
    kapat()
    # son satırdaki boş eleman yüzünden oluşan fazla "\n" temizlenir
    return {ad: p.rstrip("\n") + "\n" for ad, p in parcalar.items()}
