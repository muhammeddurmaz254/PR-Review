"""halka_bench ve zincir_bench veri kümelerini salt okunur okur.

Her vaka için değişen dosyaları ve dosya düzeyi altın etiketleri üretir:
- pozitif: bir kusurun `konum.dosya` değeri
- notr: yalnızca `esdeger_konumlar` içinde geçen, pozitif olmayan dosya
"""
from __future__ import annotations

import json
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
            eksik_pozitif = sorted(pozitif - set(diffteki))
            if eksik_pozitif:
                # Sessizce düşürülürse ne TP ne FN sayılır ve skor yanlış olur.
                raise ValueError(f"{etiket['vaka_kimligi']}: kusurlu dosya diff'te yok: "
                                 f"{', '.join(eksik_pozitif)}")
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


_DIFF_BASI = "diff --git "


def _tirnak_coz(yol: str) -> str:
    """Git'in tırnaklı yolunu çözer: "b/\303\274.py" -> b/ü.py (C tarzı kaçışlar)."""
    yol = yol.strip()
    if not (len(yol) >= 2 and yol[0] == '"' and yol[-1] == '"'):
        return yol
    ic, bayt, i = yol[1:-1], bytearray(), 0
    kacis = {"n": b"\n", "t": b"\t", '"': b'"', "\\": b"\\", "a": b"\a", "b": b"\b",
             "f": b"\f", "r": b"\r", "v": b"\v"}
    while i < len(ic):
        c = ic[i]
        if c == "\\" and i + 1 < len(ic):
            sonraki = ic[i + 1]
            if sonraki in "01234567":
                bayt.append(int(ic[i + 1:i + 4], 8))
                i += 4
                continue
            bayt += kacis.get(sonraki, sonraki.encode("utf-8"))
            i += 2
            continue
        bayt += c.encode("utf-8")
        i += 1
    return bayt.decode("utf-8", "replace")


def _on_eki_at(yol: str) -> str:
    yol = _tirnak_coz(yol.rstrip("\t"))
    return yol[2:] if yol[:2] in ("a/", "b/") else yol


def _parca_adi(satirlar: list[str]) -> str:
    """Dosya adı önce `+++ b/...` satırından (silinmişse `--- a/...`) okunur.

    İçerik satırı yoksa (yalnızca mod değişikliği) `diff --git` başlığından çözülür:
    tırnaklı yollar ve "a/P b/P" biçimi (P içinde " b/" geçse bile) desteklenir.
    """
    eski = yeni = None
    for satir in satirlar[1:]:
        if satir.startswith("@@"):
            break
        if satir.startswith("--- "):
            eski = satir[4:]
        elif satir.startswith("+++ "):
            yeni = satir[4:]
    if yeni is not None and yeni.strip() != "/dev/null":
        return _on_eki_at(yeni)
    if eski is not None and eski.strip() != "/dev/null":
        return _on_eki_at(eski)
    kalan = satirlar[0][len(_DIFF_BASI):]
    if kalan.startswith('"'):
        son = kalan.index('"', 1) + 1
        return _on_eki_at(kalan[son:].strip() or kalan[:son])
    n = (len(kalan) - len("a/ b/")) // 2
    if n > 0 and kalan == f"a/{kalan[2:2 + n]} b/{kalan[2:2 + n]}":
        return kalan[2:2 + n]
    return _on_eki_at(kalan.rsplit(" b/", 1)[-1] if " b/" in kalan else kalan)


def diffi_bol(metin: str) -> dict[str, str]:
    """Birleşik diff'i dosya başına parçalara böler. Anahtar: dosya yolu.

    Yeni dosyada ad `+++ b/` satırından, silinen dosyada (`+++ /dev/null`) `--- a/`
    satırından alınır. Git'in tırnaklı (Türkçe karakterli) yolları çözülür.
    """
    metin = metin.replace("\r\n", "\n")
    bloklar: list[list[str]] = []
    for satir in metin.split("\n"):
        if satir.startswith(_DIFF_BASI):
            bloklar.append([satir])
        elif bloklar:
            bloklar[-1].append(satir)
    # son satırdaki boş eleman yüzünden oluşan fazla "\n" temizlenir
    return {_parca_adi(b): "\n".join(b).rstrip("\n") + "\n" for b in bloklar}
