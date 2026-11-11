"""Deney-3a: herhangi bir dosya düzeyi sonucun kırılımı (API çağrısı yok).

Eksenler (yalnızca pozitif dosyalar, TP/FN): kusur ailesi, dayanak, dosya rolü,
PR boyutu. Ayrıca temiz vakalarda yanlış alarm ve halka çift skoru.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from veri import VARSAYILAN_REPOS, Vaka  # noqa: E402


def _rol(kusur: dict, dosya: str) -> str:
    rol = kusur.get("file_role")
    if rol == "test" or dosya.startswith("tests/") or "/test_" in dosya:
        return "test"
    if rol == "config" or dosya.rsplit("/", 1)[-1] in ("settings.py", "defaults.py", "profiles.py") \
            or "config" in dosya:
        return "yapilandirma"
    return "kod"


def _etiketler(repos: Path) -> dict[str, dict]:
    sonuc = {}
    for bench in ("halka_bench", "zincir_bench"):
        for yol in (Path(repos) / bench / "labels").glob("*.json"):
            e = json.loads(yol.read_text(encoding="utf-8"))
            sonuc[e["vaka_kimligi"]] = e
    return sonuc


def _isaretliler(sonuc: dict) -> set[tuple[str, str]]:
    return {(v["vaka_kimligi"], d["dosya"]) for v in sonuc["vakalar"]
            for d in v["dosyalar"] if d.get("kusurlu_mu")}


def kirilim(vakalar: list[Vaka], sonuc: dict,
            repos: Path = VARSAYILAN_REPOS) -> dict[str, dict[str, tuple[int, int]]]:
    """eksen → grup → (TP, FN)."""
    etiketler = _etiketler(repos)
    isaretli = _isaretliler(sonuc)
    sayac: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for v in vakalar:
        etiket = etiketler.get(v.vaka_kimligi)
        if etiket is None:      # veri kümesinde olmayan (ör. test) vaka
            continue
        for dosya in sorted(v.pozitif):
            kusur = next(k for k in etiket["kusurlar"] if k["konum"]["dosya"] == dosya)
            aile = kusur.get("aile") or etiket.get("aile")
            gruplar = {
                "aile": f"{v.veri_kumesi.split('_')[0]}:{aile}",
                "dayanak": kusur.get("dayanak", "?"),
                "rol": _rol(kusur, dosya),
                "pr_boyutu": "tek_dosya" if len(v.degisen_dosyalar) == 1 else "cok_dosya",
                "kusur_sayisi": "tek_kusur" if len(etiket["kusurlar"]) == 1 else "cok_kusur",
            }
            indeks = 0 if (v.vaka_kimligi, dosya) in isaretli else 1
            for eksen, grup in gruplar.items():
                sayac[eksen][grup][indeks] += 1
    return {e: {g: tuple(s) for g, s in sorted(gr.items())} for e, gr in sayac.items()}


def cift_skoru(vakalar: list[Vaka], sonuc: dict) -> tuple[int, int]:
    """halka: kusurlu varyantta pozitif dosya işaretli VE temiz ikizde hiç işaret yok."""
    isaretli = _isaretliler(sonuc)
    kimlikle = {v.vaka_kimligi: v for v in vakalar}
    basarili = toplam = 0
    for v in vakalar:
        if v.veri_kumesi != "halka" or not v.vaka_kimligi.endswith("-kusurlu"):
            continue
        ikiz = kimlikle.get(v.vaka_kimligi.removesuffix("-kusurlu") + "-temiz")
        if ikiz is None:
            continue
        toplam += 1
        buldu = any((v.vaka_kimligi, d) in isaretli for d in v.pozitif)
        sessiz = not any((ikiz.vaka_kimligi, d) in isaretli for d in ikiz.degisen_dosyalar)
        basarili += buldu and sessiz
    return basarili, toplam


def temiz_yanlis_alarm(vakalar: list[Vaka], sonuc: dict) -> dict[str, tuple[int, int]]:
    """küme → (en az bir dosyası işaretlenen temiz vaka, temiz vaka sayısı)."""
    isaretli = _isaretliler(sonuc)
    sonuclar: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for v in vakalar:
        if v.pozitif or v.notr:
            continue
        sonuclar[v.veri_kumesi][1] += 1
        sonuclar[v.veri_kumesi][0] += any((v.vaka_kimligi, d) in isaretli
                                          for d in v.degisen_dosyalar)
    return {k: tuple(s) for k, s in sorted(sonuclar.items())}
