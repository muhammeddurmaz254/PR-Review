"""Jev istemcileri ve soru tanımları (soru metinleri yalnızca bu dosyadadır).

TypeSafe API: POST https://api.typesafe.ai/v1/systemone, Bearer anahtar.
Kaynak: docs.typesafe.ai (quick start, API reference, models; 10.11.2026).

Tasarım kararları (TypeSafe skill rehberine göre):
- Her dosya ayrı bir istektir; state yalnızca dosya yolu ve o dosyanın diff
  parçasıdır (ilgisiz içerik doğruluğu düşürür).
- Tek soruda tek yargı: "kusur" ve "güvenlik zafiyeti" iki ayrı Noul'dur.
  Dosyanın kusurlu sayılıp sayılmayacağına kod karar verir (kos.py).
- Hata veren çağrı "hayır" sayılmaz; `Yanit.hata` doldurulur.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"          # alias (jev-latest) değil: sonuç tekrar üretilebilsin
BIRIM_FIYAT_USD_1M = 0.042    # models sayfası: $0.042 / 1M giriş token, çıkış ücretsiz
ZAMAN_ASIMI_S = 30
BEKLEMELER_S = (1, 2, 4)      # 429, 5xx ve bağlantı hatalarında 3 tekrar

_ODAK = ("Judge only what this change adds or changes. "
         "Assume the code before the change was correct.")

SORULAR: dict[str, dict] = {
    "introduces_defect": {
        "type": "noul",
        "instructions": {
            "question": "Do the added or changed lines in `diff` introduce a functional "
                        "defect into `file_path`?",
            "focus": _ODAK,
        },
        "criteria": {
            "true": {"what": "The new or changed lines make the code, its tests or its "
                             "configuration behave incorrectly"},
            "false": {"what": "The new or changed lines behave correctly",
                      "not_for": "Style, naming or formatting preferences"},
        },
    },
    "introduces_vulnerability": {
        "type": "noul",
        "instructions": {
            "question": "Do the added or changed lines in `diff` introduce a security "
                        "vulnerability into `file_path`?",
            "focus": _ODAK,
        },
        "criteria": {
            "true": {"what": "The change lets an attacker or an unauthorized user read, "
                             "change or run something they should not"},
            "false": {"what": "The change keeps security as it was",
                      "not_for": "A bug with no security impact"},
        },
    },
}


def istek_govdesi(dosya: str, diff_parcasi: str, model: str = MODEL) -> dict:
    """API'ye giden tam gövde. Önbellek anahtarı da bundan üretilir."""
    return {"model": model,
            "state": {"file_path": dosya, "diff": diff_parcasi},
            "questions": SORULAR}


@dataclass
class Yanit:
    olasilik: float | None                      # dosya düzeyi birleşik olasılık (max)
    sure_ms: int | None
    giris_token: int | None
    hata: str | None = None
    olasiliklar: dict[str, float] = field(default_factory=dict)
    model_yanit: str | None = None


@dataclass
class HamYanit:
    """Herhangi bir istek gövdesinin yanıtı: soru adı → noul olasılığı."""
    cevaplar: dict[str, float] | None
    sure_ms: int | None
    giris_token: int | None
    model_yanit: str | None
    hata: str | None = None


def _dosya_yaniti(ham: HamYanit) -> Yanit:
    """Ana sorular (SORULAR) için ham yanıtı dosya düzeyi Yanit'a çevirir."""
    if ham.hata:
        return Yanit(None, ham.sure_ms, None, hata=ham.hata)
    eksik = [ad for ad in SORULAR if ad not in ham.cevaplar]
    if eksik:
        return Yanit(None, ham.sure_ms, None, hata=f"yanitta eksik cevap: {', '.join(eksik)}")
    olasiliklar = {ad: ham.cevaplar[ad] for ad in SORULAR}
    return Yanit(_birlestir(olasiliklar), ham.sure_ms, ham.giris_token,
                 olasiliklar=olasiliklar, model_yanit=ham.model_yanit)


class JevIstemci(Protocol):
    ad: str
    model: str

    def sor(self, dosya: str, diff_parcasi: str) -> Yanit: ...

    def gonder(self, govde: dict) -> HamYanit: ...


def _birlestir(olasiliklar: dict[str, float]) -> float:
    return max(olasiliklar.values())


class SahteIstemci:
    """Ağa çıkmaz. Olasılıklar isteğin SHA-256 özetinden türetilir (anlamsızdır)."""

    ad = "sahte"
    model = "sahte"
    birim_fiyat_usd_1m = None

    def gonder(self, govde: dict) -> HamYanit:
        durum = json.dumps(govde["state"], sort_keys=True)
        cevaplar = {}
        for ad in govde["questions"]:
            ozet = hashlib.sha256(f"{ad}\0{durum}".encode("utf-8")).digest()
            cevaplar[ad] = int.from_bytes(ozet[:8], "big") / 2**64
        return HamYanit(cevaplar, 0, len(json.dumps(govde)) // 4, self.model)

    def sor(self, dosya: str, diff_parcasi: str) -> Yanit:
        return _dosya_yaniti(self.gonder(istek_govdesi(dosya, diff_parcasi, self.model)))


class GercekIstemci:
    """TypeSafe /v1/systemone istemcisi. Yalnızca standart kütüphane."""

    ad = "gercek"
    birim_fiyat_usd_1m = BIRIM_FIYAT_USD_1M

    def __init__(self, anahtar: str, model: str = MODEL,
                 acici: Callable = urllib.request.urlopen,
                 bekle: Callable[[float], None] = time.sleep):
        self._anahtar = anahtar
        self.model = model
        self._acici = acici
        self._bekle = bekle

    def __repr__(self) -> str:
        return f"GercekIstemci(model={self.model!r})"

    def _istek(self, govde: dict) -> urllib.request.Request:
        return urllib.request.Request(
            URL, data=json.dumps(govde).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self._anahtar}",
                     "Content-Type": "application/json"})

    def sor(self, dosya: str, diff_parcasi: str) -> Yanit:
        return _dosya_yaniti(self.gonder(istek_govdesi(dosya, diff_parcasi, self.model)))

    def gonder(self, govde: dict) -> HamYanit:
        son_hata = "bilinmeyen hata"
        for deneme in range(len(BEKLEMELER_S) + 1):
            if deneme:
                self._bekle(BEKLEMELER_S[deneme - 1])
            baslangic = time.perf_counter()
            try:
                with self._acici(self._istek(govde), timeout=ZAMAN_ASIMI_S) as yanit:
                    veri = json.loads(yanit.read().decode("utf-8"))
                sure = round((time.perf_counter() - baslangic) * 1000)
                return self._coz(veri, sure, govde)
            except urllib.error.HTTPError as h:
                ayrinti = h.read().decode("utf-8", "replace")[:300]
                son_hata = f"HTTP {h.code}: {ayrinti}"
                if h.code != 429 and h.code < 500:
                    break                       # 400/401/403/422: tekrar denemenin anlamı yok
            except (urllib.error.URLError, TimeoutError, ConnectionError) as h:
                son_hata = f"{type(h).__name__}: {getattr(h, 'reason', h)}"
        return HamYanit(None, None, None, None, hata=son_hata)

    @staticmethod
    def _coz(veri: dict, sure: int, govde: dict) -> HamYanit:
        cevaplar = veri.get("answers", {})
        eksik = [ad for ad in govde["questions"] if "noul" not in cevaplar.get(ad, {})]
        if eksik:
            return HamYanit(None, sure, None, veri.get("model"),
                            hata=f"yanitta eksik cevap: {', '.join(eksik)}")
        return HamYanit({ad: float(cevaplar[ad]["noul"]) for ad in govde["questions"]}, sure,
                        veri.get("usage", {}).get("input_tokens"), veri.get("model"))


def anahtar_oku(klasor: Path, ortam: dict | None = None) -> str | None:
    """Önce TYPESAFE_API_KEY ortam değişkeni, sonra <klasor>/.env dosyası."""
    ortam = os.environ if ortam is None else ortam
    if ortam.get("TYPESAFE_API_KEY"):
        return ortam["TYPESAFE_API_KEY"].strip()
    env = Path(klasor) / ".env"
    if not env.exists():
        return None
    for satir in env.read_text(encoding="utf-8-sig").splitlines():
        anahtar, _, deger = satir.partition("=")
        if anahtar.strip() == "TYPESAFE_API_KEY" and deger.strip():
            return deger.strip().strip('"').strip("'")
    return None
