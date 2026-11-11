"""istemci.py testleri (ağa çıkmaz)."""
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from istemci import (MODEL, SORULAR, URL, GercekIstemci, SahteIstemci,  # noqa: E402
                     anahtar_oku, istek_govdesi)

PARCA = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"


class SoruTasarimiTestleri(unittest.TestCase):
    def test_iki_ayri_noul_sorusu(self):
        # Skill: tek soruda tek yargı; "or" ile birleştirilmez.
        self.assertEqual(set(SORULAR), {"introduces_defect", "introduces_vulnerability"})
        for soru in SORULAR.values():
            self.assertEqual(soru["type"], "noul")
            self.assertEqual(set(soru["criteria"]), {"true", "false"})
            self.assertNotIn(" or a security", json.dumps(soru["instructions"]))

    def test_model_sabitlenmis(self):
        self.assertEqual(MODEL, "jev-1.14.0")  # tekrar üretilebilirlik: alias değil

    def test_istek_govdesi_yalnizca_dosya_ve_diff(self):
        govde = istek_govdesi("x.py", PARCA)
        self.assertEqual(govde["model"], MODEL)
        self.assertEqual(govde["state"], {"file_path": "x.py", "diff": PARCA})
        self.assertEqual(govde["questions"], SORULAR)


class SahteIstemciTestleri(unittest.TestCase):
    def test_sahte_belirlenimci(self):
        self.assertEqual(SahteIstemci().sor("x.py", PARCA).olasiliklar,
                         SahteIstemci().sor("x.py", PARCA).olasiliklar)

    def test_sahte_aralik_ve_birlesik_olasilik(self):
        for i in range(30):
            y = SahteIstemci().sor("x.py", PARCA + str(i))
            self.assertEqual(set(y.olasiliklar), set(SORULAR))
            for p in y.olasiliklar.values():
                self.assertTrue(0.0 <= p <= 1.0)
            self.assertEqual(y.olasilik, max(y.olasiliklar.values()))
            self.assertIsNone(y.hata)


class SahteYanit:
    def __init__(self, govde: dict):
        self._veri = json.dumps(govde).encode("utf-8")

    def read(self):
        return self._veri

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_hatasi(kod, govde=b'{"detail": "x"}'):
    return urllib.error.HTTPError(URL, kod, "hata", {}, io.BytesIO(govde))


BASARILI = {"model": "jev-1.14.0",
            "answers": {"introduces_defect": {"type": "noul", "noul": 0.8},
                        "introduces_vulnerability": {"type": "noul", "noul": 0.1}},
            "usage": {"input_tokens": 420, "output_tokens": 30}}


class GercekIstemciTestleri(unittest.TestCase):
    def _istemci(self, yanitlar):
        self.istekler, self.beklemeler = [], []
        kuyruk = list(yanitlar)

        def acici(istek, timeout):
            self.istekler.append((istek, timeout))
            y = kuyruk.pop(0)
            if isinstance(y, Exception):
                raise y
            return SahteYanit(y)

        return GercekIstemci("gizli-anahtar", acici=acici, bekle=self.beklemeler.append)

    def test_gercek_istek_govdesi(self):
        y = self._istemci([BASARILI]).sor("x.py", PARCA)
        istek, timeout = self.istekler[0]
        self.assertEqual(istek.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(istek.get_method(), "POST")
        self.assertEqual(istek.get_header("Authorization"), "Bearer gizli-anahtar")
        self.assertEqual(istek.get_header("Content-type"), "application/json")
        self.assertEqual(json.loads(istek.data), istek_govdesi("x.py", PARCA))
        self.assertEqual(timeout, 30)
        self.assertEqual(y.olasiliklar, {"introduces_defect": 0.8, "introduces_vulnerability": 0.1})
        self.assertEqual((y.olasilik, y.giris_token, y.hata), (0.8, 420, None))

    def test_anahtar_repr_icinde_gorunmez(self):
        self.assertNotIn("gizli-anahtar", repr(self._istemci([])))

    def test_429_ve_5xx_tekrar_denenir(self):
        y = self._istemci([_http_hatasi(429), _http_hatasi(503), BASARILI]).sor("x.py", PARCA)
        self.assertIsNone(y.hata)
        self.assertEqual(self.beklemeler, [1, 2])

    def test_baglanti_hatasi_tekrar_denenir_sonra_hata(self):
        hata = urllib.error.URLError("ag yok")
        y = self._istemci([hata] * 4).sor("x.py", PARCA)
        self.assertEqual(len(self.istekler), 4)  # 1 deneme + 3 tekrar
        self.assertIsNone(y.olasilik)
        self.assertIn("ag yok", y.hata)

    def test_401_tekrar_denenmez_hata_doner(self):
        y = self._istemci([_http_hatasi(401, b'{"detail": "bad key"}')]).sor("x.py", PARCA)
        self.assertEqual(len(self.istekler), 1)
        self.assertIn("401", y.hata)
        self.assertNotIn("gizli-anahtar", y.hata)

    def test_eksik_cevap_hata_sayilir(self):
        bozuk = {"model": "jev-1.14.0", "answers": {"introduces_defect": {"noul": 0.4}},
                 "usage": {"input_tokens": 10}}
        y = self._istemci([bozuk]).sor("x.py", PARCA)
        self.assertIsNone(y.olasilik)
        self.assertIn("introduces_vulnerability", y.hata)


class AnahtarOkumaTestleri(unittest.TestCase):
    def test_env_dosyasindan_okunur(self):
        with tempfile.TemporaryDirectory() as g:
            (Path(g) / ".env").write_text("# yorum\nTYPESAFE_API_KEY = abc123 \n", encoding="utf-8")
            self.assertEqual(anahtar_oku(Path(g), ortam={}), "abc123")

    def test_ortam_degiskeni_once_gelir(self):
        with tempfile.TemporaryDirectory() as g:
            (Path(g) / ".env").write_text("TYPESAFE_API_KEY=dosya\n", encoding="utf-8")
            self.assertEqual(anahtar_oku(Path(g), ortam={"TYPESAFE_API_KEY": "ortam"}), "ortam")

    def test_bom_ve_tirnak_temizlenir(self):
        with tempfile.TemporaryDirectory() as g:
            (Path(g) / ".env").write_text('﻿TYPESAFE_API_KEY="abc"\n', encoding="utf-8")
            self.assertEqual(anahtar_oku(Path(g), ortam={}), "abc")

    def test_yoksa_none(self):
        with tempfile.TemporaryDirectory() as g:
            self.assertIsNone(anahtar_oku(Path(g), ortam={}))


if __name__ == "__main__":
    unittest.main()
