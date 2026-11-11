"""Komut satırı testleri: Windows konsolunda (cp1254) Türkçe/Unicode çıktı çökmemeli."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

KLASOR = Path(__file__).resolve().parents[1]


def _calistir(*args):
    ortam = dict(os.environ, PYTHONIOENCODING="cp1254")
    return subprocess.run([sys.executable, *args], cwd=KLASOR, env=ortam,
                          capture_output=True, timeout=120)


class KomutSatiriTestleri(unittest.TestCase):
    def test_kos_cp1254_konsolda_cokmez(self):
        with tempfile.TemporaryDirectory() as g:
            r = _calistir("kos.py", "--istemci", "sahte", "--vaka", "authz-01-kusurlu",
                          "--cikti", g)
            self.assertEqual(r.returncode, 0, r.stderr.decode("utf-8", "replace"))
            # tek vakalık koşu, tam koşunun dosyasının üzerine yazmamalı
            self.assertTrue(list(Path(g).glob("jev_sahte_authz-01-kusurlu_*.json")))
            # test, gerçek sonuç klasörüne dosya bırakmamalı
            self.assertTrue(list(Path(g, "onbellek").glob("sahte.jsonl")))

    def test_degerlendir_cp1254_konsolda_cokmez(self):
        with tempfile.TemporaryDirectory() as g:
            sonuc = Path(g) / "s.json"
            sonuc.write_text(json.dumps({"arac": "deneme", "vakalar": []}), encoding="utf-8")
            r = _calistir("degerlendir.py", str(sonuc), "--cikti", g)
            self.assertEqual(r.returncode, 0, r.stderr.decode("utf-8", "replace"))
            self.assertTrue((Path(g) / "ozet_deneme.md").exists())


    def test_hatali_kosu_gecersiz_tablo_yazilmaz(self):
        # Hata veren çağrı "hayır" sayılmaz: sonuç geçersizdir, rapora tablo üretilmez.
        sonuc = {"arac": "hatali", "vakalar": [
            {"veri_kumesi": "halka", "vaka_kimligi": "authz-01-kusurlu",
             "dosyalar": [{"dosya": "halka/billing/views.py", "kusurlu_mu": False,
                           "hata": "HTTP 503: x"}]}]}
        with tempfile.TemporaryDirectory() as g:
            yol = Path(g) / "s.json"
            yol.write_text(json.dumps(sonuc), encoding="utf-8")
            r = _calistir("degerlendir.py", str(yol), "--cikti", g)
            self.assertEqual(r.returncode, 2)
            ozet = (Path(g) / "ozet_hatali.md").read_text(encoding="utf-8")
            self.assertIn("GEÇERSİZ", ozet)
            self.assertFalse((Path(g) / "tablo_hatali.tex").exists())


if __name__ == "__main__":
    unittest.main()


class AnaOzetKirilimTestleri(unittest.TestCase):
    def test_ozet_kirilim_ve_cift_skoru_icerir(self):
        sys.path.insert(0, str(KLASOR))
        from veri import vakalari_yukle
        vakalar = vakalari_yukle()
        sonuc = {"arac": "kahin", "vakalar": [
            {"veri_kumesi": v.veri_kumesi, "vaka_kimligi": v.vaka_kimligi,
             "dosyalar": [{"dosya": d, "kusurlu_mu": d in v.pozitif} for d in v.degisen_dosyalar]}
            for v in vakalar]}
        with tempfile.TemporaryDirectory() as g:
            yol = Path(g) / "s.json"
            yol.write_text(json.dumps(sonuc), encoding="utf-8")
            r = _calistir("degerlendir.py", str(yol), "--cikti", g)
            self.assertEqual(r.returncode, 0, r.stderr.decode("utf-8", "replace"))
            ozet = (Path(g) / "ozet_kahin.md").read_text(encoding="utf-8")
        for parca in ("## Çift skoru", "49/49", "## Temiz vakalarda yanlış alarm",
                      "## Kırılım: aile", "## Kırılım: rol", "halka:authz"):
            self.assertIn(parca, ozet)
