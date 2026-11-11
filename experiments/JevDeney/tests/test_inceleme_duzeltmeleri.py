"""Bağımsız incelemenin bulduğu 7 açığın testleri (2026-11-11).

Her test, düzeltmeden önce başarısız olan bir girdiyi tarif eder.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))

from degerlendir import _dosya_adi, _tex_kacis, puanla, tablo_tex  # noqa: E402
from kirilim import cift_skoru, kirilim, temiz_yanlis_alarm  # noqa: E402
from veri import Vaka, diffi_bol, vakalari_yukle  # noqa: E402


def _vakalar():
    return [Vaka("halka", "x-kusurlu", ["a.py", "b.py"], pozitif={"a.py"}),
            Vaka("halka", "x-temiz", ["a.py", "b.py"])]


def _sonuc(*vakalar):
    return {"arac": "t", "vakalar": list(vakalar)}


class Bulgu1MetinBoolean(unittest.TestCase):
    def test_metin_false_hata_verir(self):
        s = _sonuc({"veri_kumesi": "halka", "vaka_kimligi": "x-kusurlu",
                    "dosyalar": [{"dosya": "a.py", "kusurlu_mu": "false"}]})
        with self.assertRaisesRegex(ValueError, "kusurlu_mu"):
            puanla(_vakalar(), s)

    def test_sayi_bir_de_hata_verir(self):
        s = _sonuc({"veri_kumesi": "halka", "vaka_kimligi": "x-kusurlu",
                    "dosyalar": [{"dosya": "a.py", "kusurlu_mu": 1}]})
        with self.assertRaises(ValueError):
            puanla(_vakalar(), s)

    def test_gercek_boolean_calisir(self):
        s = _sonuc({"veri_kumesi": "halka", "vaka_kimligi": "x-kusurlu",
                    "dosyalar": [{"dosya": "a.py", "kusurlu_mu": True}]})
        self.assertEqual(puanla(_vakalar(), s)["halka"].tp, 1)


class Bulgu2DosyalarAlaniYok(unittest.TestCase):
    def test_kirilim_ve_cift_skoru_cokmez(self):
        s = _sonuc({"veri_kumesi": "halka", "vaka_kimligi": "x-kusurlu"})
        self.assertEqual(cift_skoru(_vakalar(), s), (0, 1))
        self.assertEqual(temiz_yanlis_alarm(_vakalar(), s), {"halka": (0, 1)})
        kirilim(_vakalar(), s)  # veri kümesinde olmayan vaka: atlanır, çökmez

    def test_vakalar_alani_yok(self):
        self.assertEqual(cift_skoru(_vakalar(), {"arac": "t"}), (0, 1))


class Bulgu3BelgeNotrCumlesi(unittest.TestCase):
    def test_belge_ne_fp_ne_fn_der(self):
        metin = (KOK / "ORTAK_FORMAT.md").read_text(encoding="utf-8")
        self.assertIn("ne FP ne FN", metin)
        self.assertNotIn("İşaretlenirse FP, işaretlenmezse FN sayılmaz", metin)


class Bulgu4TekrarVeKume(unittest.TestCase):
    def test_ayni_vaka_iki_kez_hata(self):
        v = {"veri_kumesi": "halka", "vaka_kimligi": "x-kusurlu",
             "dosyalar": [{"dosya": "a.py", "kusurlu_mu": True}]}
        with self.assertRaisesRegex(ValueError, "iki kez"):
            puanla(_vakalar(), _sonuc(v, dict(v, dosyalar=[])))

    def test_ayni_dosya_iki_kez_hata(self):
        v = {"veri_kumesi": "halka", "vaka_kimligi": "x-kusurlu",
             "dosyalar": [{"dosya": "a.py", "kusurlu_mu": True},
                          {"dosya": "a.py", "kusurlu_mu": False}]}
        with self.assertRaisesRegex(ValueError, "iki kez"):
            puanla(_vakalar(), _sonuc(v))

    def test_yanlis_veri_kumesi_hata(self):
        v = {"veri_kumesi": "zincir_dev", "vaka_kimligi": "x-kusurlu",
             "dosyalar": [{"dosya": "a.py", "kusurlu_mu": True}]}
        with self.assertRaisesRegex(ValueError, "veri_kumesi"):
            puanla(_vakalar(), _sonuc(v))


class Bulgu5LatexVeDosyaAdi(unittest.TestCase):
    def test_ad_korunur_ve_kacislanir(self):
        m = puanla(_vakalar(), _sonuc())
        tex = tablo_tex(m, "PR-Agent & Co_1")
        self.assertIn("PR-Agent \\& Co\\_1 taban", tex)
        self.assertIn("\\label{tbl:pr_agent_co_1_dosya}", tex)

    def test_kacis_ozel_karakterler(self):
        self.assertEqual(_tex_kacis("a_b%c#d$e{f}"), "a\\_b\\%c\\#d\\$e\\{f\\}")

    def test_dosya_adi_guvenli(self):
        self.assertEqual(_dosya_adi("PR-Agent/v2 beta"), "PR-Agent_v2_beta")


class Bulgu6PozitifDosyaDifftYok(unittest.TestCase):
    def test_pozitif_dosya_diffte_yoksa_hata(self):
        with tempfile.TemporaryDirectory() as g:
            kok = Path(g) / "halka_bench"
            (kok / "labels").mkdir(parents=True)
            (kok / "diffs").mkdir()
            etiket = {"vaka_kimligi": "v-kusurlu", "degisen_dosyalar": ["a.py", "b.py"],
                      "kusurlar": [{"konum": {"dosya": "b.py"}}]}
            (kok / "labels" / "v-kusurlu.json").write_text(json.dumps(etiket), encoding="utf-8")
            (kok / "diffs" / "v-kusurlu.diff").write_text(
                "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n",
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "b.py"):
                vakalari_yukle(Path(g))


class Bulgu7DosyaYollari(unittest.TestCase):
    def test_yolda_bosluk_b_egik_cizgi(self):
        metin = ("diff --git a/dir b/x.py b/dir b/x.py\n--- a/dir b/x.py\n+++ b/dir b/x.py\n"
                 "@@ -1 +1 @@\n-a\n+b\n")
        self.assertEqual(list(diffi_bol(metin)), ["dir b/x.py"])

    def test_tirnakli_turkce_yol(self):
        metin = ('diff --git "a/\\303\\274.py" "b/\\303\\274.py"\n'
                 '--- "a/\\303\\274.py"\n+++ "b/\\303\\274.py"\n@@ -1 +1 @@\n-a\n+b\n'
                 "diff --git a/z.py b/z.py\n--- a/z.py\n+++ b/z.py\n@@ -1 +1 @@\n-a\n+b\n")
        self.assertEqual(list(diffi_bol(metin)), ["ü.py", "z.py"])

    def test_icerigi_olmayan_mod_degisikligi(self):
        metin = ("diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n"
                 "diff --git a/z.py b/z.py\n--- a/z.py\n+++ b/z.py\n@@ -1 +1 @@\n-a\n+b\n")
        self.assertEqual(list(diffi_bol(metin)), ["run.sh", "z.py"])


class KomutSatiriGirdiHatasi(unittest.TestCase):
    def test_gecersiz_girdi_acik_mesaj_ve_cikis_3(self):
        import os
        import subprocess
        s = {"arac": "PR-Agent", "vakalar": [
            {"veri_kumesi": "halka", "vaka_kimligi": "authz-01-kusurlu",
             "dosyalar": [{"dosya": "halka/billing/views.py", "kusurlu_mu": "true"}]}]}
        with tempfile.TemporaryDirectory() as g:
            yol = Path(g) / "s.json"
            yol.write_text(json.dumps(s), encoding="utf-8")
            r = subprocess.run([sys.executable, "degerlendir.py", str(yol), "--cikti", g],
                               cwd=KOK, capture_output=True, timeout=120,
                               env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            self.assertEqual(r.returncode, 3, r.stderr.decode("utf-8", "replace"))
            self.assertIn("kusurlu_mu", r.stdout.decode("utf-8") + r.stderr.decode("utf-8"))
            self.assertNotIn("Traceback", r.stderr.decode("utf-8"))

    def test_gorunen_ad_ve_guvenli_dosya_adi(self):
        import os
        import subprocess
        s = {"arac": "PR-Agent/v2", "vakalar": []}
        with tempfile.TemporaryDirectory() as g:
            yol = Path(g) / "s.json"
            yol.write_text(json.dumps(s), encoding="utf-8")
            r = subprocess.run([sys.executable, "degerlendir.py", str(yol), "--cikti", g,
                                "--ad", "PR-Agent"], cwd=KOK, capture_output=True, timeout=120,
                               env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            self.assertEqual(r.returncode, 0, r.stderr.decode("utf-8", "replace"))
            tex = (Path(g) / "tablo_PR-Agent_v2.tex").read_text(encoding="utf-8")
            self.assertIn("\\caption{PR-Agent taban", tex)


if __name__ == "__main__":
    unittest.main()
