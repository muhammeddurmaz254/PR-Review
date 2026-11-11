"""veri.py testleri. Gerçek veri kümelerini salt okunur okur."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veri import VARSAYILAN_REPOS, diffi_bol, vakalari_yukle  # noqa: E402


class GercekVeriTestleri(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vakalar = vakalari_yukle(VARSAYILAN_REPOS)
        cls.kimlikle = {v.vaka_kimligi: v for v in cls.vakalar}

    def _kume(self, ad):
        return [v for v in self.vakalar if v.veri_kumesi == ad]

    def test_vaka_sayilari(self):
        self.assertEqual(len(self._kume("halka")), 110)
        self.assertEqual(len(self._kume("zincir_dev")), 26)
        self.assertEqual(len(self._kume("zincir_holdout")), 18)

    def test_dosya_sayilari(self):
        halka = sum(len(v.degisen_dosyalar) for v in self._kume("halka"))
        zincir = sum(len(v.degisen_dosyalar) for v in self.vakalar
                     if v.veri_kumesi.startswith("zincir"))
        self.assertEqual(halka, 158)
        # Etiket 127 dosya sayar; 3 temiz ikizin etiketinde diff'te olmayan
        # birer dosya fazladan yazılı (git ile doğrulandı). Ölçüm birimi diff'tir.
        self.assertEqual(zincir, 124)

    def test_etikette_olup_diffte_olmayan_dosya_ayiklanir(self):
        beklenen = {"ckpt-01-temiz": "tests/test_worker.py",
                    "log-01-temiz": "tests/conftest.py",
                    "timeout-01-temiz": "zincir/defaults.py"}
        for kimlik, dosya in beklenen.items():
            vaka = self.kimlikle[kimlik]
            self.assertNotIn(dosya, vaka.degisen_dosyalar)
            self.assertEqual(vaka.diffte_olmayan, [dosya])

    def test_pozitif_dosya_sayilari(self):
        halka = sum(len(v.pozitif) for v in self._kume("halka"))
        zincir = sum(len(v.pozitif) for v in self.vakalar
                     if v.veri_kumesi.startswith("zincir"))
        self.assertEqual(halka, 49)
        self.assertEqual(zincir, 26)

    def test_notr_dosya(self):
        self.assertEqual(self.kimlikle["maint-02-kusurlu"].notr,
                         {"halka/billing/services.py"})
        self.assertIn("zincir/settings.py", self.kimlikle["dlq-01-kusurlu"].notr)

    def test_temiz_vakada_pozitif_yok(self):
        self.assertEqual(self.kimlikle["authz-01-temiz"].pozitif, set())

    def test_diffi_bol_authz01(self):
        vaka = self.kimlikle["authz-01-kusurlu"]
        parcalar = diffi_bol(vaka.diff_yolu.read_text(encoding="utf-8"))
        self.assertEqual(len(parcalar), 3)
        self.assertEqual(sorted(parcalar), sorted(vaka.degisen_dosyalar))

    def test_her_degisen_dosyanin_diff_parcasi_var(self):
        for v in self.vakalar:
            parcalar = diffi_bol(v.diff_yolu.read_text(encoding="utf-8"))
            for d in v.degisen_dosyalar:
                self.assertIn(d, parcalar, v.vaka_kimligi)


class DiffBolmeTestleri(unittest.TestCase):
    def test_yeni_dosya_adi_b_tarafindan(self):
        metin = ("diff --git a/x/yeni.py b/x/yeni.py\nnew file mode 100644\n"
                 "--- /dev/null\n+++ b/x/yeni.py\n@@ -0,0 +1 @@\n+a = 1\n")
        self.assertEqual(list(diffi_bol(metin)), ["x/yeni.py"])

    def test_silinen_dosya_adi_a_tarafindan(self):
        metin = ("diff --git a/x/eski.py b/x/eski.py\ndeleted file mode 100644\n"
                 "--- a/x/eski.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-a = 1\n")
        self.assertEqual(list(diffi_bol(metin)), ["x/eski.py"])

    def test_crlf_diff_bolunur(self):
        metin = ("diff --git a/a.py b/a.py\r\n--- a/a.py\r\n+++ b/a.py\r\n@@ -1 +1 @@\r\n-x\r\n+y\r\n"
                 "diff --git a/b.py b/b.py\r\n--- a/b.py\r\n+++ b/b.py\r\n@@ -1 +1 @@\r\n-x\r\n+z\r\n")
        parcalar = diffi_bol(metin)
        self.assertEqual(sorted(parcalar), ["a.py", "b.py"])
        self.assertNotIn("\r", parcalar["a.py"])
        self.assertTrue(parcalar["b.py"].startswith("diff --git a/b.py"))


if __name__ == "__main__":
    unittest.main()
