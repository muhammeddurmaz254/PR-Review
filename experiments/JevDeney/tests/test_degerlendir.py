"""degerlendir.py testleri: ortak değerlendiricinin puanlama kuralı."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from degerlendir import puanla, tablo_md, tablo_tex  # noqa: E402
from veri import VARSAYILAN_REPOS, Vaka, vakalari_yukle  # noqa: E402


def _sonuc(vakalar, karar, **dosya_alanlari):
    """karar(vaka, dosya) -> bool ile ortak biçimde sonuç üretir."""
    return {
        "arac": "test", "surum": "t", "tarih": "2026-11-10",
        "birim_fiyat_usd_1m": dosya_alanlari.pop("fiyat", None),
        "vakalar": [
            {"veri_kumesi": v.veri_kumesi, "vaka_kimligi": v.vaka_kimligi,
             "dosyalar": [dict({"dosya": d, "kusurlu_mu": karar(v, d)}, **dosya_alanlari)
                          for d in v.degisen_dosyalar]}
            for v in vakalar
        ],
    }


class GercekVeriPuanlama(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vakalar = vakalari_yukle(VARSAYILAN_REPOS)

    def test_kahin_tam_puan(self):
        m = puanla(self.vakalar, _sonuc(self.vakalar, lambda v, d: d in v.pozitif))
        for kume in ("halka", "zincir_dev", "zincir_holdout", "toplam"):
            self.assertEqual((m[kume].p, m[kume].r, m[kume].f1), (1.0, 1.0, 1.0), kume)
        self.assertEqual(m["halka"].tp, 49)

    def test_hepsi_kusurlu(self):
        m = puanla(self.vakalar, _sonuc(self.vakalar, lambda v, d: True))
        halka = [v for v in self.vakalar if v.veri_kumesi == "halka"]
        notr = sum(len(v.notr & set(v.degisen_dosyalar)) for v in halka)
        self.assertEqual(m["halka"].r, 1.0)
        self.assertAlmostEqual(m["halka"].p, 49 / (158 - notr))

    def test_hic_isaret_yok(self):
        m = puanla(self.vakalar, _sonuc(self.vakalar, lambda v, d: False))
        self.assertEqual((m["toplam"].p, m["toplam"].r, m["toplam"].f1), (0.0, 0.0, 0.0))
        self.assertEqual(m["toplam"].fn, 75)

    def test_eksik_vaka_isaretlenmemis_sayilir(self):
        m = puanla(self.vakalar, {"arac": "t", "vakalar": []})
        self.assertEqual(m["toplam"].fn, 75)
        self.assertEqual(m["toplam"].fp, 0)


class KucukOrnekPuanlama(unittest.TestCase):
    def setUp(self):
        self.vakalar = [Vaka("halka", "v1", ["a.py", "b.py", "c.py"],
                             pozitif={"a.py"}, notr={"b.py"})]

    def _sonuc(self, dosyalar, fiyat=None):
        return {"arac": "t", "birim_fiyat_usd_1m": fiyat,
                "vakalar": [{"veri_kumesi": "halka", "vaka_kimligi": "v1",
                             "dosyalar": dosyalar}]}

    def test_notr_dosya_fp_sayilmaz(self):
        m = puanla(self.vakalar, self._sonuc([
            {"dosya": "a.py", "kusurlu_mu": True},
            {"dosya": "b.py", "kusurlu_mu": True}]))
        self.assertEqual((m["halka"].tp, m["halka"].fp, m["halka"].fn), (1, 0, 0))

    def test_bilinmeyen_dosya_uyari(self):
        m = puanla(self.vakalar, self._sonuc([
            {"dosya": "a.py", "kusurlu_mu": True},
            {"dosya": "yok.py", "kusurlu_mu": True}]))
        self.assertEqual(m["halka"].fp, 0)
        self.assertEqual(m["halka"].uyarilar, ["v1: etikette olmayan dosya yok.py"])

    def test_hata_sayisi_ve_sure(self):
        m = puanla(self.vakalar, self._sonuc([
            {"dosya": "a.py", "kusurlu_mu": False, "hata": "zaman asimi", "sure_ms": None},
            {"dosya": "c.py", "kusurlu_mu": False, "sure_ms": 100},
            {"dosya": "b.py", "kusurlu_mu": False, "sure_ms": 300}]))
        self.assertEqual(m["halka"].hata_sayisi, 1)
        self.assertEqual(m["halka"].sure_ort_ms, 400)   # PR başına toplam süre
        self.assertEqual(m["halka"].sure_medyan_ms, 400)

    def test_maliyet_hesaplanir(self):
        m = puanla(self.vakalar, self._sonuc(
            [{"dosya": "a.py", "kusurlu_mu": True, "giris_token": 1_000_000}], fiyat=0.042))
        self.assertAlmostEqual(m["halka"].maliyet_usd, 0.042)

    def test_maliyet_null_ise_tire(self):
        m = puanla(self.vakalar, self._sonuc([{"dosya": "a.py", "kusurlu_mu": True}]))
        self.assertIsNone(m["halka"].maliyet_usd)
        satir = [s for s in tablo_md(m).splitlines() if s.startswith("| halka")][0]
        self.assertTrue(satir.rstrip().endswith("| — |"), satir)

    def test_tex_tablo_baslik_ustte(self):
        m = puanla(self.vakalar, self._sonuc([{"dosya": "a.py", "kusurlu_mu": True}]))
        tex = tablo_tex(m, "Jev")
        self.assertLess(tex.index("\\caption"), tex.index("\\begin{tabular"))
        self.assertIn("\\toprule", tex)


if __name__ == "__main__":
    unittest.main()
