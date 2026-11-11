"""Yan deneylerin testleri (ağa çıkmaz)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
sys.path.insert(0, str(KOK / "deneyler"))

from degerlendir import puanla  # noqa: E402
from deney_kos import birimler, deney_kos  # noqa: E402
from istemci import SORULAR, SahteIstemci  # noqa: E402
from kirilim import cift_skoru, kirilim, temiz_yanlis_alarm  # noqa: E402
from satir import Hunk, hunklari_bol, satir_degerlendir, satir_kos  # noqa: E402
from varyantlar import ATOMIK_SORULAR, pr_baglam_govde  # noqa: E402
from veri import VARSAYILAN_REPOS, Vaka, diffi_bol, vakalari_yukle  # noqa: E402

IKI_HUNK = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
            "@@ -1,3 +1,4 @@ def a():\n a\n+b\n c\n d\n"
            "@@ -40,5 +41,2 @@ def z():\n-q\n-r\n-s\n t\n u\n")


class SabitIstemci(SahteIstemci):
    """Soru adı belirli bir alt dizgeyi içeriyorsa 0.9, değilse 0.1 döner."""
    ad = "sabit"
    model = "sabit"

    def __init__(self, yuksek):
        self.yuksek = yuksek
        self.govdeler = []

    def gonder(self, govde):
        from istemci import HamYanit
        self.govdeler.append(govde)
        return HamYanit({q: (0.9 if self.yuksek in q else 0.1) for q in govde["questions"]},
                        5, 100, "sabit")


class VaryantTestleri(unittest.TestCase):
    def test_pr_baglam_tum_dosyalar_tek_state(self):
        govde, harita = pr_baglam_govde({"a.py": "da", "b/c.py": "dc"}, "m")
        self.assertEqual(govde["state"], {"changed_files": {
            "f0": {"path": "a.py", "diff": "da"}, "f1": {"path": "b/c.py", "diff": "dc"}}})
        self.assertEqual(set(govde["questions"]),
                         {"f0_defect", "f0_vulnerability", "f1_defect", "f1_vulnerability"})
        self.assertEqual(harita["f1_defect"], "b/c.py")
        self.assertIn("`changed_files.f1.diff`",
                      json.dumps(govde["questions"]["f1_defect"]["instructions"]))

    def test_atomik_sorular_tek_yargi_ve_noul(self):
        self.assertEqual(len(ATOMIK_SORULAR), 9)
        for ad, s in ATOMIK_SORULAR.items():
            self.assertEqual(s["type"], "noul", ad)
            self.assertEqual(set(s["criteria"]), {"true", "false"}, ad)

    def test_birimler_ana_ve_pr(self):
        parcalar = {"a.py": "da", "b.py": "db"}
        vaka = Vaka("halka", "v", ["a.py", "b.py"])
        self.assertEqual(len(birimler("atomik", vaka, parcalar, "m")), 2)
        self.assertEqual(len(birimler("pr_baglam", vaka, parcalar, "m")), 1)


class DeneyKosTestleri(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tum = vakalari_yukle(VARSAYILAN_REPOS)
        cls.vakalar = [v for v in tum if v.vaka_kimligi in ("authz-01-kusurlu", "audit-01-kusurlu")]

    def test_pr_baglam_dosya_karari_ve_sure_ilk_dosyada(self):
        with tempfile.TemporaryDirectory() as g:
            ist = SabitIstemci("f0_")
            sonuc = deney_kos(self.vakalar, ist, "pr_baglam", Path(g) / "o.jsonl")
        self.assertEqual(len(ist.govdeler), 2)  # PR başına 1 istek
        self.assertEqual(sonuc["arac"], "jev_pr_baglam_sabit")
        d = sonuc["vakalar"][0]["dosyalar"]
        self.assertEqual([x["kusurlu_mu"] for x in d], [True] + [False] * (len(d) - 1))
        self.assertEqual(d[0]["sure_ms"], 5)
        self.assertTrue(all(x["sure_ms"] is None for x in d[1:]))
        self.assertEqual(puanla(self.vakalar, sonuc)["toplam"].uyarilar, [])

    def test_atomik_maks_karar(self):
        with tempfile.TemporaryDirectory() as g:
            sonuc = deney_kos(self.vakalar, SabitIstemci("weakened_test"), "atomik", Path(g) / "o.jsonl")
        d = sonuc["vakalar"][0]["dosyalar"][0]
        self.assertEqual(d["skor"], 0.9)
        self.assertTrue(d["kusurlu_mu"])
        self.assertEqual(len(d["olasiliklar"]), 9)

    def test_hata_tum_birim_dosyalarina_yazilir(self):
        class Hatali(SahteIstemci):
            def gonder(self, govde):
                from istemci import HamYanit
                return HamYanit(None, None, None, None, hata="HTTP 503: x")
        with tempfile.TemporaryDirectory() as g:
            sonuc = deney_kos(self.vakalar[:1], Hatali(), "pr_baglam", Path(g) / "o.jsonl")
        d = sonuc["vakalar"][0]["dosyalar"]
        self.assertTrue(all(x.get("hata") and x["kusurlu_mu"] is False for x in d))

    def test_onbellek_tekrar_sormaz(self):
        with tempfile.TemporaryDirectory() as g:
            ist = SabitIstemci("x")
            deney_kos(self.vakalar, ist, "atomik", Path(g) / "o.jsonl")
            n = len(ist.govdeler)
            deney_kos(self.vakalar, ist, "atomik", Path(g) / "o.jsonl")
        self.assertEqual(len(ist.govdeler), n)


class HunkTestleri(unittest.TestCase):
    def test_hunklari_bol_yeni_taraf_araligi(self):
        h = hunklari_bol(IKI_HUNK)
        self.assertEqual([(x.yeni_bas, x.yeni_son) for x in h], [(1, 4), (41, 42)])
        self.assertTrue(h[0].metin.startswith("@@ -1,3 +1,4 @@"))

    def test_saf_silme_hunk_tek_satir(self):
        h = hunklari_bol("diff --git a/x b/x\n@@ -10,2 +9,0 @@\n-a\n-b\n")
        self.assertEqual((h[0].yeni_bas, h[0].yeni_son), (9, 9))

    def test_sayisiz_baslik(self):
        h = hunklari_bol("diff --git a/x b/x\n@@ -3 +3 @@\n-a\n+b\n")
        self.assertEqual((h[0].yeni_bas, h[0].yeni_son), (3, 3))


class SatirTestleri(unittest.TestCase):
    def setUp(self):
        self.vaka = Vaka("halka", "v1", ["x.py"], pozitif={"x.py"})
        self.etiket = {"v1": [{"dosya": "x.py", "bas": 41, "son": 41}]}
        self.kaynak = {"vakalar": [{"veri_kumesi": "halka", "vaka_kimligi": "v1",
                                    "dosyalar": [{"dosya": "x.py", "kusurlu_mu": True}]}]}
        self.parcalar = {"v1": {"x.py": IKI_HUNK}}

    def test_satir_kos_en_yuksek_hunk(self):
        with tempfile.TemporaryDirectory() as g:
            ist = SabitIstemci("h1")
            s = satir_kos([self.vaka], self.kaynak, ist, Path(g) / "o.jsonl",
                          parcalar=self.parcalar)
        tahmin = s["vakalar"][0]["dosyalar"][0]
        self.assertEqual((tahmin["tahmin_bas"], tahmin["tahmin_son"]), (41, 42))
        self.assertEqual(len(ist.govdeler), 1)

    def test_tek_hunk_cagri_yapmaz(self):
        tek = {"v1": {"x.py": "diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-a\n+b\n"}}
        with tempfile.TemporaryDirectory() as g:
            ist = SabitIstemci("h0")
            s = satir_kos([self.vaka], self.kaynak, ist, Path(g) / "o.jsonl", parcalar=tek)
        self.assertEqual(ist.govdeler, [])
        self.assertTrue(s["vakalar"][0]["dosyalar"][0]["tek_hunk"])

    def test_degerlendir_isabet_ve_rastgele_taban(self):
        sonuc = {"vakalar": [{"veri_kumesi": "halka", "vaka_kimligi": "v1", "dosyalar": [
            {"dosya": "x.py", "tahmin_bas": 41, "tahmin_son": 42, "tek_hunk": False,
             "hunk_araliklari": [[1, 4], [41, 42]]}]}]}
        m = satir_degerlendir([self.vaka], sonuc, self.etiket)
        self.assertEqual((m["dogru_dosya"], m["isabet"]), (1, 1))
        self.assertAlmostEqual(m["rastgele_beklenen"], 0.5)
        self.assertAlmostEqual(m["rastgele_cok_hunk"], 0.5)
        # en büyük birim [1,4] kuralı etiketi (41) ıskalar
        self.assertEqual(m["en_buyuk_cok_hunk_isabet"], 0)
        self.assertAlmostEqual(m["uctan_uca_satir_r"], 1.0)

    def test_pozitif_olmayan_dosya_sayilmaz(self):
        vaka = Vaka("halka", "v1", ["x.py"], pozitif=set())
        sonuc = {"vakalar": [{"veri_kumesi": "halka", "vaka_kimligi": "v1", "dosyalar": [
            {"dosya": "x.py", "tahmin_bas": 1, "tahmin_son": 4, "tek_hunk": False,
             "hunk_araliklari": [[1, 4]]}]}]}
        self.assertEqual(satir_degerlendir([vaka], sonuc, {})["dogru_dosya"], 0)


class KirilimTestleri(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vakalar = vakalari_yukle(VARSAYILAN_REPOS)

    def _sonuc(self, karar):
        return {"vakalar": [{"veri_kumesi": v.veri_kumesi, "vaka_kimligi": v.vaka_kimligi,
                             "dosyalar": [{"dosya": d, "kusurlu_mu": karar(v, d)}
                                          for d in v.degisen_dosyalar]} for v in self.vakalar]}

    def test_kahin_her_grupta_tam_r(self):
        k = kirilim(self.vakalar, self._sonuc(lambda v, d: d in v.pozitif), VARSAYILAN_REPOS)
        for eksen, gruplar in k.items():
            for grup, (tp, fn) in gruplar.items():
                self.assertEqual(fn, 0, (eksen, grup))
        self.assertEqual(sum(tp for tp, _ in k["aile"].values()), 75)
        self.assertEqual(set(k["rol"]), {"kod", "test", "yapilandirma"})

    def test_cift_skoru_kahin_ve_hepsi(self):
        kahin = self._sonuc(lambda v, d: d in v.pozitif)
        hepsi = self._sonuc(lambda v, d: True)
        self.assertEqual(cift_skoru(self.vakalar, kahin), (49, 49))
        self.assertEqual(cift_skoru(self.vakalar, hepsi), (0, 49))

    def test_temiz_yanlis_alarm(self):
        hepsi = self._sonuc(lambda v, d: True)
        oranlar = temiz_yanlis_alarm(self.vakalar, hepsi)
        self.assertEqual(oranlar["halka"], (61, 61))  # 49 ikiz + 12 bağımsız temiz


if __name__ == "__main__":
    unittest.main()


class KarsilastirmaTestleri(unittest.TestCase):
    def test_rapor_her_araci_ve_bolumleri_icerir(self):
        from karsilastir import rapor_md
        vakalar = vakalari_yukle(VARSAYILAN_REPOS)

        def sonuc(arac, karar):
            return {"arac": arac, "vakalar": [
                {"veri_kumesi": v.veri_kumesi, "vaka_kimligi": v.vaka_kimligi,
                 "dosyalar": [{"dosya": d, "kusurlu_mu": karar(v, d)} for d in v.degisen_dosyalar]}
                for v in vakalar]}
        md = rapor_md(vakalar, [sonuc("kahin", lambda v, d: d in v.pozitif),
                                sonuc("hepsi", lambda v, d: True)], [], VARSAYILAN_REPOS)
        for parca in ("kahin", "hepsi", "Çift skoru", "Temiz vakalarda yanlış alarm",
                      "Kırılım: aile", "49/49", "0/49"):
            self.assertIn(parca, md)

    def test_gecersiz_sonuc_isaretlenir(self):
        from karsilastir import rapor_md
        v = Vaka("halka", "v1", ["a.py"], pozitif={"a.py"})
        hatali = {"arac": "h", "vakalar": [{"veri_kumesi": "halka", "vaka_kimligi": "v1",
                  "dosyalar": [{"dosya": "a.py", "kusurlu_mu": False, "hata": "x"}]}]}
        self.assertIn("GEÇERSİZ", rapor_md([v], [hatali], [], VARSAYILAN_REPOS))


IKI_BLOKLU_HUNK = ("diff --git a/y.py b/y.py\n--- a/y.py\n+++ b/y.py\n"
                   "@@ -10,7 +10,8 @@ def f():\n a\n-b\n+B\n c\n d\n e\n+x\n+y\n f\n")


class BlokTestleri(unittest.TestCase):
    def test_bloklar_ardisik_degisiklikler(self):
        from satir import bloklari_bol
        b = bloklari_bol(IKI_BLOKLU_HUNK)
        self.assertEqual([(x.yeni_bas, x.yeni_son) for x in b], [(11, 11), (15, 16)])
        self.assertIn("+B", b[0].metin)
        self.assertNotIn("+x", b[0].metin)

    def test_saf_silme_blogu_konumu(self):
        from satir import bloklari_bol
        b = bloklari_bol(IKI_HUNK)
        self.assertEqual([(x.yeni_bas, x.yeni_son) for x in b], [(2, 2), (41, 41)])

    def test_blok_kosu_en_yuksek_blok_ve_tam_diff_baglam(self):
        vaka = Vaka("halka", "v1", ["y.py"], pozitif={"y.py"})
        kaynak = {"vakalar": [{"veri_kumesi": "halka", "vaka_kimligi": "v1",
                               "dosyalar": [{"dosya": "y.py", "kusurlu_mu": True}]}]}
        with tempfile.TemporaryDirectory() as g:
            ist = SabitIstemci("b1")
            s = satir_kos([vaka], kaynak, ist, Path(g) / "o.jsonl",
                          parcalar={"v1": {"y.py": IKI_BLOKLU_HUNK}}, birim="blok")
        d = s["vakalar"][0]["dosyalar"][0]
        self.assertEqual((d["tahmin_bas"], d["tahmin_son"]), (15, 16))
        self.assertEqual(s["arac"], "jev_blok_sabit")
        govde = ist.govdeler[0]
        self.assertEqual(govde["state"]["diff"], IKI_BLOKLU_HUNK)
        self.assertEqual(set(govde["state"]["blocks"]), {"b0", "b1"})


class AtomikV2Testleri(unittest.TestCase):
    def test_v2_siniflari(self):
        from varyantlar import ATOMIK_V2_SORULAR
        self.assertEqual(len(ATOMIK_V2_SORULAR), 15)
        for ayrilan in ("missing_check", "unsafe_input_use", "race_condition"):
            self.assertNotIn(ayrilan, ATOMIK_V2_SORULAR)
        for yeni in ("missing_authorization", "missing_input_validation", "injection",
                     "outbound_request", "unsafe_deserialization", "shared_state_race",
                     "weak_crypto", "boundary_case", "inefficient_data_access"):
            self.assertIn(yeni, ATOMIK_V2_SORULAR)
        for s in ATOMIK_V2_SORULAR.values():
            self.assertEqual(s["type"], "noul")

    def test_v2_birimi_dosya_basina(self):
        vaka = Vaka("halka", "v", ["a.py", "b.py"])
        b = birimler("atomik_v2", vaka, {"a.py": "da", "b.py": "db"}, "m")
        self.assertEqual(len(b), 2)
        self.assertEqual(len(b[0][0]["questions"]), 15)


class KuralSecimiTestleri(unittest.TestCase):
    def setUp(self):
        self.vakalar = [Vaka("halka", "d1", ["a.py", "b.py"], pozitif={"a.py"}),
                        Vaka("zincir_holdout", "t1", ["c.py"], pozitif={"c.py"})]
        self.sonuc = {"arac": "jev_atomik_v2", "vakalar": [
            {"veri_kumesi": "halka", "vaka_kimligi": "d1", "dosyalar": [
                {"dosya": "a.py", "olasiliklar": {"x": 0.65}, "kusurlu_mu": True, "skor": 0.65},
                {"dosya": "b.py", "olasiliklar": {"x": 0.55}, "kusurlu_mu": True, "skor": 0.55}]},
            {"veri_kumesi": "zincir_holdout", "vaka_kimligi": "t1", "dosyalar": [
                {"dosya": "c.py", "olasiliklar": {"x": 0.58}, "kusurlu_mu": True, "skor": 0.58}]}]}

    def test_esik_yalnizca_gelistirme_kumesinde_secilir(self):
        from kurallar import esik_sec
        secim = esik_sec(self.vakalar, self.sonuc, (0.5, 0.6, 0.7), ("halka", "zincir_dev"))
        # dev: 0.5 → P .5 F1 .67; 0.6 → P 1 R 1 F1 1; 0.7 → F1 0
        self.assertEqual(secim["esik"], 0.6)
        self.assertEqual(set(secim["tarama"]), {0.5, 0.6, 0.7})

    def test_esik_uygula_kararlari_yeniden_yazar(self):
        from kurallar import esik_uygula
        yeni = esik_uygula(self.sonuc, 0.6)
        kararlar = [d["kusurlu_mu"] for v in yeni["vakalar"] for d in v["dosyalar"]]
        self.assertEqual(kararlar, [True, False, False])
        self.assertEqual(yeni["esik"], 0.6)
        self.assertEqual(yeni["arac"], "jev_atomik_v2_e060")
        self.assertTrue(self.sonuc["vakalar"][0]["dosyalar"][1]["kusurlu_mu"])  # kaynak bozulmaz


class SinifSecimiTestleri(unittest.TestCase):
    def test_ileri_secim_gurultulu_sinifi_almaz(self):
        from kurallar import sinif_sec, siniflari_uygula
        vakalar = [Vaka("halka", "d1", ["a.py", "b.py", "c.py"], pozitif={"a.py", "b.py"}),
                   Vaka("zincir_holdout", "t1", ["z.py"], pozitif={"z.py"})]
        # iyi1 a'yı, iyi2 b'yi yakalar; gurultu her şeyi işaretler
        o = {"a.py": {"v1:iyi1": .9, "v1:iyi2": .1, "v2:gurultu": .9},
             "b.py": {"v1:iyi1": .1, "v1:iyi2": .9, "v2:gurultu": .9},
             "c.py": {"v1:iyi1": .1, "v1:iyi2": .1, "v2:gurultu": .9}}
        sonuc = {"arac": "havuz", "vakalar": [
            {"veri_kumesi": "halka", "vaka_kimligi": "d1",
             "dosyalar": [{"dosya": d, "olasiliklar": p} for d, p in o.items()]},
            {"veri_kumesi": "zincir_holdout", "vaka_kimligi": "t1",
             "dosyalar": [{"dosya": "z.py", "olasiliklar": {"v1:iyi1": .1, "v1:iyi2": .1,
                                                            "v2:gurultu": .9}}]}]}
        secim = sinif_sec(vakalar, sonuc)
        self.assertEqual(sorted(secim["siniflar"]), ["v1:iyi1", "v1:iyi2"])
        self.assertAlmostEqual(secim["dev_f1"], 1.0)
        uygulanan = siniflari_uygula(sonuc, secim["siniflar"], secim["esik"], "jev_secili")
        self.assertFalse(uygulanan["vakalar"][1]["dosyalar"][0]["kusurlu_mu"])  # holdout'a dokunulmadı, kural uygulandı
        self.assertEqual(uygulanan["arac"], "jev_secili")

    def test_havuz_birlestirir(self):
        from kurallar import havuz_olustur
        a = {"arac": "a", "vakalar": [{"veri_kumesi": "halka", "vaka_kimligi": "v", "dosyalar": [
            {"dosya": "x.py", "olasiliklar": {"p": .2}, "giris_token": 10, "sure_ms": 5}]}]}
        b = {"arac": "b", "vakalar": [{"veri_kumesi": "halka", "vaka_kimligi": "v", "dosyalar": [
            {"dosya": "x.py", "olasiliklar": {"p": .3, "q": .4}, "giris_token": 20, "sure_ms": 7}]}]}
        h = havuz_olustur({"v1": a, "v2": b})
        d = h["vakalar"][0]["dosyalar"][0]
        self.assertEqual(d["olasiliklar"], {"v1:p": .2, "v2:p": .3, "v2:q": .4})
        self.assertEqual((d["giris_token"], d["sure_ms"]), (30, 12))
