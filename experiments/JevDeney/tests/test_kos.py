"""kos.py testleri: önbellek, hata dayanıklılığı, ortak biçim."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from degerlendir import puanla  # noqa: E402
from istemci import SORULAR, SahteIstemci, istek_govdesi  # noqa: E402
from kos import kos  # noqa: E402
from veri import VARSAYILAN_REPOS, diffi_bol, vakalari_yukle  # noqa: E402


class SayacliIstemci(SahteIstemci):
    ad = "sayacli"

    def __init__(self):
        super().__init__()
        self.cagri = 0

    def sor(self, dosya, diff_parcasi):
        self.cagri += 1
        return super().sor(dosya, diff_parcasi)


class HataliIstemci(SahteIstemci):
    """'views.py' içeren parçada istisna fırlatır, diğerlerinde çalışır."""
    ad = "hatali"

    def sor(self, dosya, diff_parcasi):
        if dosya.endswith("views.py"):
            raise TimeoutError("zaman asimi")
        return super().sor(dosya, diff_parcasi)


class KosTestleri(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tum = vakalari_yukle(VARSAYILAN_REPOS)
        cls.vakalar = [v for v in tum if v.vaka_kimligi in
                       ("authz-01-kusurlu", "authz-01-temiz", "audit-01-kusurlu")]
        cls.dosya_sayisi = sum(len(v.degisen_dosyalar) for v in cls.vakalar)
        # Kusurlu ve temiz ikizde birebir aynı diff parçaları tek kez sorulur.
        cls.benzersiz_parca = len({diffi_bol(v.diff_yolu.read_text(encoding="utf-8"))[d]
                                   for v in cls.vakalar for d in v.degisen_dosyalar})

    def setUp(self):
        self.gecici = tempfile.TemporaryDirectory()
        self.onbellek = Path(self.gecici.name) / "onbellek.jsonl"

    def tearDown(self):
        self.gecici.cleanup()

    def test_onbellekte_olan_tekrar_sorulmaz(self):
        ist = SayacliIstemci()
        ilk = kos(self.vakalar, ist, self.onbellek)
        self.assertEqual(ist.cagri, self.benzersiz_parca)
        self.assertLess(self.benzersiz_parca, self.dosya_sayisi)
        ist2 = SayacliIstemci()
        ikinci = kos(self.vakalar, ist2, self.onbellek)
        self.assertEqual(ist2.cagri, 0)
        self.assertEqual(ilk["vakalar"], ikinci["vakalar"])

    def test_hata_veren_dosya_kosuyu_durdurmaz(self):
        sonuc = kos(self.vakalar, HataliIstemci(), self.onbellek)
        dosyalar = [d for v in sonuc["vakalar"] for d in v["dosyalar"]]
        self.assertEqual(len(dosyalar), self.dosya_sayisi)
        hatalilar = [d for d in dosyalar if d.get("hata")]
        self.assertEqual(len(hatalilar), 2)  # iki authz vakasının views.py dosyası
        for d in hatalilar:
            self.assertEqual((d["skor"], d["kusurlu_mu"]), (None, False))
            self.assertIn("zaman asimi", d["hata"])

    def test_hata_onbellege_yazilmaz(self):
        kos(self.vakalar, HataliIstemci(), self.onbellek)
        ist = SayacliIstemci()
        kos(self.vakalar, ist, self.onbellek)
        self.assertEqual(ist.cagri, 2)  # yalnızca hatalı olanlar yeniden sorulur

    def test_esik_yarim(self):
        sonuc = kos(self.vakalar, SahteIstemci(), self.onbellek)
        for v in sonuc["vakalar"]:
            for d in v["dosyalar"]:
                self.assertEqual(d["kusurlu_mu"], d["skor"] >= 0.5)

    def test_cikti_ortak_bicimde(self):
        sonuc = kos(self.vakalar, SahteIstemci(), self.onbellek)
        self.assertEqual(sonuc["arac"], "jev_sahte")  # sahte tablo rapora karışmasın
        self.assertEqual(sonuc["surum"], "sahte")
        m = puanla(self.vakalar, sonuc)
        self.assertEqual(m["toplam"].dosya, self.dosya_sayisi)
        self.assertEqual(m["toplam"].uyarilar, [])

    def test_yalnizca_diff_parcasi_gonderilir(self):
        gonderilen = []

        class Kayitci(SahteIstemci):
            def sor(self, dosya, diff_parcasi):
                gonderilen.append((dosya, diff_parcasi))
                return super().sor(dosya, diff_parcasi)

        kos(self.vakalar[:1], Kayitci(), self.onbellek)
        for dosya, parca in gonderilen:
            self.assertTrue(parca.startswith(f"diff --git a/{dosya} "))
            self.assertEqual(parca.count("diff --git "), 1)
            self.assertNotIn("Fatura iptal ucu", parca)  # PR başlığı gönderilmez

    def test_sonuc_soru_ve_model_kaydeder(self):
        sonuc = kos(self.vakalar, SahteIstemci(), self.onbellek)
        self.assertEqual(sonuc["sorular"], SORULAR)
        self.assertEqual(sonuc["model"], "sahte")
        d = sonuc["vakalar"][0]["dosyalar"][0]
        self.assertEqual(set(d["olasiliklar"]), set(SORULAR))
        self.assertEqual(d["skor"], max(d["olasiliklar"].values()))

    def test_onbellek_anahtari_modeli_icerir(self):
        kos(self.vakalar, SahteIstemci(), self.onbellek)
        baska = SayacliIstemci()
        baska.model = "baska-model"
        kos(self.vakalar, baska, self.onbellek)
        self.assertEqual(baska.cagri, self.benzersiz_parca)

    def test_onbellek_tam_istegi_saklar(self):
        kos(self.vakalar[:1], SahteIstemci(), self.onbellek)
        kayit = json.loads(self.onbellek.read_text(encoding="utf-8").splitlines()[0])
        v = self.vakalar[0]
        dosya = v.degisen_dosyalar[0]
        parca = diffi_bol(v.diff_yolu.read_text(encoding="utf-8"))[dosya]
        self.assertEqual(kayit["istek"], istek_govdesi(dosya, parca, model="sahte"))


if __name__ == "__main__":
    unittest.main()
