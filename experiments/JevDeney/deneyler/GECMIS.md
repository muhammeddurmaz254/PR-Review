# Jev deney günlüğü

Yapılmış ama ana yola eklenmemiş deneyler burada durur. Sonuçlar yerinde kalır, silinmez. Her kaydın sonuç dosyası ve karar gerekçesi vardır.

## 2026-11-10 — Deney-1A: PR bağlamı (`jev_pr_baglam`) — ANA YOLA EKLENMEDİ

> **Bağlam vermek de yetmedi.** PR'daki bütün dosyaların diff'i Jev'e bağlam olarak verildiğinde zincir'in ayrılmış test kümesinde F1 değişmedi (0,31 → 0,31). Toplam F1 0,49'dan 0,50'ye çıktı, yani anlamlı bir kazanç yok.

- **Ne yapıldı:** PR başına tek istek. State'te PR'daki bütün dosyaların diff'i vardı. Her dosya için ana koşuyla aynı iki Noul sorusu soruldu. Karar: maks. olasılık ≥ 0,5.
- **Sonuç (P / R / F1):**
  - halka: 0,80 / 0,41 / 0,54. Ana koşu 0,81 / 0,45 / 0,58 idi.
  - zincir_dev: 0,83 / 0,31 / 0,45. Ana koşu 0,50 / 0,19 / 0,27 idi.
  - zincir_holdout: 0,67 / 0,20 / 0,31. Ana koşuyla aynı.
  - Toplam: 0,79 / 0,36 / 0,50.
- **Diğer ölçüler:** çift skoru %33 (ana %37). Temiz halka vakalarında yanlış alarm %8 (ana %7). Faturalanan tutar 0,0092 dolar.
- **Yorum:**
  - Çok kusurlu PR'larda R %17'den %33'e çıktı ve zincir_dev'deki artış buradan geliyor. Ama holdout'ta tekrarlanmadı.
  - Bu, Jev'in zincir'deki zayıflığının yalnızca eksik bağlamdan değil, modelin dolaylı ve çok adımlı akıl yürütmedeki sınırından da kaynaklandığını gösteriyor. TypeSafe zayıf noktalar listesinde "indirection" maddesi bu sınırı anlatıyor.
- **Karar (kullanıcı, 2026-11-11):** Ana yola eklenmedi. Cümle burada saklanıyor.
- **Dosyalar:**
  - `deneyler/sonuclar/jev_pr_baglam_hepsi_2026-11-10.json`
  - `deneyler/onbellek/pr_baglam_gercek.jsonl`
  - `deneyler/sonuclar/karsilastirma.md`

## 2026-11-11 — Deney-1B v2: 15 atomik sınıf (`jev_atomik_v2`) — TEK BAŞINA KULLANILMAYACAK

> **Sınıfları bölmek ve yeni sınıf eklemek tek başına işe yaramadı.** Yanlış alarm, temiz halka vakalarında %36'dan %69'a çıktı.

- **Ne yapıldı:**
  - v1'de "veya" ile birleşmiş sorular bölündü: `missing_check` iki soruya, `unsafe_input_use` üç soruya.
  - `race_condition` gözlenebilir bir duruma daraltıldı.
  - Üç genel sınıf eklendi: zayıf kriptografi, sınır durumu, verimsiz veri erişimi.
- **Sonuç:**
  - Toplam P / R / F1: 0,40 / 0,77 / 0,52.
  - Ayrılmış test F1: 0,36. Çift skoru: %10.
  - Faturalanan tutar: 0,0251 dolar.
- **Neden başarısız oldu:** Geliştirme kümesinde üç sınıf temiz kodda çok sık evet dedi:
  - `missing_input_validation`: 66 temiz dosyada alarm, 37 isabet.
  - `shared_state_race`: 15 temiz dosyada alarm, 8 isabet.
  - `boundary_case`: 12 temiz dosyada alarm, 15 isabet.
  - Ders: "Bu kod girdiyi doğruluyor mu?" gibi sorular neredeyse her diff'te kısmen doğrudur. Soru somut olsa bile, temiz kodda da sık geçerli olan bir durumu soruyorsa ayırt edici değildir.
- **Nerede kullanıldı:** v2'nin olasılıkları seçili sınıf havuzuna girdi (`jev_atomik_secili`). Oradan yalnızca şu sınıflar seçildi: `weak_crypto`, `unsafe_deserialization`, `missing_authorization` ve korunan altı sınıf.
- **Dosyalar:**
  - `deneyler/sonuclar/jev_atomik_v2_hepsi_2026-11-11.json`
  - `deneyler/onbellek/atomik_v2_gercek.jsonl`

## 2026-11-11 — Deney-1B seçili sınıflar (`jev_atomik_secili`) — YAN SONUÇ, GELİŞTİRME DURDURULDU

- **Kural:** 18 sınıf (ana + v1 + v2 havuzundan), eşik 0,6. Seçim yalnızca halka ve zincir_dev üzerinde yapıldı.
- **Adil sonuç (zincir_holdout):** P 0,80, R 0,40, F1 0,53. v1 ile aynı. Ana koşu 0,31.
- **Örneklem içi sonuçlar (iyimser):** toplam F1 0,66, çift skoru %47, temiz halka vakalarında yanlış alarm %18.
- **Karar (kullanıcı):** (a) yan sonuç olarak kabul edildi, yeni sınıf denenmeyecek. Kalan seçenekler (b) yalnızca seçili soruları koşup gerçek maliyeti ölçmek ve (c) yeni sınıflar. İkisi de gerekirse buradan devam edilebilir.
- **Dosyalar:**
  - `deneyler/sonuclar/jev_atomik_secili_2026-11-11.json`
  - `deneyler/sonuclar/secim_2026-11-11.json`

## 2026-11-11 — Bağımsız inceleme ve düzeltmeler (ortak değerlendirici)

- **İnceleme:** Ayrı bir ajan (Sonnet), yalnızca `veri.py`, `degerlendir.py`, `kirilim.py` ve `ORTAK_FORMAT.md` dosyalarını okudu. Yaklaşık 67 bin token tuttu. Kritik hata çıkmadı. 7 gizli açık bulundu.
- **Düzeltilenler** (her biri önce hatayı gösteren bir testle, `tests/test_inceleme_duzeltmeleri.py`):
  1. `kusurlu_mu` gerçek true/false değilse değerlendirici açık hata verir (çıkış kodu 3). Önceden `"false"` metni işaretli sayılıyordu.
  2. `dosyalar` ya da `vakalar` alanı olmayan sonuç kırılım adımında çökmüyor.
  3. `ORTAK_FORMAT.md` belgesinde nötr dosya kuralı netleştirildi ("ne FP ne FN") ve girdi kuralları eklendi.
  4. Aynı vaka ya da aynı dosya iki kez yazılırsa ve `veri_kumesi` etiketle uyuşmazsa hata verilir.
  5. Araç adı LaTeX başlığında olduğu gibi kalır ve kaçışlanır. Etiket ve dosya adı güvenli hâle getirildi. `--ad` seçeneği eklendi.
  6. Kusurlu (pozitif) bir dosya diff'te yoksa veri yükleyici hata verir.
  7. Diff bölücü dosya adını `+++ b/` ya da `--- a/` satırından okur. Tırnaklı, Türkçe karakterli ve `" b/"` içeren yollar doğru çözülür.
- **Doğrulama:**
  - 95 test geçiyor.
  - Gerçek koşuların 1000 isteğinin hepsi yeni kodla önbellekte birebir eşleşti, yani sonuçlar değişmedi.
  - Bütün sonuç dosyaları yeni doğrulamadan geçti.
- **Bilerek bırakılan küçük sınırlar:**
  - Bir dosyada birden çok kusur varsa kırılımda ilk kusurun ailesi sayılır. Bugünkü veride böyle bir dosya yok.
  - `_rol` fonksiyonu yolda "config" geçen dosyayı yapılandırma sayar.
  - PR başına süre ortalaması yalnızca süre bildiren vakalar üzerinden hesaplanır.
  - LaTeX tablosunda yalnızca medyan süre var, ortalama `.md` özetinde.
