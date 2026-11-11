# JevDeney — Jev taban çizgisi ve ortak değerlendirici

Ara Rapor 1'de Jev'e tek bir rol verildi: karşılaştırmada **dosya düzeyinde ölçülen ucuz bir taban çizgisi** (Bölüm 1.2, Tablo 2, İP8). Bu klasör o ölçümü yapar. Ayrıca partnerin araçlarının (statik analiz, OCR vb.) da aynı kuralla puanlanabilmesi için **ortak sonuç biçimini ve değerlendiriciyi** içerir.

- Yalnızca Python standart kütüphanesi kullanılır, kurulum gerekmez.
- Veri kümeleri (`Desktop\Repos\halka_bench`, `zincir_bench`) yalnızca okunur, hiçbir dosyaları değiştirilmez.

## Ne ölçülüyor
- **Sağlayıcı:** TypeSafe API (`POST https://api.typesafe.ai/v1/systemone`). Model sabittir: `jev-1.14.0`, yani `jev-latest` alias'ı kullanılmaz ve sonuç tekrar üretilebilir.
- Değişen her dosya için ayrı bir istek atılır. State yalnızca `{"file_path", "diff"}` alanlarından oluşur, yani o dosyanın diff parçasıdır. PR başlığı ve açıklaması gönderilmez, çünkü Türkçe ve ipucu verir.
- Her istekte iki İngilizce Noul sorusu sorulur (`istemci.py` → `SORULAR`): `introduces_defect` ve `introduces_vulnerability`. TypeSafe rehberi tek soruda tek yargı istediği için bunlar iki ayrı sorudur.
- **Karar kodda verilir:** `max(olasılıklar) >= 0.5` ise dosya kusurlu sayılır. Eşik ayarlanmaz.
- **Hata veren çağrı "hayır" sayılmaz.** Hata varsa değerlendirici sonucu GEÇERSİZ sayar ve LaTeX tablosu yazmaz. Koşu tekrarlanır; önbellek yalnızca hatalı dosyaları yeniden sorar.
- Soru metinleri veri kümesine göre ayarlanmaz. Bu bir taban çizgisidir; benchmark üzerinde ince ayar test kümesini sızdırır.
- Sonuçlar halka, zincir dev, zincir holdout ve toplam için ayrı verilir: dosya düzeyi P/R/F1, PR başına süre, maliyet. Satır ve tür düzeyi ölçülmez (Tablo 2: "Bulgu konumu: Yok").
- Puanlama kuralı `ORTAK_FORMAT.md` dosyasında anlatılır.

## Dosyalar
| Dosya | Görev |
|---|---|
| `veri.py` | Etiketleri ve diff'leri okur, diff'i dosyalara böler, pozitif ve nötr dosyaları çıkarır |
| `istemci.py` | `SORULAR`, `MODEL`, `istek_govdesi`, `SahteIstemci`, `GercekIstemci`, `anahtar_oku` |
| `kos.py` | Koşu: soruları sorar, `onbellek/` klasörüne yazar, sonucu `sonuclar/` klasörüne koyar |
| `degerlendir.py` | **Ortak değerlendirici.** Herhangi bir aracın sonuç dosyasını puanlar |
| `ORTAK_FORMAT.md` | Partnerle paylaşılacak sonuç biçimi ve puanlama kuralı |
| `tests/` | `unittest` testleri |

## Komutlar (proje kökünden)
```powershell
py -m unittest discover -s JevDeney/tests              # testler
py JevDeney/kos.py --istemci sahte --kume hepsi         # sahte koşu (anlamsız sonuç, boru hattı sınaması)
py JevDeney/degerlendir.py JevDeney/sonuclar/jev_sahte_hepsi_<tarih>.json
py JevDeney/kos.py --istemci gercek --vaka authz-01-kusurlu   # gerçek duman testi (anahtar gerekir)
```
Çıktılar `sonuclar/` klasörüne yazılır:
- `jev_<istemci>_<küme|vaka>_<tarih>.json`: ortak biçimdeki ham sonuç
- `ozet_<arac>.md` (sahte koşuda araç adı `jev_sahte`): tablo, veri notları ve uyarılar
- `tablo_<arac>.tex`: rapora girecek LaTeX tablosu; başlığı tablonun üstünde, booktabs biçiminde

## API anahtarı
- Anahtar `JevDeney/.env` dosyasına tek satır olarak yazılır: `TYPESAFE_API_KEY=...`. İstersen aynı adla ortam değişkeni olarak da verebilirsin. Anahtarı sohbete yapıştırma.
- `.env` ve `onbellek/` git'e girmez (`.gitignore`).
- Önbellek kaydı tam istek gövdesini saklar (anahtar hariç). Böylece her sonuç yeniden koşulabilir.
- Tam koşu yaklaşık 282 çağrı ve 150 bin giriş token'ı tutar. $0.042 / 1M fiyatıyla bu 0,01 dolardan azdır (tahmin).

## Veri notu
zincir'deki 3 temiz vakanın etiketinde (`ckpt-01-temiz`, `log-01-temiz`, `timeout-01-temiz`), diff'te bulunmayan birer dosya `degisen_dosyalar` listesine yazılmış. `git diff --stat` diff'in doğru olduğunu gösteriyor. Ölçüm birimi diff'teki dosyalardır, bu yüzden zincir'de 127 değil 124 dosya sayılır. Pozitif dosya kaybı yoktur. Bu, veri kümesinin kendisinde düzeltilmesi gereken bir etiket hatası olabilir. Veri kümesi salt okunur olduğu için burada düzeltilmedi.

## Yan deneyler (`deneyler/`)
Ana koşuya dokunmayan deneyler burada durur. Her deneyin kendi araç adı, önbelleği ve sonuç dosyası vardır (`deneyler/sonuclar/`). Tasarım `deneyler/TASARIM.md` dosyasında anlatılır. Ana yola eklenmeyen deneyler `deneyler/GECMIS.md` dosyasında kayıtlıdır.

| Deney | Komut (proje kökünden) | Araç adı |
|---|---|---|
| 1A. PR bağlamı (ana yola eklenmedi, `GECMIS.md`) | `py JevDeney/deneyler/deney_kos.py --varyant pr_baglam --istemci gercek` | `jev_pr_baglam` |
| 1B. Atomik v1 (9 sınıf) / v2 (15 sınıf) | `py JevDeney/deneyler/deney_kos.py --varyant atomik\|atomik_v2 --istemci gercek` | `jev_atomik`, `jev_atomik_v2` |
| 1B. Sınıf seçimi (yeni çağrı yok, yalnızca geliştirme kümeleri) | `py JevDeney/deneyler/atomik_sec.py --ana A.json --v1 B.json --v2 C.json` | `jev_atomik_secili` |
| 3b. Satır bulma, hunk ya da blok | `py JevDeney/deneyler/satir.py --kaynak K.json --istemci gercek --birim hunk\|blok` | `jev_satir`, `jev_blok` |
| Karşılaştırma | `py JevDeney/deneyler/karsilastir.py <sonuc.json>... --satir <satir.json>...` | `karsilastirma.md` |

**Ana yolda (2026-11-11):** kırılım. `degerlendir.py` artık her araç için çift skorunu, temiz vakalardaki yanlış alarmı ve aile / dayanak / rol / PR boyutu kırılımını `ozet_<arac>.md` dosyasına yazıyor. `kirilim.py` bu yüzden ana klasöre taşındı.

**Sonuçların özeti:**
- Ana koşu: F1 0,49.
- Seçili atomik sınıflar: ayrılmış testte F1 0,53 (ana 0,31). Toplam F1 0,66 ve çift skoru %47 örneklem içi, yani iyimser. Yan sonuç olarak kabul edildi, geliştirme durduruldu.
- Blok bazlı satır bulma: çok bloklu dosyada isabet %92. Rastgele seçim %41, "en büyük blok" kuralı %80.
- Ayrıntı: `sonuclar/rapor_taslak.md`.
