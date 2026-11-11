# Rapor taslağı — Jev taban çizgisi (Ara Rapor 2: Bölüm 3, Final: Bölüm 4)

> Taslaktır. AR2 çalışma klasörü açılınca `.tex` dosyasına aktarılır. Aktarırken sürüm kuralları, `check_tex.py` ve atıf/kaynakça eşleşmesi uygulanır.
> - Ana kaynak: `JevDeney/sonuclar/jev_gercek_hepsi_2026-11-10.json`. Kırılımlı özet: `ozet_jev.md`.
> - Yan deneyler: `JevDeney/deneyler/sonuclar/karsilastirma.md`.
> - Ana yola eklenmeyen deneyler: `JevDeney/deneyler/GECMIS.md`.

## Bölüm 3 için: Kullanılan yöntem (Jev taban çizgisi)

Jev \cite{systemone}, tek geçişli bir karar modelidir. Kapalı ağırlıklıdır ve yalnızca bulut API'si üzerinden kullanılabilir. Bu nedenle PR Kod Analizi'nin parçası değildir ve karşılaştırmada yalnızca dosya düzeyinde ucuz bir taban çizgisi olarak yer alır. Ölçüm şöyle yapılmıştır:

- **Girdi:** PR'daki her değişen dosya için yalnızca o dosyanın diff parçası ve dosya yolu gönderilmiştir. PR başlığı ve açıklaması gönderilmemiştir.
- **Sorular:** Her dosya için iki evet/hayır sorusu sorulmuş, her soru için bir olasılık alınmıştır. İlki değişikliğin işlevsel bir kusur ekleyip eklemediğini, ikincisi bir güvenlik zafiyeti ekleyip eklemediğini sorar. Üreticinin tek soruda tek yargı önerisine uyulmuştur.
- **Karar:** İki olasılıktan büyük olanı 0,5 veya üzerindeyse dosya kusurlu sayılmıştır. Eşik veri kümesine göre ayarlanmamıştır. Soru metinleri sonuçlar görülmeden sabitlenmiştir.
- **Model:** `jev-1.14.0` sürümü sabitlenmiştir. Her istek ve yanıt kaydedilmiştir; ölçüm tekrarlanabilir.
- **Değerlendirme:** Birim, (PR, değişen dosya) çiftidir.
  - Bir kusurun konumunun bulunduğu dosya pozitif sayılır. Yalnızca kusurun eşdeğer konumu olarak geçen dosya puana katılmaz.
  - zincir'de üç temiz vakanın etiketinde diff'te bulunmayan birer dosya yer aldığı için ölçüm diff'teki dosyalarla yapılmıştır. Bu nedenle halka'da 158, zincir'de 124 dosya vardır.
  - P, R ve F1'e ek olarak iki ölçü verilmiştir:
    - **Çift skoru:** Kusurlu varyantta doğru dosyanın işaretlenmesi ve aynı PR'ın temiz ikizinde hiçbir dosyanın işaretlenmemesi.
    - **Temiz vakada yanlış alarm oranı.**
  - Sonuçlar ayrıca kusur ailesi, dayanak, dosya rolü ve PR boyutuna göre kırılmıştır.
- **Kapsam dışı:** Jev yalnızca dosya düzeyinde kusurlu/temiz kararı için kullanılmıştır. Kusur türü sınıflaması ve satır konumu bu taban çizgisinde ölçülmemiştir.

## Bölüm 4 için: Sonuçlar (ana koşu, 10 Kasım 2026)

Tablo: Jev taban çizgisinin dosya düzeyi sonuçları (`JevDeney/sonuclar/tablo_jev.tex`)

| Veri kümesi | Dosya | P | R | F1 | Medyan süre/PR (ms) | Maliyet (USD) |
|---|---|---|---|---|---|---|
| halka | 158 | 0,815 | 0,449 | 0,579 | 334 | 0,0056 |
| zincir (geliştirme) | 73 | 0,500 | 0,188 | 0,273 | 829 | 0,0025 |
| zincir (ayrılmış test) | 51 | 0,667 | 0,200 | 0,308 | 638 | 0,0018 |
| Toplam | 282 | 0,750 | 0,360 | 0,486 | 405 | 0,0098 |

**Yorum:**
1. **halka'da kesinlik yüksek, duyarlılık orta.** Jev işaretlediği dosyalarda çoğunlukla haklıdır, ancak kusurlu dosyaların yarısından fazlasını kaçırır.
2. **zincir'de duyarlılık 0,20'ye düşer.** zincir'de kusurlar dosyalar ve aşamalar arasına yayılır, test ve yapılandırma dosyalarında da bulunur. Tek dosyanın diff'ini gören tek geçişli bir model bu bağları kuramaz. Bu düşüş, değişikliği depo bağlamıyla doğrulayan ajan tabanlı bir yaklaşımın neden gerektiğini gösterir.
3. **Ayırt etme gücü sınırlıdır.**
   - halka'da çift skoru %37'dir (18/49). Kusurlu PR'ların ancak üçte birinden biraz fazlasında hem doğru dosya bulunmuş hem temiz ikizde sessiz kalınmıştır.
   - Temiz vakalarda yanlış alarm düşüktür: halka %7, zincir geliştirme %11, zincir ayrılmış test %8.
4. **Kırılım, Jev'in güçlü ve kör olduğu yerleri ayırır.**
   - **Güçlü:** gizli veri, SSRF, enjeksiyon ve yetkilendirme ailelerinde 3/3. Bunlar diff yüzeyinde görülen, kalıp benzeri kusurlardır.
   - **Kör:** test kalitesi (halka 0/3), sınır durumları (0/3), kaynak ve başarım (0/3).
   - **Dosya rolüne göre:** test dosyalarında R %15, kod dosyalarında %42.
   - **Dayanak (c)'de R %39:** Deponun başka yerlerindeki bir kuralı bozan kusurlar, Jev o yerleri görmediği için düşük kalır.
5. **Maliyet ve süre çok düşüktür.** 282 dosyalık koşu yaklaşık 0,01 dolar tutmuş, PR başına medyan süre 0,4 saniyede kalmıştır.
   - Tablodaki maliyet her dosyanın token sayısıyla hesaplanmıştır. Kusurlu ve temiz ikizlerin birebir aynı diff parçaları bir kez sorulduğu için faturalanan tutar 0,0085 dolardır.
6. **Satır ve tür düzeyinde Jev'in karşılığı yoktur.** Karşılaştırma tablosunda bu hücreler "uygulanamaz" olarak gösterilir.

**Sınırlılıklar:**
- Sonuçlar tek koşudandır. Pozitif dosya sayısı küçüktür: zincir'in ayrılmış test kümesinde 10, kırılımdaki her halka ailesinde 3.
- Diff'lerdeki kod yorumları Türkçedir; Jev ağırlıklı olarak İngilizce eğitilmiştir.
- Jev'in olasılıkları bu veri kümesi için kalibre edilmemiştir. 0,5 eşiği varsayılandır.

## Yan sonuçlar (ana taban çizgisinin yerine geçmez)

### Atomik sorular: soru tasarımının etkisi
Geniş iki soru yerine somut ve gözlenebilir sorular sorulmuştur. Seçim ve ayar yalnızca halka ve zincir geliştirme kümesinde yapılmış, ayrılmış test kümesine bir kez uygulanmıştır.

| Varyant | Toplam P / R / F1 | zincir ayrılmış test F1 | Çift skoru | Temizde yanlış alarm (halka) |
|---|---|---|---|---|
| Ana (2 soru, eşik 0,5) | 0,75 / 0,36 / 0,49 | 0,31 | %37 | %7 |
| Atomik v1 (9 sınıf, eşik 0,5) | 0,52 / 0,67 / 0,58 | 0,53 | %31 | %36 |
| Atomik v1, eşik geliştirmede seçildi (0,6)\* | 0,64 / 0,59 / 0,61\* | 0,53 | %37\* | %23\* |
| Atomik v2 (15 sınıf: ayrılmış ve yeni sınıflar) | 0,40 / 0,77 / 0,52 | 0,36 | %10 | %69 |
| **Seçili sınıflar** (18 sınıf, eşik 0,6)\* | 0,71 / 0,61 / 0,66\* | **0,53** | %47\* | %18\* |

\* **Örneklem içi:** Bu satırlarda eşik ve sınıflar halka ile zincir geliştirme kümesine bakılarak seçilmiştir. Bu yüzden toplam sütunu, çift skoru ve yanlış alarm sütunları iyimserdir. Adil karşılaştırma yalnızca **zincir ayrılmış test** sütunudur.

1. **Soru tasarımı en büyük etkendir.** Hiçbir ayar yapılmayan v1, aynı model ve aynı girdiyle ana koşunun F1 değerini toplamda 0,49'dan 0,58'e, ayrılmış testte 0,31'den 0,53'e çıkarmıştır.
2. **Her sınıf eklemek iyi değildir.** v2'de "veya" ile birleşmiş sorular bölünmüş ve yeni sınıflar eklenmiştir. Ancak "girdi doğrulaması eksik mi?", "paylaşılan veride yarış var mı?" ve "sınır durumu yanlış mı?" gibi sorular temiz kodda da sık evet dediği için yanlış alarm %69'a çıkmıştır.
3. **Seçim yöntemi:**
   - Sınıflar geliştirme kümesindeki kesinliklerine göre sıralanmış ve bütün önekler denenmiştir.
   - Seçilen kural, temiz kodda sık alarm veren sınıfları dışarıda bırakır.
   - Ayrılmış testte v1 ile aynı sonucu verir (P 0,80, R 0,40). Seçim ayrılmış testte ek kazanç sağlamamış, ama zarar da vermemiştir.
   - Geliştirme kümesinde görülen yanlış alarm düşüşü (%36'dan %18'e) ve çift skoru artışı (%47) bu yüzden yeni veride doğrulanmamıştır.
   - **Karar (kullanıcı, 2026-11-11):** Seçili sınıflar yan sonuç olarak kabul edildi ve geliştirme bu turda durduruldu.
4. **Kullanım alanı:** Hızlı ve ucuz bir ön tarama ya da PR Kod Analizi'nin LLM aşamasından önce "bu dosyaya bak" sinyali olarak kullanılabilir.
5. **Uyarılar:**
   - Atomik sınıflar genel kusur sınıflarıdır, ancak halka aileleri de bu sınıflara benzer. Geliştirme kümesinde yapılan seçim oraya fazla uymuş olabilir.
   - Ayrılmış test kümesi yalnızca 10 pozitif dosyadır.
   - Seçili kuralın maliyeti burada üç koşunun toplamıdır (0,059 dolar, üst sınır). Yalnızca seçili sorular sorulsaydı yaklaşık atomik v1 kadar (0,02 dolar) olurdu.

### Satır bulma: hunk ve blok düzeyi
Kusurlu denen dosyanın diff'i parçalara bölünmüş ve her parça için "sorun bu parçada mı?" diye sorulmuştur. Parça olarak iki birim denenmiştir:
- **Hunk:** diff'in değişiklik bölümü, medyan 16 satır.
- **Blok:** bir hunk içindeki ardışık değişen satırlar, medyan 3–5 satır.

| Kaynak | Birim | Doğru dosya | İsabet | Birden çok birimli dosyada isabet | Rastgele seçim | "En büyük birim" kuralı |
|---|---|---|---|---|---|---|
| Ana | hunk | 27 | 27/27 | 11/11 (%100) | %48 | 9/11 (%82) |
| Atomik v1 | hunk | 50 | 50/50 | 21/21 (%100) | %48 | 16/21 (%76) |
| Ana | blok | 27 | 26/27 | 12/13 (%92) | %39 | 11/13 (%85) |
| Atomik v1 | blok | 50 | 47/50 | 23/25 (%92) | %41 | 20/25 (%80) |

1. **Jev doğru dosyada sorunlu parçayı büyük ölçüde bulur.** Blok düzeyinde %92 isabet, rastgele seçimin iki katından fazladır.
2. **Ancak basit bir kural da güçlüdür.** "En büyük parçayı seç" kuralı blok düzeyinde %80–85 isabet sağlar. Jev bu kuralı 25 dosyada yalnızca 3 dosya farkla geçmiştir. Yani kusurlar çoğunlukla değişikliğin en büyük parçasındadır ve Jev'in katkısı sınırlıdır.
3. **Konum inceliği:** Blok düzeyinde tahmin edilen bölge medyan 9–10 satırdır, hunk düzeyinde 18–19 satır. Blok daha dar bir konum verir, karşılığında isabet %100'den %92'ye iner.
4. **Uçtan uca etki:** Satır ancak dosya doğru bulunduysa aranır. Uçtan uca satır duyarlılığı ana kaynakla 0,35, atomik kaynakla 0,63'tür.
5. **Uyarı:** Örnek küçüktür (13 ve 25 dosya). Blok, PR Kod Analizi'nin verdiği tek satırlık konumla birebir aynı ölçü değildir.
