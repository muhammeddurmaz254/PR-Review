# Jev yan deneyleri — tasarım (2026-11-10)

Kullanıcı kararı: ana koşu (`kos.py`, araç adı `jev`) değişmez. Aşağıdaki deneyler ayrı araç adlarıyla ve ayrı sonuç dosyalarıyla koşulur, `deneyler/sonuclar/` klasörüne yazılır. Beğenilen deney sonra ana yola eklenir.

## Deney-1: güçlü ve adil taban çizgisi
| Varyant | Araç adı | İstek birimi | State | Sorular | Dosya kararı |
|---|---|---|---|---|---|
| A. PR bağlamı | `jev_pr_baglam` | PR başına 1 istek | `changed_files.f<i>.{path,diff}`: PR'daki bütün dosyalar | Her dosya için `f<i>_defect` ve `f<i>_vulnerability` (ana sorularla aynı ölçüt) | o dosyanın iki olasılığının maks. değeri ≥ 0.5 |
| B. Atomik sorular | `jev_atomik` | Dosya başına 1 istek | `{file_path, diff}` (ana koşuyla aynı) | 9 somut ve gözlenebilir Noul (`varyantlar.ATOMIK_SORULAR`) | 9 olasılığın maks. değeri ≥ 0.5 |

**Adillik kuralları:**
- Soru metinleri sonuçlar görülmeden bu tarihte sabitlendi.
- Atomik kategoriler kod incelemesinde yaygın kusur sınıflarıdır: mantık, hata yönetimi, eksik kontrol, güvenilmeyen girdi, gizli veri, test, yapılandırma, eşzamanlılık, kaynak. Benchmark tür adları ve örnekleri kullanılmaz.
- **Sızıntı riski:** halka aileleri de bu genel sınıflara benzediği için risk sıfır değil. Raporda bu yazılır.
- Her varyant bir kez koşulur ve tüm varyantlar raporlanır.

## Deney-3: kırılım ve satır bulma
- **Kırılım (`kirilim.py`, API çağrısı yok):** herhangi bir sonuç dosyası için pozitif dosyalarda R değerleri şu eksenlerde ayrılır:
  - kusur ailesi
  - dayanak (a/b/c)
  - dosya rolü (kod/test/yapılandırma)
  - PR boyutu (tek dosyalı / çok dosyalı)

  Ayrıca temiz vakalarda yanlış alarm oranı ve halka çift skoru hesaplanır. Çift skoru: kusurlu varyantta pozitif dosya işaretlenmiş olmalı ve temiz ikizde hiçbir dosya işaretlenmemiş olmalı.
- **Satır bulma (`satir.py`, araç adı `jev_satir`):**
  - Kaynak koşuda kusurlu denen her dosyanın diff'i hunk'lara bölünür.
  - Birden çok hunk varsa tek istekte, her hunk için bir Noul sorulur: `h<i>`.
  - En yüksek olasılıklı hunk tahmindir. Tek hunk varsa çağrı yapılmaz.
  - Tahmin edilen aralık, hunk'ın yeni taraftaki satır aralığıdır.
  - **İsabet:** bu aralık o dosyadaki bir kusurun `konum` ya da `esdeger_konumlar` aralığıyla çakışırsa.
  - **Ölçüler:**
    - doğru dosyada isabet oranı, hem tüm dosyalarda hem çok hunk'lı dosyalarda
    - rastgele hunk seçiminin beklenen isabeti (taban)
    - uçtan uca satır duyarlılığı: isabet / pozitif dosya

## Karşılaştırma
`karsilastir.py`, ana ve deney sonuç dosyalarından `deneyler/sonuclar/karsilastirma.md` dosyasını üretir. İçinde şunlar yer alır:
- P/R/F1, süre ve maliyet
- çift skoru
- temiz vakalarda yanlış alarm
- kırılım tabloları
- satır bulma sonucu

Bütçe tavanı 0,10 dolar. Tahmini toplam 0,05 doların altında.
