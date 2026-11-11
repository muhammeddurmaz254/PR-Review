# Ortak sonuç biçimi ve değerlendirici

Karşılaştırmadaki her araç (PR Kod Analizi, statik analiz, PR-Agent, Alibaba OCR, Jev) aynı değerlendiriciyle puanlanır (Ara Rapor 1, İP8). Bunun için her araç çıktısını aşağıdaki JSON biçimine dönüştürür ve `degerlendir.py` ile puanlatır.

## Biçim

```json
{
  "arac": "statik",
  "surum": "semgrep 1.x, kural seti p/python",
  "tarih": "2026-11-12",
  "birim_fiyat_usd_1m": null,
  "vakalar": [
    {
      "veri_kumesi": "halka",
      "vaka_kimligi": "authz-01-kusurlu",
      "dosyalar": [
        {"dosya": "halka/billing/views.py", "kusurlu_mu": true,
         "skor": null, "sure_ms": 850, "giris_token": null}
      ]
    }
  ]
}
```

| Alan | Zorunlu | Açıklama |
|---|---|---|
| `arac` | evet | Kısa ad. Çıktı dosyalarının adında kullanılır (`ozet_<arac>.md`) |
| `surum` | evet | Araç ve model sürümü, ayarlar |
| `tarih` | evet | Koşu tarihi (YYYY-AA-GG) |
| `birim_fiyat_usd_1m` | hayır | 1 milyon giriş token'ının dolar fiyatı. Yerel araçlarda `null` |
| `vakalar[].veri_kumesi` | evet | `halka`, `zincir_dev` ya da `zincir_holdout` |
| `vakalar[].vaka_kimligi` | evet | Etiket dosyasındaki `vaka_kimligi` (ör. `authz-01-kusurlu`) |
| `dosyalar[].dosya` | evet | Depo köküne göre yol, etiketteki yazımla aynı |
| `dosyalar[].kusurlu_mu` | evet | Araç bu dosyada en az bir kusur bildirdi mi |
| `dosyalar[].skor` | hayır | Varsa güven ya da olasılık (0–1) |
| `dosyalar[].sure_ms` | hayır | Bu dosya için harcanan süre. Araç PR başına süre veriyorsa, süreyi ilk dosyaya yazın, diğerlerini `null` bırakın |
| `dosyalar[].giris_token` | hayır | Modele giden token sayısı |
| `dosyalar[].hata` | hayır | Araç bu dosyada hata verdiyse kısa açıklama |
| `vakalar[].bulgular` | hayır | **Ayrılmış.** `[{dosya, satir, tur}]` biçiminde satır/tür düzeyi bulgular. Şu an puanlanmaz |

Sonuçta yer almayan bir değişen dosya **işaretlenmedi** sayılır. Yalnızca bulgu üreten araçlarda (statik analiz, OCR) sadece bulgu çıkan dosyaları `kusurlu_mu: true` ile yazmak yeterlidir.

## Puanlama kuralı

- **Birim:** (vaka, değişen dosya). Değişen dosya, vakanın diff'inde bulunan dosyadır. halka'da 158, zincir'de 124 dosya vardır. 3 temiz zincir vakasının etiketi diff'te olmayan bir dosya daha listeler. Bu dosyalar ölçüme katılmaz ve özet dosyasında listelenir.
- **Pozitif dosya:** bir kusurun `konum.dosya` değeri. halka'da 49, zincir'de 26 pozitif dosya vardır.
- **Nötr dosya:** yalnızca bir kusurun `esdeger_konumlar` listesinde geçen dosya. İşaretlenirse FP, işaretlenmezse FN sayılmaz.
- **TP / FP / FN:** pozitif dosya işaretlendiyse TP, işaretlenmediyse FN. Diğer dosyalar işaretlendiyse FP.
- P, R ve F1 şu kümeler için ayrı ayrı verilir: `halka`, `zincir_dev`, `zincir_holdout`, `toplam`.
- **Süre:** PR başına toplam süre, ortalama ve medyan olarak verilir. **Maliyet:** toplam giriş token × birim fiyat.

## Çalıştırma

Gereken dosyalar `JevDeney/` klasöründe aynı yerde durmalıdır: `degerlendir.py`, `veri.py`, `kirilim.py`. Yalnızca standart kütüphane kullanılır, Python 3.11 veya üstü gerekir. Özet dosyası P/R/F1 tablosuna ek olarak çift skorunu, temiz vakalardaki yanlış alarm oranını ve aile / dayanak / rol / PR boyutu kırılımını içerir.

```powershell
py JevDeney\degerlendir.py yol\statik_sonuc.json
```

Çıktılar `JevDeney\sonuclar\ozet_<arac>.md` (Markdown tablo, veri notları, uyarılar) ve `tablo_<arac>.tex` (rapora girecek LaTeX tablosu) dosyalarıdır. Veri kümesi başka bir yerdeyse `--repos C:\yol\Repos` kullanılır.

## Örnek dönüştürücü (bulgu listesinden ortak biçime)

```python
import json
from collections import defaultdict

# bulgular: [("halka", "authz-01-kusurlu", "halka/billing/views.py"), ...]
def ortak_bicime(bulgular, arac, surum, tarih):
    vakalar = defaultdict(set)
    for kume, vaka, dosya in bulgular:
        vakalar[(kume, vaka)].add(dosya)
    return {"arac": arac, "surum": surum, "tarih": tarih, "birim_fiyat_usd_1m": None,
            "vakalar": [{"veri_kumesi": k, "vaka_kimligi": v,
                         "dosyalar": [{"dosya": d, "kusurlu_mu": True} for d in sorted(ds)]}
                        for (k, v), ds in sorted(vakalar.items())]}
```
