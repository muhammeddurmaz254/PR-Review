# jev — dosya düzeyi sonuçlar

Kaynak: `jev_gercek_hepsi_2026-11-10.json` · sürüm: jev-1.14.0 · tarih: 2026-11-10

| Küme | Vaka | Dosya | TP | FP | FN | P | R | F1 | Hata | Ort. süre/PR (ms) | Medyan (ms) | Maliyet (USD) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| halka | 110 | 158 | 22 | 5 | 27 | 0.815 | 0.449 | 0.579 | 0 | 547 | 334 | 0.0056 |
| zincir_dev | 26 | 73 | 3 | 3 | 13 | 0.500 | 0.188 | 0.273 | 0 | 932 | 829 | 0.0025 |
| zincir_holdout | 18 | 51 | 2 | 1 | 8 | 0.667 | 0.200 | 0.308 | 0 | 1072 | 638 | 0.0018 |
| toplam | 154 | 282 | 27 | 9 | 48 | 0.750 | 0.360 | 0.486 | 0 | 673 | 405 | 0.0098 |

## Çift skoru (halka)

Kusurlu varyantta pozitif dosya işaretli ve temiz ikizde hiç işaret yok: 18/49 (37%)

## Temiz vakalarda yanlış alarm

| Küme | En az bir dosyası işaretlenen temiz vaka |
|---|---|
| halka | 4/61 (7%) |
| zincir_dev | 2/18 (11%) |
| zincir_holdout | 1/12 (8%) |

## Kırılım: aile (pozitif dosyalarda R)

| Grup | TP / pozitif |
|---|---|
| halka:authz | 3/3 (100%) |
| halka:concurrency | 1/3 (33%) |
| halka:config | 1/3 (33%) |
| halka:correctness | 1/3 (33%) |
| halka:crypto | 1/3 (33%) |
| halka:data_layer | 0/3 (0%) |
| halka:deserialization | 2/3 (67%) |
| halka:disclosure | 2/3 (67%) |
| halka:edge_cases | 0/3 (0%) |
| halka:error_handling | 1/3 (33%) |
| halka:injection | 3/3 (100%) |
| halka:maintainability | 1/4 (25%) |
| halka:resource_perf | 0/3 (0%) |
| halka:secrets | 3/3 (100%) |
| halka:ssrf | 3/3 (100%) |
| halka:test_quality | 0/3 (0%) |
| zincir:concurrency | 1/3 (33%) |
| zincir:config | 2/10 (20%) |
| zincir:error_handling | 0/3 (0%) |
| zincir:test_quality | 2/10 (20%) |

## Kırılım: dayanak (pozitif dosyalarda R)

| Grup | TP / pozitif |
|---|---|
| a | 2/15 (13%) |
| b | 8/16 (50%) |
| c | 17/44 (39%) |

## Kırılım: rol (pozitif dosyalarda R)

| Grup | TP / pozitif |
|---|---|
| kod | 21/50 (42%) |
| test | 2/13 (15%) |
| yapilandirma | 4/12 (33%) |

## Kırılım: pr_boyutu (pozitif dosyalarda R)

| Grup | TP / pozitif |
|---|---|
| cok_dosya | 13/40 (32%) |
| tek_dosya | 14/35 (40%) |

## Kırılım: kusur_sayisi (pozitif dosyalarda R)

| Grup | TP / pozitif |
|---|---|
| cok_kusur | 3/18 (17%) |
| tek_kusur | 24/57 (42%) |

## Veri notları
- ckpt-01-temiz: etikette olup diff'te olmayan tests/test_worker.py ölçüme katılmadı
- log-01-temiz: etikette olup diff'te olmayan tests/conftest.py ölçüme katılmadı
- timeout-01-temiz: etikette olup diff'te olmayan zincir/defaults.py ölçüme katılmadı
