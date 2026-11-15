# CLI iş akışları

CLI komutları ve makine tarafından okunan dosya şemaları İngilizce adlarıyla korunur. TR/EN seçimi yalnızca gösterge panelinin sunum katmanını değiştirir. Bu UI değişikliği kapsamında gerçek veriyle eğitim veya yeniden değerlendirme yapılmaz.

## Kurgusal demo

```bash
gridguard demo --customers 120 --days 90 --output data/local-demo.csv --output-dir artifacts/local-demo
```

Bu komut bağımsız kurgusal okumalar üretir, eğitim yapar ve kurgusal test sonuçlarını kaydeder. Sentetik sonuçlar gerçek dünyada başarım kanıtı değildir. Paneli göstermek için önceden üretilmiş dosyalar yeterlidir.

## Kayıtlı deneylerle araştırma protokolü

Bu komutlar yalnızca yeni, bilinçli olarak tasarlanmış bir araştırma deneyi gerektiğinde kullanılır. Tarihsel test grubuna tekrar tekrar bakarak model seçmeyin; ayrıntılar [değerlendirme protokolünde](real-data-evaluation.md).

```bash
gridguard experiment --input data/consumption.csv --source custom --sample-size 10000 --output-dir artifacts/new-validation
gridguard finalize --input data/consumption.csv --experiment-dir artifacts/new-validation --output-dir artifacts/new-final
```

`experiment` yalnızca doğrulama sonuçlarını kullanır; `finalize` dondurulmuş kazananı yeniden eğitmeden değerlendirir. `train --validation-only`, tek bir özellik yapılandırmasının geliştirme akışıdır. Örneklem boyutunun değişmesi veri grubunu değiştirir; farklı grupları doğrudan karşılaştırmayın.

## İsteğe bağlı geliştirme analizleri

```bash
gridguard calibrate --input data/consumption.csv --reference-dir artifacts/new-final --output-dir artifacts/new-calibration
gridguard benchmark-anomaly --input data/consumption.csv --reference-dir artifacts/new-final --output-dir artifacts/new-anomaly
gridguard explain --input data/consumption.csv --reference-dir artifacts/new-final --output-dir artifacts/new-explainability
```

Bu akışlar kayıtlı referansın müşteri bölümlerini doğrular ve nihai test grubunu dışlar. Geliştirme deneyleri tarihsel nihai modelin bağımsız performans kanıtı değildir. Panel sadece üretilmiş raporları okur; bunları arayüzden çalıştırmaz.

## Tahmin ve isteğe bağlı SQL dışa aktarımı

```bash
gridguard score --input data/consumption.csv --model artifacts/new-final/model.joblib --output reports/risk_ranking.csv
gridguard export-sqlite --input data/local-demo.csv --output data/local-demo.sqlite --limit-customers 200
```

`score`, sağlanan tüm müşterileri sıralar; eğitim grubundaki tahminler bağımsız test sonucu değildir. SQLite yalnızca sınırlandırılmış keşif örneğidir; panel veritabanı gerektirmez.

## Bağımsız sentetik operasyonel bağlam

```bash
gridguard generate-context --input data/consumption.csv --reference-metrics artifacts/pipeline-final-20261111/metrics.json --output-dir artifacts/new-context --seed 42 --regions 20
gridguard context-gate --predictions artifacts/pipeline-final-20261111/test_predictions.csv --context-dir artifacts/new-context --evaluation-date 2016-11-07 --window-start 2016-10-01 --window-end 2016-10-31 --output-dir reports/new-context
```

Olaylar kaynak SGCC kimlikleri ve tarih başlıklarından bağımsız operasyonel varsayımlarla üretilir. Etiket, tahmin veya anomali sırası olay yerleştirmeyi etkilemez. Karar kapısı özgün puanları ve kayıt sırasını korur. Şemalar, yapılandırma, dönem ve kanıt kuralları [bağlam belgesindedir](operational-context.md).

Kaynak verileri, etiketleri, eğitim/test bölümlerini ve tarihsel dosyaları değiştirmeyin. Yeni/boş çıktı dizinleri kullanın; müşteri düzeyindeki çıktıları paylaşmayın. Tüm seçenekler için `gridguard --help` ve `gridguard <komut> --help` kullanılabilir.
