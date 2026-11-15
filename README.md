# GridGuard AI

**Elektrik tüketimini analiz eden, müşterileri inceleme önceliğine göre sıralayan ve operasyonel bağlamı görünür kılan karar destek prototipi.**

GridGuard AI, bir elektrik dağıtım şirketinin “Hangi müşteriyi önce incelemeliyiz, karar öncesinde hangi kayıtları doğrulamalıyız?” sorularını araştırır. Python, scikit-learn ve Streamlit kullanır; kayıtlı sonuçları salt okunur bir gösterge panelinde sunar.

> Risk puanı **doğrulanmış kaçak kullanım kanıtı değildir**. Bu bir portföy ve araştırma projesidir; otomatik yaptırım veya üretim sistemi olarak tasarlanmamıştır.

## Temel yetenekler

| Bölüm | Yanıtladığı soru |
| --- | --- |
| **Genel Bakış** | Genel risk durumu nedir? |
| **Risk Analizi** | Önce kimler incelenmeli? Liste, filtreler ve müşteri tüketim geçmişi. |
| **Operasyonel İnceleme** | Kesinti, sayaç ve abonelik bağlamında ne doğrulanmalı? Kapasite nasıl planlanmalı? |
| **Model Performansı** | Geçmiş değerlendirme, kalibrasyon, anomali karşılaştırması ve permütasyon önemi ne söylüyor? |
| **Veri Kalitesi** | Okumalar ne kadar tam ve güvenilir? |

Arayüz yeni oturumda Türkçe açılır; **TR / EN** seçimi oturum boyunca korunur. Ayrıntılı filtreler ve teknik açıklamalar gerektiğinde açılır. On analitik görünümün tamamı bu beş bölüm altında erişilebilir.

## Mimari ve teknolojiler

```text
SGCC CSV → doğrulama → tüketim özellikleri → kayıtlı model / tahmin / değerlendirme
                                                    ↓
Bağımsız SENTETİK olaylar → deterministik karar kapısı → salt okunur Streamlit paneli
```

Python 3.11+ · pandas · NumPy · scikit-learn · Streamlit · Plotly · pytest · Ruff · GitHub Actions. İsteğe bağlı SQLite dışa aktarımı ve SQL örnekleri korunur. Panel model yüklemez, eğitmez veya yeniden tahmin üretmez; veritabanı, API veya bulut servisi gerektirmez.

## Veri ve önemli sınırlar

- Yerel SGCC dosyası: **42.372 müşteri**, **1.034 okuma sütunu**, 2014-01-01–2016-10-31. Takvimde bir günün sütunu eksiktir. Kaynak şeması `CONS_NO`, `FLAG` ve günlük tarih sütunlarıdır; `customer_id` eşdeğeri de desteklenir.
- `FLAG`, geçmiş müşteri etiketidir; doğrulanmış olay tarihi veya gelecekteki inceleme sonucu değildir. Eksik okumalar sıfır tüketim sayılmaz.
- Risk puanı, ham model puanının 100 katının yuvarlanmasıdır; **kalibre edilmiş olasılık değildir**. Risk düzeyleri örnektir, şirket politikası değildir.
- Operasyonel olaylar **SENTETİK / SYNTHETIC** veridir. Gerçek SGCC kimlikleri ve tarih aralığıyla uyumludur; etiketler, model tahminleri veya anomaliler kullanılarak üretilmez. Sentetik bölgeler gerçek coğrafya değildir. Olay eşleşmemesi, olay yaşanmadığını kanıtlamaz.
- Gerçek müşteri verileri, tahminler, modeller ve müşteri düzeyindeki sentetik çıktılar depoya eklenmez. Yerel paneli localhost üzerinde kullanın; kimlikler gösterildiğinde ekran görüntüsü veya dışa aktarım paylaşmayın.

## Kayıtlı model değerlendirmesi

`pipeline-final-20261111`: 10.000 müşterilik sabit örneklem; eğitim/doğrulama/test **6.000/2.000/2.000**. `quality` özellikleri kullanan Random Forest, **doğrulama AP'sine** göre seçilmiş ve yalnızca eğitim verisine uydurulmuştur. Aşağıdaki değerler kayıtlı 2.000 müşterilik **geçmiş nihai test** sonucudur:

| Ölçüm | Kayıtlı değer |
| --- | ---: |
| Average Precision (AP) | 0.32281366523385974 |
| ROC-AUC | 0.7883066514472805 |
| Precision@50 | 0.50 |
| Precision@100 | 0.43 |
| Precision@200 | 0.345 |
| Pozitif etiket oranı | 0.0855 |

0,5 eşiğindeki sınıflandırma matrisi: `[[1779, 50], [130, 41]]`. AP, trapez PR-AUC ile aynı ölçüm değildir. Test grubu daha önce incelenmiştir; bu sonuçlar ileriye dönük, coğrafi veya gerçek operasyonel başarıyı kanıtlamaz. Sentetik bağlamdan ML başarım artışı çıkarılmaz.

Tam ölçümler: [kayıtlı nihai sonuç](docs/experiments/pipeline-final.json), [yöntem ve değerlendirme](docs/pipeline-evaluation.md). Bağımsız geliştirme deneylerinin kalibrasyon, anomali ve açıklanabilirlik sonuçları bu nihai testle birleştirilmez.

## Kurulum

```bash
git clone https://github.com/GorkemErtas/gridguard-ai.git
cd gridguard-ai
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[app,dev]"
```

## Gösterge panelini çalıştırma

Mevcut yerel deney dosyalarıyla PowerShell üzerinden:

```powershell
$env:GRIDGUARD_ARTIFACT_DIR = 'artifacts/pipeline-final-20261111'
streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

`http://localhost:8501` adresini açın. **Deney kaynağı** panelinden başka bir kayıtlı deney seçebilirsiniz. Dosyalar bulunmuyorsa uygulama açıklayıcı bir boş durum gösterir. Üçüncü taraf SGCC verisi depoyla dağıtılmaz; kullanım ve yeniden dağıtım haklarını ayrıca doğrulayın.

Operasyonel bağlamın dizinleri, deney seçiminden **ayrıdır**:

```powershell
$env:GRIDGUARD_CONTEXT_DIR = 'artifacts/context-sgcc-phase1'
$env:GRIDGUARD_CONTEXT_DECISIONS = 'reports/context-sgcc-phase1'
$env:GRIDGUARD_CONTEXT_PREDICTIONS = 'artifacts/pipeline-final-20261111/test_predictions.csv'
```

Kurulum, bölüm eşlemesi ve önce/sonra ekran görüntüleri: [TR/EN panel rehberi](docs/dashboard-tr-en.md). Dosya güvenliği ve kaynak geçmişi: [operasyonel bağlam rehberi](docs/operational-context-dashboard.md).

## CLI ve yeniden üretilebilirlik

```bash
# İsteğe bağlı: yalnızca kurgusal okumalarla bağımsız demo deneyi oluşturur.
gridguard demo --customers 120 --days 90 --output data/local-demo.csv --output-dir artifacts/local-demo

# Mevcut SGCC kaynağına uyumlu sentetik bağlam üretir; modeli yeniden eğitmez.
gridguard generate-context --input data/consumption.csv --reference-metrics artifacts/pipeline-final-20261111/metrics.json --output-dir artifacts/new-context --seed 42 --regions 20
gridguard context-gate --predictions artifacts/pipeline-final-20261111/test_predictions.csv --context-dir artifacts/new-context --evaluation-date 2016-11-07 --window-start 2016-10-01 --window-end 2016-10-31 --output-dir reports/new-context
```

Yeni/boş çıktı dizinleri kullanın; geçmiş dosyaları değiştirmeyin. `train`, `experiment`, `finalize`, `score`, `calibrate`, `benchmark-anomaly`, `explain` ve `export-sqlite` komutları korunur. Ayrıntılı komutlar ve protokol: [CLI iş akışları](docs/cli-workflows.md), [gerçek veri değerlendirmesi](docs/real-data-evaluation.md), [sentetik bağlam şemaları](docs/operational-context.md).

## Test ve doğrulama

```bash
pytest -q
ruff check .
ruff format --check .
```

CI; güncel bağımlılıkları ve minimum uyumluluk yapılandırmasını çalıştırır. Testler veri/sıralama bütünlüğünü, salt okunur görünümleri, TR/EN geçişini, karar kapısını ve eksik/kısmi dosya durumlarını kapsar. Küçük kurgusal test fikstürleri gerçek modellerin veya tarihsel çıktıların yerine yazılmaz.

## Proje yapısı ve belgeler

```text
app/streamlit_app.py     TR/EN gösterge paneli
src/gridguard/          veri, özellikler, ML, değerlendirme ve bağlam modülleri
src/gridguard/i18n.py   sunum çevirileri ve görüntüleme yardımcıları
tests/                 birim, bütünleşim ve Streamlit AppTest testleri
docs/                  yöntem, kullanım ve sınırlamalar
data/ artifacts/ reports/   yerel; Git tarafından yok sayılır
```

- [Mimari](docs/architecture.md) · [Model kartı](docs/model-card.md) · [Panel](docs/dashboard-tr-en.md)
- [Kapasite planlama](docs/inspection-capacity-planning.md) · [Kalibrasyon](docs/probability-calibration.md)
- [Anomali karşılaştırması](docs/anomaly-benchmark.md) · [Model açıklanabilirliği](docs/model-explainability.md)
- [Operasyonel bağlam](docs/operational-context.md) · [Kaynak veri atfı ve sınırlar](docs/pipeline-evaluation.md#research-data-attribution)

İngilizce teknik belgeler ayrıntılı başvuru olarak korunur. Yazılım için henüz lisans seçilmemiştir; kaynağın herkese açık olması yeniden kullanım hakkı vermez. Üçüncü taraf verilerin ayrı koşulları olabilir.
