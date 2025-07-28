# Vivollo Yanıtlanmamış Soru Tespiti

**Vivollo NLP**, kullanıcı‑bot sohbetlerinde “yanıtlanmamış” olarak kalan soruları tespit edip; duygu, kategori ve intent etiketleriyle zenginleştiren bir pipeline’dır.

## 📦 İçindekiler

- **data/**  
  - `yorumlar.json` – orijinal sohbet mesajları  
  - `full_messages.csv|.parquet` – tam etiketli + yeni JSON satırları  
- **models/**  
  - `cat_model/`, `sentiment_model/`, `intent_model/` – eğitilmiş model paketleri  
- **src/** – adım adım Python script’leri:  
  - `00_merge_json_excel.py` → JSON + Excel’i birleştir  
  - `01_prepare_dataset.py` → kategori veri seti hazırla  
  - `01_prepare_sentiment_intent.py` → duygu & intent veri seti hazırla  
  - `02_train_*.py` → modelleri eğit  
  - `03_predict_*.py` → tüm tahminleri al  
  - `04_format_output.py` → temiz Excel üret  
  - `05_split_safe.py` → train/val/test böl  
  - `06_diag_manual_tune.py` → hiperparametre tanı  
  - `07_recalc_yanitlandi.py` → yanitlandi_mi etiketini güncelle  
  - `08_simulation.py` → hataları simüle et (`outputs/simulation_result.csv`)  
  - `09_group_highlight_conversations.py` → preview Excel’de gruplandırma & renklendirme  
  - `10_recheck_yanitlandi.py` → son yanitlandi_mi kontrolleri  
  - **`plot_confusion_matrix.py`** (outside notebook) – doğrulama seti için karışıklık matrisi  
- **notebooks/**  
  - `01_eda.ipynb` – dağılımlar, trend, simülasyon görselleri  
- **outputs/** – tüm CSV/Excel çıktı dosyaları ve PNG grafikleri  
- **reports/** – (önerilen) summary PDF veya PPT slaytları  

## 🚀 Kurulum & Çalıştırma

1. Proje kök dizininde sanal ortamı oluşturun ve etkinleştirin:  
   ```bash
    python -m venv venv
    venv\Scripts\activate     # Windows
    source venv/bin/activate  # macOS/Linux
    pip install -r requirements.txt
    python src/00_merge_json_excel.py
    python src/01_prepare_dataset.py
    python src/01_prepare_sentiment_intent.py
    python src/02_train_category.py
    python src/02_train_sentiment.py
    python src/02_train_intent.py
    python src/03_predict_unlabeled.py
    python src/03_predict_all.py
    python src/04_format_output.py
    python src/05_split_safe.py
    python src/06_diag_manual_tune.py
    python src/07_recalc_yanitlandi.py
    python src/08_simulation.py
    python src/09_group_highlight_conversations.py
    python src/10_recheck_yanitlandi.py


##  📊 Analiz Özeti
Kategori/Sentiment/Intent dağılımları: notebooks/01_eda.ipynb

Aylık yanıtlanma oranı trendi: %41 aralığında dalgalanıyor.

Hata simülasyonu:

Senaryo	Hata Oranı (%)
Original slot	80.6
Thresh = 2	39.2
Model fallback	41.4
API retry	0.4

Yorum: Slot‑filling eşiğini 2’ye düşürmek ve API retry mekanizması en büyük iyileşmeyi sağlıyor.

Karışıklık matrisi script ile outputs/confusion_matrix_validation.png olarak kaydedildi.

##  ⚙️ Limitations & Next Steps
Veri kalitesi: Yeni kategoriler eklenebilir, neg / pos dengesizliği optimize edilebilir.

Model geliştirme: BERT fine‑tuning veya ensembel yöntemleri denenebilir.

Demo: Senior sunum slaytları reports/summary.pdf içinde yer alacak.

