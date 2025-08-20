# Makine Öğrenmesi Kursu

YouTube'da yayınlanan Makine Öğrenmesi Kursu'nun materyallerini içermektedir. 

| Modül | Açıklama                                | Colab Linki |
|-------|-----------------------------------------|-------------|
| 01    | Giriş ve Proje Yaşam Döngüsü             | <a target="_blank" href="https://colab.research.google.com/github/Frightera/ML-Kursu-Youtube/blob/main/Notebooklar/ML_Modul01_Giris_ve_Proje_Yasam_Dongusu.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 02    | Python ML Ekosistemi                     | <a target="_blank" href="https://colab.research.google.com/github/Frightera/ML-Kursu-Youtube/blob/main/Notebooklar/ML_Modul02_Python_ML_Ekosistemi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 03    | Ön İşleme ve Özellik Mühendisliği        | <a target="_blank" href="https://colab.research.google.com/github/Frightera/ML-Kursu-Youtube/blob/main/Notebooklar/ML_Modul03_OnIsleme_ve_Ozellik_Muhendisligi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 04    | Ön İşleme ve Feature Engineering (Sigorta Verisi) | <a target="_blank" href="https://colab.research.google.com/github/Frightera/ML-Kursu-Youtube/blob/main/Notebooklar/ML_Modul04_OnIsleme_ve_FE_SigortaVerisi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 05    | Sigorta Verisi Üzerinde Model Fit        | <a target="_blank" href="https://colab.research.google.com/github/Frightera/ML-Kursu-Youtube/blob/main/Notebooklar/ML_Modul05_SigortaVerisi_ModelFit.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## Kurs İçeriği

5 ana modül mevcuttur (küçük bilgisel materyalleri hariç):

- **Modül 1: Makine Öğrenmesine Giriş ve Proje Yaşam Döngüsü**
  - Neden Makine Öğrenmesi?
  - Uçtan Uca ML Yaşam Döngüsü (CRISP-DM + MLOps)
  - Başarı Metrikleri ve Kabul Kriterleri
  - Veri Hazırlık Seviyeleri ve Veri Sözleşmeleri
  - Doğrulama Stratejileri
  - Baseline (Temel) Modeller
  - Deney Kaydı ve Sürümleme

- **Modül 2: Python ML Ekosistemi**
  - NumPy, Pandas, Matplotlib ekosistemine pratik giriş:
    - NumPy
      - ndarray/dtype/shape/axis ve eksen mantığı
      - Broadcasting (matris + vektör işlemleri), vektörizasyon vs. Python döngüleri performans karşılaştırması
      - İndisleme: slice, boolean mask, fancy indexing
      - Şekil dönüşümü ve view vs. copy (reshape/ravel, np.shares_memory)
      - Rastgele sayı üretimi (np.random.default_rng) ve doğrusal cebir (np.linalg.solve)
    - Pandas
      - Series ve DataFrame temelleri; .loc vs .iloc ile seçim
      - Atama ve tip yönetimi; kategorik tip (astype('category')), pd.cut ile binleme
      - groupby/agg ile özetleme, pivot_table ile yeniden şekillendirme
      - merge/join ile tabloları birleştirme
      - Eksik değerleri yönetme (fillna, interpolate); bellek/tip optimizasyonu
      - Zaman serileri: resample ve rolling; temel çizimler
      - Büyük veriler: chunksize ile parçalı okuma (chunked I/O)
    - Matplotlib
      - Temel iş akışı: figure → plot/scatter → title/xlabel/ylabel → tight_layout → show
      - Temel grafik: çizgi grafiği (line plot) ve benzerleri

- **Modül 3: Ön İşleme ve Özellik Mühendisliği**
  - Model öncesi veri hazırlığına kapsamlı yaklaşım:
    - Sentetik karma veri üretimi (sayısal + kategorik + tarih + kısa metin) ve hedef (`churn`) tasarımı; kontrollü eksik değer serpiştirme
    - Keşifsel analiz (EDA)
      - Dağılımlar: `usage_count`, `session_minutes`, `spend` histogramları
      - Eksik değer analizi: sütun bazında eksik sayıları grafiği
    - Eksik değer imputasyonu
      - Sayısal: SimpleImputer (median/mean), KNNImputer; strateji seçimi ve çarpıklık etkisi
      - Kategorik: most_frequent ve constant (ör. "Bilinmiyor")
      - Boyutsallık laneti uyarıları (KNN için)
    - Aykırı değerler ve ölçekleme/dönüşüm
      - IQR ve Z-skoru ile tespit; boxplot ile görselleştirme
      - StandardScaler, RobustScaler, MinMaxScaler; log1p ile çarpıklığı giderme
    - Kategorik kodlama (encoding)
      - One-Hot (handle_unknown=ignore) ve yüksek kardinalite uyarıları
      - Ordinal encoding kullanımı ve yanlış kullanım riskleri; pipeline içinde `get_dummies` kullanmama notu
    - Özellik türetme (feature engineering)
      - Tarih: year/month/dow/is_weekend, döngüsel sin/cos temsil (ay/gün)
      - Metin: karakter/kelime uzunluğu, negatif/pozitif anahtar kelime sinyalleri
      - Özel sklearn dönüştürücüleri: DateFeatureExtractor, TextLengthExtractor
    - Uçtan uca ML boru hattı (pipeline)
      - ColumnTransformer ile sayısal/kategorik/tarih/metin akışları
      - PolynomialFeatures + LogisticRegression; StratifiedKFold ile F1 çapraz doğrulama
    - Artefakt yönetimi
      - Tüm pipeline'ı `joblib` ile kaydetme/yükleme

- **Modül 4: Sigorta Verisi ile Ön İşleme ve Özellik Mühendisliği**
  - Gerçek sigorta talepleri verisiyle model öncesi derin EDA ve özellik mühendisliği (No-Model):
    - Veri yükleme ve şema özeti
      - `Insurance claims data.csv` yükleme; `claim_status` hedefi
      - Sütun tipleri, missing%, n_unique; dtale ile ön-izleme
    - String→sayısal özellik türetme (Regex)
      - `max_power`, `max_torque` metinlerinden `power_bhp`, `power_rpm`, `torque_nm`, `torque_rpm`
      - Etkileşim: `bhp_per_nm`
    - Tek değişkenli dağılımlar
      - Sayısallar: histogram/kde (örn. `subscription_length`, `vehicle_age`, `customer_age`, `region_density`)
      - Kategorikler: countplot (örn. `region_code`, `segment`, `model`, `fuel_type`)
    - Hedef dağılımı ve dengesizlik
      - `claim_status` sınıf oranı (~%6.4 pozitif); doğruluk paradoksu
      - Doğru metrikler: Precision/Recall/F1/AUC; `class_weight` ve loss notları
    - Kategorik × hedef
      - Claim oranı bar grafikleri; Ki-kare testi (chi2) ve p-değeri yorumu
    - Sayısal × hedef
      - Violin plotlar; Mann–Whitney U testi ve yorumu
    - Korelasyon ve çoklu doğrusal bağlantı
      - Sayısal korelasyon ısı haritası; VIF hesaplama, yorum ve çözüm stratejileri (eleme/birleştirme/PCA/Ridge)
    - Aykırı değer analizi
      - IQR ve Z-score ile tespit; log1p, winsorizing, bırak/sil stratejileri


- **Modül 5: Sigorta Verisi ile Modelleme**
  - Hazırlanan veri seti ile uçtan uca modelleme, eşik (threshold) optimizasyonu ve HPO (Optuna):
    - Kurulum notu: `catboost`, `optuna` (opsiyonel hızlandırma: `scikit-learn-intelex`).
    - Veri ve hedef:
      - `Insurance claims data.csv` yüklenir; `policy_id` (kimlik) atılır.
      - Hedef: `claim_status` ikili sınıfa dönüştürülür (0/1).
    - Özel özellik mühendisliği (Regex Transformer):
      - `PowerTorqueRegexExtractor` ile `max_power`/`max_torque` metinlerinden sayısal alanlar türetilir: `power_bhp`, `power_rpm`, `torque_nm`, `torque_rpm`, `bhp_per_nm`.
      - Sklearn Pipeline içinde ilk adım olarak kullanılır (leakage’siz tekrar üretilebilirlik).
    - Eğitim/Test ayrımı: `train_test_split` (stratified, SEED=42).
    - Sütun tipleri ve dinamik seçim:
      - `dataclass ColumnInfo` ile her sütun için `dtype` ve `n_unique` özetlenir.
      - Gruplar: `numeric_cols`, `binary_cols`, `ohe_cols`, `ordinal_cols = ["ncap_rating", "airbags", "cylinder"]`.
      - Hedef sütun gruplardan çıkarılır; özel FE sütunları (`max_power`, `max_torque`) ayrı tutulur.
    - Ön işleme hattı (ColumnTransformer):
      - Sayısal: `SimpleImputer(median)` + `StandardScaler`.
      - İkili/Ordinal: `SimpleImputer(most_frequent)` + `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`.
      - Kategorik: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=10, sparse_output=False)`.
      - Remainder=`passthrough`; pipeline: `PowerTorqueRegexExtractor` → `preprocessor`.
    - Temel modeller ve eşik optimizasyonu:
      - Adaylar: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`.
      - `TunedThresholdClassifierCV(scoring='f1')` ile her fold için en iyi karar eşiğini bul.
      - `StratifiedKFold(n_splits=5)` ile `cross_validate`: `f1`, `recall`, `roc_auc`, `threshold` metriklerinin dağılımları ve kutu grafikleri.
    - Test değerlendirmesi:
      - Seçilen temel model için `classification_report`, `ROC-AUC`/`AP` ve `ConfusionMatrixDisplay` ile raporlama.
    - Gradient Boosting + HPO (CatBoost + Optuna):
      - Sabitler: `loss_function='Logloss'`, `auto_class_weights='SqrtBalanced'` (sınıf dengesizliği için), mümkünse `task_type='GPU'`.
      - `cat_features` otomatik çıkarılır (metin FE sütunları hariç); `Pool` + `cv` ile AUC odaklı iç CV ve `early_stopping`.
      - En iyi parametrelerle `CatBoost` modeli, `TunedThresholdClassifierCV` ile sarılır; eğitim seti tamamında en iyi eşik öğrenilir; testte `F1(macro)` ve `ROC-AUC` raporlanır.
    - Eşik (threshold) stratejisi ve iş senaryoları:
      - Eşik taraması (0.05–0.35) ile yüksek Recall, dengeli, yüksek Precision senaryoları; iş hedefine göre seçim rehberi.
    - Karşılaştırma ve karar:
      - Optimize Decision Tree vs. CatBoost karşılaştırması; CatBoost’un daha esnek ve stabil karar eğrileri nedeniyle tercih gerekçeleri.
    - Artefaktlar:
      - Tam pipeline’ı `joblib` ile kaydetme (`best_model_pipeline.pkl`) ve örnek tahmin demosu.
  