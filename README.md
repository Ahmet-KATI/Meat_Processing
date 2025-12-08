# 🥩 Et Bozulma Tespit Sistemi

Makine öğrenmesi ve görüntü işleme kullanarak et tazeligini tespit eden kapsamlı bir sistem. MobileNetV2 tabanlı CNN modeli ile etin bozulma seviyesini 0-1 arası skor olarak tahmin eder ve kullanıcıya görsel bir bar (yeşilden kırmızıya) ile gösterir.

## 🎯 Özellikler

- ✅ **Regresyon tabanlı model**: 0.0 (taze) ile 1.0 (bozuk) arası sürekli skor
- ✅ **Transfer learning**: MobileNetV2 ile hafif ve etkili model
- ✅ **Masaüstü UI**: Tkinter tabanlı kullanıcı dostu arayüz
- ✅ **Raspberry Pi desteği**: TensorFlow Lite ile optimize edilmiş
- ✅ **Kamera entegrasyonu**: picamera2 ve OpenCV desteği
- ✅ **Görsel feedback**: Yeşil-sarı-kırmızı gradient bar
- ✅ **Veri augmentation**: Eğitim performansını artıran veri çoğaltma

## 📁 Proje Yapısı

```
meat_virtual_image_processing/
│
├── data/                          # Veri setleri
│   ├── raw/                       
│   │   ├── images/                # Et görselleri
│   │   └── labels.csv             # Görsel-skor eşleştirmeleri
│   └── processed/                 
│
├── src/                           # Kaynak kodlar
│   ├── data_utils.py              # Veri yükleme ve ön işleme
│   ├── model.py                   # Model mimarisi
│   ├── train.py                   # Eğitim scripti
│   ├── predict.py                 # Tahmin fonksiyonları
│   └── visualization.py           # Veri görselleştirme
│
├── ui/                            # Kullanıcı arayüzü
│   ├── components.py              # UI bileşenleri (bar widget)
│   └── desktop_app.py             # Tkinter uygulaması
│
├── raspi/                         # Raspberry Pi kodları
│   ├── camera_capture.py          # Kamera entegrasyonu
│   ├── raspi_app.py               # Ana uygulama
│   └── setup_instructions.md      # Kurulum talimatları
│
├── models/                        # Kaydedilmiş modeller
│   ├── model.h5                   # Keras formatı
│   └── model.tflite               # TFLite (Raspberry Pi)
│
├── outputs/                       # Çıktılar
│   ├── plots/                     # Grafikler
│   └── reports/                   # Raporlar
│
├── requirements.txt               # Python bağımlılıkları
├── requirements_raspi.txt         # Raspberry Pi bağımlılıkları
├── main.py                        # Ana çalıştırma dosyası
└── README.md                      
```

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
# Repository'yi klonlayın (veya indirin)
git clone <repo-url>
cd meat_virtual_image_processing

# Sanal ortam oluşturun (önerilen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 2. Veri Seti Hazırlama

Veri setinizi organize edin. İki seçenek:

#### Seçenek A: Klasör bazlı (skorlarla)

Klasörlerinizi şu şekilde organize edin:

```
data/raw/
  ├── fresh/          # Taze et görselleri
  ├── medium/         # Orta seviye bozulmuş
  └── spoiled/        # Bozuk et görselleri
```

Sonra CSV oluşturun:

```bash
python main.py --mode prepare_data --folders fresh medium spoiled --scores 0.0 0.5 1.0
```

#### Seçenek B: Manuel CSV

`data/raw/labels.csv` dosyası oluşturun:

```csv
image_path,freshness_score
images/img001.jpg,0.1
images/img002.jpg,0.4
images/img003.jpg,0.8
```

### 3. Veri Görselleştirme

```bash
python main.py --mode visualize
```

Bu komut şunları oluşturur:
- Veri seti istatistikleri
- Skor dağılımı grafikleri
- Örnek görüntüler

### 4. Model Eğitimi

```bash
# Varsayılan ayarlarla (50 epoch, batch size 16)
python main.py --mode train

# Özel ayarlarla
python main.py --mode train --epochs 100 --batch_size 32 --lr 0.0001
```

Eğitim sonrası oluşturulanlar:
- `models/model.h5` - Keras modeli
- `models/model.tflite` - TFLite modeli (Raspberry Pi için)
- `outputs/plots/training_history.png` - Eğitim grafikleri
- `outputs/reports/training_report.txt` - Detaylı rapor

### 5. Masaüstü Uygulaması

```bash
python main.py --mode desktop
```

veya doğrudan:

```bash
python ui/desktop_app.py
```

### 6. Tek Görüntü Tahmini

```bash
python main.py --mode predict --image data/raw/images/test.jpg
```

## 🖥️ Masaüstü Uygulaması Kullanımı

1. **Görsel Seç** butonuna tıklayın
2. Bir et görseli seçin
3. **Tahmin Et** butonuna tıklayın
4. Sonuç bar ve metin ile gösterilir:
   - **Yeşil bölge** (0.00-0.33): Yenilebilir
   - **Sarı bölge** (0.33-0.67): Dikkatli olun
   - **Kırmızı bölge** (0.67-1.00): Yenmemeli

## 🍓 Raspberry Pi Deployment

Detaylı talimatlar için: [raspi/setup_instructions.md](raspi/setup_instructions.md)

### Hızlı Özet

```bash
# Raspberry Pi'de bağımlılıkları yükle
pip3 install -r requirements_raspi.txt

# Tek tahmin
cd raspi
python3 raspi_app.py --mode single --camera picamera

# Sürekli mod (her 5 saniye)
python3 raspi_app.py --mode continuous --interval 5

# LED feedback modu
python3 raspi_app.py --mode led
```

## 📊 Model Detayları

### Mimari

- **Base**: MobileNetV2 (ImageNet pre-trained)
- **Custom Head**: 
  - GlobalAveragePooling2D
  - Dense(128, ReLU)
  - BatchNormalization
  - Dropout(0.3)
  - Dense(1, Sigmoid) → Output: 0-1 arası skor

### Eğitim

- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=0.001)
- **Metrics**: MAE, MSE, RMSE
- **Callbacks**: 
  - ModelCheckpoint (en iyi modeli kaydet)
  - EarlyStopping (patience=10)
  - ReduceLROnPlateau (factor=0.5, patience=5)
- **Data Augmentation**: 
  - Random flip
  - Random brightness/contrast
  - Random rotation

### Performans

Tipik sonuçlar (veri setine bağlı):
- Validation MAE: ~0.05-0.10
- Validation RMSE: ~0.08-0.15
- Inference time (TFLite - Raspberry Pi 4): ~0.5-1 saniye

## 🔧 Konfigürasyon

### Veri Augmentation

`src/train.py` içinde:

```python
use_augmentation=True  # Varsayılan: True
```

### Model Hiperparametreleri

```bash
python main.py --mode train \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.0005
```

### Görüntü Boyutu

`src/data_utils.py` ve `src/model.py` içinde `img_size` parametresi:

```python
img_size = (224, 224)  # Varsayılan (MobileNetV2 için)
```

## 📝 Veri Seti Gereksinimleri

- **Format**: JPG, JPEG, PNG, BMP
- **Boyut**: Herhangi bir boyut (otomatik resize edilir)
- **Etiketler**: 0.0-1.0 arası float skorlar
- **Önerilen miktar**: En az 500-1000 görüntü (daha fazlası daha iyi)
- **Dağılım**: Farklı bozulma seviyelerinden dengeli örnekler

## 🐛 Sorun Giderme

### Model bulunamadı hatası

```bash
# Önce modeli eğitin
python main.py --mode train
```

### CUDA/GPU hatası

TensorFlow CPU versiyonu kullanılıyor. GPU istemiyorsanız:

```bash
# GPU'yu devre dışı bırak
export CUDA_VISIBLE_DEVICES="-1"  # Linux/Mac
set CUDA_VISIBLE_DEVICES=-1       # Windows
```

### Bellek hatası

Batch size'ı küçültün:

```bash
python main.py --mode train --batch_size 8
```

### Tkinter bulunamadı (Linux)

```bash
sudo apt-get install python3-tk
```

## 📄 Lisans

[MIT License](LICENSE)

## 👨‍💻 Geliştirici

**Ahmet** - Et Bozulma Tespit Sistemi

## 🙏 Teşekkürler

- MobileNetV2 için Google Research
- TensorFlow ve Keras ekipleri
- Raspberry Pi Foundation

## 📮 İletişim

Sorularınız için:
- GitHub Issues: [https://github.com/Ahmet-KATI/Meat_Processing]
- Email: [ahmetkati16@gmail.com]

---

## 📚 Ek Kaynaklar

- [Model Eğitim Detayları](docs/training_guide.md)
- [Raspberry Pi Setup](raspi/setup_instructions.md)
- [API Dokümantasyonu](docs/api_reference.md)

---

**Not**: Bu proje eğitim ve araştırma amaçlıdır. Gerçek gıda güvenliği kararları için profesyonel analiz gereklidir.
