# Et Bozulma Tespit Sistemi - Hızlı Başlangıç

## 🎯 Ne Yaptık?

Etin bozulma seviyesini fotoğraflardan tahmin eden **tam teşekküllü bir makine öğrenmesi sistemi** oluşturduk.

## 📦 Sistem Bileşenleri

### 1. Veri İşleme (`src/`)
- ✅ CSV'den veri yükleme
- ✅ Görüntü ön işleme (resize, normalize)
- ✅ Veri augmentation (flip, brightness, contrast)
- ✅ Veri görselleştirme ve istatistikler

### 2. Model (`src/model.py`)
- ✅ MobileNetV2 tabanlı transfer learning
- ✅ Regresyon: 0.0 (taze) → 1.0 (bozuk)
- ✅ Hafif mimari (~4M parametre)
- ✅ TensorFlow Lite dönüşümü

### 3. Eğitim (`src/train.py`)
- ✅ Otomatik callbacks (checkpoint, early stopping)
- ✅ Eğitim grafikleri ve raporlar
- ✅ Multi-format kayıt (.h5, .tflite)

### 4. Masaüstü UI (`ui/`)
- ✅ Tkinter tabanlı arayüz
- ✅ Yeşil-kırmızı gradient bar
- ✅ Dosya seçme ve tahmin gösterimi

### 5. Raspberry Pi (`raspi/`)
- ✅ TFLite ile optimize edilmiş
- ✅ Kamera entegrasyonu (picamera2/OpenCV)
- ✅ LED feedback modu
- ✅ Detaylı kurulum talimatları

## 🚀 İlk Kullanım

### Adım 1: Bağımlılıkları Kur
```bash
pip install -r requirements.txt
```

### Adım 2: Veri Hazırla
```bash
# Klasörlerinizi organize edin:
# data/raw/fresh/     -> taze et görselleri
# data/raw/medium/    -> orta seviye
# data/raw/spoiled/   -> bozuk et

# CSV oluştur
python main.py --mode prepare_data --folders fresh medium spoiled --scores 0.0 0.5 1.0
```

### Adım 3: Veriyi İncele
```bash
python main.py --mode visualize
```

### Adım 4: Model Eğit
```bash
python main.py --mode train --epochs 50
```

### Adım 5: UI'ı Başlat
```bash
python main.py --mode desktop
```

## 📊 Dosya Listesi

| Dosya | Açıklama |
|-------|----------|
| `main.py` | 🎮 Ana çalıştırma scripti |
| `src/data_utils.py` | 📦 Veri yükleme ve işleme |
| `src/visualization.py` | 📈 Veri görselleştirme |
| `src/model.py` | 🧠 Model mimarisi |
| `src/train.py` | 🏋️ Eğitim pipeline'ı |
| `src/predict.py` | 🔮 Tahmin fonksiyonları |
| `ui/components.py` | 🎨 UI widget'ları |
| `ui/desktop_app.py` | 🖥️ Tkinter uygulaması |
| `raspi/camera_capture.py` | 📷 Kamera entegrasyonu |
| `raspi/raspi_app.py` | 🍓 Raspberry Pi uygulaması |

## 🎯 Temel Komutlar

```bash
# Veri hazırlama
python main.py --mode prepare_data

# Veri görselleştirme
python main.py --mode visualize

# Model eğitimi
python main.py --mode train --epochs 50 --batch_size 16

# Tek görsel tahmini
python main.py --mode predict --image path/to/image.jpg

# Masaüstü UI
python main.py --mode desktop
```

## 📚 Dokümantasyon

- **README.md**: Detaylı kullanım kılavuzu
- **raspi/setup_instructions.md**: Raspberry Pi kurulum
- **data/raw/labels_example.md**: Veri etiketleme rehberi

## 🎨 Görsel Feedback

Sistem, bozulma skorunu görsel bar ile gösterir:

```
[████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
 TAZE                ORTA                  BOZUK
 (0.0)              (0.5)                 (1.0)
```

- **Yeşil** (0.00-0.33): Bu et tazedir ve güvenle yenilebilir ✅
- **Sarı** (0.33-0.67): Bu et orta seviyede bozulmuş. Dikkatli olun ⚠️
- **Kırmızı** (0.67-1.00): Bu et bozulmuş. Tüketilmemelidir ❌

## 🔄 Proje Akışı

```
1. VERİ HAZIRLA
   └─> CSV oluştur (klasörlerden veya manuel)

2. VERİYİ İNCELE
   └─> Görselleştir ve istatistikler

3. MODEL EĞİT
   └─> MobileNetV2 + Transfer Learning
   └─> Callbacks (checkpoint, early stopping)
   └─> Kaydet (.h5, .tflite)

4. TEST ET
   ├─> Masaüstü UI ile test
   └─> Tek görsel tahminleri

5. DEPLOY ET
   └─> Raspberry Pi'ye transfer
   └─> Kamera ile canlı tahmin
```

## 💾 Model Formatları

- **model.h5**: Masaüstü için Keras formatı
- **model.tflite**: Raspberry Pi için optimize edilmiş
- **best_model.h5**: En iyi checkpoint

## 🍓 Raspberry Pi Hızlı Başlangıç

```bash
# Raspberry Pi'de
pip3 install -r requirements_raspi.txt

# Tek tahmin
cd raspi
python3 raspi_app.py --mode single --camera picamera

# Sürekli mod
python3 raspi_app.py --mode continuous --interval 5
```

## 📞 Yardım

Tüm detaylar için:
- **README.md** - Genel kullanım
- **walkthrough.md** - Teknik detaylar
- **raspi/setup_instructions.md** - Raspberry Pi

---

**🎉 Proje hazır! Veri setinizi ekleyip eğitime başlayabilirsiniz!**
