# Raspberry Pi Kurulum Talimatları

Bu döküman, Et Bozulma Tespit Sistemi'nin Raspberry Pi üzerinde nasıl kurulacağını ve çalıştırılacağını açıklar.

---

## ⚙️ Gereksinimler

### Donanım
- **Raspberry Pi 4 Model B** (önerilen) veya Raspberry Pi 3
- **Raspberry Pi Camera Module** veya USB Kamera
- **microSD Kart** (en az 16GB, Class 10)
- **Güç Kaynağı** (5V, 3A)
- **(Opsiyonel)** LED'ler ve dirençler (feedback için)

### Yazılım
- **Raspberry Pi OS** (Bullseye veya daha üst)
- **Python 3.9+**

---

## 📋 Kurulum Adımları

### 1. Raspberry Pi OS Kurulumu

Raspberry Pi Imager kullanarak Raspberry Pi OS'i yükleyin:

```bash
# Raspberry Pi Imager'ı indirin (Windows/Mac/Linux):
# https://www.raspberrypi.com/software/
```

**Önerilen ayarlar:**
- OS: Raspberry Pi OS (64-bit) - Bullseye
- SSH etkinleştirin
- WiFi bilgilerini önceden ayarlayın

### 2. Sistem Güncellemesi

```bash
# Raspberry Pi'ye SSH ile bağlanın veya terminali açın
sudo apt update
sudo apt upgrade -y
```

### 3. Python ve Bağımlılıkların Kurulumu

```bash
# Python 3 ve pip
sudo apt install -y python3 python3-pip python3-venv

# OpenCV ve numpy (sistem paketleri)
sudo apt install -y python3-opencv python3-numpy

# picamera2 (Raspberry Pi Camera Module için)
sudo apt install -y python3-picamera2

# TensorFlow Lite runtime
pip3 install tensorflow-lite-runtime==2.13.0

# Pillow
pip3 install pillow
```

**Not:** TensorFlow Lite, tam TensorFlow'dan çok daha hafiftir ve Raspberry Pi için önerilir.

### 4. Kamera Aktivasyonu

#### Raspberry Pi Camera Module kullanıyorsanız:

```bash
# Camera interface'i etkinleştir
sudo raspi-config
# Interface Options > Camera > Enable seçin

# Yeniden başlat
sudo reboot
```

#### USB Kamera kullanıyorsanız:

```bash
# Kameranın tanındığını kontrol edin
ls /dev/video*
# /dev/video0 görmelisiniz
```

### 5. Proje Dosyalarının Transferi

#### Method 1: Git (Önerilen)

```bash
# Projeyi klonlayın (GitHub'a yüklediyseniz)
git clone https://github.com/your-username/meat_freshness_detection.git
cd meat_freshness_detection
```

#### Method 2: SCP ile Transfer

Bilgisayarınızdan model ve kod dosyalarını Raspberry Pi'ye aktarın:

```bash
# Bilgisayarınızda (Windows PowerShell veya Linux/Mac Terminal)
scp -r meat_virtual_image_processing pi@raspberrypi.local:~/

# Şifre girdikte Raspberry Pi'nin şifresini girin
```

#### Method 3: USB Bellek

USB belleğe kopyalayıp Raspberry Pi'ye takın.

---

## 🚀 Çalıştırma

### Model Dosyasının Varlığını Kontrol Edin

```bash
cd ~/meat_virtual_image_processing
ls models/model.tflite

# Dosya yoksa masaüstü bilgisayardan transfer edin
```

### Tek Tahmin Modu

```bash
cd raspi
python3 raspi_app.py --mode single --camera picamera

# veya USB kamera ile:
python3 raspi_app.py --mode single --camera opencv
```

### Sürekli Tahmin Modu

```bash
# Her 5 saniyede bir tahmin yapar
python3 raspi_app.py --mode continuous --interval 5

# Durdurmak için Ctrl+C
```

### LED Feedback Modu (Opsiyonel)

LED'leri şu şekilde bağlayın:
- **Yeşil LED**: GPIO 17 (Pin 11)
- **Sarı LED**: GPIO 27 (Pin 13)
- **Kırmızı LED**: GPIO 22 (Pin 15)
- **Ground**: GND pinlerinden herhangi biri

Her LED için 220Ω direnç kullanın.

```bash
python3 raspi_app.py --mode led --camera picamera
```

---

## 🔧 Sorun Giderme

### Kamera Bulunamıyor

```bash
# Kamera bağlantısını kontrol edin
vcgencmd get_camera

# Çıktı: "supported=1 detected=1" olmalı

# picamera2 test
python3 -c "from picamera2 import Picamera2; print('OK')"
```

### Model Yüklenemiyor

```bash
# TFLite runtime kontrolü
python3 -c "import tensorflow.lite as tflite; print('OK')"

# Model dosyası var mı?
ls -lh ../models/model.tflite
```

### OpenCV Hatası

```bash
# OpenCV yeniden kurulum
sudo apt install --reinstall python3-opencv
```

### Yetersiz Bellek

Swap alanını artırın:

```bash
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=100 satırını bulun
# CONF_SWAPSIZE=1024 yapın

sudo /etc/init.d/dphys-swapfile restart
```

---

## ⚡ Performans Optimizasyonu

### 1. Model Quantization (Daha Küçük Model)

Masaüstü bilgisayarda:

```python
# train.py içinde convert_to_tflite fonksiyonunda
# INT8 quantization ekleyin (daha küçük, biraz daha az doğru)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
```

### 2. Düşük Çözünürlük Kullanın

```python
# raspi_app.py içinde
camera = CameraCapture(resolution=(320, 240))  # Daha düşük çözünürlük
```

### 3. CPU Affinity

```bash
# Sadece belirli CPU core'ları kullan
taskset -c 0,1 python3 raspi_app.py
```

---

## 📊 Sonuç Örnekleri

### Konsol Çıktısı

```
📸 Görüntü yakalanıyor...
✓ Görüntü kaydedildi: captured.jpg
🔍 Tahmin yapılıyor...

============================================================
📊 TAHMİN SONUCU
============================================================

🎯 Bozulma Skoru: 0.2340

██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
TAZE                        ORTA                       BOZUK

📋 Kategori: FRESH
💬 Sonuç: Bu et tazedir ve güvenle yenilebilir. ✅
============================================================
```

---

## 🔄 Otomatik Başlatma (Systemd Service)

Raspberry Pi açıldığında uygulamanın otomatik başlaması için:

```bash
# Service dosyası oluştur
sudo nano /etc/systemd/system/meat-detector.service
```

İçeriği:

```ini
[Unit]
Description=Meat Freshness Detection Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/meat_virtual_image_processing/raspi
ExecStart=/usr/bin/python3 raspi_app.py --mode continuous --interval 10
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Servisi aktifleştir:

```bash
sudo systemctl daemon-reload
sudo systemctl enable meat-detector.service
sudo systemctl start meat-detector.service

# Durumu kontrol et
sudo systemctl status meat-detector.service
```

---

## 📝 Notlar

- **İlk çalıştırma**: Model yüklemesi 5-10 saniye sürebilir
- **Tahmin süresi**: TFLite ile ~0.5-1 saniye
- **Kamera ısınma**: İlk tahmin daha uzun sürebilir
- **Işık koşulları**: İyi aydınlatma daha iyi sonuç verir

---

## 🆘 Yardım

Sorun yaşarsanız:

1. Log dosyalarını kontrol edin
2. Verbose mod ile çalıştırın: `python3 -v raspi_app.py`
3. Kamera ve model dosyalarını doğrulayın

**İletişim:** [Projenizin GitHub/Email bilgisi]

---

© 2025 Et Bozulma Tespit Sistemi
