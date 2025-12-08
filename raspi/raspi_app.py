"""
Raspberry Pi ana uygulaması.
Kameradan görüntü alıp TFLite model ile tahmin yapar.
"""

import os
import sys
import time
import argparse

# Kamera modülünü import et
from camera_capture import CameraCapture

# src modülünü import etmek için path ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.predict import TFLitePredictor


class RaspiFreshnessApp:
    """
    Raspberry Pi Et Bozulma Tespit Uygulaması.
    """
    
    def __init__(self, model_path='models/model.tflite', use_picamera=True):
        """
        Args:
            model_path (str): TFLite model yolu
            use_picamera (bool): picamera2 kullan (False ise OpenCV)
        """
        self.model_path = model_path
        self.use_picamera = use_picamera
        
        # Model yükle
        print("📦 Model yükleniyor...")
        self.predictor = TFLitePredictor(model_path=model_path)
        
        # Kamera başlat
        print("📷 Kamera başlatılıyor...")
        self.camera = CameraCapture(use_picamera=use_picamera, resolution=(640, 480))
        
        print("\n✅ Sistem hazır!\n")
    
    def capture_and_predict(self, save_image=False, image_path='captured.jpg'):
        """
        Kameradan görüntü alır ve tahmin yapar.
        
        Args:
            save_image (bool): Görüntüyü kaydet
            image_path (str): Kayıt yolu
        
        Returns:
            dict: Tahmin sonuçları
        """
        print("📸 Görüntü yakalanıyor...")
        
        # Kameradan kare yakala
        frame = self.camera.capture_frame()
        
        if frame is None:
            print("⚠ Görüntü yakalanamadı!")
            return None
        
        # İsteğe bağlı kaydet
        if save_image:
            self.camera.capture_and_save(image_path)
        
        # Tahmin yap
        print("🔍 Tahmin yapılıyor...")
        result = self.predictor.predict(frame)
        
        return result
    
    def display_result_console(self, result):
        """
        Tahmin sonucunu konsolda gösterir.
        
        Args:
            result (dict): Tahmin sonuçları
        """
        if result is None:
            return
        
        score = result['score']
        category = result['category']
        label = result['label']
        
        # Başlık
        print("\n" + "=" * 60)
        print("📊 TAHMİN SONUCU")
        print("=" * 60)
        
        # Skor
        print(f"\n🎯 Bozulma Skoru: {score:.4f}")
        
        # Bar gösterimi (ASCII)
        bar_length = 50
        filled_length = int(bar_length * score)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\n{bar}")
        print("TAZE                        ORTA                       BOZUK")
        
        # Kategori ve mesaj
        print(f"\n📋 Kategori: {category.upper()}")
        print(f"💬 Sonuç: {label}")
        
        print("=" * 60 + "\n")
    
    def run_single_prediction(self, save_image=True):
        """
        Tek bir tahmin yapar ve sonucu gösterir.
        
        Args:
            save_image (bool): Görüntüyü kaydet
        """
        result = self.capture_and_predict(save_image=save_image)
        self.display_result_console(result)
    
    def run_continuous(self, interval=5):
        """
        Sürekli tahmin modu (belirli aralıklarla).
        
        Args:
            interval (int): Tahminler arası süre (saniye)
        """
        print(f"🔄 Sürekli tahmin modu başlatıldı (Her {interval} saniye)")
        print("Durdurmak için Ctrl+C'ye basın\n")
        
        count = 0
        
        try:
            while True:
                count += 1
                print(f"\n--- TAHMİN #{count} ---")
                
                result = self.capture_and_predict(
                    save_image=True,
                    image_path=f'captures/capture_{count}.jpg'
                )
                
                self.display_result_console(result)
                
                # Bekle
                print(f"⏳ {interval} saniye bekleniyor...\n")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\n⏹ Sürekli mod durduruldu")
    
    def run_led_feedback(self, led_pins={'green': 17, 'yellow': 27, 'red': 22}):
        """
        LED feedback ile tahmin (Raspberry Pi GPIO).
        
        Args:
            led_pins (dict): LED pin numaraları
        """
        try:
            import RPi.GPIO as GPIO
            
            # GPIO setup
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            for pin in led_pins.values():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            
            print("💡 LED feedback modu aktif")
            print(f"  Yeşil LED: Pin {led_pins['green']}")
            print(f"  Sarı LED: Pin {led_pins['yellow']}")
            print(f"  Kırmızı LED: Pin {led_pins['red']}\n")
            
            # Tahmin yap
            result = self.capture_and_predict(save_image=True)
            self.display_result_console(result)
            
            if result:
                # LED'leri kapat
                for pin in led_pins.values():
                    GPIO.output(pin, GPIO.LOW)
                
                # Kategoriye göre LED yak
                category = result['category']
                
                if category == 'fresh':
                    GPIO.output(led_pins['green'], GPIO.HIGH)
                    print("💡 YEŞİL LED yanıyor (Taze)")
                elif category == 'medium':
                    GPIO.output(led_pins['yellow'], GPIO.HIGH)
                    print("💡 SARI LED yanıyor (Orta)")
                else:
                    GPIO.output(led_pins['red'], GPIO.HIGH)
                    print("💡 KIRMIZI LED yanıyor (Bozuk)")
                
                # 5 saniye bekle
                time.sleep(5)
                
                # LED'leri kapat
                for pin in led_pins.values():
                    GPIO.output(pin, GPIO.LOW)
            
            # Cleanup
            GPIO.cleanup()
            
        except ImportError:
            print("⚠ RPi.GPIO kütüphanesi bulunamadı!")
            print("Bu özellik sadece Raspberry Pi'de çalışır.")
        except Exception as e:
            print(f"⚠ LED feedback hatası: {e}")
    
    def cleanup(self):
        """Kaynakları temizle."""
        self.camera.release()
        print("✓ Kaynaklar temizlendi")


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(
        description='Raspberry Pi Et Bozulma Tespit Sistemi'
    )
    
    parser.add_argument('--model', type=str, default='../models/model.tflite',
                       help='TFLite model yolu')
    parser.add_argument('--camera', type=str, default='picamera',
                       choices=['picamera', 'opencv'],
                       help='Kamera tipi')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'continuous', 'led'],
                       help='Çalışma modu')
    parser.add_argument('--interval', type=int, default=5,
                       help='Sürekli modda tahminler arası süre (saniye)')
    parser.add_argument('--no_save', action='store_true',
                       help='Görüntüleri kaydetme')
    
    args = parser.parse_args()
    
    # Kamera tipini belirle
    use_picamera = (args.camera == 'picamera')
    
    try:
        # Uygulama oluştur
        app = RaspiFreshnessApp(
            model_path=args.model,
            use_picamera=use_picamera
        )
        
        # Çalışma moduna göre çalıştır
        if args.mode == 'single':
            app.run_single_prediction(save_image=not args.no_save)
        
        elif args.mode == 'continuous':
            # captures klasörünü oluştur
            os.makedirs('captures', exist_ok=True)
            app.run_continuous(interval=args.interval)
        
        elif args.mode == 'led':
            app.run_led_feedback()
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'app' in locals():
            app.cleanup()


if __name__ == "__main__":
    main()
