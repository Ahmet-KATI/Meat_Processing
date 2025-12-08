"""
Raspberry Pi kamera entegrasyonu.
picamera2 veya OpenCV ile kameradan görüntü yakalama.
"""

import os
import numpy as np
import cv2


class CameraCapture:
    """
    Raspberry Pi kamera yakalama sınıfı.
    Hem picamera2 hem de OpenCV'yi destekler.
    """
    
    def __init__(self, use_picamera=True, resolution=(640, 480)):
        """
        Args:
            use_picamera (bool): True ise picamera2 kullan, False ise OpenCV
            resolution (tuple): Kamera çözünürlüğü (width, height)
        """
        self.use_picamera = use_picamera
        self.resolution = resolution
        self.camera = None
        
        # Kamerayı başlat
        self.setup_camera()
    
    def setup_camera(self):
        """Kamerayı başlatır."""
        try:
            if self.use_picamera:
                # picamera2 kullan
                self._setup_picamera()
            else:
                # OpenCV kullan
                self._setup_opencv_camera()
        except Exception as e:
            print(f"⚠ Kamera başlatılamadı: {e}")
            print("Başka bir yöntem deneyin.")
    
    def _setup_picamera(self):
        """picamera2 ile kamerayı başlatır (Raspberry Pi Camera Module için)."""
        try:
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            
            # Kamera konfigürasyonu
            config = self.camera.create_still_configuration(
                main={"size": self.resolution}
            )
            self.camera.configure(config)
            
            # Başlat
            self.camera.start()
            
            print(f"✓ Picamera2 başlatıldı: {self.resolution}")
            
        except ImportError:
            print("⚠ picamera2 kütüphanesi bulunamadı!")
            print("Kurulum: sudo apt install python3-picamera2")
            raise
    
    def _setup_opencv_camera(self):
        """OpenCV ile kamerayı başlatır (USB kamera için)."""
        # Varsayılan kamera (genellikle /dev/video0)
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            raise Exception("Kamera açılamadı! /dev/video0 mevcut değil.")
        
        # Çözünürlük ayarla
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        print(f"✓ OpenCV kamera başlatıldı: {self.resolution}")
    
    def capture_frame(self):
        """
        Kameradan tek bir kare yakalar.
        
        Returns:
            np.ndarray: RGB formatında görüntü (H, W, 3)
        """
        if self.camera is None:
            raise Exception("Kamera başlatılmamış!")
        
        try:
            if self.use_picamera:
                # picamera2 ile yakala
                frame = self.camera.capture_array()
                
                # RGB'ye çevir (picamera2 varsayılan olarak RGB verir)
                if len(frame.shape) == 2:  # Grayscale ise
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
                return frame
            
            else:
                # OpenCV ile yakala
                ret, frame = self.camera.read()
                
                if not ret:
                    raise Exception("Kare yakalanamadı!")
                
                # BGR'den RGB'ye çevir
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                return frame
        
        except Exception as e:
            print(f"⚠ Kare yakalama hatası: {e}")
            return None
    
    def capture_and_save(self, save_path='captured_image.jpg'):
        """
        Kameradan görüntü yakalar ve kaydeder.
        
        Args:
            save_path (str): Kayıt yolu
        
        Returns:
            str: Kaydedilen dosya yolu
        """
        frame = self.capture_frame()
        
        if frame is None:
            return None
        
        # Kaydet (RGB -> BGR dönüşümü yaparak)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, frame_bgr)
        
        print(f"✓ Görüntü kaydedildi: {save_path}")
        
        return save_path
    
    def preview_stream(self, window_name='Camera Preview', duration=10):
        """
        Kamera önizlemesini gösterir (test için).
        
        Args:
            window_name (str): Pencere adı
            duration (int): Önizleme süresi (saniye), 0 ise sürekli
        """
        print(f"Kamera önizlemesi başlatılıyor... (ESC veya 'q' ile çıkış)")
        
        import time
        start_time = time.time()
        
        while True:
            frame = self.capture_frame()
            
            if frame is None:
                break
            
            # BGR'ye çevir (OpenCV imshow için)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Göster
            cv2.imshow(window_name, frame_bgr)
            
            # Çıkış kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' veya ESC
                break
            
            # Süre kontrolü
            if duration > 0 and (time.time() - start_time) > duration:
                break
        
        cv2.destroyAllWindows()
        print("Önizleme kapatıldı")
    
    def release(self):
        """Kamerayı serbest bırakır."""
        if self.camera is not None:
            try:
                if self.use_picamera:
                    self.camera.stop()
                else:
                    self.camera.release()
                
                print("✓ Kamera serbest bırakıldı")
            except Exception as e:
                print(f"⚠ Kamera kapatma hatası: {e}")
    
    def __del__(self):
        """Destructor: Kamerayı otomatik kapat."""
        self.release()


if __name__ == "__main__":
    # Test kodu
    import sys
    
    print("🧪 Kamera Test\n")
    
    # Kullanım: python camera_capture.py [picamera|opencv]
    use_picamera = True
    
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'opencv':
            use_picamera = False
    
    try:
        # Kamera oluştur
        camera = CameraCapture(use_picamera=use_picamera, resolution=(640, 480))
        
        # Test 1: Tek kare yakala ve kaydet
        print("\n📸 Test 1: Tek kare yakalama")
        image_path = camera.capture_and_save('test_capture.jpg')
        
        if image_path:
            print(f"✓ Test başarılı: {image_path}")
        
        # Test 2: Önizleme (opsiyonel)
        # camera.preview_stream(duration=5)
        
    except Exception as e:
        print(f"❌ Test başarısız: {e}")
    
    finally:
        print("\nKamera kapatılıyor...")
