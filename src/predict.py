"""
Tahmin (inference) fonksiyonları.
Eğitilmiş model ile et bozulma skoru tahmini yapar.
"""

import os
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
import tempfile
import shutil


def load_trained_model(model_path='models/model.h5'):
    """
    Eğitilmiş modeli yükler.
    
    Args:
        model_path (str): Model dosya yolu (.h5 veya SavedModel dizini)
    
    Returns:
        keras.Model: Yüklenmiş model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    
    try:
        # Windows path encoding fix: Copy to temp file if needed
        # or just always do it to be safe and robust
        fd, temp_path = tempfile.mkstemp(suffix='.h5')
        os.close(fd)
        
        print(f"Model gecici dosyaya kopyalaniyor: {temp_path}")
        shutil.copy2(model_path, temp_path)
        
        try:
            model = keras.models.load_model(temp_path, compile=False)
            print(f"Model yuklendi: {model_path}")
            return model
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        raise Exception(f"Model yuklenirken hata: {e}")


def preprocess_image_for_prediction(image_path, img_size=(224, 224)):
    """
    Tahmin için görüntüyü ön işler.
    
    Args:
        image_path (str): Görüntü dosya yolu veya numpy array
        img_size (tuple): Hedef boyut
    
    Returns:
        np.ndarray: İşlenmiş görüntü (batch dimension ile)
    """
    # Eğer numpy array ise
    if isinstance(image_path, np.ndarray):
        img = image_path
    else:
        # Dosyadan yükle
        # Windows path encoding fix: Use imdecode instead of imread
        try:
            with open(image_path, 'rb') as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Görüntü okuma hatası (imdecode): {e}")
            img = None
            
        # Fallback to imread if imdecode fails (though imdecode is usually better)
        if img is None:
            img = cv2.imread(image_path)
            
        if img is None:
            raise ValueError(f"Görüntü yüklenemedi: {image_path}")
        
        # RGB'ye çevir
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Yeniden boyutlandır
    img = cv2.resize(img, img_size)
    
    # Float32    # Normalize et (ARTIK MODEL İÇİNDE YAPILIYOR)
    img = img.astype(np.float32)
    
    # Batch dimension ekle
    img = np.expand_dims(img, axis=0)
    
    return img


def predict_freshness(model, image_path, return_category=True):
    """
    Et bozulma skorunu tahmin eder.
    
    Args:
        model: Yüklenmiş Keras modeli
        image_path (str or np.ndarray): Görüntü yolu veya array
        return_category (bool): Kategori de döndür
    
    Returns:
        dict: Tahmin sonuçları
            - score: 0-1 arası bozulma skoru
            - category: "Taze", "Orta", "Bozuk"
            - label: Kullanıcıya gösterilecek metin
            - color: UI rengi (rgb tuple)
    """
    # Görüntüyü hazırla
    img = preprocess_image_for_prediction(image_path)
    
    # Tahmin
    score = model.predict(img, verbose=0)[0][0]
    
    # --- KALİBRASYON (YUMUŞATMA) ---
    # Model sadece 0 ve 1 etiketleriyle eğitildiği için çok "keskin" (overconfident) sonuçlar verir.
    # Temperature scaling (T > 1) ile bu keskinliği yumuşatarak ara değerleri daha görünür kılıyoruz.
    temp = 5.0 
    # Sigmoid'in tersi olan logit üzerinden işlem yapıyoruz:
    # epsilon ekleyerek log(0) hatasını önlüyoruz
    eps = 1e-7
    score = np.clip(score, eps, 1.0 - eps)
    logit = np.log(score / (1.0 - score))
    # Logiti yumuşatıp tekrar sigmoid'e sokuyoruz
    calibrated_score = 1.0 / (1.0 + np.exp(-logit / temp))
    
    # Sonuç dictionary'si
    result = {
        'score': float(calibrated_score),
        'raw_score': float(score) # Ham skoru da saklayalım
    }
    
    if return_category:
        category, label, color = score_to_category(score)
        result['category'] = category
        result['label'] = label
        result['color'] = color
    
    return result


def score_to_category(score):
    """
    Skoru kategoriye çevirir.
    
    Args:
        score (float): 0-1 arası bozulma skoru
    
    Returns:
        tuple: (category, label, color)
            - category: "fresh", "medium", "spoiled"
            - label: Kullanıcıya gösterilecek Türkçe metin
            - color: (R, G, B) renk tuple'ı
    """
    if score <= 0.33:
        return (
            "fresh",
            "Bu et tazedir ve güvenle yenilebilir.",
            (46, 204, 113)  # Yeşil
        )
    elif score <= 0.67:
        return (
            "medium",
            "Bu et orta seviyede bozulmuş. Dikkatli olun!",
            (241, 196, 15)  # Sarı
        )
    else:
        return (
            "spoiled",
            "Bu et bozulmuş durumda. Tüketilmemelidir!",
            (231, 76, 60)  # Kırmızı
        )


def batch_predict(model, image_paths, batch_size=32):
    """
    Birden fazla görüntü için toplu tahmin yapar.
    
    Args:
        model: Yüklenmiş model
        image_paths (list): Görüntü yolları listesi
        batch_size (int): Batch boyutu
    
    Returns:
        list: Her görüntü için tahmin sonuçları
    """
    results = []
    
    print(f"{len(image_paths)} görüntü için tahmin yapılıyor...")
    
    for i, img_path in enumerate(image_paths):
        try:
            result = predict_freshness(model, img_path)
            results.append({
                'image_path': img_path,
                **result
            })
            
            if (i + 1) % 10 == 0:
                print(f"  İşlendi: {i+1}/{len(image_paths)}")
                
        except Exception as e:
            print(f"Hata ({img_path}): {e}")
            results.append({
                'image_path': img_path,
                'error': str(e)
            })
    
    print(f"Tahmin tamamlandı")
    
    return results


class TFLitePredictor:
    """
    TensorFlow Lite model için tahmin sınıfı (Raspberry Pi için).
    """
    
    def __init__(self, model_path='models/model.tflite'):
        """
        Args:
            model_path (str): TFLite model yolu
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model bulunamadı: {model_path}")
        
        # Interpreter oluştur
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Input/Output detayları
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Input shape
        self.input_shape = self.input_details[0]['shape']
        self.img_size = (self.input_shape[1], self.input_shape[2])
        
        print(f"TFLite model yüklendi: {model_path}")
        print(f"  Input shape: {self.input_shape}")
    
    def predict(self, image_path):
        """
        TFLite model ile tahmin yapar.
        
        Args:
            image_path (str or np.ndarray): Görüntü
        
        Returns:
            dict: Tahmin sonuçları
        """
        # Görüntüyü hazırla
        img = preprocess_image_for_prediction(image_path, img_size=self.img_size)
        
        # Input tensor'ü ayarla
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        
        # Inference çalıştır
        self.interpreter.invoke()
        
        # Output al
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        score = float(output[0][0])
        
        # Kategori bilgisi
        category, label, color = score_to_category(score)
        
        return {
            'score': score,
            'category': category,
            'label': label,
            'color': color
        }


if __name__ == "__main__":
    # Test kodu
    import sys
    
    print("🧪 Predict Test\n")
    
    model_path = 'models/model.h5'
    
    if not os.path.exists(model_path):
        print(f"⚠ Model bulunamadı: {model_path}")
        print("Önce modeli eğitin: python src/train.py")
        sys.exit(1)
    
    # Model yükle
    model = load_trained_model(model_path)
    
    # Test tahmini
    test_image = 'data/raw/images/test.jpg'
    
    if os.path.exists(test_image):
        result = predict_freshness(model, test_image)
        print(f"\n📊 Tahmin Sonucu:")
        print(f"  Skor: {result['score']:.4f}")
        print(f"  Kategori: {result['category']}")
        print(f"  Mesaj: {result['label']}")
    else:
        print(f"⚠ Test görüntüsü bulunamadı: {test_image}")
