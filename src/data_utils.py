"""
Veri yükleme, ön işleme ve augmentation fonksiyonları.
Bu modül et görüntülerini yüklemek ve model eğitimi için hazırlamak amacıyla kullanılır.
"""

import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


class MeatDataset:
    """Et veri seti yönetimi için ana sınıf."""
    
    def __init__(self, data_dir, img_size=(224, 224)):
        """
        Args:
            data_dir (str): Veri seti ana dizini
            img_size (tuple): Hedef görüntü boyutu (height, width)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.images = []
        self.scores = []
        
    def load_from_csv(self, csv_path):
        """
        CSV dosyasından veri yükler.
        CSV formatı: image_path, freshness_score
        
        Args:
            csv_path (str): CSV dosya yolu
        
        Returns:
            tuple: (image_paths, scores)
        """
        df = pd.read_csv(csv_path)
        print(f"✓ CSV'den {len(df)} kayıt yüklendi")
        
        # Görüntü yollarını tam yol haline getir
        image_paths = [os.path.join(self.data_dir, path) for path in df['image_path']]
        scores = df['freshness_score'].values
        
        # Skorların 0-1 arasında olduğunu kontrol et
        if scores.min() < 0 or scores.max() > 1:
            print(f"⚠ Uyarı: Skorlar 0-1 aralığı dışında! Min: {scores.min()}, Max: {scores.max()}")
            # Normalizasyon yap
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            print(f"✓ Skorlar 0-1 aralığına normalize edildi")
        
        return image_paths, scores
    
    def create_csv_from_folders(self, folder_mapping, output_csv='data/raw/labels.csv'):
        """
        Klasör yapısından CSV oluşturur.
        
        Args:
            folder_mapping (dict): Klasör adı -> skor eşleştirmesi
                Örnek: {'fresh': 0.0, 'medium': 0.5, 'spoiled': 1.0}
            output_csv (str): Oluşturulacak CSV dosya yolu
        
        Returns:
            pd.DataFrame: Oluşturulan DataFrame
        """
        data = []
        
        for folder_name, score in folder_mapping.items():
            folder_path = os.path.join(self.data_dir, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"⚠ Klasör bulunamadı: {folder_path}")
                continue
            
            # Klasördeki tüm görüntüleri bul
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"✓ {folder_name}: {len(image_files)} görüntü bulundu (skor: {score})")
            
            for img_file in image_files:
                rel_path = os.path.join(folder_name, img_file)
                data.append({'image_path': rel_path, 'freshness_score': score})
        
        # DataFrame oluştur ve kaydet
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"\n✓ CSV dosyası oluşturuldu: {output_csv}")
        print(f"  Toplam {len(df)} görüntü kaydedildi")
        
        return df
    
    def preprocess_image(self, image_path):
        """
        Tek bir görüntüyü önişler.
        
        Args:
            image_path (str): Görüntü dosya yolu
        
        Returns:
            np.ndarray: İşlenmiş görüntü (normalized)
        """
        try:
            # Görüntüyü yükle
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠ Görüntü yüklenemedi: {image_path}")
                return None
            
            # RGB'ye çevir (OpenCV BGR kullanır)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Yeniden boyutlandır
            img = cv2.resize(img, self.img_size)
            
            # 0-1 aralığına normalize et
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            print(f"⚠ Hata ({image_path}): {e}")
            return None
    
    def load_and_preprocess(self, image_paths, scores, test_size=0.2, random_state=42):
        """
        Tüm veri setini yükler ve train/validation'a böler.
        
        Args:
            image_paths (list): Görüntü yolları listesi
            scores (np.ndarray): Bozulma skorları
            test_size (float): Validation oranı
            random_state (int): Random seed
        
        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        print(f"\n📊 Görüntüler yükleniyor...")
        
        images = []
        valid_scores = []
        
        for i, (img_path, score) in enumerate(zip(image_paths, scores)):
            if (i + 1) % 100 == 0:
                print(f"  İşlendi: {i+1}/{len(image_paths)}")
            
            img = self.preprocess_image(img_path)
            if img is not None:
                images.append(img)
                valid_scores.append(score)
        
        images = np.array(images)
        valid_scores = np.array(valid_scores)
        
        print(f"\n✓ Toplam {len(images)} görüntü başarıyla yüklendi")
        print(f"  Görüntü boyutu: {images.shape}")
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            images, valid_scores, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"\n📂 Veri bölünmesi:")
        print(f"  Training: {len(X_train)} görüntü")
        print(f"  Validation: {len(X_val)} görüntü")
        
        return X_train, X_val, y_train, y_val


def get_augmentation_pipeline(rotation_range=20, 
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              horizontal_flip=True,
                              zoom_range=0.2,
                              brightness_range=(0.8, 1.2)):
    """
    Veri artırma (augmentation) pipeline'ı oluşturur.
    
    Args:
        rotation_range (int): Rastgele döndürme açısı
        width_shift_range (float): Yatay kaydırma oranı
        height_shift_range (float): Dikey kaydırma oranı
        horizontal_flip (bool): Yatay çevirme
        zoom_range (float): Zoom oranı
        brightness_range (tuple): Parlaklık değişim aralığı
    
    Returns:
        ImageDataGenerator: Augmentation pipeline
    """
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        zoom_range=zoom_range,
        brightness_range=brightness_range,
        fill_mode='nearest'
    )
    
    return datagen


def create_tf_dataset(X, y, batch_size=16, augment=False):
    """
    TensorFlow Dataset oluşturur.
    
    Args:
        X (np.ndarray): Görüntüler
        y (np.ndarray): Skorlar
        batch_size (int): Batch boyutu
        augment (bool): Augmentation uygula
    
    Returns:
        tf.data.Dataset: TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if augment:
        # Veri artırma fonksiyonu
        def augment_fn(image, label):
            # Random flip
            image = tf.image.random_flip_left_right(image)
            # Random brightness
            image = tf.image.random_brightness(image, max_delta=0.2)
            # Random contrast
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            # Clip to [0, 1]
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label
        
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    # Test kodu
    print("🧪 Data Utils Test")
    
    # Örnek kullanım
    dataset = MeatDataset(data_dir='data/raw')
    
    # Klasör bazlı CSV oluşturma örneği
    # folder_mapping = {
    #     'fresh': 0.0,      # Taze
    #     'medium': 0.5,     # Orta
    #     'spoiled': 1.0     # Bozuk
    # }
    # df = dataset.create_csv_from_folders(folder_mapping)
