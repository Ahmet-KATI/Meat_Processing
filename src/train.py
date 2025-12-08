"""
Model eğitim scripti.
Et bozulma tahmini modelini eğitir ve kaydeder.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Modülleri içe aktar
from data_utils import MeatDataset, create_tf_dataset
from model import create_meat_freshness_model, compile_model, get_model_summary


def plot_training_history(history, save_path='outputs/plots/training_history.png'):
    """
    Eğitim geçmişini (loss ve metrics) grafikle gösterir.
    
    Args:
        history: Keras History objesi
        save_path (str): Grafik kayıt yolu
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # MAE
    axes[0, 1].plot(epochs, history.history['mae'], 'b-', label='Training MAE', linewidth=2)
    axes[0, 1].plot(epochs, history.history['val_mae'], 'r-', label='Validation MAE', linewidth=2)
    axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # RMSE
    axes[1, 0].plot(epochs, history.history['rmse'], 'b-', label='Training RMSE', linewidth=2)
    axes[1, 0].plot(epochs, history.history['val_rmse'], 'r-', label='Validation RMSE', linewidth=2)
    axes[1, 0].set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # MSE
    axes[1, 1].plot(epochs, history.history['mse'], 'b-', label='Training MSE', linewidth=2)
    axes[1, 1].plot(epochs, history.history['val_mse'], 'r-', label='Validation MSE', linewidth=2)
    axes[1, 1].set_title('Mean Squared Error', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Kaydet
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Eğitim grafiği kaydedildi: {save_path}")


def save_training_report(history, model_params, save_path='outputs/reports/training_report.txt'):
    """
    Eğitim raporunu metin dosyası olarak kaydeder.
    
    Args:
        history: Keras History objesi
        model_params (dict): Model parametreleri
        save_path (str): Rapor kayıt yolu
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ET BOZULMA TAHMİN MODELİ - EĞİTİM RAPORU\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model parametreleri
        f.write("MODEL PARAMETRELERİ:\n")
        f.write("-" * 80 + "\n")
        for key, value in model_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Eğitim sonuçları
        f.write("EĞİTİM SONUÇLARI:\n")
        f.write("-" * 80 + "\n")
        
        final_epoch = len(history.history['loss'])
        f.write(f"Toplam epoch: {final_epoch}\n\n")
        
        # Son epoch metrikleri
        f.write("SON EPOCH METRİKLERİ:\n")
        f.write(f"  Training Loss (MSE): {history.history['loss'][-1]:.6f}\n")
        f.write(f"  Validation Loss (MSE): {history.history['val_loss'][-1]:.6f}\n")
        f.write(f"  Training MAE: {history.history['mae'][-1]:.6f}\n")
        f.write(f"  Validation MAE: {history.history['val_mae'][-1]:.6f}\n")
        f.write(f"  Training RMSE: {history.history['rmse'][-1]:.6f}\n")
        f.write(f"  Validation RMSE: {history.history['val_rmse'][-1]:.6f}\n\n")
        
        # En iyi sonuçlar
        f.write("EN İYİ SONUÇLAR:\n")
        best_val_loss_epoch = np.argmin(history.history['val_loss']) + 1
        f.write(f"  En düşük validation loss: {min(history.history['val_loss']):.6f} (Epoch {best_val_loss_epoch})\n")
        best_val_mae_epoch = np.argmin(history.history['val_mae']) + 1
        f.write(f"  En düşük validation MAE: {min(history.history['val_mae']):.6f} (Epoch {best_val_mae_epoch})\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✓ Eğitim raporu kaydedildi: {save_path}")


def convert_to_tflite(model, save_path='models/model.tflite'):
    """
    Modeli TensorFlow Lite formatına çevirir (Raspberry Pi için).
    
    Args:
        model: Keras modeli
        save_path (str): TFLite model kayıt yolu
    """
    # TFLite converter oluştur
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimizasyonlar (boyut küçültme)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Float16 quantization (daha da küçük model)
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    tflite_model = converter.convert()
    
    # Kaydet
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    
    # Boyut bilgisi
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✓ TFLite modeli kaydedildi: {save_path} ({size_mb:.2f} MB)")


def train_model(csv_path='data/raw/labels.csv',
                data_dir='data/raw',
                epochs=50,
                batch_size=16,
                learning_rate=0.001,
                test_size=0.2,
                use_augmentation=True,
                model_save_path='models/model.h5'):
    """
    Ana eğitim fonksiyonu.
    
    Args:
        csv_path (str): Etiket CSV dosyası
        data_dir (str): Veri dizini
        epochs (int): Epoch sayısı
        batch_size (int): Batch boyutu
        learning_rate (float): Öğrenme oranı
        test_size (float): Validation oranı
        use_augmentation (bool): Veri artırma kullan
    """
    
    print("\n" + "=" * 80)
    print("🚀 MODEL EĞİTİMİ BAŞLIYOR")
    print("=" * 80 + "\n")
    
    # ============= 1. VERİ YÜKLEME =============
    print("📂 ADIM 1: Veri Yükleme")
    print("-" * 80)
    
    dataset = MeatDataset(data_dir=data_dir, img_size=(224, 224))
    image_paths, scores = dataset.load_from_csv(csv_path)
    
    print(f"✓ {len(image_paths)} görüntü yüklendi")
    
    # Veri ön işleme ve bölme
    X_train, X_val, y_train, y_val = dataset.load_and_preprocess(
        image_paths, scores, test_size=test_size
    )
    
    # ============= 2. MODEL OLUŞTURMA =============
    print("\n🏗️ ADIM 2: Model Oluşturma")
    print("-" * 80)
    
    model = create_meat_freshness_model(input_shape=(224, 224, 3))
    model = compile_model(model, learning_rate=learning_rate)
    
    model_params = get_model_summary(model)
    model_params['epochs'] = epochs
    model_params['batch_size'] = batch_size
    model_params['learning_rate'] = learning_rate
    model_params['augmentation'] = use_augmentation
    
    # ============= 3. CALLBACKS =============
    print("\n⚙️ ADIM 3: Callbacks Ayarlama")
    print("-" * 80)
    
    callbacks = []
    
    # Model checkpoint - en iyi modeli kaydet
    model_dir = os.path.dirname(model_save_path)
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(model_dir, 'best_model.h5')
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)
    print(f"✓ ModelCheckpoint: {checkpoint_path}")
    
    # Early stopping - overfitting önleme
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    print("✓ EarlyStopping: patience=10")
    
    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    print("✓ ReduceLROnPlateau: factor=0.5, patience=5")
    
    # TensorBoard - Devre dışı (Windows encoding sorunu)
    # log_dir = f"outputs/logs/fit_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # callbacks.append(tensorboard)
    # print(f"✓ TensorBoard: {log_dir}")
    print("⚠ TensorBoard devre dışı (encoding sorunu nedeniyle)")
    
    # ============= 4. EĞİTİM =============
    print("\n🎯 ADIM 4: Model Eğitimi")
    print("-" * 80)
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print()
    
    # TF Dataset oluştur
    train_dataset = create_tf_dataset(X_train, y_train, batch_size=batch_size, augment=use_augmentation)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size=batch_size, augment=False)
    
    # Eğit
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # ============= 5. SONUÇLAR =============
    print("\n💾 ADIM 5: Modeli Kaydetme")
    print("-" * 80)
    
    # Keras formatında kaydet
    # Keras formatında kaydet
    model.save(model_save_path)
    print(f"✓ Keras model kaydedildi: {model_save_path}")
    
    # SavedModel formatında kaydet
    # SavedModel formatında kaydet
    saved_model_path = os.path.join(model_dir, 'saved_model')
    # Eski SavedModel klasörünü temizle
    import shutil
    if os.path.exists(saved_model_path):
        shutil.rmtree(saved_model_path)
    model.save(saved_model_path)
    print(f"✓ SavedModel kaydedildi: {saved_model_path}")
    
    # TFLite formatına çevir (Raspberry Pi için)
    tflite_path = os.path.splitext(model_save_path)[0] + '.tflite'
    convert_to_tflite(model, save_path=tflite_path)
    
    # Grafikleri kaydet
    plot_training_history(history)
    
    # Rapor kaydet
    save_training_report(history, model_params)
    
    print("\n" + "=" * 80)
    print("✅ EĞİTİM TAMAMLANDI!")
    print("=" * 80)
    
    # Final metrikler
    print("\n📊 FINAL METRİKLER:")
    print(f"  Validation Loss (MSE): {history.history['val_loss'][-1]:.6f}")
    print(f"  Validation MAE: {history.history['val_mae'][-1]:.6f}")
    print(f"  Validation RMSE: {history.history['val_rmse'][-1]:.6f}")
    
    return model, history


if __name__ == "__main__":
    # Komut satırı argümanları
    parser = argparse.ArgumentParser(description='Et Bozulma Tahmin Modeli Eğitimi')
    
    parser.add_argument('--csv', type=str, default='data/raw/labels.csv',
                       help='CSV etiket dosyası yolu')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Veri dizini')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch boyutu')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Öğrenme oranı')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Validation oranı')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Veri artırma kullanma')
    
    args = parser.parse_args()
    
    # Eğitimi başlat
    train_model(
        csv_path=args.csv,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        test_size=args.test_size,
        use_augmentation=not args.no_augmentation,
        model_save_path='models/model.h5'  # Default for standalone run
    )
