"""
Et bozulma tahmini için CNN model mimarisi.
MobileNetV2 tabanlı transfer learning ile hafif ve etkili model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def create_meat_freshness_model(input_shape=(224, 224, 3), 
                                 trainable_base_layers=0,
                                 dropout_rate=0.3):
    """
    Et tazeliği tahmini için regresyon modeli oluşturur.
    MobileNetV2 kullanarak transfer learning yapar.
    
    Args:
        input_shape (tuple): Giriş görüntü boyutu (height, width, channels)
        trainable_base_layers (int): Base model'den kaç katman eğitilebilir (0 = hepsi frozen)
        dropout_rate (float): Dropout oranı (overfitting önlemek için)
    
    Returns:
        keras.Model: Derlenmiş model
    """
    
    # Base model: MobileNetV2 (ImageNet weights ile)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,  # Classification head'i dahil etme
        weights='imagenet'
    )
    
    # Base model'i freeze et (transfer learning için)
    base_model.trainable = False
    
    # Eğer belirtilmişse, bazı katmanları trainable yap (fine-tuning için)
    if trainable_base_layers > 0:
        # Son N katmanı trainable yap
        for layer in base_model.layers[-trainable_base_layers:]:
            layer.trainable = True
        print(f"✓ Base model'in son {trainable_base_layers} katmanı trainable yapıldı")
    else:
        print("✓ Base model tamamen frozen (transfer learning)")
    
    # Custom head (regresyon için)
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing (MobileNetV2 için [-1, 1] normalizasyonu)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='batch_norm')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    
    # Output layer (regresyon: 0-1 arası skor)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Model oluştur
    model = models.Model(inputs=inputs, outputs=outputs, name='MeatFreshnessModel')
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Modeli derler (compile).
    
    Args:
        model (keras.Model): Derlenecek model
        learning_rate (float): Öğrenme oranı
    
    Returns:
        keras.Model: Derlenmiş model
    """
    
    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Loss: Mean Squared Error (regresyon için)
    loss = keras.losses.MeanSquaredError()
    
    # Metrics
    metrics = [
        keras.metrics.MeanAbsoluteError(name='mae'),
        keras.metrics.MeanSquaredError(name='mse'),
        keras.metrics.RootMeanSquaredError(name='rmse')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print("✓ Model derlendi")
    print(f"  Optimizer: Adam (lr={learning_rate})")
    print(f"  Loss: Mean Squared Error")
    print(f"  Metrics: MAE, MSE, RMSE")
    
    return model


def get_model_summary(model):
    """
    Model özetini gösterir.
    
    Args:
        model (keras.Model): Model
    """
    print("\n" + "=" * 70)
    print("📋 MODEL ÖZETİ")
    print("=" * 70)
    
    model.summary()
    
    # Parametre sayıları
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print("\n" + "=" * 70)
    print(f"📊 Toplam parametreler: {total_params:,}")
    print(f"🔓 Eğitilebilir parametreler: {trainable_params:,}")
    print(f"🔒 Eğitilemez parametreler: {non_trainable_params:,}")
    print("=" * 70 + "\n")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params
    }


def create_lightweight_model(input_shape=(224, 224, 3)):
    """
    Daha hafif bir custom CNN modeli (Raspberry Pi için alternatif).
    MobileNetV2 yerine sıfırdan küçük bir CNN.
    
    Args:
        input_shape (tuple): Giriş görüntü boyutu
    
    Returns:
        keras.Model: Derlenmiş hafif model
    """
    
    inputs = keras.Input(shape=input_shape)
    
    # Normalizasyon
    x = layers.Rescaling(1./255)(inputs)
    
    # Conv Block 1
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Conv Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Conv Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Conv Block 4
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='LightweightMeatModel')
    
    print("✓ Hafif custom CNN modeli oluşturuldu")
    
    return model


if __name__ == "__main__":
    # Test kodu
    print("🧪 Model Test\n")
    
    # Model oluştur
    model = create_meat_freshness_model()
    
    # Derle
    model = compile_model(model)
    
    # Özet
    get_model_summary(model)
    
    # Test prediction
    import numpy as np
    test_input = np.random.rand(1, 224, 224, 3)
    prediction = model.predict(test_input, verbose=0)
    print(f"Test prediction: {prediction[0][0]:.4f}")
