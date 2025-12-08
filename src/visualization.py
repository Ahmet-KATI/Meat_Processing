"""
Veri görselleştirme ve istatistiksel analiz fonksiyonları.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2


def show_dataset_statistics(csv_path):
    """
    Veri seti istatistiklerini gösterir.
    
    Args:
        csv_path (str): CSV dosya yolu
    """
    df = pd.read_csv(csv_path)
    
    print("=" * 60)
    print("📊 VERİ SETİ İSTATİSTİKLERİ")
    print("=" * 60)
    
    # Temel bilgiler
    print(f"\n📌 Genel Bilgiler:")
    print(f"  Toplam görüntü sayısı: {len(df)}")
    print(f"  Özellik sayısı: {df.shape[1]}")
    
    # Skor istatistikleri
    scores = df['freshness_score']
    print(f"\n📈 Bozulma Skoru İstatistikleri:")
    print(f"  Ortalama: {scores.mean():.4f}")
    print(f"  Std. Sapma: {scores.std():.4f}")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")
    print(f"  Medyan: {scores.median():.4f}")
    
    # Skor dağılımı (kategorize edilmiş)
    fresh_count = len(df[df['freshness_score'] <= 0.33])
    medium_count = len(df[(df['freshness_score'] > 0.33) & (df['freshness_score'] <= 0.67)])
    spoiled_count = len(df[df['freshness_score'] > 0.67])
    
    print(f"\n🎯 Kategorik Dağılım:")
    print(f"  Taze (0.00-0.33): {fresh_count} (%{fresh_count/len(df)*100:.1f})")
    print(f"  Orta (0.33-0.67): {medium_count} (%{medium_count/len(df)*100:.1f})")
    print(f"  Bozuk (0.67-1.00): {spoiled_count} (%{spoiled_count/len(df)*100:.1f})")
    
    print("=" * 60)
    
    return df


def plot_score_distribution(csv_path, save_path='outputs/plots/score_distribution.png'):
    """
    Skor dağılımını histogram ve box plot ile gösterir.
    
    Args:
        csv_path (str): CSV dosya yolu
        save_path (str): Grafik kayıt yolu
    """
    df = pd.read_csv(csv_path)
    scores = df['freshness_score']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Ortalama: {scores.mean():.3f}')
    axes[0].axvline(scores.median(), color='green', linestyle='--', linewidth=2, label=f'Medyan: {scores.median():.3f}')
    axes[0].set_xlabel('Bozulma Skoru', fontsize=12)
    axes[0].set_ylabel('Frekans', fontsize=12)
    axes[0].set_title('Bozulma Skoru Dağılımı', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot
    axes[1].boxplot(scores, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightcoral', alpha=0.7),
                    medianprops=dict(color='darkred', linewidth=2))
    axes[1].set_ylabel('Bozulma Skoru', fontsize=12)
    axes[1].set_title('Bozulma Skoru Box Plot', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Kaydet
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Grafik kaydedildi: {save_path}")
    
    plt.show()


def plot_sample_images(data_dir, csv_path, num_samples=12, save_path='outputs/plots/sample_images.png'):
    """
    Rastgele örnek görüntüleri ve skorlarını gösterir.
    
    Args:
        data_dir (str): Veri dizini
        csv_path (str): CSV dosya yolu
        num_samples (int): Gösterilecek örnek sayısı
        save_path (str): Grafik kayıt yolu
    """
    df = pd.read_csv(csv_path)
    
    # Rastgele örnekler seç
    samples = df.sample(n=min(num_samples, len(df)))
    
    # Grid boyutunu hesapla
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= num_samples:
            break
        
        # Görüntüyü yükle
        img_path = os.path.join(data_dir, row['image_path'])
        
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Skor ve kategori
            score = row['freshness_score']
            if score <= 0.33:
                category = "TAZE"
                color = 'green'
            elif score <= 0.67:
                category = "ORTA"
                color = 'orange'
            else:
                category = "BOZUK"
                color = 'red'
            
            # Göster
            axes[idx].imshow(img)
            axes[idx].set_title(f'{category}\nSkor: {score:.3f}', 
                              fontsize=12, fontweight='bold', color=color)
            axes[idx].axis('off')
            
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Yüklenemedi\n{e}', 
                          ha='center', va='center', fontsize=10)
            axes[idx].axis('off')
    
    # Boş axis'leri gizle
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Örnek Et Görüntüleri ve Bozulma Skorları', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Kaydet
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Örnek görüntüler kaydedildi: {save_path}")
    
    plt.show()


def plot_category_distribution(csv_path, save_path='outputs/plots/category_distribution.png'):
    """
    Kategorik dağılımı pasta grafiği ile gösterir.
    
    Args:
        csv_path (str): CSV dosya yolu
        save_path (str): Grafik kayıt yolu
    """
    df = pd.read_csv(csv_path)
    
    # Kategorilere ayır
    fresh = len(df[df['freshness_score'] <= 0.33])
    medium = len(df[(df['freshness_score'] > 0.33) & (df['freshness_score'] <= 0.67)])
    spoiled = len(df[df['freshness_score'] > 0.67])
    
    # Pasta grafiği
    labels = ['Taze\n(0.00-0.33)', 'Orta\n(0.33-0.67)', 'Bozuk\n(0.67-1.00)']
    sizes = [fresh, medium, spoiled]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.05, 0.05, 0.05)
    
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    plt.title('Et Bozulma Kategorileri Dağılımı', fontsize=16, fontweight='bold', pad=20)
    
    # Kaydet
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Kategori dağılımı grafiği kaydedildi: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test kodu
    print("🧪 Visualization Test")
    
    csv_path = 'data/raw/labels.csv'
    
    if os.path.exists(csv_path):
        show_dataset_statistics(csv_path)
        # plot_score_distribution(csv_path)
        # plot_category_distribution(csv_path)
        # plot_sample_images('data/raw', csv_path)
    else:
        print(f"⚠ CSV dosyası bulunamadı: {csv_path}")
