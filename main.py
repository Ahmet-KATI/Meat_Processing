"""
Et Bozulma Tespit Sistemi - Ana Çalıştırma Dosyası

Bu dosya, projenin tüm özelliklerini tek noktadan çalıştırmak için kullanılır.

Kullanım:
    python main.py --mode [visualize|train|predict|desktop] [opsiyonlar]

Modlar:
    - visualize: Veri seti görselleştirme ve istatistikler
    - train: Model eğitimi
    - predict: Tek görsel için tahmin
    - desktop: Masaüstü UI başlatma
    - prepare_data: Klasör yapısından CSV oluşturma
"""

import os
import sys
import argparse

# Modülleri ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ui'))


def visualize_dataset(csv_path, data_dir):
    """Veri seti görselleştirme."""
    from src.visualization import (
        show_dataset_statistics,
        plot_score_distribution,
        plot_category_distribution,
        plot_sample_images
    )
    
    print("\n" + "=" * 80)
    print("📊 VERİ SETİ GÖRSELLEŞTİRME")
    print("=" * 80 + "\n")
    
    if not os.path.exists(csv_path):
        print(f"⚠ CSV dosyası bulunamadı: {csv_path}")
        print("Önce 'prepare_data' modunu çalıştırın.")
        return
    
    # İstatistikler
    show_dataset_statistics(csv_path)
    
    # Grafikler
    print("\n📈 Grafikler oluşturuluyor...")
    plot_score_distribution(csv_path)
    plot_category_distribution(csv_path)
    plot_sample_images(data_dir, csv_path, num_samples=12)
    
    print("\n✅ Görselleştirme tamamlandı!")
    print("Grafikler 'outputs/plots/' klasöründe")


def prepare_data(data_dir, folder_mapping, output_csv):
    """Klasör yapısından CSV oluşturur."""
    from src.data_utils import MeatDataset
    
    print("\n" + "=" * 80)
    print("📁 VERİ SETİ HAZIRLAMA")
    print("=" * 80 + "\n")
    
    dataset = MeatDataset(data_dir=data_dir)
    
    print(f"Klasör eşleştirmeleri:")
    for folder, score in folder_mapping.items():
        print(f"  {folder} → {score}")
    
    print()
    df = dataset.create_csv_from_folders(folder_mapping, output_csv=output_csv)
    
    print(f"\n✅ CSV dosyası oluşturuldu: {output_csv}")
    print(f"   Toplam {len(df)} görüntü")


def train_model(csv_path, data_dir, epochs, batch_size, lr, model_path):
    """Model eğitimi."""
    from src.train import train_model
    
    print("\n" + "=" * 80)
    print("🚀 MODEL EĞİTİMİ")
    print("=" * 80 + "\n")
    
    train_model(
        csv_path=csv_path,
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        model_save_path=model_path
    )


def predict_single(image_path, model_path):
    """Tek görsel için tahmin."""
    from src.predict import load_trained_model, predict_freshness
    
    print("\n" + "=" * 80)
    print("🔍 TAHMİN")
    print("=" * 80 + "\n")
    
    if not os.path.exists(image_path):
        print(f"⚠ Görüntü bulunamadı: {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"⚠ Model bulunamadı: {model_path}")
        print("Önce modeli eğitin: python main.py --mode train")
        return
    
    # Model yükle
    model = load_trained_model(model_path)
    
    # Tahmin
    result = predict_freshness(model, image_path)
    
    # Sonuç göster
    print(f"📸 Görüntü: {image_path}")
    print(f"🎯 Bozulma Skoru: {result['score']:.4f}")
    print(f"📋 Kategori: {result['category'].upper()}")
    print(f"💬 Sonuç: {result['label']}")
    
    # ASCII bar
    bar_length = 50
    filled = int(bar_length * result['score'])
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\n{bar}")
    print("TAZE                        ORTA                       BOZUK\n")


def run_desktop_app(model_path):
    """Masaüstü UI başlat."""
    from ui.desktop_app import main
    
    print("\n" + "=" * 80)
    print("🖥️ MASAÜSTÜ UYGULAMASI BAŞLATILIYOR")
    print("=" * 80 + "\n")
    
    main()


def main():
    """Ana fonksiyon."""
    
    # Ana parser
    parser = argparse.ArgumentParser(
        description='Et Bozulma Tespit Sistemi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Veri hazırlama
  python main.py --mode prepare_data
  
  # Veri görselleştirme
  python main.py --mode visualize
  
  # Model eğitimi
  python main.py --mode train --epochs 50 --batch_size 16
  
  # Tek görsel tahmini
  python main.py --mode predict --image data/raw/images/test.jpg
  
  # Masaüstü UI
  python main.py --mode desktop
        """
    )
    
    # Genel argümanlar
    parser.add_argument('--mode', type=str, required=True,
                       choices=['prepare_data', 'visualize', 'train', 'predict', 'desktop'],
                       help='Çalışma modu')
    
    parser.add_argument('--csv', type=str, default='data/raw/labels.csv',
                       help='CSV dosya yolu')
    
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Veri dizini')
    
    parser.add_argument('--model', type=str, default='models/model.h5',
                       help='Model dosya yolu')
    
    # prepare_data için
    parser.add_argument('--folders', type=str, nargs='+',
                       help='Klasör adları (ör: fresh medium spoiled)')
    
    parser.add_argument('--scores', type=float, nargs='+',
                       help='Klasör skorları (ör: 0.0 0.5 1.0)')
    
    # train için
    parser.add_argument('--epochs', type=int, default=50,
                       help='Epoch sayısı')
    
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch boyutu')
    
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Öğrenme oranı')
    
    # predict için
    parser.add_argument('--image', type=str,
                       help='Tahmin için görüntü yolu')
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "=" * 80)
    print("🥩 ET BOZULMA TESPİT SİSTEMİ")
    print("=" * 80)
    
    # Moda göre çalıştır
    if args.mode == 'prepare_data':
        if args.folders and args.scores:
            if len(args.folders) != len(args.scores):
                print("⚠ Hata: Klasör ve skor sayıları eşit olmalı!")
                return
            
            folder_mapping = dict(zip(args.folders, args.scores))
        else:
            # Varsayılan mapping
            print("⚠ Klasör mapping belirtilmedi, varsayılan kullanılıyor:")
            folder_mapping = {
                'fresh': 0.0,
                'medium': 0.5,
                'spoiled': 1.0
            }
        
        prepare_data(args.data_dir, folder_mapping, args.csv)
    
    elif args.mode == 'visualize':
        visualize_dataset(args.csv, args.data_dir)
    
    elif args.mode == 'train':
        train_model(args.csv, args.data_dir, args.epochs, args.batch_size, args.lr, args.model)
    
    elif args.mode == 'predict':
        if not args.image:
            print("⚠ Hata: --image argümanı gerekli!")
            return
        
        predict_single(args.image, args.model)
    
    elif args.mode == 'desktop':
        run_desktop_app(args.model)
    
    print("\n" + "=" * 80)
    print("✅ TAMAMLANDI")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
