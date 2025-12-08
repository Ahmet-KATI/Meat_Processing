"""
Tkinter ile masaüstü uygulaması.
Et bozulma tahmini için kullanıcı dostu arayüz.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

# src modülünü import etmek için path ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.predict import load_trained_model, predict_freshness
from ui.components import FreshnessBar, ImagePreview


class MeatFreshnessApp:
    """
    Et Bozulma Tespit Sistemi - Masaüstü Uygulaması
    """
    
    def __init__(self, root, model_path='models/model.h5'):
        """
        Args:
            root: tkinter.Tk() root window
            model_path (str): Model dosya yolu
        """
        self.root = root
        self.root.title("🥩 Et Bozulma Tespit Sistemi")
        self.root.geometry("900x800")
        self.root.configure(bg='#ecf0f1')
        
        # Model yükleme
        self.model = None
        self.model_path = model_path
        self.load_model()
        
        # Değişkenler
        self.current_image_path = None
        self.current_photo = None
        
        # UI oluştur
        self.create_ui()
    
    def load_model(self):
        """Modeli yükler."""
        if not os.path.exists(self.model_path):
            messagebox.showerror(
                "Model Bulunamadı",
                f"Model dosyası bulunamadı: {self.model_path}\n\n"
                "Lütfen önce modeli eğitin:\n"
                "python src/train.py"
            )
            return
        
        try:
            self.model = load_trained_model(self.model_path)
            print(f"✓ Model başarıyla yüklendi")
        except Exception as e:
            messagebox.showerror(
                "Model Yükleme Hatası",
                f"Model yüklenirken hata oluştu:\n{str(e)}"
            )
    
    def create_ui(self):
        """Ana UI'ı oluşturur."""
        # ===== HEADER =====
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🥩 ET BOZULMA TESPİT SİSTEMİ",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(expand=True)
        
        # ===== MAIN CONTENT =====
        content_frame = tk.Frame(self.root, bg='#ecf0f1')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        # --- Görüntü Seçme Butonu ---
        button_frame = tk.Frame(content_frame, bg='#ecf0f1')
        button_frame.pack(pady=(0, 20))
        
        self.select_btn = tk.Button(
            button_frame,
            text="📁 Görsel Seç",
            command=self.select_image,
            font=('Arial', 14, 'bold'),
            bg='#3498db',
            fg='white',
            padx=30,
            pady=15,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3
        )
        self.select_btn.pack(side=tk.LEFT, padx=10)
        
        self.predict_btn = tk.Button(
            button_frame,
            text="🔍 Tahmin Et",
            command=self.predict_image,
            font=('Arial', 14, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=30,
            pady=15,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3,
            state=tk.DISABLED  # Başlangıçta devre dışı
        )
        self.predict_btn.pack(side=tk.LEFT, padx=10)
        
        # --- Görüntü Önizleme ---
        preview_label = tk.Label(
            content_frame,
            text="Görüntü Önizleme:",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1'
        )
        preview_label.pack(anchor='w', pady=(10, 5))
        
        self.image_preview = ImagePreview(content_frame, width=600, height=400)
        self.image_preview.pack(pady=(0, 20))
        
        # --- Sonuç Bölümü ---
        result_frame = tk.Frame(content_frame, bg='white', relief=tk.RIDGE, bd=2)
        result_frame.pack(fill=tk.X, pady=(0, 20))
        
        result_header = tk.Label(
            result_frame,
            text="📊 Tahmin Sonucu",
            font=('Arial', 14, 'bold'),
            bg='white',
            pady=10
        )
        result_header.pack()
        
        # Freshness Bar
        self.freshness_bar = FreshnessBar(result_frame, width=600, height=80)
        self.freshness_bar.pack(pady=10)
        
        # Skor metni
        self.score_label = tk.Label(
            result_frame,
            text="Skor: --",
            font=('Arial', 12),
            bg='white'
        )
        self.score_label.pack(pady=5)
        
        # Sonuç metni
        self.result_label = tk.Label(
            result_frame,
            text="Henüz tahmin yapılmadı",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#7f8c8d',
            wraplength=700,
            pady=15
        )
        self.result_label.pack(pady=10)
        
        # ===== FOOTER =====
        footer_frame = tk.Frame(self.root, bg='#34495e', height=50)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        footer_frame.pack_propagate(False)
        
        footer_label = tk.Label(
            footer_frame,
            text="© 2025 Et Bozulma Tespit Sistemi | MobileNetV2 Tabanlı CNN",
            font=('Arial', 9),
            bg='#34495e',
            fg='white'
        )
        footer_label.pack(expand=True)
    
    def select_image(self):
        """Dosyadan görüntü seçer."""
        file_path = filedialog.askopenfilename(
            title="Et Görseli Seçin",
            filetypes=[
                ("Görüntü Dosyaları", "*.jpg *.jpeg *.png *.bmp"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if not file_path:
            return  # Kullanıcı iptal etti
        
        try:
            # Görüntüyü yükle ve göster
            self.current_image_path = file_path
            
            # PIL ile yükle
            img = Image.open(file_path)
            
            # Önizleme boyutuna göre resize
            max_size = (600, 400)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # PhotoImage oluştur
            self.current_photo = ImageTk.PhotoImage(img)
            
            # Önizlemede göster
            self.image_preview.display_image(self.current_photo)
            
            # Tahmin butonunu aktif et
            self.predict_btn.config(state=tk.NORMAL)
            
            # Önceki sonuçları temizle
            self.reset_results()
            
            print(f"✓ Görüntü seçildi: {file_path}")
            
        except Exception as e:
            messagebox.showerror(
                "Görüntü Yükleme Hatası",
                f"Görüntü yüklenirken hata oluştu:\n{str(e)}"
            )
    
    def predict_image(self):
        """Seçili görüntü için tahmin yapar."""
        if not self.current_image_path:
            messagebox.showwarning("Uyarı", "Lütfen önce bir görüntü seçin!")
            return
        
        if self.model is None:
            messagebox.showerror("Hata", "Model yüklenmedi!")
            return
        
        try:
            # Tahmin yap
            result = predict_freshness(self.model, self.current_image_path)
            
            score = result['score']
            category = result['category']
            label = result['label']
            
            # Bar'ı güncelle
            self.freshness_bar.update_score(score)
            
            # Skor metnini güncelle
            self.score_label.config(
                text=f"Bozulma Skoru: {score:.4f}",
                font=('Arial', 12, 'bold')
            )
            
            # Sonuç metnini güncelle
            if category == 'fresh':
                color = '#2ecc71'  # Yeşil
            elif category == 'medium':
                color = '#f39c12'  # Sarı
            else:
                color = '#e74c3c'  # Kırmızı
            
            self.result_label.config(
                text=label,
                fg=color,
                font=('Arial', 16, 'bold')
            )
            
            print(f"✓ Tahmin tamamlandı: Skor={score:.4f}, Kategori={category}")
            
        except Exception as e:
            messagebox.showerror(
                "Tahmin Hatası",
                f"Tahmin yapılırken hata oluştu:\n{str(e)}"
            )
    
    def reset_results(self):
        """Sonuç bölümünü sıfırlar."""
        self.freshness_bar.reset()
        self.score_label.config(text="Skor: --")
        self.result_label.config(
            text="Tahmin için 'Tahmin Et' butonuna tıklayın",
            fg='#7f8c8d',
            font=('Arial', 14, 'bold')
        )


def main():
    """Ana fonksiyon."""
    # Root window
    root = tk.Tk()
    
    # Model yolunu belirle
    model_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'models', 
        'best_model.h5'
    )
    
    # Uygulamayı başlat
    app = MeatFreshnessApp(root, model_path=model_path)
    
    # Event loop
    root.mainloop()


if __name__ == "__main__":
    main()
