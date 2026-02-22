"""
UI bileşenleri.
Yeşil-kırmızı gradient bar widget ve diğer UI elemanları.
"""

import tkinter as tk
from tkinter import Canvas
import math


class FreshnessBar(Canvas):
    """
    Et bozulma skorunu yeşilden kırmızıya gradient bar ile gösteren widget.
    """
    
    def __init__(self, parent, width=600, height=80, **kwargs):
        """
        Args:
            parent: Ana tkinter widget
            width (int): Bar genişliği
            height (int): Bar yüksekliği
        """
        super().__init__(parent, width=width, height=height, 
                        highlightthickness=0, **kwargs)
        
        self.bar_width = width
        self.bar_height = height
        self.current_score = 0.0
        
        # Gradient oluştur
        self._draw_gradient()
        
        # Etiketler
        self._draw_labels()
        
        # İndikatör (başlangıçta gizli)
        self.indicator = None
    
    def _draw_gradient(self):
        """Yeşil-sarı-kırmızı gradient çizer."""
        # Bar yüksekliği ve padding
        bar_y = 20
        bar_h = self.bar_height - 40
        
        # Gradient için ince dikdörtgenler çiz
        num_steps = self.bar_width
        
        for i in range(num_steps):
            # Normalize pozisyon (0.0 - 1.0)
            ratio = i / num_steps
            
            # Renk hesapla
            if ratio <= 0.5:
                # Yeşilden sarıya (0.0 - 0.5)
                r = int(46 + (241 - 46) * (ratio * 2))
                g = int(204 + (196 - 204) * (ratio * 2))
                b = int(113 + (15 - 113) * (ratio * 2))
            else:
                # Sarıdan kırmızıya (0.5 - 1.0)
                r = int(241 + (231 - 241) * ((ratio - 0.5) * 2))
                g = int(196 + (76 - 196) * ((ratio - 0.5) * 2))
                b = int(15 + (60 - 15) * ((ratio - 0.5) * 2))
            
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            # Çizgi çiz
            self.create_line(i, bar_y, i, bar_y + bar_h, 
                           fill=color, width=1)
        
        # Border
        self.create_rectangle(0, bar_y, self.bar_width, bar_y + bar_h, 
                            outline='black', width=2)
    
    def _draw_labels(self):
        """Alt kısma etiketler ekler."""
        label_y = self.bar_height - 10
        
        # Sol: Taze
        self.create_text(10, label_y, text="TAZE", 
                        anchor='w', font=('Arial', 10, 'bold'),
                        fill='#2ecc71')
        
        # Orta: Orta
        self.create_text(self.bar_width // 2, label_y, text="ORTA", 
                        anchor='center', font=('Arial', 10, 'bold'),
                        fill='#f39c12')
        
        # Sağ: Bozuk
        self.create_text(self.bar_width - 10, label_y, text="BOZUK", 
                        anchor='e', font=('Arial', 10, 'bold'),
                        fill='#e74c3c')
    
    def update_score(self, score):
        """
        Skora göre indikatörü günceller.
        
        Args:
            score (float): 0.0-1.0 arası bozulma skoru
        """
        self.current_score = max(0.0, min(1.0, score))
        
        # Önceki indikatör elemanlarını sil
        self.delete('indicator_element')
        
        # Yeni indikatör pozisyonu
        x_pos = int(self.current_score * self.bar_width)
        bar_y = 20
        bar_h = self.bar_height - 40
        
        # İndikatör çiz (dikey çizgi + üçgen)
        # Dikey çizgi
        self.indicator = self.create_line(
            x_pos, bar_y, x_pos, bar_y + bar_h,
            fill='black', width=4, tags='indicator_element'
        )
        
        # Üstte üçgen ok
        arrow_size = 8
        self.create_polygon(
            x_pos, bar_y - 10,
            x_pos - arrow_size, bar_y - 2,
            x_pos + arrow_size, bar_y - 2,
            fill='black', tags='indicator_element'
        )
        
        # Altta üçgen ok
        self.create_polygon(
            x_pos, bar_y + bar_h + 10,
            x_pos - arrow_size, bar_y + bar_h + 2,
            x_pos + arrow_size, bar_y + bar_h + 2,
            fill='black', tags='indicator_element'
        )
    
    def reset(self):
        """İndikatörü sıfırlar."""
        self.delete('indicator_element')
        self.indicator = None


class ImagePreview(Canvas):
    """
    Görüntü önizleme widget'ı.
    """
    
    def __init__(self, parent, width=400, height=300, **kwargs):
        """
        Args:
            parent: Ana tkinter widget
            width (int): Genişlik
            height (int): Yükseklik
        """
        super().__init__(parent, width=width, height=height,
                        bg='#ecf0f1', highlightthickness=1,
                        highlightbackground='#bdc3c7', **kwargs)
        
        self.preview_width = width
        self.preview_height = height
        self.current_image = None
        
        # Placeholder metin
        self.create_text(
            width // 2, height // 2,
            text="Görüntü seçilmedi\n\n'Görsel Seç' butonuna tıklayın",
            font=('Arial', 12),
            fill='#7f8c8d',
            justify='center',
            tags='placeholder'
        )
    
    def display_image(self, photo_image):
        """
        Görüntüyü gösterir.
        
        Args:
            photo_image: tkinter.PhotoImage objesi
        """
        # Önceki görüntüyü ve placeholder'ı temizle
        self.delete('all')
        
        # Yeni görüntüyü ortala ve göster
        self.current_image = self.create_image(
            self.preview_width // 2,
            self.preview_height // 2,
            image=photo_image,
            anchor='center'
        )
    
    def clear(self):
        """Görüntüyü temizler ve placeholder gösterir."""
        self.delete('all')
        self.create_text(
            self.preview_width // 2, self.preview_height // 2,
            text="Görüntü seçilmedi\n\n'Görsel Seç' butonuna tıklayın",
            font=('Arial', 12),
            fill='#7f8c8d',
            justify='center',
            tags='placeholder'
        )
        self.current_image = None


if __name__ == "__main__":
    # Test kodu
    root = tk.Tk()
    root.title("Components Test")
    root.geometry("700x500")
    root.configure(bg='white')
    
    # FreshnessBar test
    bar = FreshnessBar(root, width=600, height=80)
    bar.pack(pady=20)
    
    # Test skorları
    test_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    current_idx = [0]
    
    def update_bar():
        """Test için bar'ı günceller."""
        score = test_scores[current_idx[0]]
        bar.update_score(score)
        current_idx[0] = (current_idx[0] + 1) % len(test_scores)
    
    # Test butonu
    test_btn = tk.Button(root, text="Test Bar (Farklı Skorlar)", 
                        command=update_bar, 
                        font=('Arial', 12),
                        bg='#3498db', fg='white',
                        padx=20, pady=10)
    test_btn.pack(pady=10)
    
    # ImagePreview test
    preview = ImagePreview(root, width=400, height=300)
    preview.pack(pady=20)
    
    root.mainloop()
