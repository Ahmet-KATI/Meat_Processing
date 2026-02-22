# Derin Öğrenme Kavramları: Genel Uygulamalar vs. Bizim Projemiz

Bu doküman, gıda işleme ve doku analizi projelerinde kullanılan standart yöntemler ile `meat_virtual_image_processing` projesindeki özel tercihlerimizi karşılaştırır.

---

## 1. Görüntü Yeniden Boyutlandırma (Resize)

| Özellik | Genel Gıda Projeleri | Bizim Projemiz (Meat Processing) | Neden? |
| :--- | :--- | :--- | :--- |
| **Boyut** | 128x128 ile 512x512 arası | **224x224** | Et dokusundaki (texture) bozulmaları yakalamak için en verimli denge noktasıdır. |
| **Hız** | 128x128 (Çok hızlı) | Orta / Segmentasyon seviyesi | Çok küçük boyutlar etteki mikro renk değişimlerini "bulanıklaştırabilir". |
| **Model Uyumu** | Çeşitli | **MobileNetV2 Standartı** | Kullandığımız ana mimari bu boyutta eğitildiği için ağırlıklar tam uyum sağlar. |

**Proje Uygulaması:** [`src/data_utils.py`](file:///c:/Users/ahmet/OneDrive/Masaüstü/Projects/meat_virtual_image_processing/src/data_utils.py) içinde OpenCV `cv2.resize` ile sabitlenmiştir.

---

## 2. Öğrenme Oranı (Learning Rate) ve Epoch

| Kavram | Standart Yaklaşım | Bizim Projemiz | Avantajı |
| :--- | :--- | :--- | :--- |
| **Learning Rate** | Genelde sabittir (0.001) | **Dinamik / Azalan (Adaptive)** | `ReduceLROnPlateau` ile model zorlandığında hız keserek hedefi ıskalamaz. |
| **Epoch Stratejisi** | Sabit bir sayı (örn. 50) | **Otomatik Durdurmalı (EarlyStopping)** | Model gelişim göstermediği an eğitimi keser, zaman ve güç tasarrufu sağlar. |
| **Optimum Bulma** | Manuel Grafik Takibi | **Otomatik Geri Yükleme** | Eğitim dursa bile en iyi epoch'taki ağırlıkları bulur ve geri yükler. |

### Gerçek Eğitim Sonuçlarımız & Optimum Kanıtı:
Yaptığımız analizler sonucunda elde ettiğimiz en iyi verileri içeren özel grafiğimiz:

![En İyi Eğitim Grafiği](file:///c:/Users/ahmet/OneDrive/Masaüstü/Projects/meat_virtual_image_processing/outputs/plots/training_history_best.png)

- **Tespit Edilen Optimum:** **21. Epoch** (En düşük Validation Loss: `0.01358`)
- **Kanıt:** 21. epoch'tan sonra turuncu çizginin (validation) yükselmeye başlaması, modelin "ezberleme" (overfitting) evresine girdiğini kanıtlar.

---

## 3. Aktivasyon Fonksiyonları (ReLU ve Sigmoid)

| Fonksiyon | Genel Kullanım Alanı | Bizim Projemizdeki Rolü | Neden Kritik? |
| :--- | :--- | :--- | :--- |
| **ReLU** | Gizli Katmanlar | `dense_1` katmanında | Gıda dokusundaki karmaşık desenleri hızlı ve gürültüsüz öğrenmeyi sağlar. |
| **Sigmoid** | Sınıflandırma | **Çıkış Katmanında (Score)** | Çıktıyı 0-1 arasına hapseder. Bu, "Tazelik Skoru" için mükemmel bir metriktir. |

**Neden Sigmoid?** Diğer projelerde bazen "Softmax" (Kategorik) kullanılır. Ancak biz bir **derecelendirme** (0:Taze, 1:Bozuk) istediğimiz için Sigmoid kullanarak hassas bir skor elde ediyoruz.

---

> [!TIP]
> **Özet Karşılaştırma:** Genel projeler sadece "Bu ne?" sorusuna yanıt ararken (Sınıflandırma), bizim projemiz **"Ne kadar taze?"** sorusuna yanıt arar (Regresyon). Bu yüzden Sigmoid çıkışı ve 224x224 model uyumu projemizin bel kemiğidir.

---

## 🚨 Kritik Güncelleme: Çifte Normalizasyon (Double-Normalization) ve Çözümü

Modelin ilk versiyonlarında taze etler için bile "orta bozulmuş" (0.6 - 0.76) gibi yüksek skorlar ürettiği gözlemlenmiştir. Yapılan derinlemesine incelemede bunun **Çifte Normalizasyon** hatasından kaynaklandığı tespit edilmiştir.

### Sorun Neydi?
1. **Manuel Normalizasyon:** Kodda görüntüler yüklenirken manuel olarak 255'e bölünüyordu (0-1 aralığı).
2. **Model İçi Normalizasyon:** Kullanılan `MobileNetV2` mimarisi, kendi `preprocess_input` katmanıyla bu veriyi tekrar işliyor ve [-1, 1] aralığına çekmeye çalışıyordu.

Zaten normalize edilmiş (0-1) bir verinin tekrar normalize edilmesi, modelin karakteristik özelliklerini "ezmiş" ve ciddi bir tahmin sapmasına (bias) yol açmıştır.

### Çözüm
Manuel `/ 255.0` işlemi eğitim ve tahmin pipeline'ından kaldırılmıştır. Bu düzeltme sonrası model tekrar eğitilerek tahmin hassasiyeti normale döndürülmüştür.
