# Et Bozulma Tespit Sistemi - Örnek Veri Etiketleme

Bu dosya, veri setinizi etiketlemek için örnek bir şablon içerir.

## CSV Formatı

CSV dosyanız şu sütunları içermelidir:

```csv
image_path,freshness_score
```

- **image_path**: Görüntü dosyasının `data/raw/` dizinine göre göreceli yolu
- **freshness_score**: 0.0 (çok taze) ile 1.0 (tamamen bozuk) arası skor

## Örnek Etiketleme

```csv
image_path,freshness_score
images/meat_001.jpg,0.05
images/meat_002.jpg,0.12
images/meat_003.jpg,0.28
images/meat_004.jpg,0.45
images/meat_005.jpg,0.52
images/meat_006.jpg,0.68
images/meat_007.jpg,0.75
images/meat_008.jpg,0.89
images/meat_009.jpg,0.95
```

## Skor Rehberi

### Taze (0.00 - 0.33)
- Parlak kırmızı renk
- Hoş koku veya koku yok
- Sıkı doku
- Az veya hiç sıvı birikmesi yok

### Orta Seviye (0.33 - 0.67)
- Kahverengimsi-kırmızı renk
- Hafif ekşi koku
- Yumuşamış doku
- Orta düzeyde sıvı birikmesi

### Bozuk (0.67 - 1.00)
- Gri, yeşil veya kahverengi renk
- Kuvvetli kötü koku
- Çok yumuşak veya yapışkan doku
- Fazla sıvı birikmesi
- Görünür küf veya lekeler

## Kullanım

Bu dosyayı şuraya kaydedin:
```
data/raw/labels.csv
```

Sonra eğitimi başlatın:
```bash
python main.py --mode train
```
