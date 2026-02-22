import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
csv_path = r'c:\Users\ahmet\OneDrive\Masaüstü\Projects\meat_virtual_image_processing\outputs\reports\training_history_best.csv'
plot_path = r'c:\Users\ahmet\OneDrive\Masaüstü\Projects\meat_virtual_image_processing\outputs\plots\training_history_best.png'

print(f"Reading logs from: {csv_path}")
df = pd.read_csv(csv_path)

# Ensure plot directory exists
os.makedirs(os.path.dirname(plot_path), exist_ok=True)

# Plotting
plt.figure(figsize=(12, 8))

# Loss Plot (İstediğiniz asıl grafik bu)
plt.plot(df['epoch'], df['loss'], label='Training Loss (Egitim Kaybi)', linewidth=2)
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss (Dogrulama Kaybi)', linewidth=2)

# En iyi noktayı işaretle (Epoch 21)
best_val_loss = df.loc[21, 'val_loss']
plt.scatter(21, best_val_loss, color='red', s=100, zorder=5, label=f'Best Epoch: 21 (Loss: {best_val_loss:.4f})')
plt.annotate('Optimum Nokta', xy=(21, best_val_loss), xytext=(25, best_val_loss + 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('Egitim ve Dogrulama Kaybi (Loss vs Val_Loss)')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (Loss - MSE)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(plot_path)
print(f"✓ Grafik başarıyla oluşturuldu: {plot_path}")
