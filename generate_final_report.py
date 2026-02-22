import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
csv_path = r'c:\Users\ahmet\OneDrive\Masaüstü\Projects\meat_virtual_image_processing\outputs\reports\training_history.csv'
plot_path = r'c:\Users\ahmet\OneDrive\Masaüstü\Projects\meat_virtual_image_processing\outputs\plots\training_history.png'

print(f"Reading logs from: {csv_path}")
if not os.path.exists(csv_path):
    print("Error: CSV file not found!")
    exit(1)

df = pd.read_csv(csv_path)

# Ensure plot directory exists
os.makedirs(os.path.dirname(plot_path), exist_ok=True)

# Plotting
plt.figure(figsize=(15, 10))

# Loss Plot
plt.subplot(2, 2, 1)
plt.plot(df['epoch'], df['loss'], label='Training Loss')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# MAE Plot
plt.subplot(2, 2, 2)
plt.plot(df['epoch'], df['mae'], label='Training MAE')
plt.plot(df['epoch'], df['val_mae'], label='Validation MAE')
plt.title('MAE History')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

# MSE Plot
plt.subplot(2, 2, 3)
plt.plot(df['epoch'], df['mse'], label='Training MSE')
plt.plot(df['epoch'], df['val_mse'], label='Validation MSE')
plt.title('MSE History')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# RMSE Plot
plt.subplot(2, 2, 4)
plt.plot(df['epoch'], df['rmse'], label='Training RMSE')
plt.plot(df['epoch'], df['val_rmse'], label='Validation RMSE')
plt.title('RMSE History')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(plot_path)
print(f"✓ Plots saved to: {plot_path}")

# Find best epoch in THIS run
best_epoch_idx = df['val_loss'].idxmin()
best_epoch = df.loc[best_epoch_idx, 'epoch']
best_val_loss = df.loc[best_epoch_idx, 'val_loss']

print(f"\nReport for Current Run:")
print(f"Total Epochs: {len(df)}")
print(f"Best Epoch in this run: {int(best_epoch)}")
print(f"Best Val Loss in this run: {best_val_loss:.6f}")
