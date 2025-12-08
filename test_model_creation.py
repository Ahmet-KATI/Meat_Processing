
import os
import tensorflow as keras
from tensorflow.keras import layers, models
import sys
import tempfile
import shutil

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

print("Creating dummy model...")
model = models.Sequential([
    layers.Dense(10, input_shape=(5,), activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Create a temp file with .h5 extension
fd, temp_path = tempfile.mkstemp(suffix='.h5')
os.close(fd)

print(f"Temp path: {temp_path}")

try:
    print(f"Saving to {temp_path}...")
    model.save(temp_path)
    print("Save successful.")
    
    print("Loading back...")
    loaded_model = models.load_model(temp_path)
    print("Load successful.")
    
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
finally:
    if os.path.exists(temp_path):
        os.remove(temp_path)

print("Test Complete.")
