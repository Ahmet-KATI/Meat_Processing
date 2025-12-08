
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

def test_save():
    print("Creating dummy model...")
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(5,)),
        keras.layers.Dense(1)
    ])
    
    save_path = 'test_output_debug/model.h5'
    model_dir = os.path.dirname(save_path)
    
    print(f"Creating directory: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Saving H5 model to {save_path}...")
    try:
        model.save(save_path)
        print("Success: H5 saved.")
    except Exception as e:
        print(f"Failed H5: {e}")

    saved_model_path = os.path.join(model_dir, 'saved_model')
    print(f"Saving SavedModel to {saved_model_path}...")
    if os.path.exists(saved_model_path):
        print(f"Path {saved_model_path} already exists. Is dir: {os.path.isdir(saved_model_path)}")
    
    try:
        model.save(saved_model_path)
        print("Success: SavedModel saved.")
    except Exception as e:
        print(f"Failed SavedModel: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_save()
