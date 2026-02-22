import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

# Add src to path
sys.path.append('src')
from predict import load_trained_model, preprocess_image_for_prediction

def test_sensitivity_gradient():
    model_path = 'models/best_model.h5'
    if not os.path.exists(model_path): return
    
    model = load_trained_model(model_path)
    
    # Path to a fresh image
    fresh_path = 'data/raw/fresh/Fresh (1).jpg'
    if not os.path.exists(fresh_path):
        print("Test image missing")
        return

    img = cv2.imread(fresh_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # We will simulate "spoilage" by slowly shifting colors towards a grayish/greenish tint 
    # and adding some dark texture noise.
    
    results = []
    print("Testing Sensitivity Gradient (Artificial Spoilage):")
    
    for i in range(11):
        factor = i / 10.0 # 0.0 to 1.0 intensity of "spoilage"
        
        # Modify the image: darken and desaturate
        degraded = img.astype(np.float32)
        # Shift towards a more "spoiled" color profile (empirical simplification)
        degraded[:,:,0] *= (1.0 - factor * 0.3) # Less Red
        degraded[:,:,1] *= (1.0 - factor * 0.1) # Mixed Green
        degraded[:,:,2] *= (1.0 - factor * 0.4) # Significantly Less Blue
        
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        
        # Preprocess and predict
        inp = preprocess_image_for_prediction(degraded)
        # Using prediction without UI calibration to see raw sensitivity
        raw_score = model.predict(inp, verbose=0)[0][0]
        
        line = f"Spoilage Level {factor*100:3.0f}%: Model Score = {raw_score:.4f}"
        print(line)
        results.append(line)

    with open('sensitivity_test.txt', 'w') as f:
        f.write("Model Sensitivity Gradient Proof:\n")
        f.write("\n".join(results))

if __name__ == "__main__":
    test_sensitivity_gradient()
