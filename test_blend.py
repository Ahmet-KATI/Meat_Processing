import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

# Add src to path
sys.path.append('src')
from predict import load_trained_model, preprocess_image_for_prediction

def test_regression_blending():
    model_path = 'models/best_model.h5'
    if not os.path.exists(model_path): return
    
    model = load_trained_model(model_path)
    
    # Path to one fresh and one spoiled image
    # Using specific samples that are likely representative
    fresh_path = 'data/raw/fresh/Fresh (1).jpg'
    spoiled_path = 'data/raw/spoiled/Rotten (1).jpg'
    
    if not os.path.exists(fresh_path) or not os.path.exists(spoiled_path):
        print("Test images missing")
        return

    # Load images and convert to RGB
    f_img = cv2.imread(fresh_path)
    f_img = cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB)
    
    s_img = cv2.imread(spoiled_path)
    s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)
    
    # Resize to same size for blending
    size = (224, 224)
    f_img = cv2.resize(f_img, size)
    s_img = cv2.resize(s_img, size)
    
    results = []
    
    print("Testing Alpha Blending (Fresh -> Spoiled) in RGB:")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # blended = (1-alpha)*fresh + alpha*spoiled
        blended = cv2.addWeighted(f_img, 1 - alpha, s_img, alpha, 0)
        
        # Preprocess manually (Passing RGB array directly)
        inp = preprocess_image_for_prediction(blended)
        
        prediction = model.predict(inp, verbose=0)[0][0]
        line = f"Alpha {alpha:.2f} (Spoiled Ratio): Score = {prediction:.4f}"
        print(line)
        results.append(line)

    with open('blending_test.txt', 'w') as f:
        f.write("Alpha Blending Proof of Regression (RGB Fixed):\n")
        f.write("\n".join(results))

if __name__ == "__main__":
    test_regression_blending()
