import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

# Add src to path
sys.path.append('src')
from predict import load_trained_model, predict_freshness

def test_bias():
    model_path = 'models/best_model.h5'
    if not os.path.exists(model_path):
        content = f"Model not found at {model_path}"
        print(content)
        with open('diag_results.txt', 'w') as f: f.write(content)
        return

    model = load_trained_model(model_path)
    
    test_samples = {
        'Fresh 1': 'data/raw/fresh/Fresh (1).jpg',
        'Fresh 100': 'data/raw/fresh/Fresh (100).jpg',
        'Rotten 1': 'data/raw/spoiled/Rotten (1).jpg',
        'Rotten 1640': 'data/raw/spoiled/Rotten (1640).jpg'
    }
    
    results_str = "--- Diagnostic Results ---\n"
    for name, path in test_samples.items():
        if os.path.exists(path):
            result = predict_freshness(model, path)
            line = f"{name}: Score = {result['score']:.4f} ({result['category']})\n"
            print(line, end='')
            results_str += line
        else:
            line = f"{name}: File not found at {path}\n"
            print(line, end='')
            results_str += line
            
    with open('diag_results.txt', 'w') as f:
        f.write(results_str)

if __name__ == "__main__":
    test_bias()
