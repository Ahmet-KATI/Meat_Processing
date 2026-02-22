import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sys

# Add src to path
sys.path.append('src')
from predict import load_trained_model, preprocess_image_for_prediction

def analyze_distribution():
    model_path = 'models/best_model.h5'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model = load_trained_model(model_path)
    df = pd.read_csv('data/raw/labels.csv')
    
    # Load all images into memory (approx 3k * 224 * 224 * 3 * 4 bytes = 1.8 GB)
    # This might be tight, so let's do it in chunks of 500
    all_scores = []
    chunk_size = 500
    
    print(f"Analyzing {len(df)} images in chunks...")
    
    for start_idx in range(0, len(df), chunk_size):
        end_idx = min(start_idx + chunk_size, len(df))
        chunk_df = df.iloc[start_idx:end_idx]
        
        chunk_images = []
        for _, row in chunk_df.iterrows():
            img_path = os.path.join('data/raw', row['image_path'])
            if os.path.exists(img_path):
                img = preprocess_image_for_prediction(img_path)
                chunk_images.append(img[0]) # Remove batch dim added by preprocess
        
        if chunk_images:
            chunk_array = np.array(chunk_images)
            predictions = model.predict(chunk_array, verbose=0)
            all_scores.extend(predictions.flatten())
            print(f"  Processed {end_idx}/{len(df)}")

    scores = np.array(all_scores)
    
    print("\n--- Prediction Distribution Analysis ---")
    print(f"Min Score: {scores.min():.4f}")
    print(f"Max Score: {scores.max():.4f}")
    print(f"Mean Score: {scores.mean():.4f}")
    
    # Count in ranges
    fresh = np.sum(scores <= 0.33)
    medium = np.sum((scores > 0.33) & (scores <= 0.67))
    spoiled = np.sum(scores > 0.67)
    
    print(f"Fresh (<=0.33): {fresh} ({fresh/len(scores)*100:.1f}%)")
    print(f"Medium (0.33-0.67): {medium} ({medium/len(scores)*100:.1f}%)")
    print(f"Spoiled (>0.67): {spoiled} ({spoiled/len(scores)*100:.1f}%)")
    
    # Find some intermediate scores if they exist
    intermediate_indices = np.where((scores > 0.2) & (scores < 0.8))[0]
    print(f"\nFound {len(intermediate_indices)} images with intermediate scores (0.2-0.8)")
    
    if len(intermediate_indices) > 0:
        print("Sample intermediate scores:")
        for idx in intermediate_indices[:5]:
            print(f"  {df.iloc[idx]['image_path']}: {scores[idx]:.4f}")

    # Save results
    with open('score_distribution.txt', 'w') as f:
        f.write(f"Distribution of {len(scores)} images:\n")
        f.write(f"Ranges: Fresh: {fresh}, Medium: {medium}, Spoiled: {spoiled}\n")
        f.write(f"Intermediate scores (0.1 to 0.9 range count): {np.sum((scores > 0.1) & (scores < 0.9))}\n")
        if len(intermediate_indices) > 0:
            f.write("\nSamples:\n")
            for idx in intermediate_indices[:10]:
                 f.write(f"{df.iloc[idx]['image_path']}: {scores[idx]:.4f}\n")

if __name__ == "__main__":
    analyze_distribution()
