
import os
import sys
import traceback

# Force UTF-8 for stdout/stderr just in case
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

try:
    from src.predict import load_trained_model, predict_freshness
except Exception as e:
    print(f"Import Error: {repr(e)}")
    sys.exit(1)

# Setup paths
model_path = 'models/best_model.h5'
image_path = 'data/raw/fresh/Fresh (1).jpg'

print(f"Testing prediction with model: {model_path}")
print(f"Testing on image: {image_path}")

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    sys.exit(1)

if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    sys.exit(1)

try:
    # Load model
    print("Loading model...")
    model = load_trained_model(model_path)
    print("Model loaded.")
    
    # Predict
    print("Predicting...")
    result = predict_freshness(model, image_path)
    
    print("\nPrediction Result:")
    print(f"Score: {result['score']}")
    print(f"Category: {result['category']}")
    print(f"Label: {result['label']}")
    print("SUCCESS")
    
except Exception as e:
    print(f"\nFAILED with error: {repr(e)}")
    traceback.print_exc()
