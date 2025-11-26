# Updated imports for Keras 3 compatibility
import keras
from keras.preprocessing import image
import numpy as np
import os
import joblib
import pandas as pd 

CLASSIFIER_NAME = "ResNet50" 
IMG_SIZE = 224 

# Paths
classifier_path = os.path.join("..", "models", f"{CLASSIFIER_NAME}_best.h5")
regressor_path = os.path.join("..", "models", "rf_regressor.joblib")
train_dir = os.path.join("..", "data", "train")

try:
    # Use keras.models.load_model
    classifier_model = keras.models.load_model(classifier_path)
    regressor_model = joblib.load(regressor_path)
    
    class_names = sorted(os.listdir(train_dir))
    
except Exception as e:
    print(f"Error loading models: {e}")
    print("Ensure all models are trained and present in the '../models' folder.")
    exit()

def predict_disease_and_yield(img_path):
    try:
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
    except FileNotFoundError:
        print(f"Error: Image not found at path: {img_path}")
        return

    pred_probabilities = classifier_model.predict(img_array)[0]
    class_index = np.argmax(pred_probabilities)
    confidence = np.max(pred_probabilities)
    
    predicted_disease = class_names[class_index]
    
    print("\n--- CLASSIFICATION RESULTS ---")
    print(f"Predicted Class: {predicted_disease}")
    print(f"Confidence: {confidence * 100:.2f}%")
    
    if "healthy" in predicted_disease.lower():
        severity_score = 0.1 
    elif confidence > 0.8:
        severity_score = 0.65 
    else:
        severity_score = 0.85 
        
    input_data = {
        'Severity_Score': [severity_score],
        'Crop_Type_Potato': [0],
        'Crop_Type_Tomato': [0],
    }
    
    if 'Potato' in predicted_disease:
        input_data['Crop_Type_Potato'] = [1]
    elif 'Tomato' in predicted_disease:
        input_data['Crop_Type_Tomato'] = [1]

    regressor_input = pd.DataFrame(input_data)

    yield_loss_prediction = regressor_model.predict(regressor_input)[0]
    
    print("\n--- REGRESSION RESULTS ---")
    print(f"Input Severity Score (Simulated): {severity_score:.2f}")
    print(f"Predicted Yield Loss: {yield_loss_prediction:.2f}%")
    print(f"Actionable Insight: Apply treatment for {predicted_disease} and monitor crop.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_leaf_image>")
        print("Example: python predict.py ../data/test/Tomato_Bacterial_spot/1.JPG")
        
    else:
        predict_disease_and_yield(sys.argv[1])