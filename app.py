import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import joblib
import pandas as pd
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Crop Disease & Yield Predictor",
    page_icon="🌱",
    layout="centered"
)

# --- HEADER ---
st.title("🌱 AI Crop Doctor")
st.markdown("Upload a leaf image to detect diseases and estimate potential yield loss.")

# --- SIDEBAR ---
st.sidebar.title("Options")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Load the Classifier (Ensure the filename matches what is in your models folder)
    # If your file is named 'ResNet50_best.h5', change this line!
    
    classifier_path = os.path.join("models", "ResNet50_best.h5") 
    regressor_path = os.path.join("models", "rf_regressor.joblib")
    
    
    if not os.path.exists(classifier_path):
        st.error(f"❌ Classifier model not found at: {classifier_path}")
        return None, None
        
    if not os.path.exists(regressor_path):
        st.error(f"❌ Regressor model not found at: {regressor_path}")
        return None, None

    model_clf = tf.keras.models.load_model(classifier_path)
    model_reg = joblib.load(regressor_path)
    return model_clf, model_reg

classifier, regressor = load_models()

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]
# NOTE: The list above is a standard PlantVillage list. 
# If your model predicts the wrong class names, check this list against your original 'data/train' folders.

def predict_yield_loss(disease_name, confidence_score, regressor_model):
    """
    Calculates yield loss based on the logic from predict.py
    """
    # 1. Determine Severity Score
    if "healthy" in disease_name.lower():
        severity_score = 0.1
    elif confidence_score > 0.8:
        severity_score = 0.65
    else:
        severity_score = 0.85

    # 2. Prepare Input Data for Regressor
    # Initialize all columns with 0
    input_data = {
        'Severity_Score': [severity_score],
        'Crop_Type_Pepper': [0],
        'Crop_Type_Potato': [0],
        'Crop_Type_Tomato': [0],
        'Crop_Type_Apple': [0],
        'Crop_Type_Coffee': [0],
    }

    # Set the specific crop flag to 1
    if 'Potato' in disease_name:
        input_data['Crop_Type_Potato'] = [1]
    elif 'Tomato' in disease_name:
        input_data['Crop_Type_Tomato'] = [1]
    elif 'Pepper' in disease_name:
        input_data['Crop_Type_Pepper'] = [1]
    elif 'Apple' in disease_name:
        input_data['Crop_Type_Apple'] = [1]
    elif 'Coffee' in disease_name:
        input_data['Crop_Type_Coffee'] = [1]
    
    # Create DataFrame
    df_input = pd.DataFrame(input_data)
    
    # Predict
    predicted_loss = regressor_model.predict(df_input)[0]
    return severity_score, predicted_loss

# --- MAIN APP INTERFACE ---
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and classifier is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button("Analyze Crop"):
        with st.spinner('Analyzing...'):
            # Preprocess Image
            size = (224, 224)
            image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image_resized)
            img_array = img_array / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

            # Predict Disease
            predictions = classifier.predict(img_array)
            class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Get Class Name (Safeguard against index out of bounds)
            if class_index < len(CLASS_NAMES):
                predicted_class = CLASS_NAMES[class_index]
            else:
                predicted_class = f"Unknown Class (Index {class_index})"

            # Display Results
            if confidence >= confidence_threshold:
                st.success(f"**Diagnosis:** {predicted_class}")
                st.info(f"**Confidence:** {confidence*100:.2f}%")
                
                # Run Regression Logic
                severity, loss = predict_yield_loss(predicted_class, confidence, regressor)
                
                st.markdown("---")
                st.subheader("📉 Yield Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Estimated Severity", f"{severity*100:.0f}%")
                with col2:
                    st.metric("Potential Yield Loss", f"{loss:.2f}%")
                
                if "healthy" not in predicted_class.lower():
                    st.warning("⚠️ Action Required: Applying treatment is recommended.")
                else:
                    st.balloons()
                    st.success("✅ Crop looks healthy!")
            else:
                st.warning(f"⚠️ Prediction Confidence ({confidence*100:.2f}%) is low. The model is unsure.")

