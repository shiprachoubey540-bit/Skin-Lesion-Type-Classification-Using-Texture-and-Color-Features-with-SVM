import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import io

# --- 1. ASSET LOADING ---
st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load('models/svm.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_assets()

classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis', 
           'Dermatofibroma', 'Melanoma', 'Nevus', 'Vascular lesions']

# --- 2. FORCED FRESH FEATURE EXTRACTION ---
def extract_features(img_input):
    # Ensure it's a fresh copy
    img = np.copy(img_input)
    img_resized = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # 32 bins for H and S = 64 features
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 256]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    color_feats = np.concatenate([h_hist, s_hist])

    # 6 Texture features
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    texture_feats = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0]
    ]

    return np.hstack([color_feats, texture_feats]).reshape(1, -1)

# --- 3. UI & LOGIC ---
st.title("🔬 Skin Lesion Classification")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="file_uroller")

if uploaded_file is not None:
    # FORCE CLEAR PREVIOUS DATA
    uploaded_file.seek(0) 
    bytes_data = uploaded_file.read()
    image = Image.open(io.BytesIO(bytes_data)).convert('RGB')
    
    st.image(image, caption=f"Processing: {uploaded_file.name}", use_container_width=True)
    
    # We use the file name in the button key to force Streamlit to refresh the state
    if st.button('Perform Classification', key=f"btn_{uploaded_file.name}"):
        with st.spinner('Calculating fresh features...'):
            img_array = np.array(image)
            
            # Get fresh prediction
            current_features = extract_features(img_array)
            scaled = scaler.transform(current_features)
            
            prediction = model.predict(scaled)[0]
            probs = model.predict_proba(scaled)[0]
            
            st.success(f"Final Prediction: {classes[prediction]}")
            st.metric("Confidence Level", f"{max(probs)*100:.2f}%")
            
            # Show the probability distribution to see the variation
            st.bar_chart(dict(zip(classes, probs)))
            
            # Clean up memory
            del img_array
            del current_features