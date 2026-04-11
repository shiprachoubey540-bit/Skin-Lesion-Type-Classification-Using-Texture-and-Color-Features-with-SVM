import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import io
import pandas as pd # Added for better chart handling

# --- 1. ASSET LOADING ---
# Updated page config for a more professional feel
st.set_page_config(
    page_title="Skin Lesion AI Diagnostic", 
    page_icon="🔬", 
    layout="wide"
)

# Custom CSS for a clean "Medical Dashboard" look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stAlert { border-radius: 10px; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model  = joblib.load('models/svm.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_assets()

classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
           'Dermatofibroma', 'Melanoma', 'Nevus', 'Vascular lesions']

# --- 2. FEATURE EXTRACTION (Logic preserved exactly) ---
def extract_features(img_input):
    img = np.copy(img_input)
    img_resized = cv2.resize(img, (224, 224))
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    glcm = graycomatrix(gray, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    glcm_feats = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        glcm_feats.extend(graycoprops(glcm, prop).flatten().tolist())

    hsv_feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [16], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-7)
        hsv_feats.extend(hist.tolist())

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    flip_h = cv2.flip(mask, 0)
    flip_v = cv2.flip(mask, 1)
    asym_h = np.sum(mask != flip_h) / (mask.size + 1e-7)
    asym_v = np.sum(mask != flip_v) / (mask.size + 1e-7)
    asymmetry = float(np.clip((asym_h + asym_v) / 2, 0, 1))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        border = float(np.clip((perimeter ** 2) / (4 * np.pi * area + 1e-7), 1, 50))
    else:
        border = 1.0

    return np.array(glcm_feats + hsv_feats + [asymmetry, border]).reshape(1, -1)

# --- 3. UI LAYOUT ---

# Sidebar for Project Details
with st.sidebar:
    st.title("About Project")
    st.info("""
    **Multi-Feature Extraction:**
    - 20 Texture Features (GLCM)
    - 48 Color Features (HSV)
    - 2 Shape Features (ABCD)
    """)
    st.markdown("---")
    st.write("🏥 **Clinical Decision Support System**")

# Main Content Header
st.title("🔬 Skin Lesion Classification System")
st.write("Artificial Intelligence for Dermatological Image Analysis")
st.divider()

# Layout: Two columns for Input and Output
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Image Input")
    uploaded_file = st.file_uploader("Drop a lesion image here (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        uploaded_file.seek(0)
        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data)).convert('RGB')
        
        # Display image inside a neat container
        with st.container(border=True):
            st.image(image, caption=f"Source: {uploaded_file.name}", use_container_width=True)
            
        predict_btn = st.button('🚀 Start Classification Analysis', use_container_width=True, type="primary")

with col2:
    st.subheader("📊 Diagnostic Results")
    
    if uploaded_file is not None and predict_btn:
        with st.spinner('Running Feature Extraction & SVM Inference...'):
            img_array = np.array(image)
            features = extract_features(img_array)
            scaled = scaler.transform(features)

            prediction = model.predict(scaled)[0]
            probs = model.predict_proba(scaled)[0]
            
            # Show the main result in a success card
            st.success(f"### Predicted Class: **{classes[prediction]}**")
            
            # Confidence metric
            st.metric("Model Confidence", f"{max(probs)*100:.2f}%")
            
            # Better probability visualization
            st.write("**Probability Distribution:**")
            chart_data = pd.DataFrame({
                'Lesion Type': classes,
                'Probability': probs
            }).set_index('Lesion Type')
            st.bar_chart(chart_data)
            
            # Expandable section for ABCD details
            with st.expander("🔍 View Extracted Shape Metrics"):
                asym_val = features[0][-2]
                border_val = features[0][-1]
                st.write(f"- **Asymmetry Score:** {asym_val:.4f}")
                st.write(f"- **Border Irregularity (Compactness):** {border_val:.4f}")

            del img_array, features
    else:
        st.info("Awaiting image upload and analysis command.")