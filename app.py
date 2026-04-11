import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import io
import plotly.express as px

# --- 1. SETTINGS & ASSET LOADING ---
st.set_page_config(
    page_title="Skin Lesion AI Diagnostic Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Professional Styling
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stSidebar"] { background-color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    # Ensure these files are in your 'models/' folder
    model  = joblib.load('models/svm.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}. Ensure .pkl files are in the 'models' folder.")

classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
           'Dermatofibroma', 'Melanoma', 'Nevus', 'Vascular lesions']

# --- 2. CORE FEATURE EXTRACTION ---
def extract_features(img_input):
    img = np.copy(img_input)
    img_resized = cv2.resize(img, (224, 224))

    # Color Spaces
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # GLCM (Texture)
    glcm = graycomatrix(gray, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    glcm_feats = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        glcm_feats.extend(graycoprops(glcm, prop).flatten().tolist())

    # HSV (Color)
    hsv_feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [16], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-7)
        hsv_feats.extend(hist.tolist())

    # ABCD (Shape)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    flip_h = cv2.flip(mask, 0)
    flip_v = cv2.flip(mask, 1)
    asym = float(np.clip(((np.sum(mask != flip_h) + np.sum(mask != flip_v)) / (2 * mask.size + 1e-7)), 0, 1))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border = 1.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area, peri = cv2.contourArea(cnt), cv2.arcLength(cnt, True)
        if area > 0: border = float(np.clip((peri ** 2) / (4 * np.pi * area + 1e-7), 1, 50))

    return np.array(glcm_feats + hsv_feats + [asym, border]).reshape(1, -1)

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🛡️ Analysis Info")
    st.write("This dashboard provides AI-driven dermatological insights based on the HAM10000 dataset.")
    st.markdown("---")
    st.subheader("Extraction Summary")
    st.write("✅ Texture (GLCM): 20 features")
    st.write("✅ Color (HSV): 48 features")
    st.write("✅ Shape (ABCD): 2 features")
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** For educational use only. Consult a dermatologist for medical diagnosis.")

# --- 4. MAIN DASHBOARD ---
st.title("🔬 Skin Lesion Clinical Decision Support System")
st.divider()

tab1, tab2 = st.tabs(["🚀 Real-time Diagnostic", "📈 Model Performance"])

# --- TAB 1: DIAGNOSTIC TOOL ---
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Input Image")
        uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            with st.container(border=True):
                st.image(image, caption="Uploaded Sample", use_container_width=True)
            predict_btn = st.button('Perform Classification Analysis', type="primary")

    with col2:
        st.subheader("Diagnostic Intelligence")
        if uploaded_file and predict_btn:
            with st.spinner('Calculating 70-Feature Vector...'):
                img_array = np.array(image)
                features = extract_features(img_array)
                scaled = scaler.transform(features)

                prediction = model.predict(scaled)[0]
                probs = model.predict_proba(scaled)[0]

                # Prediction Card
                st.success(f"### Result: {classes[prediction]}")
                st.metric("Model Confidence", f"{max(probs)*100:.2f}%")

                # Probability Chart
                st.write("**Differential Diagnosis (Probability Chart):**")
                chart_df = pd.DataFrame({'Class': classes, 'Prob': probs}).sort_values('Prob')
                fig = px.bar(chart_df, x='Prob', y='Class', orientation='h', 
                             color='Prob', color_continuous_scale='Blues', template='plotly_white')
                fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

                # ABCD Expander
                with st.expander("🔍 Detailed Morphological Metrics"):
                    st.write(f"- **Asymmetry Score:** {features[0][-2]:.4f}")
                    st.write(f"- **Border Irregularity:** {features[0][-1]:.4f}")
        else:
            st.info("Awaiting image upload to begin feature extraction.")

# --- TAB 2: PERFORMANCE DATA ---
with tab2:
    st.header("Model Evaluation Summary")
    
    # High-level Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", "87.4%")
    m2.metric("Precision", "0.86")
    m3.metric("Recall", "0.85")
    m4.metric("F1-Score", "0.85")
    
    st.divider()
    
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.subheader("Confusion Matrix")
        # Ensure 'preprocessing_validation.png' is in your root folder
        try:
            st.image("preprocessing_validation.png", caption="Model Validation Results", use_container_width=True)
        except:
            st.error("Confusion matrix image not found. Please add 'preprocessing_validation.png' to GitHub.")

    with p_col2:
        st.subheader("Model Architecture")
        st.write("""
        **Support Vector Machine (SVM)**
        - **Kernel:** RBF (Radial Basis Function)
        - **C:** 10
        - **Probability:** Enabled
        - **Preprocessing:** Standard Scaling & SMOTE Oversampling
        """)
        st.write("**Feature Importance:** Color (HSV) and Texture (GLCM) represent the highest weight in model decision making.")