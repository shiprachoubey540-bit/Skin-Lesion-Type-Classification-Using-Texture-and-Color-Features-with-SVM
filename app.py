import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

# --- 1. PAGE CONFIG & THEMING ---
st.set_page_config(page_title="DermAI Clinical Dashboard", layout="wide")

# Custom CSS for the unique "DermAI" Dark-Card Theme
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card { background-color: #1e293b; color: white; padding: 20px; border-radius: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f1f5f9; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_models():
    # Loading all 3 models for comparison
    models = {
        "SVM": joblib.load('models/svm.pkl'),
        "Random Forest": joblib.load('models/random_forest.pkl'),
        "XGBoost": joblib.load('models/xgboost.pkl')
    }
    scaler = joblib.load('models/scaler.pkl')
    return models, scaler

models, scaler = load_models()
classes = ['Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis', 
           'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesions']

# --- 3. FEATURE EXTRACTION (70 Features) ---
def get_advanced_features(img_array):
    img_res = cv2.resize(img_array, (224, 224))
    gray = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(cv2.resize(img_array, (224, 224)), cv2.COLOR_RGB2HSV)

    # GLCM Texture
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, True, True)
    glcm_props = {p: graycoprops(glcm, p).mean() for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']}
    
    # HSV Color
    hsv_list = [cv2.calcHist([hsv], [i], None, [16], [0, 256]).flatten() for i in range(3)]
    hsv_feats = np.concatenate(hsv_list) / (img_res.size/3)

    # ABCD Shape
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Asymmetry calculation
    asym = (np.sum(mask != cv2.flip(mask, 1)) / mask.size)
    # Border calculation
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border = 1.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        border = (cv2.arcLength(cnt, True)**2) / (4 * np.pi * cv2.contourArea(cnt) + 1e-7)

    full_vector = np.concatenate([list(glcm_props.values()), hsv_feats, [asym, border]])
    return full_vector.reshape(1, -1), glcm_props, (asym, border), hsv.mean(axis=(0,1))

# --- 4. SIDEBAR & NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=80)
    st.title("DermAI")
    selected_model_name = st.selectbox("🎯 Select Model", list(models.keys()))
    
    st.subheader("Model Details")
    if selected_model_name == "SVM":
        st.info("RBF kernel (C=10). Excellent boundary separation.")
    
    st.markdown("---")
    st.write("📊 **Dataset:** HAM10000\n\n**Features:** 70 (GLCM+HSV+ABCD)")

# --- 5. TABS INTERFACE ---
t1, t2, t3 = st.tabs(["🔍 Classify", "📊 Dashboard", "ℹ️ About"])

with t1:
    col1, col2 = st.columns([1, 1])
    with col1:
        u_file = st.file_uploader("Upload Lesion Image", type=['jpg', 'png', 'jpeg'])
        if u_file:
            img = Image.open(u_file).convert('RGB')
            st.image(img, use_container_width=True)
    
    with col2:
        if u_file:
            feats, g_props, abcd, hsv_means = get_advanced_features(np.array(img))
            scaled_feats = scaler.transform(feats)
            
            # Prediction
            curr_model = models[selected_model_name]
            prob = curr_model.predict_proba(scaled_feats)[0]
            pred_idx = np.argmax(prob)
            
            st.markdown(f"""
                <div class='metric-card'>
                    <p style='text-align:center;'>PREDICTION</p>
                    <h2 style='text-align:center;'>{classes[pred_idx]}</h2>
                    <h1 style='text-align:center; color:#38bdf8;'>{prob[pred_idx]*100:.1f}%</h1>
                    <p style='text-align:center; opacity:0.8;'>confidence</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Radar Chart for GLCM
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=list(g_props.values()),
                theta=list(g_props.keys()),
                fill='toself'
            ))
            st.plotly_chart(fig_radar, use_container_width=True)

with t2:
    # Model Comparison Graphs
    st.subheader("Model Comparison Overview")
    comp_data = pd.DataFrame({
        'Model': ['SVM', 'Random Forest', 'XGBoost'],
        'Accuracy': [71.9, 71.9, 72.8],
        'Macro F1': [45.7, 45.5, 48.0]
    })
    fig_comp = px.bar(comp_data, x='Model', y='Accuracy', color='Model', barmode='group')
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Class Distribution
    st.subheader("Dataset Class Distribution")
    dist_data = {'Melanocytic Nevi': 6705, 'Melanoma': 1113, 'Benign Keratosis': 1099}
    st.bar_chart(dist_data)