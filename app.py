import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

# --- 1. DATA CONFIG & CONSTANTS ---
CLASSES = [
    "Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis",
    "Dermatofibroma", "Melanoma", "Melanocytic Nevi", "Vascular Lesions",
]

MODEL_INFO = {
    "SVM": {
        "desc": "RBF kernel (C=10). Excellent boundary separation in high-dimensional space.",
        "params": "kernel=rbf, C=10, probability=True",
        "strengths": ["High specificity", "Good generalization", "Robust to overfitting"],
        "acc": "71.9%", "f1": "0.457", "sens": "0.448", "spec": "0.930"
    },
    "Random Forest": {
        "desc": "Ensemble of 200 decision trees with bootstrap aggregation.",
        "params": "n_estimators=200, max_depth=None",
        "strengths": ["Feature importance", "No scaling needed", "Handles imbalance"],
        "acc": "71.9%", "f1": "0.455", "sens": "0.451", "spec": "0.933"
    },
    "XGBoost": {
        "desc": "Gradient boosted trees with regularization for optimal performance.",
        "params": "n_estimators=300, lr=0.1, max_depth=6",
        "strengths": ["Best accuracy", "Regularization", "Fast inference"],
        "acc": "72.8%", "f1": "0.480", "sens": "0.475", "spec": "0.930"
    }
}

# Data for Dashboard charts (Fixed the NameError by defining these)
METRICS_DATA = pd.DataFrame([
    {"name": "SVM", "accuracy": 71.9, "f1": 45.7, "sensitivity": 44.8, "specificity": 93.0},
    {"name": "Random Forest", "accuracy": 71.9, "f1": 45.5, "sensitivity": 45.1, "specificity": 93.3},
    {"name": "XGBoost", "accuracy": 72.8, "f1": 48.0, "sensitivity": 47.5, "specificity": 93.0},
])

CLASS_DISTRIBUTION = pd.DataFrame([
    {"name": "Melanocytic Nevi", "count": 6705},
    {"name": "Melanoma", "count": 1113},
    {"name": "Benign Keratosis", "count": 1099},
    {"name": "Basal Cell Carcinoma", "count": 514},
    {"name": "Actinic Keratoses", "count": 327},
    {"name": "Vascular Lesions", "count": 142},
    {"name": "Dermatofibroma", "count": 115},
])

RISK_MAP = {
    "Melanoma": {"level": "High", "color": "#ff4b4b", "advice": "Potentially malignant; seek immediate dermatologist evaluation"},
    "Basal Cell Carcinoma": {"level": "Moderate", "color": "#ffa500", "advice": "Slow-growing cancer; schedule a dermatology appointment"},
    "Actinic Keratoses": {"level": "Moderate", "color": "#ffa500", "advice": "Pre-cancerous; monitor and consult dermatologist"},
    "Benign Keratosis": {"level": "Low", "color": "#00d1b2", "advice": "Benign growth; monitor changes"},
    "Dermatofibroma": {"level": "Low", "color": "#00d1b2", "advice": "Benign; usually no treatment needed"},
    "Melanocytic Nevi": {"level": "Low", "color": "#00d1b2", "advice": "Common mole; benign but monitor changes"},
    "Vascular Lesions": {"level": "Low", "color": "#00d1b2", "advice": "Benign vascular condition; typically harmless"}
}

# --- 2. PAGE SETUP & STYLING ---
st.set_page_config(page_title="DermAI Classifier", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #00d1b2; }
    .risk-box { padding: 15px; border-radius: 10px; border: 1px solid #2d3748; background-color: #1a1f2e; margin-top: 10px; }
    .glass-card {
        background: #1a1f2e; padding: 20px; border-radius: 15px; 
        border: 1px solid #2d3748; text-align: center; margin-bottom: 10px;
    }
    .stat-label { font-size: 0.9rem; opacity: 0.7; margin-bottom: 5px; color: white; }
    .stat-val { font-size: 2rem; font-weight: bold; color: #00d1b2; margin: 0; }
    .feature-card { text-align: left; min-height: 120px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    # Note: Ensure these files exist in a 'models/' folder
    try:
        svm = joblib.load('models/svm.pkl')
        rf = joblib.load('models/random_forest.pkl')
        xgb = joblib.load('models/xgboost.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return {"SVM": svm, "Random Forest": rf, "XGBoost": xgb}, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure .pkl files are in the 'models/' directory.")
        return None, None

models, scaler = load_assets()

# --- 3. FEATURE EXTRACTION LOGIC ---
def extract_features(img_input):
    img = np.array(img_input)
    img_res = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)
    
    # GLCM - 20 features
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, True, True)
    glcm_props = {p: graycoprops(glcm, p).mean() for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']}
    
    # HSV - 48 features
    hsv = cv2.cvtColor(img_res, cv2.COLOR_RGB2HSV)
    hsv_feats = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [16], [0, 256]).flatten()
        hsv_feats.extend((hist / (hist.sum() + 1e-7)).tolist())
    
    # ABCD - 2 features
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    asym = np.sum(mask != cv2.flip(mask, 1)) / (mask.size + 1e-7)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border = 1.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 0:
            border = (cv2.arcLength(cnt, True)**2) / (4 * np.pi * cv2.contourArea(cnt) + 1e-7)

    vector = list(glcm_props.values()) + hsv_feats + [asym, border]
    return np.array(vector).reshape(1, -1), glcm_props, (asym, border)

# --- 4. SIDEBAR ---
if models:
    with st.sidebar:
        st.markdown("### 🔬 DermAI")
        st.caption("SKIN LESION CLASSIFIER")
        selected_name = st.selectbox("⚙️ Select Model", list(models.keys()))
        
        info = MODEL_INFO[selected_name]
        st.markdown(f"**Model Details:** {info['desc']}")
        st.code(info['params'], language="text")
        
        st.markdown("**Strengths:**")
        for s in info['strengths']:
            st.markdown(f"✅ <span style='color:#00d1b2; font-size:0.8rem'>{s}</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### 📊 Model Stats")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", info['acc'])
        c2.metric("Macro F1", info['f1'])
        c3, c4 = st.columns(2)
        c3.metric("Sensitivity", info['sens'])
        c4.metric("Specificity", info['spec'])

# --- 5. MAIN CONTENT ---
tab1, tab2 = st.tabs(["🔍 Classify", "📈 Dashboard"])

with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Click to upload (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
            classify_btn = st.button("🚀 Classify Image", type="primary")

    with col_right:
        st.subheader("Results")
        if uploaded_file and 'classify_btn' in locals() and classify_btn:
            with st.spinner('Analyzing...'):
                feats, g_props, abcd = extract_features(image)
                scaled_feats = scaler.transform(feats)
                
                probs = models[selected_name].predict_proba(scaled_feats)[0]
                pred_idx = np.argmax(probs)
                pred_name = CLASSES[pred_idx]
                confidence = probs[pred_idx] * 100
                
                st.markdown(f"""
                <div style="background-color: #1a1f2e; padding: 25px; border-radius: 12px; text-align: center; border: 1px solid #00d1b2;">
                    <p style="margin:0; opacity:0.7; font-size:0.8rem;">PREDICTION</p>
                    <h2 style="margin:5px 0; color: white;">{pred_name}</h2>
                    <h1 style="color:#00d1b2; margin:0;">{confidence:.1f}%</h1>
                    <p style="opacity:0.7; color: white;">confidence</p>
                </div>
                """, unsafe_allow_html=True)
                
                risk = RISK_MAP.get(pred_name, {"level": "Unknown", "color": "gray", "advice": "N/A"})
                st.markdown(f"""
                <div class="risk-box">
                    <span style="color:{risk['color']}; font-weight:bold;">● {risk['level']} Risk Level</span><br>
                    <span style="font-size:0.85rem; opacity:0.8; color: white;">— {risk['advice']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### GLCM Radar Analysis")
                radar_fig = go.Figure(data=go.Scatterpolar(
                    r=list(g_props.values()),
                    theta=list(g_props.keys()),
                    fill='toself',
                    line_color='#00d1b2'
                ))
                radar_fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True), bgcolor="rgba(0,0,0,0)"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False, margin=dict(l=40, r=40, t=20, b=20), height=300
                )
                st.plotly_chart(radar_fig, use_container_width=True)

with tab2:
    st.subheader("Model Comparison Overview")
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.markdown('<div class="glass-card"><p class="stat-label">Models</p><p class="stat-val">3</p></div>', unsafe_allow_html=True)
    ov2.markdown('<div class="glass-card"><p class="stat-label">Features</p><p class="stat-val">70</p></div>', unsafe_allow_html=True)
    ov3.markdown('<div class="glass-card"><p class="stat-label">Images</p><p class="stat-val">10K</p></div>', unsafe_allow_html=True)
    ov4.markdown('<div class="glass-card"><p class="stat-label">Classes</p><p class="stat-val">7</p></div>', unsafe_allow_html=True)

    st.subheader("Overall Metrics Comparison")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        fig_acc = px.bar(METRICS_DATA, x="name", y="accuracy", title="Accuracy (%)", color="name", template="plotly_dark")
        st.plotly_chart(fig_acc, use_container_width=True)
    with m_col2:
        fig_f1 = px.bar(METRICS_DATA, x="name", y="f1", title="Macro F1 Score (%)", color="name", template="plotly_dark")
        st.plotly_chart(fig_f1, use_container_width=True)

    st.subheader("Feature Engineering Breakdown")
    f_col1, f_col2, f_col3 = st.columns(3)
    f_col1.markdown('<div class="glass-card feature-card"><b>20 GLCM Features</b><br><small>Contrast, Homogeneity, Energy, Dissimilarity, Correlation at 4 angles.</small></div>', unsafe_allow_html=True)
    f_col2.markdown('<div class="glass-card feature-card"><b>48 HSV Features</b><br><small>16-bin histograms for Hue, Saturation, and Value channels.</small></div>', unsafe_allow_html=True)
    f_col3.markdown('<div class="glass-card feature-card"><b>2 ABCD Features</b><br><small>Asymmetry Index and Border Irregularity via Otsu Segmentation.</small></div>', unsafe_allow_html=True)

    st.subheader("Dataset Class Distribution (HAM10000)")
    fig_dist = px.bar(CLASS_DISTRIBUTION, x="count", y="name", orientation='h', color="count", color_continuous_scale="Viridis", template="plotly_dark")
    fig_dist.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_dist, use_container_width=True)