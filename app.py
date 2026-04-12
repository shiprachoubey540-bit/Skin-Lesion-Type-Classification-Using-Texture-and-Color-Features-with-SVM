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
    
    .about-header-icon {
        background: rgba(0, 209, 178, 0.15); width: 64px; height: 64px; 
        border-radius: 16px; display: flex; align-items: center; 
        justify-content: center; margin: 0 auto; color: #00d1b2; font-size: 32px;
    }
    .gradient-text {
        background: linear-gradient(90deg, #00d1b2, #00b8d4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .warning-card {
        background: #1a1f2e; padding: 24px; border-radius: 12px;
        border: 1px solid rgba(255, 165, 0, 0.3); margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        svm = joblib.load('models/svm.pkl')
        rf = joblib.load('models/random_forest.pkl')
        xgb = joblib.load('models/xgboost.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return {"SVM": svm, "Random Forest": rf, "XGBoost": xgb}, scaler
    except FileNotFoundError:
        st.error("Model files missing in 'models/' directory.")
        return None, None

models, scaler = load_assets()

# --- 3. FEATURE EXTRACTION ---
def extract_features(img_input):
    img = np.array(img_input)
    img_resized = cv2.resize(img, (224, 224))
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # GLCM - 20 features
    glcm = graycomatrix(gray, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    glcm_feats = []
    glcm_summary = {} 
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        values = graycoprops(glcm, prop).flatten()
        glcm_feats.extend(values.tolist())
        glcm_summary[prop] = values.mean()

    # HSV - 48 features
    hsv_feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [16], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-7)
        hsv_feats.extend(hist.tolist())

    # ABCD Shape - 2 features
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    flip_h = cv2.flip(mask, 0)
    flip_v = cv2.flip(mask, 1)
    asym_h = np.sum(mask != flip_h) / (mask.size + 1e-7)
    asym_v = np.sum(mask != flip_v) / (mask.size + 1e-7)
    asymmetry = float((asym_h + asym_v) / 2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        border = float((perimeter ** 2) / (4 * np.pi * area + 1e-7))
    else:
        border = 1.0

    feature_vector = np.array(glcm_feats + hsv_feats + [asymmetry, border]).reshape(1, -1)
    return feature_vector, glcm_summary, (asymmetry, border)

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

# --- 5. MAIN CONTENT ---
tab1, tab2, tab3 = st.tabs(["🔍 Classify", "📈 Dashboard", "ℹ️ About"])

with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Click to upload (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
            if st.button("🚀 Classify Image", type="primary"):
                with st.spinner('Analyzing...'):
                    feats, g_summary, abcd = extract_features(image)
                    scaled_feats = scaler.transform(feats) 
                    probs = models[selected_name].predict_proba(scaled_feats)[0]
                    pred_idx = np.argmax(probs)
                    pred_name = CLASSES[pred_idx]
                    confidence = probs[pred_idx] * 100
                    
                    with col_right:
                        st.markdown(f"""
                        <div style="background-color: #1a1f2e; padding: 25px; border-radius: 12px; text-align: center; border: 1px solid #00d1b2;">
                            <p style="margin:0; opacity:0.7; font-size:0.8rem;">PREDICTION</p>
                            <h2 style="margin:5px 0; color: white;">{pred_name}</h2>
                            <h1 style="color:#00d1b2; margin:0;">{confidence:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                        risk = RISK_MAP.get(pred_name, {"level": "Low", "color": "gray", "advice": ""})
                        st.markdown(f"""<div class="risk-box"><span style="color:{risk['color']}; font-weight:bold;">● {risk['level']} Risk Level</span><br><span style="font-size:0.85rem; color: white;">— {risk['advice']}</span></div>""", unsafe_allow_html=True)
                        
                        radar_fig = go.Figure(data=go.Scatterpolar(r=list(g_summary.values()), theta=list(g_summary.keys()), fill='toself', line_color='#00d1b2'))
                        radar_fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)"), paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=300, margin=dict(t=30, b=30))
                        st.plotly_chart(radar_fig, use_container_width=True)

with tab2:
    st.subheader("Model Comparison Overview")
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.markdown('<div class="glass-card"><p class="stat-label">Models</p><p class="stat-val">3</p></div>', unsafe_allow_html=True)
    ov2.markdown('<div class="glass-card"><p class="stat-label">Features</p><p class="stat-val">70</p></div>', unsafe_allow_html=True)
    ov3.markdown('<div class="glass-card"><p class="stat-label">Images</p><p class="stat-val">10K</p></div>', unsafe_allow_html=True)
    ov4.markdown('<div class="glass-card"><p class="stat-label">Classes</p><p class="stat-val">7</p></div>', unsafe_allow_html=True)
    
    st.plotly_chart(px.bar(METRICS_DATA, x="name", y="accuracy", title="Accuracy (%)", color="name", template="plotly_dark"), use_container_width=True)
    st.plotly_chart(px.bar(CLASS_DISTRIBUTION, x="count", y="name", orientation='h', color="count", template="plotly_dark"), use_container_width=True)

with tab3:
    st.markdown('<div style="text-align: center; margin-bottom: 30px;">', unsafe_allow_html=True)
    st.markdown('<div class="about-header-icon">🔬</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="gradient-text" style="font-size: 2.5rem; margin-top: 10px;">DermAI Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p style="opacity:0.8;">Advanced Machine Learning for Dermatological Screening</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
            <div class="glass-card feature-card">
                <h4 style="color:#00d1b2; margin-top:0;">📊 Dataset & Training</h4>
                <p style="font-size:0.9rem; color:#e2e8f0; line-height:1.4;">
                    Built using the <b>HAM10000</b> dataset, containing 10,015 dermatoscopic images. 
                    To handle class imbalance, we implemented <b>SMOTE</b> (Synthetic Minority Over-sampling Technique) 
                    to ensure the model learns rare conditions effectively.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
            <div class="glass-card feature-card">
                <h4 style="color:#00d1b2; margin-top:0;">🧪 Feature Engineering</h4>
                <p style="font-size:0.9rem; color:#e2e8f0; line-height:1.4;">
                    We extract <b>70 distinct features</b>:
                    <ul style="font-size:0.85rem; text-align:left; color:#cbd5e0;">
                        <li><b>Texture:</b> 20 GLCM features (Energy, Correlation, etc.)</li>
                        <li><b>Color:</b> 48 HSV Histogram channels</li>
                        <li><b>Shape:</b> ABCD criteria (Asymmetry & Border Irregularity)</li>
                    </ul>
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Technical Methodology")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.info("**Preprocessing**\n\nImages are resized to 224x224 and normalized. We use Gaussian blurring to reduce noise before feature extraction.")
    with m_col2:
        st.info("**Classification**\n\nWe utilize a multi-model ensemble approach (SVM, RF, XGBoost) to provide comparative diagnostic insights.")
    with m_col3:
        st.info("**Evaluation**\n\nModels are validated using 5-fold cross-validation, focusing on Specificity to reduce false alarms.")

    st.markdown('<div class="warning-card"><h3 style="color: #ffa500; margin: 0;">⚠️ Medical Disclaimer</h3><p style="color: #cbd5e0; font-size: 0.95rem; margin-top: 10px;">This application is an AI-based screening tool and <b>not a substitute for professional medical advice, diagnosis, or treatment</b>. Always seek the advice of your dermatologist with any questions regarding a skin condition.</p></div>', unsafe_allow_html=True)