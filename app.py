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
    model  = joblib.load('models/svm.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_assets()

classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
           'Dermatofibroma', 'Melanoma', 'Nevus', 'Vascular lesions']

# --- 2. FEATURE EXTRACTION (matches training exactly: 20 + 48 + 2 = 70) ---
def extract_features(img_input):
    img      = np.copy(img_input)
    img_resized = cv2.resize(img, (224, 224))

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # GLCM: 5 props x 4 angles = 20 features
    glcm = graycomatrix(gray, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    glcm_feats = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        glcm_feats.extend(graycoprops(glcm, prop).flatten().tolist())

    # HSV: 16 bins x 3 channels = 48 features
    hsv_feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [16], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-7)
        hsv_feats.extend(hist.tolist())

    # ABCD: asymmetry + border irregularity = 2 features
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  kernel)

    flip_h    = cv2.flip(mask, 0)
    flip_v    = cv2.flip(mask, 1)
    asym_h    = np.sum(mask != flip_h) / (mask.size + 1e-7)
    asym_v    = np.sum(mask != flip_v) / (mask.size + 1e-7)
    asymmetry = float(np.clip((asym_h + asym_v) / 2, 0, 1))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt       = max(contours, key=cv2.contourArea)
        area      = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        border    = float(np.clip((perimeter ** 2) / (4 * np.pi * area + 1e-7), 1, 50))
    else:
        border = 1.0

    # Total: 20 + 48 + 2 = 70 features
    return np.array(glcm_feats + hsv_feats + [asymmetry, border]).reshape(1, -1)

# --- 3. UI ---
st.title("🔬 Skin Lesion Classification")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    uploaded_file.seek(0)
    bytes_data = uploaded_file.read()
    image = Image.open(io.BytesIO(bytes_data)).convert('RGB')

    st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

    if st.button('Perform Classification', key=f"btn_{uploaded_file.name}"):
        with st.spinner('Extracting features and classifying...'):
            img_array = np.array(image)

            features  = extract_features(img_array)
            scaled    = scaler.transform(features)

            prediction = model.predict(scaled)[0]
            probs      = model.predict_proba(scaled)[0]

            st.success(f"Final Prediction: {classes[prediction]}")
            st.metric("Confidence Level", f"{max(probs)*100:.2f}%")
            st.bar_chart(dict(zip(classes, probs)))

            del img_array, features