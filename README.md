# Skin-Lesion-Type-Classification-Using-Texture-and-Color-Features-with-SVM

A machine learning pipeline for classifying skin lesion types from the **HAM10000 dataset** using image preprocessing, handcrafted feature extraction, and classical ML classifiers (SVM, Random Forest, XGBoost).

---

## things done by member 1 till now

```
├── data_preprocess_eda.ipynb       # notebook (preprocessing + EDA)
├── HAM10000_metadata.csv           # Dataset metadata (gitignored)
├── HAM10000_images_part_1/         # Raw images part 1 (gitignored)
├── HAM10000_images_part_2/         # Raw images part 2 (gitignored)
├── processed_images.npy            # Preprocessed image cache (gitignored)
├── processed_ids.npy               # Image ID order cache (gitignored)
├── hsv_features.npy                # Extracted HSV features (gitignored)
├── eda_class_age.png               # EDA plot: class & age distribution
├── eda_feature_distributions.png   # EDA plot: feature distributions
├── eda_age_boxplots.png            # EDA plot: age boxplots & outliers
├── eda_sample_grid.png             # EDA plot: sample images per class
├── eda_sex_class_heatmap.png       # EDA plot: sex × class heatmap
├── preprocessing_validation.png    # Preprocessing validation plot
└── README.md
```

---

## Dataset — HAM10000

The **Human Against Machine with 10000 training images (HAM10000)** dataset is a large collection of dermatoscopic images of pigmented skin lesions.

| Property | Value |
|---|---|
| Total images | 10,015 |
| Image size (raw) | 450 × 600 × 3 |
| Resized to | 128 × 128 × 3 |
| Metadata columns | lesion_id, image_id, dx, dx_type, age, sex, localization |

### Lesion Classes

| Code | Full Name |
|---|---|
| `nv` | Melanocytic Nevi |
| `mel` | Melanoma |
| `bkl` | Benign Keratosis |
| `bcc` | Basal Cell Carcinoma |
| `akiec` | Actinic Keratoses |
| `vasc` | Vascular Lesions |
| `df` | Dermatofibroma |

>  The dataset is **highly imbalanced** — Melanocytic Nevi accounts for ~67% of samples. SMOTE is applied to handle this.

---

## Pipeline Overview

### Stage 1 — Setup & Imports
All required libraries are installed and imported:
- **Data**: `numpy`, `pandas`
- **Visualization**: `matplotlib`, `seaborn`
- **Image processing**: `opencv-python`, `scikit-image`, `Pillow`
- **ML**: `scikit-learn`, `xgboost`, `imbalanced-learn`

---

### Stage 2 — Dataset Overview
- Loads `HAM10000_metadata.csv`
- Maps short diagnosis codes to full label names
- Reports:
  - Total records, unique lesion/image IDs
  - Column data types and missing value counts
  - Class, sex, localization, and dx_type distributions
- Verifies all image files exist on disk and drops missing entries

---

### Stage 3 — Image Preprocessing

Each image goes through the following pipeline:

```
Raw .jpg  →  Resize (128×128)  →  BGR→RGB  →  Hair Removal  →  Normalize [0,1]
```

**Steps in detail:**

| Step | Method | Detail |
|---|---|---|
| Resize | `cv2.resize` | 128 × 128 pixels |
| Color convert | `cv2.cvtColor` | BGR → RGB |
| Hair removal | Black-hat morphology + inpainting | Removes dark thin artifacts |
| Normalization | Divide by 255 | uint8 → float32 in \[0.0, 1.0\] |

**Caching:** Preprocessed images are saved as `processed_images.npy` to avoid re-processing on subsequent runs.

**Validation output** (printed after preprocessing):
- Array shape `(N, 128, 128, 3)`
- Data type `float32`
- Pixel min/max range
- Global mean and std
- Per-channel (R, G, B) mean and std
- Side-by-side visual: original uint8 vs normalized float32

---

### Stage 4 — Exploratory Data Analysis (EDA)

#### 4.1 Statistical Summary
- `describe()` for age (mean, std, min, max, quartiles)
- Value counts + percentage for: `sex`, `localization`, `label_name`, `dx_type`

#### 4.2 Outlier Detection & Removal — Age
Uses the **IQR method**:

```
Lower fence = Q1 - 1.5 × IQR
Upper fence = Q3 + 1.5 × IQR
```

- NaN ages are filled with the **median age** before fence computation
- Rows with age outside fences are **removed** from both `df` and `images`
- Number of outliers found and removed is printed

#### 4.3 Feature Distribution Plots (`eda_feature_distributions.png`)
- Bar chart of class distribution with count labels
- Age histogram with KDE curve, mean and median lines
- Pie chart of sex distribution
- Horizontal bar chart of top-10 lesion localizations

#### 4.4 Age Boxplots (`eda_age_boxplots.png`)
- Boxplot of age grouped by lesion type
- Overall age boxplot with IQR fence lines marked

#### 4.5 Sex × Class Heatmap (`eda_sex_class_heatmap.png`)
- Heatmap of sample counts for each combination of lesion type and sex

#### 4.6 Visual Insights Interpretation
Printed analysis covering:
- Class imbalance ratio (most vs least frequent class)
- Age statistics (mean, std, range)
- Sex percentage split
- Most common localization site
- Diagnosis confirmation method breakdown

#### 4.7 Sample Image Grid (`eda_sample_grid.png`)
- 3 sample images per class displayed in a grid (normalized float32)

---

### Stage 5 — HSV Feature Extraction

For each preprocessed image:
1. Convert RGB → HSV (using `skimage.color.rgb2hsv`)
2. Compute a 16-bin normalized histogram for each of H, S, V channels
3. Concatenate → **48-dimensional feature vector** per image

Features are cached in `hsv_features.npy`.

---

### Stage 6 — Class Balancing with SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** is applied after the train/test split to balance the training set.

- Applied **only on training data** (never on test data)
- `k_neighbors=5`, `random_state=42`
- Prints class distribution before and after

---

### Stage 7 — Evaluation Visualization

Two utility functions ready for use after model training:

- **`plot_confusion_matrix`** — heatmap of predicted vs actual labels
- **`plot_per_class_accuracy`** — bar chart of per-class recall scores

---

## EDA Output Plots

| File | Description |
|---|---|
| `eda_class_age.png` | Class bar chart + age histogram |
| `eda_feature_distributions.png` | Class, age, sex, localization distributions |
| `eda_age_boxplots.png` | Age by class + IQR outlier boxplot |
| `eda_sample_grid.png` | 3 sample images per lesion class |
| `eda_sex_class_heatmap.png` | Count heatmap: sex × lesion type |
| `preprocessing_validation.png` | Before/after normalization visual |

---

## How to Run

1. Clone the repository and place the HAM10000 dataset files in the project root.
2. Open `data_preprocess_eda.ipynb` in Jupyter Notebook or VS Code.
3. Update the `META_CSV` and `IMG_DIRS` paths in the config cell to match your local setup.
4. Run all cells in order (top to bottom).

> Preprocessed images and features are cached as `.npy` files — subsequent runs load from cache and skip reprocessing.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
opencv-python
scikit-image
Pillow
scikit-learn
imbalanced-learn
xgboost
scipy
```

Install all with:
```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-image pillow scikit-learn imbalanced-learn xgboost scipy
```

---
