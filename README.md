**Skin Lesion Classification using Machine Learning**

**Problem Statement:**
The study requires skin lesion images to be classified into seven different categories which include melanoma and basal cell carcinoma through the application of conventional machine learning methods that exclude deep learning.

**Dataset:**
HAM10000 dataset: The dataset includes dermatoscopic images that show various skin lesions

**Methodology:**
1. Data collection
2. Data preprocessing:

2.1: Data cleaning: checked for missing or null values, handled missing values.
3. Exploratory Data Analysis (EDA):
Analyzed feature distributions using histograms and boxplots, identified outliers and skewness, Visualized relationships between features and the target variable.

4. Feature Engineering:

- GLCM Texture Features

GLCM (Gray Level Co-occurrence Matrix) captures how often pairs of pixel intensities appear together at a given direction and distance. We extracted 5 properties across 4 angles (0°, 45°, 90°, 135°), giving 20 features per image.

Contrast — measures intensity difference between a pixel and its neighbor. High contrast = rough texture.

Dissimilarity — similar to contrast but increases linearly, not exponentially.

Homogeneity — how uniform the texture is. Smooth regions score high.

Energy — uniformity of the GLCM. High energy = repetitive texture pattern.

Correlation — how linearly dependent a pixel is on its neighbor. High = structured texture..

  
- HSV Color Histograms

Instead of raw RGB, We converted images to HSV (Hue, Saturation, Value) because it separates color information from lighting, making it more robust to brightness variation in dermoscopy images. You computed a 16-bin histogram for each of the 3 channels, giving 48 features per image, all L1-normalised so they sum to 1.

Hue — the actual color (red, brown, black etc.) which is clinically significant in lesion diagnosis.
Saturation — how vivid or washed out the color is.

Value — brightness of the image.
  
- ABCD Shape Descriptors

These mimic the clinical ABCD criteria dermatologists use to diagnose skin lesions visually. The lesion was first segmented using Otsu thresholding + morphological operations to isolate the lesion from skin background.

Asymmetry — the lesion mask is flipped horizontally and vertically, and the pixel mismatch between original and flipped versions is measured. A perfectly symmetric lesion scores 0, a highly asymmetric one scores closer to 1. Malignant lesions tend to be asymmetric.

Border Irregularity — computed using the compactness formula: perimeter² / (4π × area). A perfect circle gives 1.0, and jagged/irregular borders push the value higher. Outliers were Winsorized (capped at the IQR upper fence) to prevent extreme values from distorting the model.
  
5.  Merging

Since GLCM was built without an image_id column but all three feature extractors looped the same folder in the same order, the three DataFrames were aligned by row position using pd.concat and the folder-based label was replaced with the real dx labels from HAM10000_metadata.csv.

After merging with the real dx labels from the HAM10000 metadata csv, we checked for outliers again before proceeding with model building. This step was essential, as it had caused some problems. Some columns got null values after feature engineering, these had to checked and removed before entering the model training part.

6. Class Imbalance — SMOTE

The dataset was heavily imbalanced (nv=6705 vs df=115). SMOTE (Synthetic Minority Oversampling Technique) was applied on the training set only after the train/test split, generating synthetic samples for minority classes so the models don't just learn to predict the majority class.

 Models Used:
* The system uses Support Vector Machine with RBF kernel package
* The system uses Random Forest algorithm
* The system uses XGBoost algorithm

7. Model Training

Model            | Key Settings
-----------------|-------------------------------
SVM              | RBF kernel, C=10
Random Forest    | 200 trees
XGBoost          | 300 rounds, lr=0.05, max depth=6

All features were StandardScaler normalized before training since SVM is sensitive to feature scale, and consistency was maintained for Random Forest and XGBoost as well.


8. Model Evaluation:
  
Evaluation Metrics:
* Accuracy
* precision
* Recall
* F1 Score

9. Confusion Matrix:
The system uses a Confusion Matrix to evaluate its performance

10. Deployment:
A Streamlit web application was developed which enables users to perform two main functions.
* Users can upload a skin lesion image
* Users receive the predicted skin lesion class along with its confidence level

* Member 1: The member handled the data preprocessing task
* Member 2: The member built the system through feature extraction and model development
* Member 3: The member performed system evaluation and deployment while creating documentation
  
**Conclusion:**
The project demonstrates machine learning's ability to classify skin lesions through its system which automatically classifies skin lesions and its development of a user-friendly system that delivers immediate prediction results.
