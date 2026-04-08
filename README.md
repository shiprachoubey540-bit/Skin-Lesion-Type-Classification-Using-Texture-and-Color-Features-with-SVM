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

- First point GLCM Texture Features

GLCM (Gray Level Co-occurrence Matrix) captures how often pairs of pixel intensities appear together at a given direction and distance. We extracted 5 properties across 4 angles (0°, 45°, 90°, 135°), giving 20 features per image.

Contrast — measures intensity difference between a pixel and its neighbor. High contrast = rough texture.

Dissimilarity — similar to contrast but increases linearly, not exponentially.

Homogeneity — how uniform the texture is. Smooth regions score high.

Energy — uniformity of the GLCM. High energy = repetitive texture pattern.

Correlation — how linearly dependent a pixel is on its neighbor. High = structured texture..

  
2. HSV Color Histograms

Instead of raw RGB, We converted images to HSV (Hue, Saturation, Value) because it separates color information from lighting, making it more robust to brightness variation in dermoscopy images. You computed a 16-bin histogram for each of the 3 channels, giving 48 features per image, all L1-normalised so they sum to 1.

Hue — the actual color (red, brown, black etc.) which is clinically significant in lesion diagnosis.
Saturation — how vivid or washed out the color is.

Value — brightness of the image.
  
4.3: ABCD Rule Features:
* Asymmetry-Measures symmetry of lesion shape across axes
* Border Irregularity-Quantifies uneven or jagged lesion boundaries
* Color Variation-Detects the presence of multiple colors within the lesion
* Diameter-Measures lesion size (important for melanoma detection)
  
5. Handling Class Imbalance: The systems applied SMOTE for their operations.
  
 Models Used:
* The system uses Support Vector Machine with RBF kernel package
* The system uses Random Forest algorithm
* The system uses XGBoost algorithm
6. Model Training
7. Model Evaluation:
  
Evaluation Metrics:
* Accuracy
* precision
* Recall
* F1 Score

8. Confusion Matrix:
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
