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
Features are derived from images instead of using deep learning.

4.1: GLCM Texture Features: Texture characteristics of the lesion were extracted using the Gray Level Co-occurrence Matrix (GLCM). The following features were computed:
* Contrast – Measures intensity variation between neighboring pixels.
* Correlation – Measures how correlated a pixel is to its neighbor.
* Energy – Represents uniformity of texture.
* Homogeneity – Measures closeness of distribution of elements.
  
4.2: HSV Color Features: Color information was extracted by converting images from RGB to HSV color space.
* Hue (color type)
* Saturation (color intensity)
* Value (brightness)
  
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

**Member3**

Model Evaluation & Validation:
To ensure high diagnostic accuracy, we evaluated three primary architectures using the extracted GLCM, HSV, and ABCD features.

Models implemented:
SVM (RBF Kernel): Optimized with C=10 for high-dimensional feature mapping.
Random Forest: Utilized 200 estimators to handle non-linear feature relationships.
XGBoost: Fine-tuned with a learning rate of 0.05 and mlogloss to handle multiclass complexity.

Evaluation metrics: 
Models were assessed using Confusion Matrices and Classification Reports (Precision, Recall, and F1-Score)

Deployment:
Pipeline Integration: 
The deployment script replicates the exact training pipeline:
Preprocessing: Resizing to 128x128.
Feature Fusion: Real-time extraction of 20 GLCM textures, 48 HSV color bins, and 2 ABCD shape features.
Normalization: Application of the saved StandardScaler.
Inference: Prediction via the serialized .pkl model.



