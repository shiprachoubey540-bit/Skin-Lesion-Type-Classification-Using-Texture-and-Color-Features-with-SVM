# Skin-Lesion-Type-Classification-Using-Texture-and-Color-Features-with-SVM
This module extracts texture-based features from dermoscopic skin lesion images using the Gray-Level Co-occurrence Matrix (GLCM) technique.

Loads images from the data/resized directory.
Converts each image to grayscale using OpenCV.
Reduces intensity levels (from 256 → 16 levels) to simplify texture computation.
Computes the GLCM matrix using scikit-image:
Distance = 1 pixel
Angle = 0° (horizontal relationship)
Normalized and symmetric matrix
Extracts four key texture features:
Contrast → Measures intensity variation (texture roughness)
Correlation → Measures pixel dependency
Energy → Measures uniformity (higher = more uniform texture)
Homogeneity → Measures closeness of pixel distribution
Stores results in a Pandas DataFrame and maps each feature set to its corresponding image_id.
Exports the final dataset as a csv file in the end.
