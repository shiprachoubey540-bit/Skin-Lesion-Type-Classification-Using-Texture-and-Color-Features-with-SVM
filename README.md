# Skin-Lesion-Type-Classification-Using-Texture-and-Color-Features-with-SVM
This module extracts texture-based features from dermoscopic skin lesion images using the Gray-Level Co-occurrence Matrix (GLCM) technique.

1. Loads images from the data/resized directory.
2. Converts each image to grayscale using OpenCV.
3. Reduces intensity levels (from 256 → 16 levels) to simplify texture computation.
4. Computes the GLCM matrix using scikit-image:
Distance = 1 pixel
Angle = 0° (horizontal relationship)
Normalized and symmetric matrix
5. Extracts four key texture features:
Contrast → Measures intensity variation (texture roughness)
Correlation → Measures pixel dependency
Energy → Measures uniformity (higher = more uniform texture)
Homogeneity → Measures closeness of pixel distribution
6. Stores results in a Pandas DataFrame and maps each feature set to its corresponding image_id.
7. Exports the final dataset as a csv file in the end.
