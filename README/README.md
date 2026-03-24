# Finger Counting Imaging Classification 👊✋✌️

## Problem Statement
In computer vision, image classifiers often achieve high accuracy under controlled conditions- such as
monochrome backgrounds and uniform lighting. However, their performance degrades in real-world scenarios 
characterized by cluttered backgrounds, varying illumination, also when the finger gestures have some 
obstructions or taking from complex angles.
This project aims to bridge this gap by building a robust finger counting model. By progressing from a baseline 
model, this system will be capable of accurately recognizing finger gestures from 0 to 5 while accounting for:
- Environmental Noise: cluttered backgrounds and inconsistent lighting.
- Perspective Variance: Gestures captured from various angles.
- Individual Diversity: Variations in hand shapes and sizes across different users.
- Interactive Application: Integration of the finger-count model into real-time environments, such as Rock-Paper-Scissors game, to evaluate robustness, usability, and reliability under dynamic, real-world conditions.

## Research Questions
- To what extent does the cascaded PCA+LDA architecture improve class separability in the feature space compared to standard PCA?
- How does a hybrid approach that integrates local gradient-based structural descriptors with global dimensionality reduction (HOG + PCA) compare to traditional pixel-level projections (PCA, PCA+LDA) in terms of classification accuracy and computational robustness for multi-pose finger gestures?
- What is the measurable "Robustness Decay" when moving from monochrome to cluttered backgrounds?
- Can ISOMAP handle complex backgrounds better than linear PCA to prevent this robustness decay?
- How does the model maintain robustness and reliability when applied to a subset of gestures used in games (0, 2, 5) compared to the full range of finger-count classes?

## Data Source 
 We will utilize two primary Kaggle datasets to simulate environmental variance:
 - Dataset 1 (Ideal): [Finger Digits 0-5](https://www.kaggle.com/datasets/roshea6/finger-digits-05). This dataset consists of 12,000 thresholded images where hand gestures are isolated against black backgrounds. This will serve as the baseline for training Model 1 and establishing core feature recognition.
 - Dataset 2 (Stressed): [Counting Fingers Dataset](https://www.kaggle.com/datasets/piyushjoshi01/counting-fingers-dataset). This dataset features hand gestures captured in natural, non-processed environments. We will use this to introduce significant environmental variations such as background clutter and varying lighting.
 
## Preprocessing
- Resizing: All images will be resized to a uniform 64x64 pixel across all pipelines.
- Train/Test Split: The dataset will be split into training (70%), validation (15%), and testing (15%) sets to ensure unbiased evaluation of model performance.
- Class Balancing: To avoid bias toward overrepresented classes, datasets will be checked for class balance. Underrepresented classes will be augmented to ensure a roughly equal number of training samples per class.
- Normalization & Scaling: Z-score Standardization at the pixel level.
- Data Augmentation: The training dataset lacks variety in hand orientation (mostly straight up). Apply random rotations and flips during training to the training set to improve robustness to orientation variations.

Note: While the above preprocessing steps apply to both datasets, Dataset 2 will additionally undergo local 
contrast enhancement (e.g., CLAHE) to mitigate the impact of inconsistent lighting.

## Methodology
 The project is structured into three progressive modules to evaluate and enhance model performance:
 ### Module 1: Gesture Recognition: Foundational Modeling under Ideal Conditions
- PCA: We implement Principal Component Analysis (PCA) as a baseline for global feature extraction. 
- PCA+LDA: Building on the comparative study of robotic hand control by (Zhang et al., 2014), this framework 
integrates PCA and LDA. This method utilizes supervised learning to find a projection that maximizes inter-class 
separability while minimizing intra-class variance, theoretically providing sharper decision boundaries than PCA 
alone.
- HOG+PCA: Drawing on the research of (Lai & Teoh, 2016), we first extracted HOG features to capture the local 
geometric structure of fingers, then applied PCA to reduce the dimensionality of the high-dimensional HOG 
vectors. 
- K-NN Classifier: K-Nearest Neighbors (K-NN) classifier will be applied only after feature extraction, to ensure 
that any performance differences can be attributed solely to the feature extraction techniques being compared.
### Module 2: Gesture Prediction: Robustness Evaluation under Complex Environments
- Complex Environment Introduction: Based on Module 1, we transition our evaluation to the "Stressed" dataset to 
assess the system's reliability against cluttered backgrounds, inconsistent lighting, perspective variance, and user 
diversity.
- Linear vs. Non-Linear Dimensionality Reduction: The feature extraction pipelines established in Module 1 (PCA, 
PCA+LDA, and HOG+PCA) all rely on linear transformations. In stressed datasets, linear PCA tends to capture 
dominant background noise rather than the actual gesture. We will introduce ISOMAP as a non-linear alternative. 
We will benchmark ISOMAP against the linear baselines to evaluate its feature separation capabilities in noisy 
environments.
- Robustness Decay Measurement: We will quantify the exact percentage drop in classification accuracy when 
moving from the ideal to the stressed dataset. This metric directly compares how well ISOMAP resists 
environmental noise compared to the linear models.
- Prediction Error Analysis: We will use confusion matrices to identify exactly which hand gestures are most 
frequently misclassified under complex conditions. This reveals the specific vulnerabilities of each pipeline before 
applying them to the interactive game in Module 3.
### Module 3: Rock-Paper-Scissors (Strategic Application) 
- Gesture Mapping: The finger-count model is adapted to the Rock-Paper-Scissors game, mapping labels 0 to Rock, 
2 to Scissors, 5 to Paper. Rock-Paper-Scissors are commonly used to evaluate image-based gesture recognition 
systems in interactive environments (Ahmed et al., 2023).
- Data Selection: To align with game mechanics, the model will be evaluated using only samples for labels 0, 2, 
and 5.
- Feature Extraction & Classification: The existing pipelines evaluated in Module 1 & 2 (PCA, PCA+LDA, 
HOG+PCA, and ISOMAP) with K-NN are deployed. This evaluates the accuracy-latency trade-off between linear 
and non-linear models in a real-time game setting.
- Game Logic Integration: Predicted gestures serve as the player’s move, while the computer move is randomly 
generated. Round outcomes follow standard Rock-Paper-Scissors rules.
