# PROJECT: # finger_counting_classifier🖐️

## TEAM: 121 
======================================================================
## 📌 DESCRIPTION
A system capable of recognizing gestures from 0 to 10, taking into account environmental noise, changes in viewing angles, and individual differences.
INSTALLATION

## 🧩 PLANNING MODULE
- Baseline Benchmarking (Module 1): Comparative analysis of global pixel-based projections vs. local structural gradients under ideal conditions.

- Environmental Stress Test (Module 2): Quantifying "Robustness Decay" by exposing trained classifiers to cluttered backgrounds and inconsistent lighting.

- Strategic Interaction (Module 3): A real-time "Rock-Paper-Scissors" game engine that validates inference latency and decision stability in an interactive setting.

## ❓ Research Questions:
- Global vs. Local Features: How do global pixel-based projections (PCA/LDA) compare to local gradient-based structural features (HOG) in terms of classification accuracy for multi-pose finger gestures?
- Discriminative Power of Cascaded Projections: To what extent does the cascaded PCA+LDA architecture (as per Raut & Humbe, 2014) improve class separability in the feature space compared to standard PCA?
- Environmental Sensitivity: Which feature extraction method maintains the highest Stability-to-Noise ratio when transitioning from monochrome backgrounds (Module 1) to cluttered real-world environments (Module 2)?
- Computational Efficiency: What is the trade-off between the dimensionality of the feature vector (e.g., PCA components vs. HOG descriptors) and the Inference Latency of the $k$-NN classifier?
