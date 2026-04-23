# Final Report: Finger Counting Imaging Classification

**Team Members:** Sichao Qiu, Yuexin Wang, Yeyan Wang <br/>
**Course:** ISYE 6740 - Computational Data Analysis (Spring 2026)

---

## 1. Problem Statement
In computer vision, image classifiers often achieve high accuracy under controlled conditions- such as monochrome backgrounds and uniform lighting. However, their performance degrades in real-world scenarios characterized by cluttered backgrounds, varying illumination, also when the finger gestures have some obstructions or taking from complex angles. 

This project aims to bridge this gap by building a robust finger counting model. By progressing from a baseline model, this system will be capable of accurately recognizing finger gestures from 0 to 5 while accounting for:
1. **Environmental Noise:** Cluttered backgrounds and inconsistent lighting.
2. **Perspective Variance:** Gestures captured from various angles.
3. **Individual Diversity:** Variations in hand shapes and sizes across different users.
4. **Interactive Application:** Integration of the finger-count model into real-time environments, such as Rock-Paper-Scissors game, to evaluate robustness, usability, and reliability under dynamic, real-world conditions. 

---

## 2. Research Questions
1. To what extent does the cascaded PCA+LDA architecture improve class separability in the feature space compared to standard PCA?
2. How does a hybrid approach that integrates local gradient-based structural descriptors with global dimensionality reduction (HOG + PCA) compare to traditional pixel-level projections (PCA, PCA+LDA) in terms of classification accuracy and computational robustness for multi-pose finger gestures?
3. What is the measurable "Robustness Decay" when moving from monochrome to cluttered backgrounds?
4. Can manifold learning methods such as ISOMAP or UMAP handle complex backgrounds better than linear PCA to prevent this robustness decay?
5. How does the model maintain robustness and reliability when applied to a subset of gestures used in games (0, 2, 5) compared to the full range of finger-count classes? 

---

## 3. Data Source & Preprocessing

### 3.1 Data Source
* **Dataset 1 (Ideal):** [Finger Digits 0-5](https://www.kaggle.com/datasets/roshea6/finger-digits-05). 12,000 thresholded images with isolated hand gestures against black backgrounds.
* **Dataset 2 (Stressed):** [Counting Fingers Dataset](https://www.kaggle.com/datasets/piyushjoshi01/counting-fingers-dataset). Gestures captured in natural environments with significant background clutter and inconsistent lighting.
* **Dataset 3 (Game):** [Fingers Dataset](https://www.kaggle.com/datasets/koryakinp/fingers). This dataset contains hand gesture images spanning classes 0–5, and will serve as unseen data for evaluating our Rock-Paper-Scissors game simulation in Module 3.

### 3.2 Preprocessing and Standardization
To ensure mathematical stability and prevent bias, the preprocessing and standardization steps are excecuted before model processing.
* **Resizing & Flattening:** All images are uniformly reshaped to 64x64 pixels and flattened into 4096-dimensional vectors for linear processing.
* **Normalization:** Pixel-level Z-score standardization is applied to ensure scale uniformity across all features.
* **Class Balancing:** The training set undergoes automated class augmentation to ensure an equal sample distribution, preventing downstream classifiers from biasing toward majority gestures.
* **Environmental Mitigation (Dataset 2 Only):** For the Stressed dataset, Gaussian smoothing is applied to reduce high-frequency noise, followed by Contrast Limited Adaptive Histogram Equalization (CLAHE) to mitigate uneven lighting.

---

## 4. Methodology
The project is structured into three progressive modules to evaluate and enhance model performance. 

### 4.1 Module 1: Foundational Modeling (`run_module_1()`)
This module establishes a baseline performance ceiling by evaluating foundational feature extractors on the noise-free, Ideal dataset. 
* We deploy global linear projections (**PCA**), supervised linear projections (**PCA+LDA**), and local gradient descriptors (**HOG+PCA**). 
* Each model is trained using a K-Nearest Neighbors (K-NN) classifier optimized via 5-Fold Cross-Validation. 
* *Note: Non-linear manifold learning models are intentionally excluded in this phase to prevent overfitting on clean data.*

### 4.2 Module 2: Robustness Evaluation (`run_module_2()`)
This core phase acts as a "Robustness Tournament" by transitioning models to the Stressed dataset to evaluate resilience against environmental noise.
* **Algorithm Showdown:** We evaluate all linear extractors against non-linear manifold learning techniques (**ISOMAP** and **UMAP**). UMAP is specifically included to test its ability to preserve local topology with higher computational stability than ISOMAP.
* **Robustness Decay Quantification:** The automated evaluation module calculates the "Robustness Decay"—the percentage drop in accuracy between the Ideal and Stressed datasets—to dynamically identify the most resilient feature extraction method.

### 4.3 Module 3: Rock-Paper-Scissors Application (`run_module_3()`)
We translate our experimental findings into a strategic deployment.
* **Data Subsetting & Downsampling:** The dataset is filtered to include only Rock (0), Scissors (2), and Paper (5). To ensure a perfectly balanced simulation, these classes are strictly downsampled to a maximum of 500 samples per class (`MAX_PER_CLASS = 500`).
* **Classifier Showdown:** Inheriting the winning feature extractor from Module 2, we initiate a final comparison between **K-NN** and a Support Vector Machine (**SVM**) using an RBF kernel. 
* **Game Simulation:** The winning classifier is deployed into a 10-round RPS simulation. Game outcomes (Win/Lose/Tie) are determined by comparing the model's predicted gesture against a randomly sampled computer move, with representative outcome examples visualized for analysis.
---

## 5. Result and Analysis

### 5.1 Module 1: Foundational Modeling under Ideal Conditions


![PCA Variance - Ideal](./results/pca_variance_ideal.png)
![KNN Tuning - Ideal PCA](./results/knn_tuning_ideal_pca.png)
![Confusion Matrix - Ideal PCA](./results/confusion_matrix_ideal_pca.png)

### 5.2 Module 2: Robustness Evaluation under Complex Environments

**Objective:**
The focus of Module 2 is the "Robustness Tournament." We evaluated how environmental complexity affects classification performance by measuring "Robustness Decay"—the percentage drop in accuracy when moving from the ideal dataset to the stressed dataset.

<p align="center">
  <img src="./results/pca_variance_stressed.png" alt="PCA Variance - Stressed"><br>
  <em>Figure: PCA Explained Variance for the Stressed Dataset</em>
</p>

<div align="center">
  
**Model Performance Result (Stressed Dataset):**
*Training Samples: 78 | Test Samples: 17*

| Feature Extractor | Stressed Accuracy | Ideal Accuracy | Robustness Decay |
| :--- | :---: | :---: | :---: |
| **LDA** | **0.8824** | **1.0000** | **11.76%** |
| HOG + PCA | 0.8824 | 1.0000 | 11.76% |
| PCA | 0.7647 | 1.0000 | 23.53% |
| ISOMAP | 0.7647 | 1.0000 | 23.53% |
| UMAP | 0.5882 | 1.0000 | 41.18% |

</div>

**Discussion of Robustness Analysis:**
* **Linear vs. Non-linear Resilience:** Surprisingly, non-linear manifold learning (**ISOMAP**) performed identically to linear **PCA** (0.7647). **UMAP** suffered the highest decay (41.18%), suggesting that for low-sample noisy environments, complex non-linear projections may overfit background noise.
* **Structural Descriptors:** The **HOG + PCA** pipeline demonstrated high accuracy performance (0.8824). Gradient-based structural descriptors effectively isolated gesture geometry from pixel-level background fluctuations.
* **Supervised Feature Extraction:** **LDA** tied for the highest accuracy (same performance with HOG + PCA). By maximizing inter-class separability, LDA successfully filtered out environmental stress that unsupervised methods (PCA) failed to ignore.

<p align="center">
  <img src="./results/robustness_decay_pca_+_knn.png" alt="Robustness Decay Chart"><br>
  <em>Figure: Robustness Decay Comparison</em>
</p>

**Conclusion for Model Selection:**
Although both LDA and HOG+PCA achieved identical accuracy (0.8824) and robustness decay (11.76%), **LDA** was selected as the final robust feature extractor for the interactive application in Module 3 due to its superior computational efficiency:
1. **Dimensionality:** LDA successfully compressed the data into just **5 dimensions** (n_classes - 1), whereas the HOG+PCA pipeline required **60 dimensions** to achieve the same result, which shows a much lower computational cost.
2. **Inference Latency:** For the real-time Rock-Paper-Scissors game, LDA requires only computationally lightweight matrix multiplications during inference, unlike HOG which requires expensive gradient calculations across the image. 

<p align="center">
  <img src="./results/confusion_matrix_stressed_pca.png" alt="Confusion Matrix - Stressed PCA"><br>
  <em>Figure: Confusion Matrix for Stressed Dataset</em>
</p>

### 5.3 Module 3: Rock-Paper-Scissors (Strategic Application)
**Objective:**
With LDA selected as the feature extraction method, this module compares KNN and SVM to determine which works better for the three-class Rock-Paper-Scissors (RPS) subset. The best model is then used in a live 10-round simulation.

**Classifier Comparison:**
Both KNN and SVM achieved very high accuracy on the RPS subset, with SVM slightly outperforming KNN (0.9997 vs. 0.9994), as shown in the classifier comparison bar chart below. While the difference is small, SVM was chosen because it can create more flexible decision boundaries in the reduced 5-dimensional LDA space, where the classes are tightly clustered. Based on this, SVM was selected as the final model for deployment.
<p align="center">
  <img src="./results/rps_classifier_accuracy_comparison.png" alt="RPS Classifier Accuracy Comparison: KNN vs SVM"><br>
  <em>Figure: RPS Classifier Accuracy Comparison: KNN vs SVM</em>
</p>

**Game Simulation Results:**
The deployed model was evaluated across a 10-round Rock-Paper-Scissors game simulation against a randomly sampling computer opponent. The outcome examples below shows one representative Win, Lose, and Tie scenario, confirming that the model correctly identifies gesture classes even under the compressed LDA feature representation. 
<p align="center">
  <img src="./results/rps_outcomes.png" alt="RPS Game Outcome Examples (Win / Lose / Tie)"><br>
  <em>Figure: RPS Game Outcome Examples (Win / Lose / Tie)</em>
</p>

---

## 6. Conclusion
This project developed a three-module pipeline for robust finger-counting gesture classification, progressively evaluating feature extraction strategies from ideal to stressed conditions and deploying the optimal model into a real-time Rock-Paper-Scissors application.

**Key Findings:**
* **Ideal conditions can be misleading:** In Module 1, all feature extractors performed almost perfectly on clean data, but their performance dropped differently under stressed conditions. This shows that evaluating only on ideal data overestimates how well models perform in real-world settings.
* **Supervised methods are more robust under stress:** LDA performed better than unsupervised methods because it focuses on separating classes. It had the lowest drop in performance (11.76%), while methods like PCA, ISOMAP, and especially UMAP degraded much more (UMAP dropped over 40%).
* **Non-linear methods didn’t help as expected:** ISOMAP performed about the same as PCA, and UMAP performed the worst under stress. This suggests that for smaller and noisier datasets, more complex non-linear methods may overfit to noise instead of learning meaningful gesture patterns.
* **LDA is the best choice for deployment:** LDA reduced the data to just 5 dimensions while still keeping 88.24% accuracy under stressed conditions. It’s also efficient, making it a good fit for real-time applications.
* **The RPS simulation shows the model works in practice:** The final LDA-based pipeline correctly classified gestures in the RPS game. Any wins or losses were due to the randomness of the opponent, not model errors, which shows the system is reliable.

**Limitations and Future Work:** 
* The stressed dataset is small, so results may not be fully reliable. Future work should include a larger dataset to better validate performance.
* Testing the model in a real-time webcam setting would provide a better understanding of LDA’s speed and practicality compared to methods like HOG.

---

## 7. References
1. Zhang, D., Zhao, X., Han, J., & Zhao, Y. (2014). A comparative study on PCA and LDA based EMG pattern recognition for anthropomorphic robotic hand. 2014 IEEE International Conference on Robotics and Automation (ICRA), 4850-4855.
2. Lai, C. Q., & Teoh, S. S. (2016). An Efficient Method of HOG Feature Extraction Using Selective Histogram Bin and PCA Feature Reduction. Advances in Electrical and Computer Engineering, 16(4), 101-108.
3. Ahmed, F., Khan, W. A., Iqbal, M., Abazeed, A. R. A., Alrababah, H., & Khan, M. F. (2023). Rock-paper-scissors image classification using transfer learning. 2023 International Conference on Business Analytics for Technology and Security (ICBATS), 1-6.
4. Reza, A. M. (2004). Realization of the Contrast Limited Adaptive Histogram Equalization (CLAHE) for Real-Time Image Enhancement. Journal of VLSI Signal Processing Systems, 38, 35-44.
