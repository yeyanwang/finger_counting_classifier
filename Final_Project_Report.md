# Final Report: Finger Counting Imaging Classification

**Team Members:** Sichao Qiu, Yuexin Wang, Yeyan Wang <br/>
**Course:** ISYE 6740 - Computational Data Analysis (Spring 2026)

---

## 1. Problem Statement
In computer vision, image classifiers often achieve high accuracy under controlled conditions- such as monochrome backgrounds and uniform lighting. However, their performance degrades in real-world scenarios characterized by cluttered backgrounds, varying illumination, also when the finger gestures have some obstructions or taking from complex angles. 

This project aims to bridge this gap by building a robust finger counting model. By progressing from a baseline model, this system will be capable of accurately recognizing finger gestures from 0 to 5 while accounting for:
- **Environmental Noise:** Cluttered backgrounds and inconsistent lighting.
- **Perspective Variance:** Gestures captured from various angles.
- **Individual Diversity:** Variations in hand shapes and sizes across different users.
- **Interactive Application:** Integration of the finger-count model into real-time environments, such as Rock-Paper-Scissors game, to evaluate robustness, usability, and reliability under dynamic, real-world conditions.

While contemporay "black-box" frameworks provide high end-to-end accuracy and robust computational results, for instances, the MediaPipe framework achieves a mean precision of 95.7% in palm identification by utilizing a coordinated machine learning pipeline to infer 3D landmarks (Roy et al., 2022) Similarly, Convolutional Neural Networks (CNN) and models like YOLOv3 are used to learn features and classify gestures directly from video frames, achieving accuracies as high as 97.68%. Futhermore, by using Skeletal and 3D Modeling can improve the detection of complex features, such as track the skeletal joints of the hand (such as 20 or 25 joints) as well as their trajectories, curvatures and angles(Oudah et al., 2020) However, their logic for a specific prediction is hidden within millions of parameters, they often obscure the relationship between raw pixel variance and environmental stressor, we want to quantify how specific real-world scenarios would degrade the underlying feature space in our project.

Therefore, our methodology is not merely a classification task but a comparative study on feature robustness, aiming to quantify the transition from raw intensity data to structural descriptors. We deliberately employs a suite of interpretable algorithms: PCA, LDA, HOG, and UMAP, to deconstruct the mechanics of hand gesture recognition.

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
Under ideal conditions, all three feature extractors achieved perfect or near-perfect classification accuracy, confirming that clean, thresholded gesture images are highly separable in a compressed feature space. 

The PCA variance plot below shows that approximately 95% of the total variance is captured within the first 100 principal components, confirming aggressive but lossless dimensionality reduction. 
<p align="center">
  <img src="./results/pca_variance_ideal.png" alt="PCA Variance - Ideal"><br>
  <em>Figure: PCA Variance - Ideal</em>
</p>

The KNN tuning curve identifies the optimal K (K=5). The confusion matrix confirms clean per-class separation with no systematic misclassifications under ideal conditions.
<p align="center">
  <img src="./results/knn_tuning_ideal_pca.png" alt="KNN Tuning - Ideal PCA"><br>
  <em>Figure: KNN Tuning - Ideal PCA</em>
</p>
<p align="center">
  <img src="./results/confusion_matrix_ideal_pca.png" alt="Confusion Matrix - Ideal PCA"><br>
  <em>Figure: Confusion Matrix - Ideal PCA</em>
</p>

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
Both KNN and SVM achieved very high accuracy on the RPS subset, with SVM slightly outperforming KNN (99.97% vs. 99.94%), as shown in the classifier comparison bar chart below. While the difference is small, SVM was chosen because it can create more flexible decision boundaries in the reduced 5-dimensional LDA space, where the classes are tightly clustered. Based on this, SVM was selected as the final model for deployment.
<p align="center">
  <img src="./results/rps_classifier_accuracy_comparison.png" alt="RPS Classifier Accuracy Comparison: KNN vs SVM"><br>
  <em>Figure: RPS Classifier Accuracy Comparison: KNN vs SVM</em>
</p>

The KNN tuning curve shows that accuracy stayed at 100% for all K values from 1 to 30. This suggests the LDA feature space separates the three RPS classes almost perfectly. As a result, the choice of K has no practical impact on training performance for this subset, and K=5 is selected as a reasonable default.
<p align="center">
  <img src="./results/knn_tuning_game_lda_knn.png" alt="knn tuning game lda knn.png"><br>
  <em>Figure: KNN Tuning Curve - RPS Game Subset, LDA Features</em>
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
1. Roy, K., & Akif, M. A. H. (2022, February). Real time hand gesture based user friendly human computer interaction system. In 2022 International Conference on Innovations in Science, Engineering and Technology (ICISET) (pp. 260-265). IEEE.
2. Oudah, M., Al-Naji, A., & Chahl, J. (2020). Hand gesture recognition based on computer vision: a review of techniques. journal of Imaging, 6(8), 73.
3. Zhang, D., Zhao, X., Han, J., & Zhao, Y. (2014). A comparative study on PCA and LDA based EMG pattern recognition for anthropomorphic robotic hand. 2014 IEEE International Conference on Robotics and Automation (ICRA), 4850-4855.
4. Lai, C. Q., & Teoh, S. S. (2016). An Efficient Method of HOG Feature Extraction Using Selective Histogram Bin and PCA Feature Reduction. Advances in Electrical and Computer Engineering, 16(4), 101-108.
5. Ahmed, F., Khan, W. A., Iqbal, M., Abazeed, A. R. A., Alrababah, H., & Khan, M. F. (2023). Rock-paper-scissors image classification using transfer learning. 2023 International Conference on Business Analytics for Technology and Security (ICBATS), 1-6.
6. Reza, A. M. (2004). Realization of the Contrast Limited Adaptive Histogram Equalization (CLAHE) for Real-Time Image Enhancement. Journal of VLSI Signal Processing Systems, 38, 35-44.
