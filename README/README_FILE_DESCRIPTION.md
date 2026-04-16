## 📦 Architecture & Pipeline Guide

To ensure data consistency across all modules and achieve strict code decoupling, the project separates low-level data operations, modeling, and evaluation into dedicated modules within the `src/` directory. 

### 1. `src/preprocessing.py` 
This module handles all pixel-level transformations. These functions are designed as "pure" utility functions, making them ideal for both batch processing in the pipeline and real-time single-image prediction.

| Function Name | Description | When to call it? |
| :--- | :--- | :--- |
| `resize_image(image)` | Resizes the image to a uniform 64x64 pixels. | Mandatory for all input images to maintain feature vector consistency. |
| **`apply_gaussian_blur(image)`** | **New:** Applies Gaussian smoothing to reduce background high-frequency noise. | **Critical for the Stressed dataset**; should be called before CLAHE. |
| `apply_clahe(image)` | Enhances contrast using CLAHE to mitigate uneven lighting conditions. | Highly recommended for the Stressed dataset after denoising. |
| `augment_image(image)` | Randomly applies horizontal flips or rotations (±15°). | Training phase only; improves model robustness to hand orientations. |
| `normalize_pixel_values(arr)` | Performs pixel-level Z-score standardization. | The final step before feeding data into dimensionality reduction models. |

---

### 2. `src/data_loader.py` 
This file acts as the "Central Command" for data management. It handles the entire lifecycle of the data, from automated downloading to delivering model-ready arrays.

| Function Name | Description | Use Case |
| :--- | :--- | :--- |
| `get_data_pipeline(dataset_type)` | **The Main Command Interface.** Handles downloading, 70/15/15 splitting, and training set balancing. | **Core Entry Point.** Call this at the start of any module to get cleaned, balanced, and normalized data. |
| `balance_classes()` | Automatically augments minority classes to ensure an equal sample distribution. | Internal: Ensures the downstream classifiers are not biased toward majority classes. |
| `flatten_and_normalize()` | Converts 2D images into 1D arrays and applies global Z-score standardization. | Internal: Prepares data for traditional Machine Learning algorithms. |

---

### 3. `src/model_*.py` 
The project implements a highly modular architecture for feature extraction, dimensionality reduction, and classification. Each technique is isolated into its own script, allowing for independent testing and seamless integration into the automated tournament pipeline.

| Script Name | Core Functionality | Purpose in Pipeline |
| :--- | :--- | :--- |
| `model_pca.py` | Principal Component Analysis (PCA) | Serves as the baseline linear dimensionality reduction technique. |
| `model_pca_lda.py` | PCA + Linear Discriminant Analysis | Supervised reduction to maximize class separability; particularly effective for the limited-class Rock-Paper-Scissors subset. |
| `model_pca_hog.py` | Histogram of Oriented Gradients (HOG) | Extracts robust edge and shape features prior to PCA; highly resilient to the complex backgrounds in the Stressed dataset. |
| `model_isomap.py` | ISOMAP (Manifold Learning) | A non-linear approach used to capture the underlying geometric structure and continuous transformations of hand poses. |
| `model_knn.py` | K-Nearest Neighbors (KNN) | The universal classifier used across all experiments. It features automated **5-fold cross-validation** to dynamically tune the optimal $K$ hyperparameter. |

---

### 4. `src/evaluation.py` (Evaluation & Visualization)
This module centralizes all academic-grade reporting metrics. It is automatically triggered by the main pipeline to ensure every experiment is rigorously and consistently documented without manual intervention.

| Feature | Description | Output Location |
| :--- | :--- | :--- |
| **Confusion Matrices** | Generates class-specific error diagnosis heatmaps for every experimental setup to identify misclassification trends (e.g., confusing '2' with '3'). | `./results/` |
| **Robustness Decay** | Automatically compares performance drops between the Ideal and Stressed datasets for a given pipeline. | `./results/` |
| **Trade-off Plots** | A comprehensive analysis charting Efficiency (Feature Dimensions) against Accuracy (Model Performance) to identify the optimal modeling strategy. | `./results/` |

---

### 5. `main.py` (Full Automated Pipeline)
The `main.py` script serves as the **Full Automated Pipeline**. It coordinates the entire workflow: data fetching, dynamic model selection, training, and the generation of evaluation plots.

#### **Structure Breakdown**
* **`run_module_1()`**: Executes foundational modeling (PCA, PCA+LDA, HOG+PCA) and **ISOMAP** on the **Ideal dataset**.
* **`run_module_2()`**: Performs **Robustness Evaluation**. Compares performance across Ideal vs. Stressed environments and automatically generates **Robustness Decay** charts.
* **`run_module_3()`**: **Strategic Application (RPS)**. Features an **Automated Tournament** that cross-evaluates all models (PCA, LDA, HOG, ISOMAP) to select and deploy the best-performing pipeline for the Rock-Paper-Scissors subset (labels 0, 2, 5).

---

### 🚀 Quick Start Example

Running the entire project is designed to be as simple as a single command. The pipeline autonomously handles data fetching, hyperparameter tuning, model selection, and reporting.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all modules and generate all plots
python main.py
