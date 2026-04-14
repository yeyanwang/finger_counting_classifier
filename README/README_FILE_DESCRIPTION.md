## Data Pipeline & Preprocessing Guide

To ensure data consistency across all modules and achieve strict code decoupling, the project separates low-level operations from high-level orchestration into `src/preprocessing.py`, `src/data_loader.py`, and `src/evaluation_plots.py`.

### 1. `src/preprocessing.py` 
This module handles all pixel-level transformations. These functions are designed to be "pure" utility functions, making them ideal for both batch processing in the pipeline and real-time single-image prediction.

| Function Name | Description | When to call it? |
| :--- | :--- | :--- |
| `resize_image(image)` | Resizes the image to a uniform 64x64 pixels. | Mandatory for all input images to maintain feature vector consistency. |
| **`apply_gaussian_blur(image)`** | **New:** Applies Gaussian smoothing to reduce background high-frequency noise. | **Critical for the Stressed dataset**; should be called before CLAHE. |
| `apply_clahe(image)` | Enhances contrast using CLAHE to mitigate uneven lighting conditions. | Highly recommended for the Stressed dataset after denoising. |
| `augment_image(image)` | Randomly applies horizontal flips or rotations (±15°). | Training phase only; improves model robustness to hand orientations. |
| `normalize_pixel_values(arr)` | Performs pixel-level Z-score standardization. | The final step before feeding data into PCA, LDA, or KNN. |

---

### 2. `src/data_loader.py`
This file acts as the "Central Command" for data. It manages the entire lifecycle of the data, from automated downloading to delivering model-ready arrays.

| Function Name | Description | Use Case |
| :--- | :--- | :--- |
| `get_data_pipeline(dataset_type)` | **The Main Command Interface.** Handles downloading, 70/15/15 splitting, and training set balancing. | **Core Entry Point.** Call this at the start of any module to get cleaned, balanced, and normalized data. |
| `balance_classes()` | Automatically augments minority classes to ensure an equal sample distribution. | Internal: Ensures the KNN classifier is not biased toward majority classes. |
| `flatten_and_normalize()` | Converts 2D images into 1D arrays and applies global Z-score standardization. | Internal: Prepares data for traditional Machine Learning algorithms. |

---

### 3. `main.py`
The `main.py` script has been upgraded to a **Full Automated Pipeline**. It coordinates the entire workflow: data fetching, model selection, training, and the generation of academic-grade evaluation plots.

#### **Structure Breakdown**
* **`run_module_1()`**: Executes foundational modeling (PCA, PCA+LDA, HOG+PCA) and **ISOMAP** on the **Ideal dataset**.
* **`run_module_2()`**: Performs **Robustness Evaluation**. Compares performance across Ideal vs. Stressed environments and automatically generates **Robustness Decay** charts.
* **`run_module_3()`**: **Strategic Application (RPS)**. Features an **Automated Tournament** that selects the best-performing pipeline for the Rock-Paper-Scissors subset (labels 0, 2, 5).

#### ** Automated Evaluation Features**
The pipeline automatically triggers `src/evaluation_plots.py` to save the following results into the `./results/` folder:
1.  **Confusion Matrices**: Detailed error diagnosis for every experimental setup.
2.  **Tuning Curves**: Visualization of the 5-fold cross-validation process for the optimal $K$ value.
3.  **Trade-off Plots**: Comprehensive analysis of Efficiency (Dimensions) vs. Accuracy (Performance).

---

### 🚀 Quick Start Example

Running the entire project is now as simple as a single command. The pipeline handles data fetching, model selection, and reporting.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all modules and generate all plots
python main.py
