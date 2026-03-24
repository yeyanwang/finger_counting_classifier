## 📦 Data Pipeline & Preprocessing Guide

To ensure data consistency across all modules and achieve code decoupling, the data loading and image preprocessing logic have been separated into two independent files: `src/preprocessing.py` and `src/data_loader.py`.

### 1. `src/preprocessing.py` 
This file contains all low-level image matrix and pixel operations. If you need to process a single image independently in your module (e.g., real-time game prediction in Module 3), call the functions in this file directly.

| Function Name | Description | When to call it? |
| :--- | :--- | :--- |
| `resize_image(image)` | Resizes the image to a uniform 64x64 pixels. | Whenever uniform dimensions are needed (DataLoader calls this by default). |
| `apply_clahe(image)` | Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to mitigate uneven lighting. | For Dataset 2 (Stressed dataset) only. |
| `augment_image(image)` | Applies a random horizontal flip or slight rotation (±15 degrees). | Used to augment the training set, improving model robustness to orientation variations. |
| `normalize_pixel_values(img_array)` | Performs pixel-level Z-score standardization. | Call this before feeding the image into the model (e.g., PCA, ISOMAP). |

---

### 2. `src/data_loader.py` (Main Data Pipeline)
This file orchestrates data downloading, loading, class balancing, and splitting. **When writing `main.py` or training your specific model, you only need to call `get_data_pipeline` from this file. The rest are internal helper functions.**

| Function Name | Description | When to call it? |
| :--- | :--- | :--- |
| `download_datasets()` | Automatically downloads the Ideal and Stressed datasets from Kaggle. | Internal use only. |
| `load_and_preprocess_data()` | Reads images and extracts labels based on Kaggle's directory structure, and performs initial Resize (and CLAHE). | Internal use only. |
| `balance_classes()` | Performs data augmentation on minority classes to ensure an equal number of training samples per class. | Internal use only (Strictly prohibited for validation/test sets). |
| `flatten_and_normalize()` | Batches Z-score standardization and flattens 2D images into 1D arrays. | Internal use only. |
| `get_data_pipeline(dataset_type)` | **The Main Command Interface**. Completes downloading, loading, 70/15/15 splitting, training set balancing, and data flattening in one go. | **Core Entry Point**. Call this directly at the start of your respective Module to get ready-to-use data. |

---

### 3. `main.py` 
This is the root execution script for the entire project. It is intentionally divided into three isolated blocks so that all team members can work on the same file simultaneously without causing code conflicts.

#### **Structure Breakdown**
* **`run_module_1()`**: For foundational modeling (PCA, PCA+LDA, HOG+PCA) on the **Ideal dataset**.
* **`run_module_2()`**: For robustness evaluation (ISOMAP vs Linear Models) on the **Stressed dataset**.
* **`run_module_3()`**: For the Rock-Paper-Scissors game application using a subset of gestures (0, 2, 5).

#### **🤝 How to Collaborate (Team Workflow)**
1.  **Write your logic separately:** Create your model class in the `src/` folder (e.g., `src/model_pca.py` or `src/model_isomap.py`).
2.  **Import your model:** Add your import statement at the top of `main.py`.
3.  **Plug it in:** Instantiate your model and write your training/evaluation logic inside your assigned `run_module_X()` function.
4.  **Test locally:** Go to the very bottom of `main.py` (inside the `if __name__ == "__main__":` block) and uncomment *only* your module to test your code without running everyone else's.

### 🚀 Quick Start Example (How to use in your module)

```python
from src.data_loader import get_data_pipeline

# Inside run_module_1(): Get data under ideal conditions
X_train_id, X_val_id, X_test_id, y_train_id, y_val_id, y_test_id = get_data_pipeline('ideal')

# Inside run_module_2(): Get data under complex environments
X_train_st, X_val_st, X_test_st, y_train_st, y_val_st, y_test_st = get_data_pipeline('stressed')
