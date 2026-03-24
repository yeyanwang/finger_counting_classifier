## 📦 Data Pipeline & Preprocessing Guide

To ensure data consistency across all modules and achieve code decoupling, the data loading and image preprocessing logic have been separated into two independent files: `src/preprocessing.py` and `src/data_loader.py`.

### 1. `src/preprocessing.py` (Image Processing Toolkit)
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
