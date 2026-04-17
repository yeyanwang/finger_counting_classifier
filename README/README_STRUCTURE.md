# Project Structure: Finger Counting Classifier

This project is organized into modular components to allow for independent development of different feature extraction techniques and seamless data integration.

```text
finger_counting_classifier
├── exp/
│   └── notebook.py            # For temporary testing and prototyping
├── src/
│   ├── data_loader.py         # Data loader, automated splitting, and balancing
│   ├── preprocessing.py       # Preprocessing (e.g., resizing to 64x64, CLAHE, Denoising)
│   ├── evaluation.py          # Model evaluation and automated plotting metrics
│   ├── model_isomap.py        # ISOMAP (Retained for vulnerability analysis)
│   ├── model_knn.py           # KNN Classifier with 5-Fold CV
│   ├── model_pca.py           # PCA Pipeline
│   ├── model_pca_hog.py       # PCA + HOG Pipeline
│   ├── model_pca_lda.py       # PCA + LDA Pipeline
│   ├── model_svm.py           # SVM Classifier with GridSearchCV tuning
│   └── model_umap.py          # UMAP (Robust non-linear manifold learning)
├── main.py                    # Main execution orchestrator (Modules 1, 2, 3)
└── README.md                  # Project documentation
