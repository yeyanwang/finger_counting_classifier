finger_counting_classifier
├── exp/
│   └── notebook.py            # For temporaty testing
├── src/
│   ├── data_loader.py         # Data loader
│   ├── preprocessing.py       # Preprocessing, e.g. resizing images to 64*64
│   ├── evaluation.py          # Model evaluation models
│   ├── model_isomap.py        # ISOMAP 
│   ├── model_knn.py           # KNN 
│   ├── model_pca.py           # PCA 
│   ├── model_pca_hog.py       # PCA + HOG 
│   └── model_pca_lda.py       # PCA + LDA 
├── main_code                  # main job（Traning and testing）
└── README.md                  # README
