import os
import numpy as np
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from src.model_knn import train_and_evaluate_knn

def run_lda_experiment(dataset_type='ideal'):
    """
    Performs Linear Discriminant Analysis (LDA) on top of PCA features.
    LDA is a supervised technique that maximizes class separation.
    """
    print(f"\n Starting LDA Dimensionality Reduction: {dataset_type}")

    # Step 1: Load pre-processed PCA features from the corresponding directory
    data_dir = f'./data/{dataset_type.capitalize()}'
    
    try:
        X_train_pca = np.load(os.path.join(data_dir, 'X_train_pca.npy'))
        X_test_pca = np.load(os.path.join(data_dir, 'X_test_pca.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        print(f" PCA features successfully loaded from {data_dir}")
    except FileNotFoundError:
        print(f" Error: PCA files not found. Please run model_pca.py first for the {dataset_type} set.")
        return

    print(f"Original PCA dimension: {X_train_pca.shape[1]}")
    
    # Step 2: Initialize and fit LDA
    # n_components must be <= (number of classes - 1). For 6 classes, max is 5.
    lda = LDA(n_components=5)
    
    # LDA is supervised, so we must provide y_train to find the optimal projection
    X_train_lda = lda.fit_transform(X_train_pca, y_train) 
    X_test_lda = lda.transform(X_test_pca)
    print(f" LDA projection complete. Reduced dimension to: {X_train_lda.shape[1]}")

    # Step 3: Save results to a specific LDA folder
    lda_save_dir = f'./data/LDA_{dataset_type.capitalize()}'
    os.makedirs(lda_save_dir, exist_ok=True)
    
    # We keep the 'X_train_pca.npy' naming convention to remain compatible with 
    # the train_and_evaluate_knn function's default search parameters.
    np.save(os.path.join(lda_save_dir, 'X_train_pca.npy'), X_train_lda)
    np.save(os.path.join(lda_save_dir, 'X_test_pca.npy'), X_test_lda)
    np.save(os.path.join(lda_save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(lda_save_dir, 'y_test.npy'), y_test)
    
    # Save the LDA transformer object
    joblib.dump(lda, os.path.join(lda_save_dir, 'lda_processor.joblib'))
    print(f" LDA features and model saved to {lda_save_dir}")

    # Step 4: Evaluation
    # Since train_and_evaluate_knn now returns 4 values, we can simply call it.
    train_and_evaluate_knn(data_dir=lda_save_dir, experiment_name=f'{dataset_type}_LDA', feature_suffix='pca')

if __name__ == "__main__":
    # Standard execution for the Ideal dataset
    run_lda_experiment(dataset_type='ideal')
