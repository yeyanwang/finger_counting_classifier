import os
import numpy as np
import joblib
from sklearn.manifold import Isomap
from src.data_loader import get_data_pipeline
from src.model_knn import train_and_evaluate_knn

def model_isomap(X_train, X_test=None, n_neighbors=5, n_components=50):
    """
    Applies Isomap dimensionality reduction to the dataset.
    """
    print(f"Fitting ISOMAP: Reducing from {X_train.shape[1]} to {n_components} dimensions...")
    
    # Initialization and Training
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    X_train_reduced = isomap.fit_transform(X_train)
    
    print("ISOMAP Training Complete!")
    
    # Test
    if X_test is not None:
        print("Transforming test data using ISOMAP...")
        X_test_reduced = isomap.transform(X_test)
        return X_train_reduced, X_test_reduced, isomap
        
    return X_train_reduced, isomap

if __name__ == "__main__":
    # 1. Load data from pipeline
    dataset_type = 'ideal' # Or 'stressed'
    X_train, _, X_test, y_train, _, y_test = get_data_pipeline(dataset_type=dataset_type)

    # 2. Run Isomap reduction
    # Note: 50 components is a common choice for high-dimensional image data
    X_train_reduced, X_test_reduced, isomap_obj = model_isomap(X_train, X_test, n_neighbors=5, n_components=50)

    # 3. Saving results for the automated evaluation pipeline
    # Folder name follows the convention expected by run_evaluations.py
    isomap_save_dir = './data/Isomap_Ideal'
    os.makedirs(isomap_save_dir, exist_ok=True)
    
    # Save features
    np.save(os.path.join(isomap_save_dir, 'X_train_pca.npy'), X_train_reduced)
    np.save(os.path.join(isomap_save_dir, 'X_test_pca.npy'), X_test_reduced)
    np.save(os.path.join(isomap_save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(isomap_save_dir, 'y_test.npy'), y_test)
    
    # Save the Isomap file
    joblib.dump(isomap_obj, os.path.join(isomap_save_dir, 'isomap_processor.joblib'))
    print(f"✅ ISOMAP results and model saved to {isomap_save_dir}")
