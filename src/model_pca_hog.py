import os
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.decomposition import PCA
from src.data_loader import get_data_pipeline
from  src.model_knn import train_and_evaluate_knn

def run_hog_experiment(dataset_type='stressed'):
    
    print(f" Starting HOG Feature Engineering for: {dataset_type}")

    X_train, _, X_test, y_train, _, y_test = get_data_pipeline(dataset_type=dataset_type)
    
    X_train_2d = X_train.reshape(-1, 64, 64)
    X_test_2d = X_test.reshape(-1, 64, 64)

    def extract_hog_batch(images):
        features = []
        for img in images:
            fd = hog(img, 
                     orientations=9, 
                     pixels_per_cell=(8, 8), 
                     cells_per_block=(2, 2), 
                     visualize=False)
            features.append(fd)
        return np.array(features)

    print("Extracting HOG descriptors (Orientations=9, Cell=8x8)...")
    X_train_hog = extract_hog_batch(X_train_2d)
    X_test_hog = extract_hog_batch(X_test_2d)
    print(f"HOG feature dimension: {X_train_hog.shape[1]}")

    print("Applying PCA on HOG features...")
    pca_hog = PCA(n_components=0.95)
    X_train_hog_pca = pca_hog.fit_transform(X_train_hog)
    X_test_hog_pca = pca_hog.transform(X_test_hog)
    print(f"Reduced dimension to: {X_train_hog_pca.shape[1]}")


    folder_name = f"HOG_{dataset_type.capitalize()}"
    save_dir = f'./data/{folder_name}'
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'X_train_pca.npy'), X_train_hog_pca)
    np.save(os.path.join(save_dir, 'X_test_pca.npy'), X_test_hog_pca)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    
    joblib.dump(pca_hog, os.path.join(save_dir, f'pca_hog_processor_{dataset_type}.joblib'))
    print(f"✅ HOG+PCA data and model saved to {save_dir}")


    train_and_evaluate_knn(data_dir=save_dir, experiment_name=f'{dataset_type}_HOG')

if __name__ == "__main__":
    run_hog_experiment(dataset_type='stressed')