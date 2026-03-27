import os
import numpy as np
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from src.model_knn import train_and_evaluate_knn

def run_lda_experiment(dataset_type='ideal'):
    
    print(f"🧬 Starting LDA Dimensionality Reduction: {dataset_type}")

    data_dir = f'./data/{dataset_type.capitalize()}'
    try:
        X_train_pca = np.load(os.path.join(data_dir, 'X_train_pca.npy'))
        X_test_pca = np.load(os.path.join(data_dir, 'X_test_pca.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        print(f"✅ Loaded PCA features from {data_dir}")
    except FileNotFoundError:
        print(f"❌ Error: PCA files not found. Please run model_pca.py first.")
        return

    print(f"Original PCA dimension: {X_train_pca.shape[1]}")
    lda = LDA(n_components=5)
    
    X_train_lda = lda.fit_transform(X_train_pca, y_train) 
    X_test_lda = lda.transform(X_test_pca)
    print(f"✨ LDA reduced dimension to: {X_train_lda.shape[1]}")


    lda_save_dir = f'./data/LDA_{dataset_type.capitalize()}'
    os.makedirs(lda_save_dir, exist_ok=True)
    
    np.save(os.path.join(lda_save_dir, 'X_train_pca.npy'), X_train_lda)
    np.save(os.path.join(lda_save_dir, 'X_test_pca.npy'), X_test_lda)
    np.save(os.path.join(lda_save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(lda_save_dir, 'y_test.npy'), y_test)
    
    joblib.dump(lda, os.path.join(lda_save_dir, 'lda_processor.joblib'))
    print(f"💾 LDA data and model saved to {lda_save_dir}")

    
    train_and_evaluate_knn(data_dir=lda_save_dir, experiment_name=f'{dataset_type}_LDA')

if __name__ == "__main__":
    run_lda_experiment(dataset_type='ideal')