from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def run_pca_experiment(X_train, X_test=None):
    pca = PCA(n_components=0.95) 
    
    print("Fitting PCA on training data...")
    X_train_pca = pca.fit_transform(X_train)
    
    print(f"Original feature dimension: {X_train.shape[1]}")
    print(f"Reduced dimension (95% variance): {pca.n_components_}")

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance - Model 1 (Ideal)')
    plt.grid()


    if X_test is not None:
            X_test_pca = pca.transform(X_test)
            return X_train_pca, X_test_pca, pca
    
    return X_train_pca, pca

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.data_loader import get_data_pipeline

    X_train, X_val, X_test, y_train, y_val, y_test = get_data_pipeline(dataset_type='ideal')
    X_train_pca, X_test_pca, pca_model = run_pca_experiment(X_train, X_test)
    
    plt.show()
