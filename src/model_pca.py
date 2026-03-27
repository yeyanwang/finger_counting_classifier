import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA
from src.data_loader import get_data_pipeline

def run_pca_experiment(dataset_type='ideal'):
    os.makedirs('./results', exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_pipeline(dataset_type=dataset_type)

    pca = PCA(n_components=0.95) 
    X_train_pca = pca.fit_transform(X_train)
    
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    print(f"Original dimension: {X_train.shape[1]}")
    print(f"Dimension with 95% variance retained: {pca.n_components_}")

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'PCA Explained Variance - {dataset_type.capitalize()}')
    plt.grid()
    plt.savefig(f'./results/pca_variance_{dataset_type}.png')
    print(f"The variance chart has been saved to ./results/")
    
    return (X_train_pca, X_val_pca, X_test_pca), (y_train, y_val, y_test), pca

def save_results(data_group, label_group, pca_obj, dataset_type='ideal'):
    
    folder_name = dataset_type.capitalize() 
    data_dir = f'./data/{folder_name}'
    os.makedirs(data_dir, exist_ok=True)

    X_train_pca, X_val_pca, X_test_pca = data_group
    y_train, y_val, y_test = label_group

    np.save(os.path.join(data_dir, 'X_train_pca.npy'), X_train_pca)
    np.save(os.path.join(data_dir, 'X_test_pca.npy'), X_test_pca)
    np.save(os.path.join(data_dir, 'X_val_pca.npy'), X_val_pca)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
    
    model_path = os.path.join(data_dir, f'pca_processor_{dataset_type}.joblib')
    joblib.dump(pca_obj, model_path)
    print(f"✅ All the results have been saved {data_dir}")

if __name__ == "__main__":
    # Module 1: Ideal dataset
    data_pca, labels, pca_model = run_pca_experiment(dataset_type='ideal')
    save_results(data_pca, labels, pca_model, dataset_type='ideal')
    
    # 2. Module 2: Stressed dataset(Robustness test for module1)
    data_pca_s, labels_s, pca_model_s = run_pca_experiment(dataset_type='stressed')
    save_results(data_pca_s, labels_s, pca_model_s, dataset_type='stressed')
