import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

def train_and_evaluate_knn(data_dir, experiment_name='PCA_Ideal', min_k=5):
    
    print(f"🚀 Starting KNN Training for Experiment: {experiment_name}")
    
    
    try:
        files = os.listdir(data_dir)
        x_train_file = [f for f in files if f.startswith('X_train') and f.endswith('.npy')][0]
        x_test_file = [f for f in files if f.startswith('X_test') and f.endswith('.npy')][0]
        
        X_train = np.load(os.path.join(data_dir, x_train_file))
        X_test = np.load(os.path.join(data_dir, x_test_file))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        print(f"✅ Data loaded successfully from {data_dir}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    k_range = range(1, 31)
    param_grid = {'n_neighbors': list(k_range)}
    knn_base = KNeighborsClassifier()
    gs = GridSearchCV(knn_base, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    
    print(f"Running 5-Fold Cross Validation for K=1~30...")
    gs.fit(X_train, y_train)
    
    auto_best_k = gs.best_params_['n_neighbors']
    print(f"📊 GridSearchCV suggested K: {auto_best_k} (CV Accuracy: {gs.best_score_:.4f})")

    final_k = max(min_k, auto_best_k)
    if final_k > auto_best_k:
        print(f"⚠️  Note: Auto-best K ({auto_best_k}) is below robustness floor. Using K={final_k} instead.")
    else:
        print(f"✅ Using Auto-best K={final_k} as it meets the robustness floor.")

    os.makedirs('../results', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), gs.cv_results_['mean_test_score'], marker='o', color='green')
    plt.axvline(x=final_k, color='red', linestyle='--', label=f'Selected K={final_k}')
    plt.title(f'K-Value vs. CV Accuracy ({experiment_name})')
    plt.xlabel('Value of K')
    plt.ylabel('Mean CV Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./results/knn_tuning_{experiment_name.lower()}.png')
    print(f"📈 Tuning chart saved to ../results/knn_tuning_{experiment_name.lower()}.png")

    final_knn = KNeighborsClassifier(n_neighbors=final_k, weights='distance')
    final_knn.fit(X_train, y_train)

    y_pred = final_knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Final Evaluation for {experiment_name} ---")
    print(f"Test Accuracy: {acc:.4f}")
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred))

    model_save_path = os.path.join(data_dir, f'knn_model_{experiment_name.lower()}.joblib')
    joblib.dump(final_knn, model_save_path)
    print(f"💾 Model saved to: {model_save_path}")
    
    return final_knn

if __name__ == "__main__":
    
    train_and_evaluate_knn(data_dir='./data/Ideal', experiment_name='Ideal_PCA')
    train_and_evaluate_knn(data_dir='./data/Stressed', experiment_name='Stressed_PCA')
    train_and_evaluate_knn(data_dir='./data/HOG_Stressed', experiment_name='Stressed_HOG')
    if os.path.exists('./data/LDA_Results'):
        train_and_evaluate_knn(data_dir='./data/LDA_Results', experiment_name='Ideal_LDA')