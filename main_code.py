"""
This is the main execution script. It orchestrates the data loading, 
model training, and evaluation for all three modules.
"""

import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# 1. Import our data pipeline
from src.data_loader import get_data_pipeline

# Model Imports
from src.model_pca import run_pca_experiment, save_results as save_pca_results
from src.model_knn import train_and_evaluate_knn
from src.model_pca_lda import run_lda_experiment
from src.model_pca_hog import run_hog_experiment
from src.model_isomap import model_isomap

# Evaluation Imports for Automated Plots
from src.evaluation import plot_confusion_matrix, plot_robustness_decay, plot_tradeoff

def run_module_1():
    """
    Module 1: Foundational Modeling under Ideal Conditions
    """
    print("MODULE 1: IDEAL CONDITIONS")
    
    # Step 1.1: PCA
    print("\n[Step 1.1] Running Basic PCA Pipeline...")
    pca_data, pca_labels, pca_model = run_pca_experiment(dataset_type='ideal')
    save_pca_results(pca_data, pca_labels, pca_model, dataset_type='ideal')
    
    # Updated to capture acc for robustness comparison in Module 2
    _, ideal_pca_acc, y_t, y_p = train_and_evaluate_knn(data_dir='./data/Ideal', experiment_name='Ideal_PCA')
    plot_confusion_matrix(y_t, y_p, 'Ideal_PCA')

    # Step 1.2: PCA + LDA
    print("\n[Step 1.2] Running PCA + LDA Pipeline...")
    run_lda_experiment(dataset_type='ideal')

    # Step 1.3: HOG + PCA
    print("\n[Step 1.3] Running HOG + PCA Pipeline...")
    run_hog_experiment(dataset_type='ideal')

    # Step 1.4: ISOMAP
    print("\n[Step 1.4] Running ISOMAP Pipeline...")
    X_train, _, X_test, y_train, _, y_test = get_data_pipeline('ideal')
    X_tr_iso, X_te_iso, _ = model_isomap(X_train, X_test)
    
    iso_dir = './data/Isomap_Ideal'
    os.makedirs(iso_dir, exist_ok=True)
    np.save(os.path.join(iso_dir, 'X_train_pca.npy'), X_tr_iso)
    np.save(os.path.join(iso_dir, 'X_test_pca.npy'), X_te_iso)
    np.save(os.path.join(iso_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(iso_dir, 'y_test.npy'), y_test)
    train_and_evaluate_knn(data_dir=iso_dir, experiment_name='Ideal_Isomap')

    print("\n MODULE 1 COMPLETE.")
    return ideal_pca_acc # Pass this to Module 2 for robustness plot


def run_module_2(ideal_baseline_acc):
    """
    Module 2: Robustness Evaluation under Complex Environments
    """
    print("STARTING MODULE 2: STRESSED CONDITIONS")

    # Step 2.1: Stressed PCA
    pca_data, labels, pca_model = run_pca_experiment(dataset_type='stressed')
    save_pca_results(pca_data, labels, pca_model, dataset_type='stressed')
    _, stressed_pca_acc, y_t, y_p = train_and_evaluate_knn(data_dir='./data/Stressed', experiment_name='Stressed_PCA', feature_suffix='pca')
    
    # Automated Robustness Decay Plot
    plot_robustness_decay(ideal_acc=ideal_baseline_acc, stressed_acc=stressed_pca_acc, model_name="PCA + KNN")
    plot_confusion_matrix(y_t, y_p, 'Stressed_PCA')

    # Step 2.2: Stressed PCA + LDA
    print("\n[Step 2.2] Running PCA + LDA on Stressed Dataset...")
    run_lda_experiment(dataset_type='stressed')
    
    # Step 2.3: Stressed HOG + PCA
    print("\n[Step 2.3] Running HOG + PCA on Stressed Dataset...")
    run_hog_experiment(dataset_type='stressed')

    print("\n MODULE 2 COMPLETE.")

# Can be updated......
def run_module_3():
    """
    Module 3: Rock-Paper-Scissors (Strategic Application)
    Automated Tournament: Compares PCA, LDA, HOG+PCA, and ISOMAP to find the best game classifier.
    """
    print("\nSTARTING MODULE 3: ROCK-PAPER-SCISSORS")

    # Step 3.1: Data Acquisition & Filtering
    # Load Stressed data for a realistic "game" scenario
    X_train, _, X_test, y_train, _, y_test = get_data_pipeline('stressed')

    # Filter labels to only include 0 (Rock), 2 (Scissors), 5 (Paper)
    game_labels = [0, 2, 5]
    mask_train = np.isin(y_train, game_labels)
    mask_test = np.isin(y_test, game_labels)

    X_train_game = X_train[mask_train]
    y_train_game = y_train[mask_train]
    X_test_game = X_test[mask_test]
    y_test_game = y_test[mask_test]
    
    print(f"Game Subset created with {len(X_train_game)} samples.")

    # Dictionary to store accuracy and transformed data for final deployment
    results = {}
    candidates_data = {}
    
    tmp_dir = './data/game_Temp'
    os.makedirs(tmp_dir, exist_ok=True)

    # Candidate A: Standard PCA Pipeline
    print("\n[Candidate A] Evaluating PCA...")
    from sklearn.decomposition import PCA
    pca_obj = PCA(n_components=0.95)
    X_tr_pca = pca_obj.fit_transform(X_train_game)
    X_te_pca = pca_obj.transform(X_test_game)
    
    # Save temporarily for KNN evaluation
    np.save(os.path.join(tmp_dir, 'X_train_pca.npy'), X_tr_pca)
    np.save(os.path.join(tmp_dir, 'X_test_pca.npy'), X_te_pca)
    np.save(os.path.join(tmp_dir, 'y_train.npy'), y_train_game)
    np.save(os.path.join(tmp_dir, 'y_test.npy'), y_test_game)
    
    _, acc_pca, _, _ = train_and_evaluate_knn(data_dir=tmp_dir, experiment_name='game_Try_PCA')
    results['PCA'] = acc_pca
    candidates_data['PCA'] = (X_tr_pca, X_te_pca)

    # Candidate B: PCA + LDA
    print("\n[Candidate B] Evaluating LDA...")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    lda_obj = LDA(n_components=2) # 3 classes - 1 = 2
    X_tr_lda = lda_obj.fit_transform(X_tr_pca, y_train_game)
    X_te_lda = lda_obj.transform(X_te_pca)
    
    np.save(os.path.join(tmp_dir, 'X_train_pca.npy'), X_tr_lda)
    np.save(os.path.join(tmp_dir, 'X_test_pca.npy'), X_te_lda)
    
    _, acc_lda, _, _ = train_and_evaluate_knn(data_dir=tmp_dir, experiment_name='game_Try_LDA')
    results['LDA'] = acc_lda
    candidates_data['LDA'] = (X_tr_lda, X_te_lda)

    # Candidate C: HOG + PCA
    print("\n[Candidate C] Evaluating HOG + PCA...")
    from skimage.feature import hog
    # Helper to extract HOG features (requires 2D reshaping)
    def extract_hog(data):
        return np.array([hog(img.reshape(64, 64), orientations=9, pixels_per_cell=(8, 8), 
                             cells_per_block=(2, 2)) for img in data])
    
    X_tr_hog = extract_hog(X_train_game)
    X_te_hog = extract_hog(X_test_game)
    
    pca_hog = PCA(n_components=0.95)
    X_tr_hog_pca = pca_hog.fit_transform(X_tr_hog)
    X_te_hog_pca = pca_hog.transform(X_te_hog)
    
    np.save(os.path.join(tmp_dir, 'X_train_pca.npy'), X_tr_hog_pca)
    np.save(os.path.join(tmp_dir, 'X_test_pca.npy'), X_te_hog_pca)
    
    _, acc_hog, _, _ = train_and_evaluate_knn(data_dir=tmp_dir, experiment_name='game_Try_HOG')
    results['HOG'] = acc_hog
    candidates_data['HOG'] = (X_tr_hog_pca, X_te_hog_pca)

    # Candidate D: ISOMAP
    print("\n[Candidate D] Evaluating ISOMAP...")
    from sklearn.manifold import Isomap
    
    scaler = StandardScaler()
    X_train_game = scaler.fit_transform(X_train_game)
    X_test_game = scaler.transform(X_test_game)

    iso_obj = Isomap(n_neighbors=15, n_components=50)
    X_tr_iso = iso_obj.fit_transform(X_train_game)
    X_te_iso = iso_obj.transform(X_test_game)
    
    np.save(os.path.join(tmp_dir, 'X_train_pca.npy'), X_tr_iso)
    np.save(os.path.join(tmp_dir, 'X_test_pca.npy'), X_te_iso)
    
    _, acc_iso, _, _ = train_and_evaluate_knn(data_dir=tmp_dir, experiment_name='game_Try_Isomap')
    results['ISOMAP'] = acc_iso
    candidates_data['ISOMAP'] = (X_tr_iso, X_te_iso)

    # Step 3.3: Deploy the best performer
    best_method = max(results, key=results.get)
    print(f"\nResults: {results}")
    print(f"Best performed model: {best_method} with {results[best_method]:.4f} accuracy.")

    game_dir = './data/game_Stressed'
    os.makedirs(game_dir, exist_ok=True)

    # Retrieve the winning data from our storage dictionary
    final_X_train, final_X_test = candidates_data[best_method]

    # Save final optimized data
    np.save(os.path.join(game_dir, 'X_train_pca.npy'), final_X_train)
    np.save(os.path.join(game_dir, 'X_test_pca.npy'), final_X_test)
    np.save(os.path.join(game_dir, 'y_train.npy'), y_train_game)
    np.save(os.path.join(game_dir, 'y_test.npy'), y_test_game)

    print(f"Final game Model ({best_method}) deployed to {game_dir}")

# Main Execution Block
if __name__ == "__main__":
    print("Initializing Project Pipeline...")
    
    # Run Module 1 and get baseline for robustness
    ideal_acc = run_module_1()
    
    # Run Module 2
    run_module_2(ideal_acc)
    
    # Run Module 3
    run_module_3()
    
    print("\n Pipeline execution finished.")
