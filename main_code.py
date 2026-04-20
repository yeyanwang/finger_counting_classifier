"""
This is the main execution script. It orchestrates the data loading, 
model training, and evaluation for all three modules.
"""

import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap
from sklearn.pipeline import Pipeline
from skimage.feature import hog
from sklearn.utils import resample

# 1. Import our data pipeline
from src.data_loader import get_data_pipeline

# Model Imports
from src.model_pca import run_pca_experiment, save_results as save_pca_results
from src.model_knn import train_and_evaluate_knn
from src.model_pca_lda import run_lda_experiment
from src.model_pca_hog import run_hog_experiment
from src.model_umap import apply_umap
from src.model_svm import train_and_evaluate_svm

# import rps game 
from src.rps import load_models, run_sim, plot_outcome_examples

# Evaluation Imports for Automated Plots
from src.evaluation import plot_confusion_matrix, plot_robustness_decay, plot_tradeoff


def run_module_1():
    """
    Module 1: Foundational Modeling under Ideal Conditions
    """
    print("MODULE 1: IDEAL CONDITIONS")
    
    # Get ideal dataset to evaluate LDA and HOG identically to PCA
    X_train, _, X_test, y_train, _, y_test = get_data_pipeline('ideal')
    tmp_dir = './data/Ideal_Temp'
    os.makedirs(tmp_dir, exist_ok=True)
    results = {}
    
    # Step 1.1: PCA
    print("\n[Step 1.1] Running Basic PCA Pipeline...")
    pca_data, pca_labels, pca_model = run_pca_experiment(dataset_type='ideal')
    save_pca_results(pca_data, pca_labels, pca_model, dataset_type='ideal')
    
    # Updated to capture acc for robustness comparison in Module 2
    _, ideal_pca_acc, y_t, y_p = train_and_evaluate_knn(data_dir='./data/Ideal', experiment_name='Ideal_PCA')
    plot_confusion_matrix(y_t, y_p, 'Ideal_PCA')
    results['PCA'] = ideal_pca_acc

    # Step 1.2: PCA + LDA
    print("\n[Step 1.2] Running PCA + LDA Pipeline...")
    run_lda_experiment(dataset_type='ideal') # Keeps the original plots
    pca_obj = PCA(n_components=0.95)
    X_tr_pca = pca_obj.fit_transform(X_train)
    X_te_pca = pca_obj.transform(X_test)
    lda_obj = LDA()
    X_tr_lda = lda_obj.fit_transform(X_tr_pca, y_train)
    X_te_lda = lda_obj.transform(X_te_pca)
    
    np.save(os.path.join(tmp_dir, 'X_train_pca.npy'), X_tr_lda)
    np.save(os.path.join(tmp_dir, 'X_test_pca.npy'), X_te_lda)
    np.save(os.path.join(tmp_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(tmp_dir, 'y_test.npy'), y_test)
    _, acc_lda, _, _ = train_and_evaluate_knn(data_dir=tmp_dir, experiment_name='Ideal_LDA')
    results['LDA'] = acc_lda

    # Step 1.3: HOG + PCA
    print("\n[Step 1.3] Running HOG + PCA Pipeline...")
    run_hog_experiment(dataset_type='ideal') # Keeps the original plots
    def extract_hog(data):
        return np.array([hog(img.reshape(64, 64), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for img in data])
    X_tr_hog = extract_hog(X_train)
    X_te_hog = extract_hog(X_test)
    pca_hog = PCA(n_components=0.95)
    X_tr_hog_pca = pca_hog.fit_transform(X_tr_hog)
    X_te_hog_pca = pca_hog.transform(X_te_hog)
    
    np.save(os.path.join(tmp_dir, 'X_train_pca.npy'), X_tr_hog_pca)
    np.save(os.path.join(tmp_dir, 'X_test_pca.npy'), X_te_hog_pca)
    _, acc_hog, _, _ = train_and_evaluate_knn(data_dir=tmp_dir, experiment_name='Ideal_HOG')
    results['HOG'] = acc_hog

    # Print a clean summary for Module 1
    print("\n" + "="*40)
    print("MODULE 1 IDEAL BASELINE RESULTS")
    print("="*40)
    for model_name, accuracy in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f" {model_name:12s} : {accuracy:.4f}")
    print("="*40)

    print("\n MODULE 1 COMPLETE.")
    return ideal_pca_acc # Pass this to Module 2 for robustness plot


def run_module_2(ideal_baseline_acc):
    """
    Module 2: Robustness Evaluation under Complex Environments
    Evaluates ALL models (PCA, LDA, HOG, ISOMAP, UMAP) on the full Stressed dataset
    to determine the most robust feature extraction method.
    """
    print("\nSTARTING MODULE 2: STRESSED CONDITIONS (ROBUSTNESS TOURNAMENT)")
    
    # Get full stressed dataset
    X_train, _, X_test, y_train, _, y_test = get_data_pipeline('stressed')
    
    tmp_dir = './data/Stressed_Temp'
    os.makedirs(tmp_dir, exist_ok=True)
    results = {}

    # Step 2.1: Stressed PCA (Baseline for decay plot)
    print("\n[Evaluating] Standard PCA...")
    pca_data, labels, pca_model = run_pca_experiment(dataset_type='stressed')
    save_pca_results(pca_data, labels, pca_model, dataset_type='stressed')
    _, acc_pca, y_t, y_p = train_and_evaluate_knn(data_dir='./data/Stressed', experiment_name='Stressed_PCA', feature_suffix='pca')
    
    plot_robustness_decay(ideal_acc=ideal_baseline_acc, stressed_acc=acc_pca, model_name="PCA + KNN")
    plot_confusion_matrix(y_t, y_p, 'Stressed_PCA')
    results['PCA'] = acc_pca

    # Step 2.2: Stressed LDA
    print("\n[Evaluating] PCA + LDA...")
    run_lda_experiment(dataset_type='stressed') # Keep original file save
    pca_obj = PCA(n_components=0.95)
    X_tr_pca = pca_obj.fit_transform(X_train)
    X_te_pca = pca_obj.transform(X_test)
    lda_obj = LDA()
    X_tr_lda = lda_obj.fit_transform(X_tr_pca, y_train)
    X_te_lda = lda_obj.transform(X_te_pca)
    
    np.save(os.path.join(tmp_dir, 'X_train_pca.npy'), X_tr_lda)
    np.save(os.path.join(tmp_dir, 'X_test_pca.npy'), X_te_lda)
    np.save(os.path.join(tmp_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(tmp_dir, 'y_test.npy'), y_test)
    _, acc_lda, _, _ = train_and_evaluate_knn(data_dir=tmp_dir, experiment_name='Stressed_LDA')
    results['LDA'] = acc_lda

    # Step 2.3: Stressed HOG
    print("\n[Evaluating] HOG + PCA...")
    run_hog_experiment(dataset_type='stressed')
    def extract_hog(data):
        return np.array([hog(img.reshape(64, 64), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for img in data])
    X_tr_hog = extract_hog(X_train)
    X_te_hog = extract_hog(X_test)
    pca_hog = PCA(n_components=0.95)
    X_tr_hog_pca = pca_hog.fit_transform(X_tr_hog)
    X_te_hog_pca = pca_hog.transform(X_te_hog)
    
    np.save(os.path.join(tmp_dir, 'X_train_pca.npy'), X_tr_hog_pca)
    np.save(os.path.join(tmp_dir, 'X_test_pca.npy'), X_te_hog_pca)
    _, acc_hog, _, _ = train_and_evaluate_knn(data_dir=tmp_dir, experiment_name='Stressed_HOG')
    results['HOG'] = acc_hog

    # Step 2.4: Stressed ISOMAP
    print("\n[Evaluating] ISOMAP...")
    # n_components=0.95 prevents crashes on small datasets
    iso_pipeline = Pipeline([
        ('pca', PCA(n_components=0.95, random_state=42)),
        ('isomap', Isomap(n_neighbors=10, n_components=10))
    ])
    X_tr_iso = iso_pipeline.fit_transform(X_train)
    X_te_iso = iso_pipeline.transform(X_test)
    np.save(os.path.join(tmp_dir, 'X_train_pca.npy'), X_tr_iso)
    np.save(os.path.join(tmp_dir, 'X_test_pca.npy'), X_te_iso)
    _, acc_iso, _, _ = train_and_evaluate_knn(data_dir=tmp_dir, experiment_name='Stressed_Isomap')
    results['ISOMAP'] = acc_iso

    # Step 2.5: Stressed UMAP
    print("\n[Evaluating] UMAP...")
    X_tr_umap, X_te_umap, _ = apply_umap(X_train, X_test, n_components=10, n_neighbors=15)
    np.save(os.path.join(tmp_dir, 'X_train_pca.npy'), X_tr_umap)
    np.save(os.path.join(tmp_dir, 'X_test_pca.npy'), X_te_umap)
    _, acc_umap, _, _ = train_and_evaluate_knn(data_dir=tmp_dir, experiment_name='Stressed_UMAP')
    results['UMAP'] = acc_umap

    # Select the ultimate feature extractor
    best_extractor = max(results, key=results.get)
    print("\n" + "="*40)
    print("MODULE 2 FULL TOURNAMENT RESULTS")
    print("="*40)
    for model_name, accuracy in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f" {model_name:12s} : {accuracy:.4f}")
        
    print(f"--> Most Robust Feature Extractor: {best_extractor}")
    print("="*40)
    
    print("\n MODULE 2 COMPLETE.")
    return best_extractor

def run_module_3(best_extractor):
    """
    Module 3: Rock-Paper-Scissors (Strategic Application)
    Takes the best feature extractor from Module 2, applies it to the game subset,
    and performs further selection between Classifiers (KNN vs SVM) for final prediction.
    Then runs RPS game sim...
    """
    print(f"\nSTARTING MODULE 3: ROCK-PAPER-SCISSORS (Deploying {best_extractor})")

    # Step 3.1: Data Acquisition & Filtering
    X_train, _, X_test, y_train, _, y_test = get_data_pipeline('rps')
    game_labels = [0, 2, 5]
    mask_train = np.isin(y_train, game_labels)
    mask_test = np.isin(y_test, game_labels)

    X_train_game = X_train[mask_train]
    y_train_game = y_train[mask_train]
    X_test_game = X_test[mask_test]
    y_test_game = y_test[mask_test]
    
    print(f"Game Subset created with {len(X_train_game)} samples.")

    MAX_PER_CLASS = 300
    X_tr_down, y_tr_down = [], []
    for cls in [0, 2, 5]:
        mask  = y_train_game == cls
        X_cls = X_train_game[mask]
        y_cls = y_train_game[mask]
        n = min(MAX_PER_CLASS, len(X_cls))
        X_s, y_s = resample(X_cls, y_cls, n_samples=n, random_state=42, replace=False)
        X_tr_down.append(X_s)
        y_tr_down.append(y_s)
    X_train_game = np.vstack(X_tr_down)
    y_train_game = np.concatenate(y_tr_down)
    print(f"Downsampled training set: {X_train_game.shape}")

    # Step 3.2: Extract Features using the Winner from Module 2
    print(f"\n[Step 3.1] Transforming data using {best_extractor}...")

    if best_extractor == 'PCA':
        pca_obj = PCA(n_components=0.95)
        X_tr_feats = pca_obj.fit_transform(X_train_game)
        X_te_feats = pca_obj.transform(X_test_game)
    elif best_extractor == 'LDA':
        pca_obj = PCA(n_components=0.95)
        X_tr_pca = pca_obj.fit_transform(X_train_game)
        X_te_pca = pca_obj.transform(X_test_game)
        lda_obj = LDA(n_components=2)
        X_tr_feats = lda_obj.fit_transform(X_tr_pca, y_train_game)
        X_te_feats = lda_obj.transform(X_te_pca)
    elif best_extractor == 'HOG':
        def extract_hog(data):
            return np.array([hog(img.reshape(64, 64), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for img in data])
        X_tr_hog = extract_hog(X_train_game)
        X_te_hog = extract_hog(X_test_game)
        pca_hog = PCA(n_components=0.95)
        X_tr_feats = pca_hog.fit_transform(X_tr_hog)
        X_te_feats = pca_hog.transform(X_te_hog)
    elif best_extractor == 'ISOMAP':
        # FIXED: n_components=0.95 here as well
        iso_pipeline = Pipeline([('pca', PCA(n_components=0.95, random_state=42)), ('isomap', Isomap(n_neighbors=10, n_components=10))])
        X_tr_feats = iso_pipeline.fit_transform(X_train_game)
        X_te_feats = iso_pipeline.transform(X_test_game)
    elif best_extractor == 'UMAP':
        X_tr_feats, X_te_feats, _ = apply_umap(X_train_game, X_test_game, n_components=10, n_neighbors=15)

    # Step 3.3: Further Classifier Selection (KNN vs SVM)
    print("\n[Step 3.2] Classifier Selection: KNN vs SVM...")
    tmp_dir = './data/game_Temp'
    os.makedirs(tmp_dir, exist_ok=True)
    np.save(os.path.join(tmp_dir, 'X_train_pca.npy'), X_tr_feats)
    np.save(os.path.join(tmp_dir, 'X_test_pca.npy'), X_te_feats)
    np.save(os.path.join(tmp_dir, 'y_train.npy'), y_train_game)
    np.save(os.path.join(tmp_dir, 'y_test.npy'), y_test_game)

    experiment_name = f'game_{best_extractor}_KNN'

    # Candidate 1: KNN
    _, acc_knn, _, _ = train_and_evaluate_knn(data_dir=tmp_dir, experiment_name=experiment_name)
    
    # Candidate 2: SVM
    _, acc_svm, _ = train_and_evaluate_svm(X_tr_feats, y_train_game, X_te_feats, y_test_game)

    best_classifier = "SVM" if acc_svm > acc_knn else "KNN"
    final_acc = max(acc_knn, acc_svm)

    print("\n" + "="*40)
    print("MODULE 3 FINAL PREDICTION RESULTS")
    print("="*40)
    print(f" Feature: {best_extractor:8s} + KNN Accuracy : {acc_knn:.4f}")
    print(f" Feature: {best_extractor:8s} + SVM Accuracy : {acc_svm:.4f}")
    print("="*40)
    print(f"--> Optimal Deployment Configuration: [{best_extractor} + {best_classifier}] with {final_acc:.4f} accuracy.")

    # trade-off plot comparing KNN vs SVM on the best extractor
    models_data = {
        f'{best_extractor} + KNN': {'dim': X_tr_feats.shape[1], 'acc': acc_knn},
        f'{best_extractor} + SVM': {'dim': X_tr_feats.shape[1], 'acc': acc_svm},
    }
    plot_tradeoff(models_data)

    # Step 3.4: Deploy the best performer
    game_dir = './data/game_Stressed'
    os.makedirs(game_dir, exist_ok=True)

    np.save(os.path.join(game_dir, 'X_train_pca.npy'), X_tr_feats)
    np.save(os.path.join(game_dir, 'X_test_pca.npy'), X_te_feats)
    np.save(os.path.join(game_dir, 'y_train.npy'), y_train_game)
    np.save(os.path.join(game_dir, 'y_test.npy'), y_test_game)

    print(f"\nFinal Game Model ({best_extractor} + {best_classifier}) deployed to {game_dir}")

    # Step 3.5: Run RPS game sim
    print(f"\nRunning RPS Game Simulation...")
    try:
        X_test_rps, y_test_rps, knn_model = load_models(path=tmp_dir, model_name=experiment_name)
        sim_results = run_sim(X_test_rps, y_test_rps, knn_model, rounds=10, show_images=False)
        print(f"\n--> Game Summary: Wins={sim_results['wins']}, Losses={sim_results['losses']}, Ties={sim_results['ties']}, Accuracy={sim_results['accuracy']:.4f}")
        plot_outcome_examples(X_test_rps, y_test_rps, knn_model, save_dir='./results')
    except FileNotFoundError as e:
        print(f"Unable to load model for simulation: {e}")

# Main Execution Block
if __name__ == "__main__":
    print("Initializing Project Pipeline...")
    
    # Run Module 1 and get baseline for robustness
    ideal_acc = run_module_1()
    
    # Run Module 2 to find the most robust feature extractor
    best_feature_model = run_module_2(ideal_acc)
    
    # Run Module 3 with the winner for further classifier selection and prediction
    run_module_3(best_feature_model)
    
    print("\n Pipeline execution finished.")