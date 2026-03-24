"""
This is the main execution script. It orchestrates the data loading, 
model training, and evaluation for all three modules.
"""

import numpy as np
# 1. Import our data pipeline
from src.data_loader import get_data_pipeline

# Placeholder Imports for Models 

# from src.model_pca import PCAModel
# from src.model_pca_lda import PCALDAModel
# from src.model_hog_pca import HOGPCAModel
# from src.model_isomap import IsomapModel
# from src.evaluation import evaluate_robustness_decay, plot_confusion_matrix


def run_module_1():
    """
    Module 1: Foundational Modeling under Ideal Conditions
    Focus: PCA, PCA+LDA, HOG+PCA on the 'Ideal' dataset.
    """
    
    print("MODULE 1: IDEAL CONDITIONS")
    
    # Step 1: Get the clean, black-background data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_pipeline('ideal')

    # TODO (Module 1): Initialize your feature extractors here
    # pca_model = PCAModel(...)
    # pca_lda_model = PCALDAModel(...)
    # hog_pca_model = HOGPCAModel(...)

    # TODO (Module 1): Train your models and apply K-NN
    # pca_model.train(X_train, y_train)
    # accuracy = pca_model.evaluate(X_test, y_test)
    # print(f"PCA Accuracy: {accuracy}")


def run_module_2():
    """
    Module 2: Robustness Evaluation under Complex Environments
    Focus: ISOMAP vs Linear Models on the 'Stressed' dataset.
    """
    print("STARTING MODULE 2: STRESSED CONDITIONS (ROBUSTNESS)")

    # Step 1: Get the noisy, complex-background data (CLAHE applied automatically)
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_pipeline('stressed')

    # TODO (Module 2): Initialize ISOMAP and Baseline models here
    # isomap_model = IsomapModel(n_neighbors=5, n_components=50)

    # TODO (Module 2): Train and evaluate
    # isomap_model.train(X_train, y_train)
    # stressed_accuracy = isomap_model.evaluate(X_test, y_test)
    
    # TODO (Module 2): Calculate Robustness Decay and plot Confusion Matrix
    # evaluate_robustness_decay(ideal_accuracy, stressed_accuracy)


def run_module_3():
    """
    Module 3: Rock-Paper-Scissors (Strategic Application)
    Focus: Testing models on a subset of gestures (0, 2, 5).
    """
    print("STARTING MODULE 3: ROCK-PAPER-SCISSORS")

    # Step 1: Get the data (Usually stressed data for a realistic game scenario)
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_pipeline('stressed')

    # TODO (Module 3: Filter the dataset to ONLY include labels 0 (Rock), 2 (Scissors), 5 (Paper))
    # Example logic:
    # mask_train = np.isin(y_train, [0, 2, 5])
    # X_train_rps, y_train_rps = X_train[mask_train], y_train[mask_train]

    # TODO (Module 3): Run the best pipeline from Module 1 & 2 on this subset
    
    # TODO (Module 3): Implement the game logic (vs. Random Computer Move)


# =========================================================
# Main Execution Block
# =========================================================
if __name__ == "__main__":
    print("Initializing Project Pipeline...")
    
    
    # run_module_1()
    
    # run_module_2()
    
    # run_module_3()
    
    print("\n Pipeline execution finished.")
