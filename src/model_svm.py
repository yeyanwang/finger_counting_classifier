"""
Module: model_svm.py
Description: Support Vector Machine (SVM) classifier with automated hyperparameter 
tuning using 5-Fold Cross Validation.
"""
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

def train_and_evaluate_svm(X_train, y_train, X_test, y_test, cv_folds=5):
    """
    Trains an SVM using the RBF kernel and tunes hyperparameters via GridSearchCV.
    
    Args:
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        cv_folds (int): Number of folds for cross-validation.
        
    Returns:
        best_svm_model, test_accuracy, y_pred
    """
    print(f"    -> Tuning and Training SVM (RBF Kernel, {cv_folds}-Fold CV)...")
    start_time = time.time()
    
    # Define parameter grid for CV (Kept concise for faster execution)
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto']
    }
    
    # Initialize base model (RBF kernel is standard for non-linear features)
    base_svm = SVC(kernel='rbf', random_state=42)
    
    # Setup GridSearch (n_jobs=-1 uses all available CPU cores)
    grid_search = GridSearchCV(base_svm, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
    
    # Fit the model (This automatically finds the best params and refits on the whole X_train)
    grid_search.fit(X_train, y_train)
    
    # Evaluate the best model on the completely unseen test set
    best_svm = grid_search.best_estimator_
    test_acc = best_svm.score(X_test, y_test)
    y_pred = best_svm.predict(X_test)
    
    print(f" Best SVM Params: {grid_search.best_params_}")
    print(f" SVM Test Accuracy: {test_acc:.2%}")
    print(f" Time taken: {time.time() - start_time:.2f}s")
    
    return best_svm, test_acc, y_pred
