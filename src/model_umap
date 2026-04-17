"""
Module: model_umap.py
Description: Non-linear dimensionality reduction using Uniform Manifold Approximation and Projection (UMAP).
Includes a PCA pre-processing step to denoise high-dimensional image data and ensure mathematical stability.
"""
import umap
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import time

def apply_umap(X_train, X_test, n_components=10, n_neighbors=15, random_state=42):
    """
    Applies the PCA-UMAP pipeline to the dataset.
    
    Args:
        X_train (array): Training feature matrix.
        X_test (array): Testing feature matrix.
        n_components (int): Target dimensions for UMAP.
        n_neighbors (int): Local neighborhood size for manifold learning.
        
    Returns:
        X_train_umap, X_test_umap, pipeline_model
    """
    print(f" Running UMAP Pipeline (PCA->100, UMAP->{n_components})...")
    start_time = time.time()
    
    # Define the robust pipeline: PCA for denoising, UMAP for manifold extraction
    pipeline = Pipeline([
        ('pca', PCA(n_components=100, random_state=random_state)),
        ('umap', umap.UMAP(n_neighbors=n_neighbors, 
                           n_components=n_components, 
                           random_state=random_state,
                           n_jobs=-1)) # Use all CPU cores for speed
    ])
    
    # Fit and transform
    X_train_umap = pipeline.fit_transform(X_train)
    X_test_umap = pipeline.transform(X_test)
    
    print(f" UMAP reduction completed in {time.time() - start_time:.2f}s")
    
    return X_train_umap, X_test_umap, pipeline
