# Model: ISOMAP

from sklearn.manifold import Isomap

def model_isomap(X_train, X_test=None, n_neighbors=5, n_components=50):

    print(f"Fitting ISOMAP: Reducing from {X_train.shape[1]} to {n_components} dimensions...")
    
    # Initialization and Training
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    X_train_reduced = isomap.fit_transform(X_train)
    
    print("ISOMAP Training Complete!")
    
    # Test
    if X_test is not None:
        print("Transforming test data using ISOMAP...")
        X_test_reduced = isomap.transform(X_test)
      
        return X_train_reduced, X_test_reduced, isomap
        
    return X_train_reduced, isomap
