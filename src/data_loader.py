import os
import cv2
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from collections import Counter
from src import preprocessing

def download_datasets():
    """Download datasets from kaggle"""
    path_ideal = kagglehub.dataset_download("roshea6/finger-digits-05")
    path_stressed = kagglehub.dataset_download("piyushjoshi01/counting-fingers-dataset")  
    return path_ideal, path_stressed 

def load_and_preprocess_data(data_path, dataset_type='ideal'):
    """Load data from the kaggle path, and extract labels"""
    X, y = [], []
    
    if dataset_type == 'ideal':
      
        # Dataset 1: ID_Label.png (e.g. 1000_5.png)
        for root, _, files in os.walk(data_path):
          
            for file in files:
              
                # Extract label，e.g. 1001_4.png -> split '4.png' to get '4'
                if file.lower().endswith('.png'):
                  
                    try:
                        label = int(file.split('_')[-1].split('.')[0])
                      
                    except ValueError:
                        continue 
                    
                    img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        X.append(preprocessing.resize_image(img))
                        y.append(label)
                        
    elif dataset_type == 'stressed':
        for root, dirs, files in os.walk(data_path):
            folder_name = os.path.basename(root)
            
            # when the folder name is a number (0, 1, 2, 3, 4, 5) is it regarded as a tag folder.
            if folder_name.isdigit():
                label = int(folder_name)
                
                for img_name in files:
                    
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(root, img_name)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        
                        if img is not None:
                            img = preprocessing.resize_image(img)
                            img = preprocessing.apply_clahe(img)
                            X.append(img)
                            y.append(label)
                        
    return np.array(X), np.array(y)

# Data Preparations for training, testing, and validation
def balance_classes(X_train, y_train):
    """
    Balances the number of images across all classes (0 to 5).
    If a class has fewer images, it generates new ones using augmentation.
    """
    # Find the target number (the count of the largest class)
    max_count = max(Counter(y_train).values())
    
    # Create lists
    X_bal, y_bal = list(X_train), list(y_train)
    
    # Go through each class (0, 1, 2, 3, 4, 5)
    for cls in set(y_train):
        # Get all original images that belong to this class
        cls_images = X_train[y_train == cls]
        
        # Calculate how many images this class is missing
        num_missing = max_count - len(cls_images)
        
        # Generate the missing amount
        for _ in range(num_missing):
            # Pick a random image from this specific class
            random_img = cls_images[np.random.randint(len(cls_images))]
            
            # Apply rotation/flip and add it to our dataset
            X_bal.append(preprocessing.augment_image(random_img))
            y_bal.append(cls)
            
    return np.array(X_bal), np.array(y_bal)


def flatten_and_normalize(image_array):
    """
    Helper function: Apply Z-score normalization and flattens 2D images to 1D.
    """
    processed_images = []
    for img in image_array:
        # Normalize the image (Z-score)
        norm_image = preprocessing.normalize_pixel_values(img)
        # Flatten from 2D to a 1D
        processed_images.append(norm_image.flatten())
        
    return np.array(processed_images)


def get_data_pipeline(dataset_type='ideal'):
    """
    Main assembly line: Download -> Split -> Balance -> Normalize & Flatten
    """
    # Get data path
    path_ideal, path_stressed = download_datasets()
    data_path = path_ideal if dataset_type == 'ideal' else path_stressed
        
    print(f"Loading {dataset_type} dataset...")
    X, y = load_and_preprocess_data(data_path, dataset_type)
    
    # Split data (70% Train, 15% Validation, 15% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Balance the training data
    print("Balancing training classes...")
    X_train, y_train = balance_classes(X_train, y_train)
    
    # Normalize and flatten all sets using helper function
    print("Normalizing and flattening data...")
    X_train = flatten_and_normalize(X_train)
    X_val   = flatten_and_normalize(X_val)
    X_test  = flatten_and_normalize(X_test)
    
    print(f"Data Preparation Pipeline Complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test
