import os
import cv2
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from collections import Counter
import preprocessing

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
        # Dataset 2: folders 0, 1, 2...
        for label_dir in os.listdir(data_path):
            label_path = os.path.join(data_path, label_dir)
            
            # 确保是一个由数字命名的文件夹
            if not os.path.isdir(label_path) or not label_dir.isdigit():
                continue
                
            label = int(label_dir)
            
            for img_name in os.listdir(label_path):
                if img_name.lower().endswith('.jpeg'):
                    img = cv2.imread(os.path.join(label_path, img_name), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = preprocessing.resize_image(img)
                        img = preprocessing.apply_clahe(img)
                        X.append(img)
                        y.append(label)
                        
    return np.array(X), np.array(y)

def balance_classes(X_train, y_train):
    """类别平衡：通过数据增强扩充少数类"""
    class_counts = Counter(y_train)
    max_count = max(class_counts.values())
    
    X_balanced = list(X_train)
    y_balanced = list(y_train)
    
    for cls, count in class_counts.items():
        if count < max_count:
            indices = np.where(y_train == cls)[0]
            num_to_add = max_count - count
            
            for _ in range(num_to_add):
                idx = np.random.choice(indices)
                aug_img = preprocessing.augment_image(X_train[idx])
                X_balanced.append(aug_img)
                y_balanced.append(cls)
                
    return np.array(X_balanced), np.array(y_balanced)

def get_data_pipeline(dataset_type='ideal'):
    """主干流水线：下载、加载、切分、平衡、标准化"""
    path_ideal, path_stressed = download_datasets()
    data_path = path_ideal if dataset_type == 'ideal' else path_stressed
    
    print(f"Loading {dataset_type} dataset...")
    X, y = load_and_preprocess_data(data_path, dataset_type)
    
    # 70% Train, 15% Val, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print("Balancing training classes...")
    X_train, y_train = balance_classes(X_train, y_train)
    
    print("Normalizing data (Z-score)...")
    X_train = np.array([preprocessing.normalize_pixel_values(img).flatten() for img in X_train])
    X_val = np.array([preprocessing.normalize_pixel_values(img).flatten() for img in X_val])
    X_test = np.array([preprocessing.normalize_pixel_values(img).flatten() for img in X_test])
    
    print(f"Pipeline Complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    X_tr, X_v, X_te, y_tr, y_v, y_te = get_data_pipeline('stressed')
