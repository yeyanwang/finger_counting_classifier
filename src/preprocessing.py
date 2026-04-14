import cv2
import numpy as np
from scipy.ndimage import rotate

def resize_image(image, size=(64, 64)):
    # Resize image size to 64*64
    return cv2.resize(image, size)

def normalize_pixel_values(img_array):
    # Z-score standardization
    img_array = img_array.astype('float32')
    return (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-7)

def augment_image(image):
    """
    Apply random data augmentation to improve the model's robustness 
    against variations in hand orientation.
    """
    # Randomly select one augmentation technique: none, horizontal flip, or rotation
    choice = np.random.choice(['none', 'flip', 'rotate'])

    if choice == 'flip':
        return cv2.flip(image, 1) 
        
    elif choice == 'rotate':
        angle = np.random.choice([-15, 15]) 
        return rotate(image, angle, reshape=False)
    
    return image

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    # Reduce noices: Apply Gaussian Blur to reduce high-frequency background noise
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_clahe(image):
    # For dataset stressed: apply CLAHE
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
