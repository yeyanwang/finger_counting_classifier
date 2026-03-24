import cv2
import numpy as np

def resize_image(image, size=(64, 64)):
    # Resize image size to 64*64
    return cv2.resize(image, size)

def apply_clahe(image):
    # For dataset stressed: apply CLAHE
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)

def normalize_pixel_values(img_array):
    # Z-score standardization[cite: 42]
    img_array = img_array.astype('float32')
    return (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-7)

def augment_image(image):
    """
    Applies random data augmentation to improve the model's robustness 
    against variations in hand orientation[cite: 43, 44].
    """
    # Randomly select one augmentation technique: none, horizontal flip, or rotation
    choice = np.random.choice(['none', 'flip', 'rotate'])
    
    if choice == 'flip':
        # Apply horizontal flip (1 means flipping around the y-axis)
        return cv2.flip(image, 1) 
        
    elif choice == 'rotate':
        # Apply a slight rotation to simulate hand gestures
        # Randomly choose to rotate either 15 degrees clockwise or counter-clockwise
        angle = np.random.choice([-15, 15]) 
        
        # Calculate the center point of the image (x, y) to act as the rotation axis
        center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # Generate the 2D rotation matrix without scaling (scale factor = 1.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply the rotation matrix to the image, keeping the original image dimensions
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
    # Return the original, unchanged image if 'none' was selected
    return image
