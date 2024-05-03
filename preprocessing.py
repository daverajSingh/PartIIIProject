import numpy as np
import os
import cv2

def preprocess_data(data_path):
    """
    Preprocesses the data by converting images to grayscale along the red channel, then reducing size of images to 1/6 of original size.
    
    Args:
    data: path to the data directory
    
    Returns:
    path to the preprocessed images
    
    """ 
    
    folder_path = "preprocessed_images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Load images from the data path and resize to 1/6 of the original size, and convert to greyscale along the red channel
    
    for classes in os.listdir(data_path):
        classPath = os.path.join(data_path, classes)
        print(classPath)
        for img in os.listdir(classPath):
            imagePath = os.path.join(classPath, img)
            if not imagePath.endswith(".JPG"):
                continue # Skip non-image files
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (image.shape[1]//6, image.shape[0]//6)) # Resize to 1/6 of the original size
            
            image = image[:,:,2] # Convert to greyscale along the red channel
            
            #Gaussian blur
            image = cv2.GaussianBlur(image, (3, 3), 0)
            
            cv2.imwrite(os.path.join(folder_path, img), image)  # Save the preprocessed image to the folder_path  
            
    #Returns path to the preprocessed images
    return folder_path


preprocess_data("dataset")