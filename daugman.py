import numpy as np
import os
import cv2

def daugmanIDO(data_path):
    """
    Performs Daugmans Integro Differential Operator on the preprocessed images.
    
    Args:
    data_path: path to the preprocessed images
    
    Returns:
    path to the processed images
    
    """
    
    folder_path = "DaugmanIDO_images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    #INSERT CODE HERE
    
    #Returns path to the processed images
    return folder_path

def daugmanRubberSheet(data_path):
    """
    Performs Daugmans Rubber Sheet Model on the localised images.
    
    Args:
    data_path: path to the localised images
    
    Returns:
    path to the processed images
    
    """
    
    folder_path = "DaugmanRubberSheet_images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    #INSERT CODE HERE
    
    #Returns path to the processed images
    return folder_path

def daugmanGaborWavelet(data_path):
    """
    Performs Daugmans Gabor Wavelet on the processed images.
    
    Args:
    data_path: path to the normalised images
    
    Returns:
    path to the processed images
    
    """
    
    folder_path = "DaugmanGaborWavelet_images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    #INSERT CODE HERE
    
    #Returns path to the processed images
    return folder_path