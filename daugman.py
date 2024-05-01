import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from util import cropCircle

def daugmanIDO(data_path, pRadiusRange, iRadiusRange, centerXRange, centerYRange, detlaRadius):
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
    # Load the image
    
    for iris in os.listdir(data_path):
        iris_path = os.path.join(data_path, iris)
        
        img = cv2.imread(iris_path)
        
        best_pupil = (0,0,0)
        best_iris = (0,0,0)
        max_gradientI = -np.inf   
        max_gradientP = -np.inf
        
        rows, cols, _ = img.shape
        for r in range(pRadiusRange[0], pRadiusRange[1], detlaRadius):
            for x in range(centerXRange[0], centerXRange[1]):
                for y in range(centerYRange[0], centerYRange[1]):
                    if x - r < 0 or x + r > cols or y - r < 0 or y + r > rows:
                        continue  # Ensure the circle mask doesn't go out of image bounds
                   # Create a circular mask
                    mask = np.zeros_like(img)
                    cv2.circle(mask, (x, y), r, 255, 1)
                    
                    # Calculate gradients along the circle
                    edges = cv2.Canny(mask, 50, 100)
                    gradient_sum = np.sum(img[edges > 0])
                    
                    if gradient_sum > max_gradientP:
                        max_gradientP = gradient_sum
                        best_pupil = (x, y, r)
                
        for r in range(iRadiusRange[0], iRadiusRange[1], detlaRadius):
            for x in range(centerXRange[0], centerXRange[1]):
                for y in range(centerYRange[0], centerYRange[1]):
                    if x - r < 0 or x + r > cols or y - r < 0 or y + r > rows:
                        continue  # Ensure the circle mask doesn't go out of image bounds
                   # Create a circular mask
                    mask = np.zeros_like(img)
                    cv2.circle(mask, (x, y), r, 255, 1)
                    
                    # Calculate gradients along the circle
                    edges = cv2.Canny(mask, 25, 100)
                    gradient_sum = np.sum(img[edges > 0])
                    
                    if gradient_sum > max_gradientI:
                        max_gradientI = gradient_sum
                        best_iris = (x, y, r)
        
    print(best_pupil, best_iris)
    
    new_image = cropCircle(iris_path, best_pupil, best_iris)

    plt.imshow(new_image, cmap='gray')
    plt.show()
    
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

daugmanIDO("test", (0, 20), (40, 70), (140, 180), (90, 120), 1)