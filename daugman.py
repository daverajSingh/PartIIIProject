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
        
        best_iris = (0,0,0)
        max_gradientI = -np.inf   
        
        best_pupil = (0,0,0)
        max_gradientP = -np.inf
        
        rows, cols, _ = img.shape
        for r in range(iRadiusRange[0], iRadiusRange[1], detlaRadius):
            for x in range(centerXRange[0], centerXRange[1]):
                for y in range(centerYRange[0], centerYRange[1]):
                    if x - r < 0 or x + r > cols or y - r < 0 or y + r > rows:
                        continue  # Ensure the circle mask doesn't go out of image bounds
                   # Create a circular mask
                    mask = np.zeros((rows, cols), dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, 255, 1)
                    # Calculate gradients along the circle
                    edges = cv2.Canny(img, 120, 150)
                    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
                    gradient_sum = np.sum(masked_edges) / (2 * np.pi * r)  # Normalize by circumference
                    
                    if gradient_sum > max_gradientI:
                        max_gradientI = gradient_sum
                        best_iris = (x, y, r)
        
        for r in range(pRadiusRange[0], pRadiusRange[1], detlaRadius):
            for x in range(centerXRange[0], centerXRange[1]):
                for y in range(centerYRange[0], centerYRange[1]):
                    if x - r < 0 or x + r > cols or y - r < 0 or y + r > rows:
                        continue  # Ensure the circle mask doesn't go out of image bounds
                   # Create a circular mask
                    mask = np.zeros((rows, cols), dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, 255, 1)
                    # Calculate gradients along the circle
                    edges = cv2.Canny(img, 40, 80)
                    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
                    gradient_sum = np.sum(masked_edges) / (2 * np.pi * r)  # Normalize by circumference
                    
                    if gradient_sum > max_gradientP:
                        max_gradientP = gradient_sum
                        best_pupil = (x, y, r)
        
        print(best_iris, best_pupil)
        
        new_image = cropCircle(iris_path, best_pupil, best_iris)

        cv2.imwrite(os.path.join(folder_path, iris), new_image)  # Save the preprocessed image to the folder_path
    
    #Returns path to the processed images
    return folder_path

def daugmanRubberSheet(iris, pupilRadius, irisRadius, pupilCenter, irisCenter):
    """
    Performs Daugmans Rubber Sheet Model on the localised images.
    
    Args:
    iris: image of cropped iris
    pupilRadius: radius of pupil
    irisRadius: radius of iris
    centers: centers of each
    
    Returns:
    path to the processed images
    
    """
    
    folder_path = "DaugmanRubberSheet_images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    #INSERT CODE HERE
    angleResolution = 360
    radiusResolution = 60
    unwrapped = np.zeros((360, 60), dtype=np.uint8)
    
    dx = irisCenter[0] - pupilCenter[0]
    dy = irisCenter[1] - pupilCenter[1]
    
    for i in range(angleResolution):
        angle = 2 * np.pi * i / angleResolution
        for j in range(radiusResolution):
            r = pupilRadius + j * (irisRadius - pupilRadius) / radiusResolution
            x = int(pupilCenter[0] + r * np.cos(angle) + dx)
            y = int(pupilCenter[1] + r * np.sin(angle) + dy)
            if 0 <= x < iris.shape[1] and 0 <= y < iris.shape[0]:
                unwrapped[j, i] = iris[y, x]
        
    return unwrapped


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

daugmanIDO("preprocessed_images", (20, 40), (65, 90), (120, 180), (80, 120), 1)