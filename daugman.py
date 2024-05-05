import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from util import cropCircle

def daugmanIDO(data_path, pRadiusRange, iRadiusRange, centerXRange, centerYRange, deltaRadius):
    """
    Performs Daugmans Integro Differential Operator on the preprocessed images.
    
    Args:
    data_path: path to the preprocessed images
    
    Returns:
    iris: segmented iris
    
    """
    # Load the image from the data_path
    img = cv2.imread(data_path)
    imgName = os.path.basename(data_path)
    
    best_iris = (0,0,0)
    max_gradientI = -np.inf   
        
    best_pupil = (0,0,0)
    max_gradientP = -np.inf
        
    rows, cols, _ = img.shape
    for r in range(iRadiusRange[0], iRadiusRange[1], deltaRadius):
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
        
        
    #Pupil is likely to be closer to the center of the iris, therefore can make range shorter
        
    pCenterXRange = (best_iris[0] - 7, best_iris[0] + 7)
    pCenterYRange = (best_iris[1] - 7, best_iris[1] + 7)
        
    for r in range(pRadiusRange[0], pRadiusRange[1], deltaRadius):
        for x in range(pCenterXRange[0], pCenterXRange[1]):
            for y in range(pCenterYRange[0], pCenterYRange[1]):
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
        
    print(best_iris, best_pupil, imgName)
        
    new_image = cropCircle(imgName, best_pupil, best_iris)
    #Returns processed image and values
    return new_image, best_pupil, best_iris

def daugmanRubberSheet(iris, pupilRadius, irisRadius, pupilCenter, irisCenter):
    """
    Performs Daugmans Rubber Sheet Model on the localised images.
    
    Args:
    iris: image of cropped iris
    pupilRadius: radius of pupil
    irisRadius: radius of iris
    centers: centers of each
    
    Returns:
    unwrapped: unwrapped image of the iris
    
    """
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


def daugmanGaborWavelet(image):
    """
    Performs Daugmans Gabor Wavelet on the processed images.
    
    Args:
    data_path: path to the normalised images
    
    Returns:
    featureVector: feature vector of the processed image
    
    """
    def gaborfilter(image, frequency, theta):
        kernel = cv2.getGaborKernel((21, 21), 5, theta, frequency, 1, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        return filtered

    freq = [0.1, 0.2, 0.3]
    theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    features = []
    
    #Apply Gabor Filter to find the magnitude of the image
    for frequency in freq:
        for angle in theta:
            filtered = gaborfilter(image, frequency, angle)
            mag, _ = cv2.cartToPolar(filtered.real, filtered.imag)
            features.append(mag)
    
    #Simple Encoding
    featureVector = np.concatenate(features)
    featureVector = np.where(featureVector > np.mean(featureVector), 1, 0)
    
    #Returns binary sequence
    return featureVector

daugmanIDO("preprocessed_images", (25, 45), (70, 90), (150, 180), (80, 120), 1)
image = cv2.imread("DaugmanIDO_images\IMG_058_R_1.JPG")