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
    image = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
    imgName = os.path.basename(data_path)

    #Gaussian blur
    img = cv2.GaussianBlur(image, (5, 5), 0)
    
    best_iris = (0,0,0)
    max_gradientI = -np.inf   
        
    best_pupil = (0,0,0)
    max_gradientP = -np.inf
        
    rows, cols = img.shape
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
        
    new_image = cropCircle(image, best_pupil, best_iris)
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
    unwrapped = np.zeros((radiusResolution, angleResolution), dtype=np.uint8)
    
    # Calculate the difference between the centers of the iris and pupil
    dx = irisCenter[0] - pupilCenter[0]
    dy = irisCenter[1] - pupilCenter[1]
    
    # If the difference is within a certain range, adjust the unwrapped image
    if -10 <= dx <= 10 and -10 <= dy <= 10:
        for i in range(angleResolution):
            angle = 2 * np.pi * i / angleResolution
            for j in range(radiusResolution):
                r = pupilRadius + j * (irisRadius - pupilRadius) / radiusResolution
                x = int(pupilCenter[0] + r * np.cos(angle) + dx)
                y = int(pupilCenter[1] + r * np.sin(angle) + dy)
                if 0 <= x < iris.shape[1] and 0 <= y < iris.shape[0]:
                    unwrapped[j, i] = iris[y, x]
    else: # Otherwise, use the iris center, as the iris has been incorrectly identified
        for i in range(angleResolution):
            angle = 2 * np.pi * i / angleResolution
            for j in range(radiusResolution):
                r = pupilRadius + j * (irisRadius - pupilRadius) / radiusResolution
                x = int(irisCenter[0] + r * np.cos(angle))
                y = int(irisCenter[1] + r * np.sin(angle))
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
        kernel = cv2.getGaborKernel((3,3),sigma=0.5, theta=theta, lambd=frequency, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        
        return filtered
        
    freq = [0.2, 0.3, 0.4]
    theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    featureVector = np.array([], dtype=np.uint8)
    
    # Apply Gabor filter to the image
    for f in freq:
        for t in theta:
            feature = gaborfilter(image, f, t)
            featureVector = np.append(featureVector, feature.ravel())

    # Basic encoding - if the pixel value is greater than the mean, it is encoded as 1, otherwise 0
    encoded = np.zeros_like(featureVector)
    encoded[featureVector > np.mean(featureVector)] = 1
    encoded[featureVector < np.mean(featureVector)] = 0
    
    return encoded

