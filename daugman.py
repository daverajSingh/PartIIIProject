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
    img = cv2.GaussianBlur(image, (7, 7), 0)
    img = cv2.equalizeHist(img)
    
    rows, cols = img.shape

    best_iris = (0,0,0)
    max_gradientI = -np.inf   
        
    best_pupil = (0,0,0)
    max_gradientP = -np.inf

    gradients = np.gradient(img.astype(float), axis=(0, 1))
    gradientMagnitude = np.sqrt(gradients[0]**2 + gradients[1]**2)

    for r in range(iRadiusRange[0], iRadiusRange[1], deltaRadius):
        for x in range(centerXRange[0], centerXRange[1]):
            for y in range(centerYRange[0], centerYRange[1]):
                if x - r < 0 or x + r > cols or y - r < 0 or y + r > rows:
                    continue  # Ensure the circle mask doesn't go out of image bounds
                gradient_sum = computeGradient(gradientMagnitude, x, y, r, deltaRadius)
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
                gradient_sum = computeGradient(gradientMagnitude, x, y, r, deltaRadius)
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
    pupilCenter: centers of pupil
    irisCenter: centers of iris
    
    Returns:
    unwrapped: unwrapped image of the iris
    
    """
    angleResolution = 512
    radiusResolution = 64
    unwrapped = np.zeros((radiusResolution, angleResolution), dtype=np.uint8)
    
    # Calculate the difference between the centers of the iris and pupil
    dx = irisCenter[0] - pupilCenter[0]
    dy = irisCenter[1] - pupilCenter[1]
    
    # If the difference is within a certain range, adjust the unwrapped image
    if not(-10 <= dx <= 10 and -10 <= dy <= 10):
        dx = dy = 0
        for i in range(angleResolution):
            angle = 2 * np.pi * i / angleResolution
            for j in range(radiusResolution):
                r = pupilRadius + j * (irisRadius - pupilRadius) / radiusResolution
                x = int(irisCenter[0] + r * np.cos(angle) + dx)
                y = int(irisCenter[1] + r * np.sin(angle) + dy)
                if 0 <= x < iris.shape[1] and 0 <= y < iris.shape[0]:
                    unwrapped[j, i] = iris[y, x]
    else:
        for i in range(angleResolution):
            angle = 2 * np.pi * i / angleResolution
            for j in range(radiusResolution):
                r = pupilRadius + j * (irisRadius - pupilRadius) / radiusResolution
                x = int(pupilCenter[0] + r * np.cos(angle) + dx)
                y = int(pupilCenter[1] + r * np.sin(angle) + dy)
                if 0 <= x < iris.shape[1] and 0 <= y < iris.shape[0]:
                    unwrapped[j, i] = iris[y, x]

    unwrapped = cv2.equalizeHist(unwrapped)

    #Crop image to remove right half - eyelid and eyelashes
    unwrapped = unwrapped[:, 0:256]
    
    #Keep only lower 75% of the image
    unwrapped = unwrapped[16:64, :]

    return unwrapped

def daugmanGaborWavelet(image):
    """
    Performs Daugmans Gabor Wavelet on the processed images.
    
    Args:
    data_path: path to the normalised images
    
    Returns:
    featureVector: feature vector of the processed image
    
    """
        
    freq = [0.1, 0.2, 0.4, 0.6]
    theta = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
    rows, cols = image.shape
    
    iris_code = np.zeros((rows, cols, len(freq) * len(theta) * 2), dtype=np.uint8)
    
    image = cv2.addWeighted(image, 1, np.zeros(image.shape, image.dtype), -0.5, 0)
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, sharpen)

        
    # Apply Gabor filter to the image
    for idx, f in enumerate(freq):
        for jdx, t in enumerate(theta):
            feature = gaborfilter(image, f, t)
            
            phase = np.arctan2(np.imag(feature), np.real(feature))
            bit1 = (phase >= 0).astype(np.uint8)
            bit2 = np.logical_or(phase >= np.pi/2, phase < -np.pi/2).astype(np.uint8)

            iris_code[:,:,2*(idx*len(theta)+jdx)] = bit1
            iris_code[:,:,2*(idx*len(theta)+jdx)+1] = bit2


    return iris_code.flatten()


def computeGradient(gradientMagnitude, centerX, centerY, radius, deltaradius):
    rows, cols = gradientMagnitude.shape
    mask = np.zeros((rows, cols), dtype = np.uint8)
    cv2.circle(mask, (centerX, centerY), radius + deltaradius, 255, -1)
    cv2.circle(mask, (centerX, centerY), radius - deltaradius, 0, -1)

    gradient = cv2.bitwise_and(gradientMagnitude, gradientMagnitude, mask=mask)
    totalGradient = np.sum(gradient)

    annulus_area = np.pi * ((radius + deltaradius)**2 - (radius-deltaradius)**2)
    return totalGradient / annulus_area


def gaborfilter(image, frequency, theta):
    kernel = cv2.getGaborKernel((3,3),sigma=0.5, theta=theta, lambd=frequency, gamma=0.5, psi=0, ktype=cv2.CV_32F)  
    if kernel is None or kernel.size == 0:
        print("Kernel creation failed.")
        
    filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
    if filtered is None or filtered.size == 0:
        print("Filtering failed.")
      
    return filtered


