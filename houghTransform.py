import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
from util import cropCircle

def houghTransform(image, pupilRadiusMax, irisRadiusMin):
    
    """
    Performs Hough Transform to detect the iris and pupil circles.
    
    Args:
    image: preprocessed image
    pupilRadiusMin: minimum radius of the pupil
    pupilRadiusMax: maximum radius of the pupil
    irisRadiusMin: minimum radius of the iris
    irisRadiusMax: maximum radius of the iris
    
    Returns:
    iris: segmented iris
    """
    
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(img,100,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,1)

    irisCircles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=1, param1=100, param2=0.8, minRadius=irisRadiusMin, maxRadius=0)
    
    pupilCircles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT_ALT, dp=1, minDist=3, param1=500, param2=0.6, minRadius=0, maxRadius=pupilRadiusMax)
          
    if not np.any(irisCircles):
        print("No iris detected.") 
        return None, None, None
    elif not np.any(pupilCircles):
        print("No pupil detected.")
        return None, None, None  
    else:
        averageIris = np.mean(irisCircles, axis=1)
        iris = np.uint16(np.around(averageIris[0]))
        
        averagePupil = np.mean(pupilCircles, axis=1)
        pupil = np.uint16(np.around(averagePupil[0]))
        
        
        print(iris, pupil)
        image = cropCircle(image, pupil, iris)
        return image, iris, pupil

def arsalanHoughTransform(image, pupilRadiusMin, pupilRadiusMax, irisRadiusMin, irisRadiusMax):
    
    """
    Performs Arsalan's Hough Transform to detect the iris and pupil circles.
    
    Args:
    image: preprocessed image
    pupilRadiusMin: minimum radius of the pupil
    pupilRadiusMax: maximum radius of the pupil
    irisRadiusMin: minimum radius of the iris
    irisRadiusMax: maximum radius of the iris
    
    Returns:
    iris: segmented iris
    """
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    img = cv2.GaussianBlur(image, (5, 5), 0)
    img = cv2.medianBlur(img, 5)
    thresh = cv2.adaptiveThreshold(img,100,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,1)

    irisCircles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=1, param1=100, param2=0.8, minRadius=irisRadiusMin, maxRadius=irisRadiusMax)
    
    pupilCircles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT_ALT, dp=1, minDist=3, param1=500, param2=0.6, minRadius=pupilRadiusMin, maxRadius=pupilRadiusMax)
          
    if not np.any(irisCircles):
        print("No iris detected.") 
        return None, None, None
    elif not np.any(pupilCircles):
        print("No pupil detected.")  
        return None, None, None
    else:
        averageIris = np.mean(irisCircles, axis=1)
        iris = np.uint16(np.around(averageIris[0]))
        
        averagePupil = np.mean(pupilCircles, axis=1)
        pupil = np.uint16(np.around(averagePupil[0]))
        
        print(iris, pupil)
        image = cropCircle(image, pupil, iris)
        return image, iris, pupil


