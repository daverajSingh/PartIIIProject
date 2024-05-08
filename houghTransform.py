import numpy as np
import cv2
from util import cropCircle

def houghTransform(img, minRadius, maxRadius):
    
    """
    Performs Hough Transform to detect circles in the image.
    
    Args:
    img: image to detect circles in
    minRadius: minimum radius of the circle
    maxRadius: maximum radius of the circle
    
    Returns:
    iris: segmented iris
    """
    
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=minRadius, maxRadius=maxRadius)
    circles = np.uint16(np.around(circles))
    circles = circles[0, :]
    return circles

def arsalanHoughTransform(img, minRadius, maxRadius):
    """
    Performs Arsalan's Improved Hough Transform to detect circles in the image.
    
    Args:
    img: image to detect circles in
    minRadius: minimum radius of the circle
    maxRadius: maximum radius of the circle
    
    Returns:
    iris: segmented iris
    """
    
    # Apply median filter to remove noise
    img = cv2.medianBlur(img, 5)
    
    # Apply canny edge and gamma adjustment to enhance contrast
    edges = cv2.Canny(img, 100, 200)
    edges = cv2.bitwise_not(edges)
    img = cv2.equalizeHist(edges) 
    
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=minRadius, maxRadius=maxRadius)
    circles = np.uint16(np.around(circles))
    circles = circles[0, :]
    return circles


def segmentIris(img):
    """
    Segments the iris from the image.
    
    Args:
    img: image to segment iris from
    
    Returns:
    image: segmented iris
    """
    
    circles = houghTransform(img, 70, 90)
    iris = circles[np.argmax(circles[:, 2])]
    
    circles = houghTransform(img, 25, 45)
    pupil = circles[np.argmax(circles[:, 2])]
    
    image = cropCircle(img, iris, pupil)
    
    return image
