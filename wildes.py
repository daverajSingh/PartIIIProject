import numpy as np
import cv2

def laplacianOfGaussian(image):
    """
    Applies the Laplacian of Gaussian filter to the image.
    
    Args:
    image: the image to apply the filter to
    
    Returns:
    filtered: the filtered image
    """
    
    filtered = cv2.Laplacian(image, cv2.CV_64F)
    filtered = filtered.ravel()
    
    #Basic encoding - if the pixel value is greater than the mean, it is encoded as 1, otherwise 0
    encoded = np.zeros_like(filtered)
    encoded[filtered > np.mean(filtered)] = 1
    encoded[filtered < np.mean(filtered)] = 0
    
    return encoded
