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

    filtered = np.where(filtered < 0, 0, 1)
    return np.asarray(filtered)