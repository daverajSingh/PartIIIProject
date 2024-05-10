import numpy as np
import cv2
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

def masekLogGabor(image):
    """
    Applies the Log-Gabor filter to the image.
    
    Args:
    image: the image to apply the filter to
    
    Returns:
    filtered: the filtered image
    """
    
    def createFilter(points, w0, a):
        w = np.fft.fftfreq(points) * points
        gaussianEnvelope = np.exp(-np.pi * ((w-w0)**2)/(a**2))
        sinComponent = np.exp(-2j * np.pi * a * (w-w0))
        return gaussianEnvelope * sinComponent

    size = image.shape[1] #applied across rows
    
    gabor = createFilter(size, 50, 15)
    
    filtered = np.zeros_like(image, dtype=np.complex128)
    for i in range(image.shape[0]):
        imageRowfft = fft(image[i, :])
        filtered[i, :] = ifft(imageRowfft * gabor)
    
    return np.abs(filtered)
    
x = cv2.imread("preprocessed_images\IMG_001_R_1.JPG", cv2.IMREAD_GRAYSCALE)
filtered = np.around(masekLogGabor(x)).astype(np.uint8)
plt.imshow(filtered, cmap='gray')   
plt.show()