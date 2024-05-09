import numpy as np
import cv2

def masekLogGabor(image):
    """
    Applies the Log-Gabor filter to the image.
    
    Args:
    image: the image to apply the filter to
    
    Returns:
    filtered: the filtered image
    """
    
    def createFilter(signalLength, centerFrequency, gaussianSpread, modulationFactor):
        freqIndices = np.linspace(-0.5,0.5, signalLength, endpoint=False)
        gaussianEnvelope = np.exp(-np.pi *((freqIndices - centerFrequency) ** 2) / gaussianSpread**2)
        sinComponent = np.exp(-2j * np.pi * modulationFactor * (freqIndices - centerFrequency) ** 2)
        filter = gaussianEnvelope * sinComponent
        return filter
    
    filter = createFilter(signalLength=360, centerFrequency=0.25, gaussianSpread=0.1, modulationFactor=0.1)
    
    filtered = cv2.filter2D(image, cv2.CV_8UC3, filter)
    
    return filtered
    
    
    
x = cv2.imread("preprocessed_images\IMG_001_R_1.JPG", cv2.IMREAD_GRAYSCALE)
filtered = masekLogGabor(x)
print(filtered)    