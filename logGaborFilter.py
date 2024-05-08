import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


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

        gaussainEnvelope = np.exp(-np.pi *((freqIndices - centerFrequency) ** 2) / gaussianSpread**2)
        sinComponent = np.exp(-2j * np.pi * modulationFactor * (freqIndices - centerFrequency) ** 2)
        filter = gaussainEnvelope * sinComponent
        return filter
    
    filter = createFilter(signalLength=360, centerFrequency=0.25, gaussianSpread=0.1, modulationFactor=0.1)
    
    signalfft = np.fft.fft2(image)
    signalfftshifted = np.fft.fftshift(signalfft)
    
    result = signalfftshifted * filter
    
    resultifft = np.fft.ifftshift(result)
    filteredSignal= np.fft.ifft(resultifft)
    
    return filteredSignal.real    
    
    
    