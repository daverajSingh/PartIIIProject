import numpy as np
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
import pywt

def LVA(image):
    
    """
    Performs Ma's Localised Variation Analysis on the normalised image, returning an encoded feature vector.
    
    Args:
    image: normalised image
    
    Returns:
    encodedFeatureVector: feature vector
    
    """

    #1D Intensity Signals
    height, width = image.shape
    step = height// 3
    signals = [np.mean(image[i:i+step], axis=0) for i in range(0, height, step)]
    encodedSignals = []
    
    #Wavelet Transform 
    for signal in signals:   
        coefficients = pywt.wavedec(signal, wavelet='db1', level=2) 
        features = []
        
        for level in coefficients:
            for index in range (1, len(level)-1):
                if level[index-1]< level[index] >level[index+1] or level[index-1]> level[index] <level[index+1]:
                    features.append((index, 'max' if level[index-1] < level[index] > level[index+1] else 'min'))
       
        #Encoding
        binary = np.zeros(len(signal), dtype=int)
        for idx, nature in features:
            binary[idx] = 1 if nature == 'max' else -1
        
        for i in range(1, len(binary)):
            binary[i] = binary[i] ^ binary[i-1]
        
        encodedSignals.append(binary)

    encodedFeatureVector = np.concatenate(encodedSignals)
    
    return encodedFeatureVector


x = cv2.imread("preprocessed_images\IMG_001_R_1.JPG", cv2.IMREAD_GRAYSCALE)
featureVector = LVA(x)