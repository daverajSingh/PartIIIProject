import numpy as np
import cv2
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
from util import cropCircle

def ACM(image_path):
    """
    Applies a Active Contour Model to a given image ot detect the iris.
    
    args:
    image_path: the image to apply the model to
    
    returns:
    new_image: the image with the iris and pupil segmented
    best_pupil: the center and radius of the pupil
    best_iris: the center and radius of the iris
    """
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(image, (7, 7), sigmaX=1, sigmaY=1)

    # Initial Guesses for iris and pupil
    best_iris = (img.shape[1]//2, img.shape[0]//2, 80)
    best_pupil = (img.shape[1]//2, img.shape[0]//2, 15)
    
    #Snake for Iris
    sI = np.linspace(0, 2*np.pi, 400)
    xI = best_iris[0] + best_iris[2]*np.cos(sI)
    yI = best_iris[1] + best_iris[2]*np.sin(sI)
    initI = np.array([xI, yI]).T
    
    #Active Contour Model
    snakeIris = active_contour(img, initI, alpha = 0.001, beta=10000, gamma=0.0001, convergence=0.01, w_line=0, w_edge=10, max_num_iter=75)
    
    #Snake for Pupil
    sP = np.linspace(0, 2*np.pi, 400)
    xP = best_pupil[0] + best_pupil[2]*np.cos(sP)
    yP = best_pupil[1] + best_pupil[2]*np.sin(sP)
    initP = np.array([xP, yP]).T
    
    #Active Contour Model
    snakePupil = active_contour(img, initP, alpha = 0.0001, beta=100000, gamma=0.00001, convergence=0.001, w_line=-100, w_edge=1000, max_num_iter=75)
        
    #Get centers and radii
    best_iris = (int(np.mean(snakeIris[:, 0])), int(np.mean(snakeIris[:, 1])), int(np.mean(np.sqrt((snakeIris[:, 0] - best_iris[0])**2 + (snakeIris[:, 1] - best_iris[1])**2))))
                 
    best_pupil = (int(np.mean(snakePupil[:, 0])), int(np.mean(snakePupil[:, 1])), int(np.mean(np.sqrt((snakePupil[:, 0] - best_pupil[0])**2 + (snakePupil[:, 1] - best_pupil[1])**2))))
    
    best_iris = np.around(best_iris).astype(int)
    best_pupil = np.around(best_pupil).astype(int)
    new_image = cropCircle(image, best_pupil, best_iris)
    
    return new_image, best_pupil, best_iris
    

