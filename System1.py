import daugman
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import util

dataPath = "preprocessed_images"
dir = os.listdir(dataPath)

if not os.path.exists("system1"):
    os.makedirs("system1")
    os.makedirs("system1/localised_images")
    os.makedirs("system1/normalized_images")
else:
    for file in os.listdir("system1/localised_images"):
        os.remove("system1/localised_images/" + file)
    for file in os.listdir("system1/normalized_images"):
        os.remove("system1/normalized_images/" + file)

features = []

for file in dir:
    if file.endswith("JPG"):
        print(file + " is being processed")
        image = os.path.join(dataPath,file)
        imgName = os.path.basename(image)
        
        #Extract Class ID from the image name
        classID = file.split("_")[1]
        
        #Perform Daugman's Integro Differential Operator
        image, pupil, iris = daugman.daugmanIDO(image, (25, 40), (60, 90), (150, 180), (80, 120), 1)
        
        #Save the localised image
        saved = cv2.imwrite("system1/localised_images/" + imgName, image)
        
        #Perform Daugman's Rubber Sheet Model
        normalizedImage = daugman.daugmanRubberSheet(image, pupil[2], iris[2], (pupil[0], pupil[1]), (iris[0], iris[1]))

        #Save the normalized image
        cv2.imwrite("system1/normalized_images/" + imgName, normalizedImage)
        
        #Feature Extraction
        gaborFeatures = daugman.daugmanGaborWavelet(normalizedImage)
        
        #Save the Gabor features with class Id
        features.append((classID, gaborFeatures))
        
#Save the features to a file
np.save("system1/features.txt", features)


        
        
        
        
        