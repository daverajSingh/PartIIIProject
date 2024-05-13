import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import houghTransform
import daugman
import logGaborFilter
import pandas as pd

dataPath = "preprocessed_images"
dir = os.listdir(dataPath)

if not os.path.exists("system3"):
    os.makedirs("system3")
    os.makedirs("system3/localised_images")
    os.makedirs("system3/normalized_images")
else:
    for file in os.listdir("system3/localised_images"):
        os.remove("system3/localised_images/" + file)
    for file in os.listdir("system3/normalized_images"):
        os.remove("system3/normalized_images/" + file)

features = []

for file in dir:
    if file.endswith("JPG"):
        print(file + " is being processed")
        image = os.path.join(dataPath,file)
        imgName = os.path.basename(image)
        
        #Extract Class ID from the image name
        classID = file.split("_")[1]
        
        #Perform Hough Transform
        image, pupil, iris = houghTransform.arsalanHoughTransform(image, 20, 40, 50, 90)
        
        if(image is None):
            continue
        
        #Save the localised image
        saved = cv2.imwrite("system3/localised_images/" + imgName, image)
        
        #Perform Daugman's Rubber Sheet Model
        normalizedImage = daugman.daugmanRubberSheet(image, pupil[2], iris[2], (pupil[0], pupil[1]), (iris[0], iris[1]))

        #Save the normalized images
        cv2.imwrite("system3/normalized_images/" + imgName, normalizedImage)
        
        #Feature Extraction
        logGabor = logGaborFilter.masekLogGabor(normalizedImage)
        print(logGabor)
        #Save the Gabor features with class Id
        features.append([classID, logGabor])

features = pd.DataFrame(features, columns = ["ClassID", "FeatureVector"])

#Save the features to a file
np.save("system3/features.npy", features)


        
        
        
        
        