import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import ACM
import daugman
import maMethod
import pandas as pd

dataPath = "preprocessed_images"
dir = os.listdir(dataPath)

if not os.path.exists("system4"):
    os.makedirs("system4")
    os.makedirs("system4/localised_images")
    os.makedirs("system4/normalized_images")
else:
    for file in os.listdir("system4/localised_images"):
        os.remove("system4/localised_images/" + file)
    for file in os.listdir("system4/normalized_images"):
        os.remove("system4/normalized_images/" + file)

features = pd.DataFrame(columns = ["ClassID", "FeatureVector"])

for file in dir:
    if file.endswith("JPG"):
        print(file + " is being processed")
        image = os.path.join(dataPath,file)
        imgName = os.path.basename(image)
        
        #Extract Class ID from the image name
        classID = file.split("_")[1]
        
        #Perform Hough Transform
        image, pupil, iris = ACM.ACM(image)
        
        #Save the localised image
        saved = cv2.imwrite("system4/localised_images/" + imgName, image)

        print("Iris Localised")
        
        #Perform Daugman's Rubber Sheet Model
        normalizedImage = daugman.daugmanRubberSheet(image, pupil[2], iris[2], (pupil[0], pupil[1]), (iris[0], iris[1]))

        #Save the normalized images
        cv2.imwrite("system4/normalized_images/" + imgName, normalizedImage)
        
        print("Iris Normalised")
        
        #Feature Extraction
        maLVA = maMethod.LVA(normalizedImage)
        
        #Save the Gabor features with class Id
        features.add({"ClassID": classID, "FeatureVector": maLVA})        
        print("Feature Extracted")
        
#Save the features to a file
np.savetxt("system4/features.npy", features)


        
        
        
        
        