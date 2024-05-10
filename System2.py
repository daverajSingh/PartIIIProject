import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import houghTransform
import daugman
import wildes

dataPath = "preprocessed_images"
dir = os.listdir(dataPath)

if not os.path.exists("system2"):
    os.makedirs("system2")
    os.makedirs("system2/localised_images")
    os.makedirs("system2/normalized_images")
else:
    for file in os.listdir("system2/localised_images"):
        os.remove("system2/localised_images/" + file)
    for file in os.listdir("system2/normalized_images"):
        os.remove("system2/normalized_images/" + file)

features = []

for file in dir:
    if file.endswith("JPG"):
        print(file + " is being processed")
        image = os.path.join(dataPath,file)
        imgName = os.path.basename(image)
        
        #Extract Class ID from the image name
        classID = file.split("_")[1]
        
        #Perform Hough Transform
        image, pupil, iris = houghTransform.houghTransform(image)
        
        if(image is None):
            continue
        
        #Save the localised image
        saved = cv2.imwrite("system2/localised_images/" + imgName, image)
        
        #Perform Daugman's Rubber Sheet Model
        normalizedImage = daugman.daugmanRubberSheet(image, pupil[2], iris[2], (pupil[0], pupil[1]), (iris[0], iris[1]))

        #Save the normalized image
        cv2.imwrite("system2/normalized_images/" + imgName, normalizedImage)
        
        #Feature Extraction
        laplacianOfGaussian = wildes.laplacianOfGaussian(normalizedImage)
        
        #Save the Gabor features with class Id
        features.append((classID, laplacianOfGaussian))
        
#Save the features to a file
np.save("system2/features.txt", features)


        
        
        
        
        