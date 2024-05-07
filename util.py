import numpy as np 
import cv2

def cropCircle(imageName, pupil, iris):
    """
    Crops the iris and pupil from the image.
    
    Args:
    imageName: name of the image
    centerX: x-coordinate of the center of the eye
    centerY: y-coordinate of the center of the eye
    radiusIris: radius of the iris
    radiusPupil: radius of the pupil
    
    Returns:
    cropped image
    """
    
    # Load the image
    if imageName is None:
        raise FileNotFoundError("The image file was not found.")

    # Create a mask with the same dimensions as the image, filled with zeros (black)
    mask = np.zeros_like(imageName)

    xP, yP, rP = pupil
    xI, yI, rI = iris

    # Draw the iris circle on the mask
    cv2.circle(mask, (xI, yI), rI, 255, -1)
    # Remove the pupil area
    cv2.circle(mask, (xP, yP), rP, 0, -1)

    # Apply the mask to isolate the iris
    iris = cv2.bitwise_and(imageName, mask)
    
    return iris

