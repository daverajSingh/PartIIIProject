import numpy as np 
import cv2

def cropCircle(imagePath, pupil, iris):
    """
    Crops the iris and pupil from the image.
    
    Args:
    imagePath: path to the image
    centerX: x-coordinate of the center of the eye
    centerY: y-coordinate of the center of the eye
    radiusIris: radius of the iris
    radiusPupil: radius of the pupil
    
    Returns:
    cropped image
    """
    
    # Load the image
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("The image file was not found.")

    # Create a mask with the same dimensions as the image, filled with zeros (black)
    mask = np.zeros_like(img)

    xP, yP, rP = pupil
    xI, yI, rI = iris

    # Draw the iris circle on the mask
    cv2.circle(mask, (xI, yI), rI, 255, -1)
    # Remove the pupil area
    cv2.circle(mask, (xP, yP), rP, 0, -1)

    # Apply the mask to isolate the iris
    iris = cv2.bitwise_and(img, mask)

    # Find bounding box coordinates for cropping
    y, x = np.where(mask > 0)
    (top_y, left_x) = (np.min(y), np.min(x))
    (bottom_y, right_x) = (np.max(y), np.max(x))
    cropped_iris = iris[top_y:bottom_y+1, left_x:right_x+1]

    return cropped_iris

