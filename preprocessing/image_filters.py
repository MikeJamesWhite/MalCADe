"""Image filtering functions for blood cell image pre-processing"""

import sys
import cv2
import numpy as np
import os


def isolate_saturation(hsv):
    """Isolate the saturation channel of an HSV image"""

    h, s, v = cv2.split(hsv)
    return s


def contrast(img, alpha = 2.0, beta = 10):
    """Apply a contrast filter with specified alpha and beta values"""

    return cv2.convertScaleAbs(img, alpha = alpha, beta = beta)


def hsv_model(img):
    """Convert RGB image to HSV colour space"""

    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def threshold(img, thresh = 200, maxval = 255, method = cv2.THRESH_BINARY):
    """Apply a binary thresholding function"""

    return cv2.threshold(img, thresh, maxval, method)[1]


# If module is run directly, allow users to select an image to see
# the effects of image filters
if __name__ == "__main__":
    import preprocessing.image_utils as iu

    # Display images in the folder, and prompt user to select one
    print("Which image would you like to display?")
    for root, dirs, files in os.walk("./images"):
        for filename in files:
            print(filename)
    print()
    imgName = input()

    # Read image and resize 
    img = cv2.imread("./images/" + imgName)
    img = iu.resize_image(img)

    # Display original image and all filtered versions
    iu.display_image(
        "normal", 
        img
    )
    iu.display_image(
        "hsv", 
        hsv_model(img)
    )
    iu.display_image(
        "isolate saturation", 
        isolate_saturation(hsv_model(img))
    )
    iu.display_image(
        "isolate saturation + contrast + threshold", 
        threshold(contrast(isolate_saturation(hsv_model(img))))
    )
    iu.display_image(
        "isolate saturation + contrast", 
        contrast(isolate_saturation(hsv_model(img)))
    )

    # Keep running until a key is pressed
    cv2.waitKey(0)
