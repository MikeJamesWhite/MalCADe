import sys
import cv2
import numpy as np
import os


def isolate_saturation(img):
    h,s,v = cv2.split(img)
    return s


def contrast(img, alpha=2.0, beta=1):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def hsv_model(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def threshold(img, thresh=150, maxval=255, method=cv2.THRESH_BINARY):
    return cv2.threshold(img, thresh, maxval, method)[1]


if __name__ == "__main__":
    import preprocessing.image_utils as iu

    print("Which image would you like to display?")
    for root, dirs, files in os.walk("./images"):
        for filename in files:
            print(filename)
    print()

    imgName = input()
    img = iu.read_image("./images/" + imgName)
    img = iu.resize_image(img)

    iu.display_image("normal", img)
    iu.display_image("hsv", hsv_model(img))
    iu.display_image("isolate saturation", isolate_saturation(hsv_model(img)))
    iu.display_image("isolate saturation + threshold", threshold(isolate_saturation(hsv_model(img))))
    iu.display_image("isolate saturation + contrast", contrast(isolate_saturation(hsv_model(img))))

    cv2.waitKey(0)
