"""Feature extraction functions for blood cell image pre-processing"""

import cv2
import copy
import mahotas
import numpy as np
import glob


def hu_moments(img):
    """Calculate and return the transformation-invariant Hu Moments"""

    # Apply greyscale conversion if image is not already single channel
    if (len(img.shape) > 2):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey = img

    # Calculate Hu moments
    feature = cv2.HuMoments(cv2.moments(grey)).flatten()
    return feature


def haralick(img):
    """Calculate and return the Haralick texture attributes of input image"""

    # Apply greyscale conversion if image is not already single channel
    if (len(img.shape) > 2):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey = img

    # Compute the haralick texture feature vector
    haralick = mahotas.features.haralick(grey).mean(axis=0)
    return haralick


def __colour_histogram(img, bins=16):    
    # Compute and normalise histogram
    hist = cv2.calcHist(
        [img], [0, 1, 2], None,
        [bins, bins, bins],
        [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)

    return hist.flatten()


def colour_histogram(bins=16):
    """Calculate and return the HSV colour histogram of input image"""
    return lambda x: __colour_histogram(x, bins)


def __greyscale_histogram(img, bins=16):
    # Apply greyscale conversion if image is not already single channel
    if (len(img.shape) > 2):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: grey = img

    # Compute and normalise histogram
    hist = cv2.calcHist(
        [grey], [0], None,
        [bins],
        [0, 256]
    )
    cv2.normalize(hist, hist)

    return hist.flatten()


def greyscale_histogram(bins=16):
    """Calculate and return the greyscale histogram of input image"""
    return lambda x: __greyscale_histogram(x, bins)
