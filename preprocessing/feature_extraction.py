import cv2
import copy
import mahotas
import numpy as np
import glob


def hu_moments(img):
    ''' Calculate and return the transformation-invariant Hu Moments'''
    if (len(img.shape) > 2):
        # convert to grey
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey = img

    # calculate Hu moments
    feature = cv2.HuMoments(cv2.moments(grey)).flatten()
    return feature


def haralick(img):
    '''Calculate and return the Haralick texture attributes of input image'''
    if (len(img.shape) > 2):
        # convert to grey
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey = img

    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(grey).mean(axis=0)
    return haralick


def __colour_histogram(img, bins=16):
    # compute and normalise histogram
    hist = cv2.calcHist(
        [img], [0, 1, 2], None,
        [bins, bins, bins],
        [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)

    return hist.flatten()


def colour_histogram(bins=16):
    '''Calculate and return the HSV colour histogram of input image'''
    return lambda x: __colour_histogram(x, bins)


def __greyscale_histogram(img, bins=16):
    if (len(img.shape) > 2):
        # convert to grey
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = img

    # compute and normalise histogram
    hist = cv2.calcHist(
        [grey], [0], None,
        [bins],
        [0, 256]
    )
    cv2.normalize(hist, hist)

    return hist.flatten()


def greyscale_histogram(bins=16):
    '''Calculate and return the greyscale histogram of input image'''
    return lambda x: __greyscale_histogram(x, bins)
