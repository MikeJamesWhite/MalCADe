'''Utility functions for working with images'''

import os
import random
import math
import numpy as np
import glob
import cv2

def read_images_in_folder(n_images, path, x=150, y=150, random_state=None):
    """Read a random set of n images from a folder, resizing each"""

    # Set random state if specified
    if not (random_state is None):
        random.setstate(random_state)

    # Get list of files in the directory
    files = os.listdir(path)

    # Notify user when there are not enough images in the directory
    if len(files) < n_images:
        print("NOT ENOUGH IMAGES IN PATH")
        print("Required:", n_images)
        print("Found:", len(files))

    # Choose a random sample of n files from the directory
    chosen = random.sample(files, n_images)

    # Load and resize each image file
    images = []
    for file in chosen:
        img = cv2.imread(path+file)
        images.append(cv2.resize(img, (x, y)))
        if len(images) == n_images:
            break

    return images


def read_dataset(n_train, n_test, path, x=150, y=150, random_state=None):
    """Read in a dataset of infected and uninfected images, and split
    these into train and test sets of specified sizes
    """

    print('Loading images from dataset:', path)
    train_dataset = []
    test_dataset = []

    # Read in the image files into infected and uninfected arrays
    infected = read_images_in_folder(
        math.floor((n_train + n_test)/2),
        path + '/infected/',
        x=x,
        y=y,
        random_state=random_state
    )
    uninfected = read_images_in_folder(
        math.floor((n_train + n_test)/2),
        path + '/uninfected/',
        x=x,
        y=y,
        random_state=random_state
    )

    # Split into train and test sets
    for img in infected:
        if len(train_dataset) < n_train // 2:
            train_dataset.append([img, "Infected"])
        else:
            test_dataset.append([img, "Infected"])
    for img in uninfected:
        if len(train_dataset) < n_train:
            train_dataset.append([img, "Uninfected"])
        else:
            test_dataset.append([img, "Uninfected"])

    # Split test and train sets into image and label arrays
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for item in train_dataset:
        train_images.append(item[0])
        train_labels.append(item[1])
    for item in test_dataset:
        test_images.append(item[0])
        test_labels.append(item[1])

    print("'Infected' training images:", train_labels.count("Infected"))
    print("'Uninfected' training images:", train_labels.count("Uninfected"))
    print("'Infected' testing images:", test_labels.count("Infected"))
    print("'Uninfected' testing images:", test_labels.count("Uninfected"))
    print('Finished loading images\n')

    return train_images, train_labels, test_images, test_labels


def resize_image(image, x=500, y=400):
    """Resize a given image based on specified x and y values"""

    return cv2.resize(image, (x, y))


def display_image(title, img):
    """Display an image with an associated title"""

    cv2.imshow(title, img)

if __name__ == "__main__":
    # Test read_images_in_folder()
    img_list = read_images_in_folder(7, "./images")
    assert(len(img_list) == 7)

    # Test read_dataset()
