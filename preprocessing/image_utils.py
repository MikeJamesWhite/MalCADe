'''Utility functions for working with images'''

import os
import random
import math
import numpy as np
import glob
import cv2

def get_images(folder):
    return glob.iglob(folder+"/*")


def read_image(name):
    return cv2.imread(name)


def read_images_in_folder(n_images, path, x=150, y=150, random_state=None):
    if not (random_state is None):
        random.setstate(random_state)
    files = os.listdir(path)

    if len(files) < n_images:
        print("NOT ENOUGH IMAGES IN PATH")
        print("Required:", n_images)
        print("Found:", len(files))

    # Choose a random set of images from the directory
    chosen = random.choices(files, k=n_images)

    # Load and resize each image
    images = []
    for file in chosen:
        img = cv2.imread(path+file)
        images.append(cv2.resize(img, (x, y)))
        if len(images) == n_images:
            break
    return images


def read_dataset(n_train, n_test, path, x=150, y=150, random_state=None):
    print('Loading images from dataset:', path)
    train_dataset = []
    test_dataset = []

    # Read in the image files
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

    # Split test and train sets into images and labels
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


def read_gray_image(name):
    return cv2.imread(name, cv2.IMREAD_GRAYSCALE)


def resize_image(image, x=500, y=400):
    return cv2.resize(image, (x, y))


def combine_images(img, img2):
    return np.concatenate((img, img2), axis=1)


def display_image(title, img):
    cv2.imshow(title, img)


def prepare_image(img, grey=False):
    img = cv2.resize(img, (256, 256))
    if (grey):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def display_all_images(images):
    preppedImages = [prepare_image(x) for x in images]

    imageRows = []
    for i in range(len(preppedImages)//4):
        print("row", i)
        row = np.hstack((
            preppedImages[i*4],
            preppedImages[i*4+1],
            preppedImages[i*4+2],
            preppedImages[i*4+3]
        ))
        imageRows.append(row)
    imageMatrix = np.vstack(imageRows)
    cv2.imshow("Filters", imageMatrix)

def rotate_images(images):
    # Mirror & rotate images
    count = len(images)
    for i in range(count):
        images.append(cv2.flip(images[i], 0))
        images.append(cv2.flip(images[i], 1))
        images.append(cv2.flip(images[i], 2))
    return images

def random_cropping(images):
    pass

if __name__ == "__main__":
    img_list = get_images("./images")
    for im in img_list:
        img = read_image(im)
        img = resize_image(img)
        display_image(im, img)
