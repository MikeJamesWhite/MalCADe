import sys
import cv2
import numpy as np
import os


def isolate_green(img):
    b, g, r = cv2.split(img)
    return g


def green_filter(img):
    g = img.copy()
    # set blue and red channels to 0
    g[:, :, 0] = 0
    g[:, :, 2] = 0
    return g


def isolate_blue(img):
    b, g, r = cv2.split(img)
    return b


def blue_filter(img):
    b = img.copy()
    # set green and red channels to 0
    b[:, :, 1] = 0
    b[:, :, 2] = 0


def isolate_red(img):
    b, g, r = cv2.split(img)
    return r


def red_filter(img):
    r = img.copy()
    # set blue and green channels to 0
    r[:, :, 0] = 0
    r[:, :, 1] = 0


def contrast(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def gaus_blur(img):
    return cv2.GaussianBlur(img,(7,7),0)

def median_filter(img, size=5):
    return cv2.medianBlur(img, size)


def adaptive_histogram(img):
    return cv2.equalizeHist(img)


def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def edge(img):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


def emboss(img):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    return cv2.filter2D(img, -1, kernel)


def hsv_model(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def threshold(img, thresh=130, maxval=255, method=cv2.THRESH_BINARY):
    return cv2.threshold(img, thresh, maxval, method)[1]

if __name__ == "__main__":
    sys.path.append('./preprocessing')
    import ImageUtils as iu

    print("Which image would you like to display?")
    for root, dirs, files in os.walk("./images"):
        for filename in files:
            print(filename)
    print()

    imgName = input()
    img = iu.read_image("./images/" + imgName)
    img_gray = iu.read_gray_image("./images/" + imgName)
    img = iu.resize_image(img)
    img_gray = iu.resize_image(img_gray)

    # Individually display images
    # iu.display_image("hsv", iu.combine_images(img, hsv_model(img)))
    # iu.display_image("hsv_green", iu.combine_images(img_gray, isolate_green(hsv_model(img))))
    # iu.display_image("hsv_green_threshold", iu.combine_images(isolate_green(hsv_model(img)), threshold(contrast(isolate_green(hsv_model(img)), 2.0, 10), thresh=200)))
    # iu.display_image("adaptive_histogram", iu.combine_images(img_gray, adaptive_histogram(img_gray)))
    # iu.display_image("Median", iu.combine_images(img, median_filter(img)))
    # iu.display_image("Contrast", iu.combine_images(img, contrast(img, 2.0, 10)))
    # iu.display_image("Green", iu.combine_images(img_gray, isolate_green(img)))
    # iu.display_image("Blue", iu.combine_images(img_gray, isolate_blue(img)))
    # iu.display_image("Red", iu.combine_images(img_gray, isolate_red(img)))
    # iu.display_image("Green Filter", iu.combine_images(img, green_filter(img)))  
    # iu.display_image("Sharpen", iu.combine_images(img, sharpen(img)))  
    # iu.display_image("Edge", iu.combine_images(img, edge(img)))
    # iu.display_image("Emboss", iu.combine_images(img_gray, emboss(img_gray)))
    # iu.display_image("Edge + Contrast", iu.combine_images(img, edge(contrast(img, 3.0, 10))))

    # Display filtered images in single window
    '''
    images = [
        iu.prepare_image(contrast(isolate_green(hsv_model(contrast(img))), 2.5, 10), grey=True),
        iu.prepare_image(adaptive_histogram(img_gray), grey=True),
        iu.prepare_image(median_filter(img)),
        iu.prepare_image(contrast(img, 2.0, 10)),
        iu.prepare_image(isolate_red(img), grey=True),
        iu.prepare_image(isolate_green(img), grey=True),
        iu.prepare_image(isolate_blue(img), grey=True),
        iu.prepare_image(green_filter(img)),
        iu.prepare_image(sharpen(img)),
        iu.prepare_image(edge(img)),
        iu.prepare_image(emboss(img_gray), grey=True),
        iu.prepare_image(edge(contrast(img, 3.0, 10))),
    ]
    iu.display_all_images(images)
    iu.display_image("Original", iu.prepare_image(img))
    '''

    cv2.waitKey(0)
