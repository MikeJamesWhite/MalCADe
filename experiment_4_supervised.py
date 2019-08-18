# Final evaluation to test performance of best models on unseen pathcare data
# Author: Michael White (mike.james.white@icloud.com)

from testing_framework.test_runner import run_tests
from preprocessing.feature_extraction import hu_moments, haralick, hsv_histogram, greyscale_histogram
from preprocessing.image_filters import contrast, edge, isolate_green, hsv_model, threshold
from models import rf, svm

# Pre-defined filter combinations
contrast_edge = [lambda x: contrast(x, 2.5, 10), edge]
hsv_green = [hsv_model, isolate_green]
hsv_green_threshold = [hsv_model, isolate_green, lambda x: contrast(x, 2.0, 10), lambda x: threshold(x, thresh=200)]
hsv_green_contrast = [hsv_model, isolate_green, lambda x: contrast(x, 2.5, 10)]
