# Final evaluation to test performance of best models on unseen pathcare data
# Author: Michael White (mike.james.white@icloud.com)

from testing_framework.test_runner import run_tests
from preprocessing.feature_extraction import hu_moments, haralick, colour_histogram, greyscale_histogram
from preprocessing.image_filters import contrast, isolate_saturation, hsv_model, threshold
from models import rf, svm

# Pre-defined filter combinations
hsv = [hsv_model]
hsv_saturation = [hsv_model, isolate_saturation]
hsv_saturation_threshold = [hsv_model, isolate_saturation, lambda x: contrast(x, 2.0, 10), lambda x: threshold(x, thresh=200)]
hsv_saturation_contrast = [hsv_model, isolate_saturation, lambda x: contrast(x, 2.5, 10)]

