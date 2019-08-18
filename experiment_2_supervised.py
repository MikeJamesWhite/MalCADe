# Initial evaluation to determine best image filtering approaches
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

def greyscale_histogram_with_preprocessing():
    rf_model_normal = rf.RF(
        label='RF with no preprocessing',
        preprocessing=[],
        features=[greyscale_histogram(bins=32)]
    )

    rf_model_hsv_ig = rf.RF(
        label='RF with hsv and isolate green',
        preprocessing=hsv_green,
        features=[greyscale_histogram(bins=32)]
    )

    rf_model_hsv_ig_thresh = rf.RF(
        label='RF with hsv, isolate green, threshold',
        preprocessing=hsv_green_threshold,
        features=[greyscale_histogram(bins=32)]
    )

    rf_model_hsv_ig_c = rf.RF(
        label='RF with hsv, isolate green, contrast',
        preprocessing=hsv_green_contrast,
        features=[greyscale_histogram(bins=32)]
    )

    rf_model_c_edge = rf.RF(
        label='RF with contrast and edge detect',
        preprocessing=contrast_edge,
        features=[greyscale_histogram(bins=32)]
    )

    run_tests('RF_Greyscale_Preprocessing', 'kaggle', [rf_model_normal, rf_model_hsv_ig, rf_model_hsv_ig_c, rf_model_hsv_ig_thresh, rf_model_c_edge], n_iterations=10, n_images=10000, training_split=0.5)

def hsv_histogram_with_preprocessing():
    pass

def hu_haralick_with_filters():
    rf_model_haralick_hu_hsv_isolate_green = rf.RF(
        label='RF, haralick and hu features + hsv, isolate_green, contrast, thresh',
        preprocessing=hsv_green_threshold,
        features=[haralick, hu_moments]
    )

    svm_model_haralick_hu_hsv_isolate_green = svm.SVM(
        label='SVM, haralick and hu features + hsv, isolate_green, contrast, thresh',
        preprocessing=hsv_green_threshold,
        features=[haralick, hu_moments]
    )

    run_tests(
        'Haralick + Hu Moment Evaluation with Filtering',
        'kaggle',
        [
         rf_model_haralick_hu_hsv_isolate_green,
         svm_model_haralick_hu_hsv_isolate_green
        ],
        n_iterations=5, n_images=10000, training_split=0.5)

