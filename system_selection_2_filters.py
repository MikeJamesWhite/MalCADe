"""System selection stage two, to determine best image filtering approaches"""

from testing_framework.test_runner import run_training_and_tests
from preprocessing.feature_extraction import hu_moments, haralick, colour_histogram, greyscale_histogram
from preprocessing.image_filters import contrast, isolate_saturation, hsv_model, threshold
from models import rf, svm

# Pre-defined filter combinations
hsv = [hsv_model]
hsv_saturation = [hsv_model, isolate_saturation]
hsv_saturation_threshold = [hsv_model, isolate_saturation, lambda x: contrast(x, 2.0, 10), lambda x: threshold(x, thresh = 200)]
hsv_saturation_contrast = [hsv_model, isolate_saturation, lambda x: contrast(x, 2.5, 10)]

def greyscale_histogram_with_filters():
    """Run evaluation of greyscale histograms with filters"""

    rf_model_normal = rf.RF(
        label = 'RF with no preprocessing', 
        preprocessing = [], 
        features = [greyscale_histogram(bins = 64)]
    )

    rf_model_hsv = rf.RF(
        label = 'RF with hsv', 
        preprocessing = [hsv_model], 
        features = [greyscale_histogram(bins = 64)]
    )

    rf_model_hsv_is = rf.RF(
        label = 'RF with hsv and isolate saturation', 
        preprocessing = hsv_saturation, 
        features = [greyscale_histogram(bins = 64)]
    )

    rf_model_hsv_is_thresh = rf.RF(
        label = 'RF with hsv, isolate saturation, threshold', 
        preprocessing = hsv_saturation_threshold, 
        features = [greyscale_histogram(bins = 64)]
    )

    rf_model_hsv_is_c = rf.RF(
        label = 'RF with hsv, isolate saturation, contrast', 
        preprocessing = hsv_saturation_contrast, 
        features = [greyscale_histogram(bins = 64)]
    )

    svm_model_normal = svm.SVM(
        label = 'SVM with no preprocessing', 
        preprocessing = [], 
        features = [greyscale_histogram(bins = 64)]
    )

    svm_model_hsv = svm.SVM(
        label = 'SVM with hsv', 
        preprocessing = [hsv_model], 
        features = [greyscale_histogram(bins = 64)]
    )

    svm_model_hsv_is = svm.SVM(
        label = 'SVM with hsv and isolate saturation', 
        preprocessing = hsv_saturation, 
        features = [greyscale_histogram(bins = 64)]
    )

    svm_model_hsv_is_thresh = svm.SVM(
        label = 'SVM with hsv, isolate saturation, threshold', 
        preprocessing = hsv_saturation_threshold, 
        features = [greyscale_histogram(bins = 64)]
    )

    svm_model_hsv_is_c = svm.SVM(
        label = 'SVM with hsv, isolate saturation, contrast', 
        preprocessing = hsv_saturation_contrast, 
        features = [greyscale_histogram(bins = 64)]
    )

    run_training_and_tests(
        'system_selection_2_grey_hist_preprocessing', 
        'kaggle', 
        [
            rf_model_normal, 
            rf_model_hsv, 
            rf_model_hsv_is, 
            rf_model_hsv_is_c, 
            rf_model_hsv_is_thresh, 
            svm_model_normal, 
            svm_model_hsv, 
            svm_model_hsv_is, 
            svm_model_hsv_is_c, 
            svm_model_hsv_is_thresh, 
        ], 
        n_iterations = 5, n_images = 10000, training_split = 0.5
    )

def colour_histogram_with_filters():
    """Run evaluation of colour histograms with filters"""

    rf_model_normal = rf.RF(
        label = 'RF with no preprocessing', 
        preprocessing = [], 
        features = [colour_histogram(bins = 16)]
    )

    rf_model_hsv = rf.RF(
        label = 'RF with hsv', 
        preprocessing = [hsv_model], 
        features = [colour_histogram(bins = 16)]
    )

    svm_model_normal = svm.SVM(
        label = 'SVM with no preprocessing', 
        preprocessing = [], 
        features = [colour_histogram(bins = 16)]
    )

    svm_model_hsv = svm.SVM(
        label = 'SVM with hsv', 
        preprocessing = [hsv_model], 
        features = [colour_histogram(bins = 16)]
    )

    run_training_and_tests(
        'system_selection_2_colour_hist_preprocessing', 
        'kaggle', 
        [
            rf_model_normal, 
            rf_model_hsv, 
            svm_model_normal, 
            svm_model_hsv
        ], 
        n_iterations = 5, n_images = 10000, training_split = 0.5
    )

def haralick_with_filters():
    """Run evaluation of haralick texture attributes with filters"""

    rf_model_normal = rf.RF(
        label = 'RF with no preprocessing', 
        preprocessing = [], 
        features = [haralick]
    )

    rf_model_hsv = rf.RF(
        label = 'RF with hsv', 
        preprocessing = hsv, 
        features = [haralick]
    )

    rf_model_hsv_is = rf.RF(
        label = 'RF with hsv and isolate saturation', 
        preprocessing = hsv_saturation, 
        features = [haralick]
    )

    rf_model_hsv_is_thresh = rf.RF(
        label = 'RF with hsv, isolate saturation, threshold', 
        preprocessing = hsv_saturation_threshold, 
        features = [haralick]
    )

    rf_model_hsv_is_c = rf.RF(
        label = 'RF with hsv, isolate saturation, contrast', 
        preprocessing = hsv_saturation_contrast, 
        features = [haralick]
    )

    svm_model_normal = svm.SVM(
        label = 'SVM with no preprocessing', 
        preprocessing = [], 
        features = [haralick]
    )

    svm_model_hsv = svm.SVM(
        label = 'SVM with hsv', 
        preprocessing = hsv, 
        features = [haralick]
    )

    svm_model_hsv_is = svm.SVM(
        label = 'SVM with hsv and isolate saturation', 
        preprocessing = hsv_saturation, 
        features = [haralick]
    )

    svm_model_hsv_is_thresh = svm.SVM(
        label = 'SVM with hsv, isolate saturation, threshold', 
        preprocessing = hsv_saturation_threshold, 
        features = [haralick]
    )

    svm_model_hsv_is_c = svm.SVM(
        label = 'SVM with hsv, isolate saturation, contrast', 
        preprocessing = hsv_saturation_contrast, 
        features = [haralick]
    )

    run_training_and_tests(
        'system_selection_2_haralick_preprocessing', 
        'kaggle', 
        [
            rf_model_normal, 
            rf_model_hsv, 
            rf_model_hsv_is, 
            rf_model_hsv_is_c, 
            rf_model_hsv_is_thresh, 
            svm_model_normal, 
            svm_model_hsv, 
            svm_model_hsv_is, 
            svm_model_hsv_is_c, 
            svm_model_hsv_is_thresh, 
        ], 
        n_iterations = 5, n_images = 10000, training_split = 0.5
    )


# Run evaluations
if __name__ == '__main__':
    greyscale_histogram_with_filters()
    haralick_with_filters()
    colour_histogram_with_filters()
