# Evaluation to determine best hyperparameters for each model
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

def svm_kernel():
    svm_rbf = svm.SVM(
        label='SVM with RBF kernel',
        preprocessing=hsv_saturation_threshold,
        features=[haralick],
        kernel='rbf'
    )
    svm_p1 = svm.SVM(
        label='SVM with polynomial kernel, degree 1',
        preprocessing=hsv_saturation_threshold,
        features=[haralick],
        kernel='poly',
        degree=1
    )
    svm_p2 = svm.SVM(
        label='SVM with polynomial kernel, degree 2',
        preprocessing=hsv_saturation_threshold,
        features=[haralick],
        kernel='poly',
        degree=2
    )
    svm_p3 = svm.SVM(
        label='SVM with polynomial kernel, degree 3',
        preprocessing=hsv_saturation_threshold,
        features=[haralick],
        kernel='poly',
        degree=3
    )
    svm_p5 = svm.SVM(
        label='SVM with polynomial kernel, degree 5',
        preprocessing=hsv_saturation_threshold,
        features=[haralick],
        kernel='poly',
        degree=5
    )

    run_tests(
        'system_selection_3_svm_kernel',
        'kaggle',
        [
            svm_rbf,
            svm_p1,
            svm_p2,
            svm_p3,
            svm_p5
        ], 
        n_iterations=5, n_images=10000, training_split=0.5
    )

def svm_gamma():
    svm_auto = svm.SVM(
        label='SVM with auto-selected gamma',
        preprocessing=hsv_saturation_threshold,
        features=[haralick],
        kernel='poly',
        degree=3
    )
    svm_thousandth = svm.SVM(
        label='SVM with 0.001 gamma',
        preprocessing=hsv_saturation_threshold,
        features=[haralick],
        kernel='poly',
        degree=3,
        gamma=0.001
    )
    svm_hundredth= svm.SVM(
        label='SVM with 0.01 gamma',
        preprocessing=hsv_saturation_threshold,
        features=[haralick],
        kernel='poly',
        degree=3,
        gamma=0.01
    )
    svm_tenth = svm.SVM(
        label='SVM with 0.1 gamma',
        preprocessing=hsv_saturation_threshold,
        features=[haralick],
        kernel='poly',
        degree=3,
        gamma=0.1
    )
    svm_one = svm.SVM(
        label='SVM with 1 gamma',
        preprocessing=hsv_saturation_threshold,
        features=[haralick],
        kernel='poly',
        degree=3,
        gamma=1.0
    )

    run_tests(
        'system_selection_3_svm_gamma',
        'kaggle',
        [
            svm_auto,
            svm_thousandth,
            svm_hundredth,
            svm_tenth,
            svm_one
        ], 
        n_iterations=5, n_images=10000, training_split=0.5
    )

def rf_forest_size():
    rf_1 = rf.RF(
        label='RF with 1 tree',
        preprocessing=hsv_saturation,
        features=[greyscale_histogram(bins=64)],
        n_estimators=1
    )
    rf_10 = rf.RF(
        label='RF with 10 trees',
        preprocessing=hsv_saturation,
        features=[greyscale_histogram(bins=64)],
        n_estimators=10
    )
    rf_50 = rf.RF(
        label='RF with 50 trees',
        preprocessing=hsv_saturation,
        features=[greyscale_histogram(bins=64)],
        n_estimators=50
    )
    rf_100 = rf.RF(
        label='RF with 100 trees',
        preprocessing=hsv_saturation,
        features=[greyscale_histogram(bins=64)],
        n_estimators=100
    )
    rf_250 = rf.RF(
        label='RF with 250 trees',
        preprocessing=hsv_saturation,
        features=[greyscale_histogram(bins=64)],
        n_estimators=250
    )

    run_tests(
        'system_selection_3_rf_forest_size',
        'kaggle',
        [
            rf_1,
            rf_10,
            rf_50,
            rf_100,
            rf_250
        ], 
        n_iterations=5, n_images=10000, training_split=0.5
    )

def rf_max_depth():
    rf_10 = rf.RF(
        label='RF with max depth 10',
        preprocessing=hsv_saturation,
        features=[greyscale_histogram(bins=64)],
        n_estimators=100,
        max_depth=10
    )
    rf_100 = rf.RF(
        label='RF with max depth 100',
        preprocessing=hsv_saturation,
        features=[greyscale_histogram(bins=64)],
        n_estimators=100,
        max_depth=100
    )
    rf_250 = rf.RF(
        label='RF with max depth 250',
        preprocessing=hsv_saturation,
        features=[greyscale_histogram(bins=64)],
        n_estimators=100,
        max_depth=250
    )
    rf_1000 = rf.RF(
        label='RF with max depth 1000',
        preprocessing=hsv_saturation,
        features=[greyscale_histogram(bins=64)],
        n_estimators=100,
        max_depth=1000
    )
    rf_none = rf.RF(
        label='RF with no max depth',
        preprocessing=hsv_saturation,
        features=[greyscale_histogram(bins=64)],
        n_estimators=100,
        max_depth=None
    )

    run_tests(
        'system_selection_3_rf_max_depth',
        'kaggle',
        [
            rf_10,
            rf_100,
            rf_250,
            rf_1000,
            rf_none
        ], 
        n_iterations=5, n_images=10000, training_split=0.5
    )

if __name__ == '__main__':
    # svm_kernel()
    # rf_forest_size()
    svm_gamma()
    rf_max_depth()
