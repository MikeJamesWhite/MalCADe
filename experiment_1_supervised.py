# Initial evaluation to determine best feature extraction approaches
# Author: Michael White (mike.james.white@icloud.com)

from testing_framework.test_runner import run_tests
from preprocessing.feature_extraction import hu_moments, haralick, hsv_histogram, greyscale_histogram
from preprocessing.image_filters import contrast, edge, isolate_green, hsv_model, threshold
from models import rf, svm

def initial_grey_histogram_bins_evaluation():
    rf_model_grey2 = rf.RF(
        label='grey 2 bin RF',
        preprocessing=[],
        features=[greyscale_histogram(bins=2)]
    )

    rf_model_grey4 = rf.RF(
        label='grey 4 bin RF',
        preprocessing=[],
        features=[greyscale_histogram(bins=4)]
    )

    rf_model_grey8 = rf.RF(
        label='grey 8 bin RF',
        preprocessing=[],
        features=[greyscale_histogram(bins=8)]
    )

    rf_model_grey16 = rf.RF(
        label='grey 16 bin RF',
        preprocessing=[],
        features=[greyscale_histogram(bins=16)]
    )

    rf_model_grey32 = rf.RF(
        label='grey 32 bin RF',
        preprocessing=[],
        features=[greyscale_histogram(bins=32)]
    )

    rf_model_grey64 = rf.RF(
        label='grey 64 bin RF',
        preprocessing=[],
        features=[greyscale_histogram(bins=64)]
    )

    svm_model_grey2 = svm.SVM(
        label='grey 2 bin SVM',
        preprocessing=[],
        features=[greyscale_histogram(bins=2)]
    )

    svm_model_grey4 = svm.SVM(
        label='grey 4 bin SVM',
        preprocessing=[],
        features=[greyscale_histogram(bins=4)]
    )


    svm_model_grey8 = svm.SVM(
        label='grey 8 bin SVM',
        preprocessing=[],
        features=[greyscale_histogram(bins=8)]
    )

    svm_model_grey16 = svm.SVM(
        label='grey 16 bin SVM',
        preprocessing=[],
        features=[greyscale_histogram(bins=16)]
    )

    svm_model_grey32 = svm.SVM(
        label='grey 32 bin SVM',
        preprocessing=[],
        features=[greyscale_histogram(bins=32)]
    )

    svm_model_grey64 = svm.SVM(
        label='grey 64 bin SVM',
        preprocessing=[],
        features=[greyscale_histogram(bins=64)]
    )

    run_tests('Histogram Bin Evaluations', 'kaggle', [rf_model_grey2, rf_model_grey4, rf_model_grey8, rf_model_grey16, rf_model_grey32, rf_model_grey64, svm_model_grey2, svm_model_grey4, svm_model_grey8, svm_model_grey16, svm_model_grey32, svm_model_grey64], n_iterations=5, n_images=10000, training_split=0.5)


def initial_hsv_histogram_bins_evaluation():
    rf_model_hsv2 = rf.RF(
        label='hsv 2 bin RF',
        preprocessing=[],
        features=[hsv_histogram(bins=2)]
    )

    rf_model_hsv4 = rf.RF(
        label='hsv 4 bin RF',
        preprocessing=[],
        features=[hsv_histogram(bins=4)]
    )

    rf_model_hsv8 = rf.RF(
        label='hsv 8 bin RF',
        preprocessing=[],
        features=[hsv_histogram(bins=8)]
    )

    rf_model_hsv16 = rf.RF(
        label='hsv 16 bin RF',
        preprocessing=[],
        features=[hsv_histogram(bins=16)]
    )

    rf_model_hsv32 = rf.RF(
        label='hsv 32 bin RF',
        preprocessing=[],
        features=[hsv_histogram(bins=32)]
    )

    svm_model_hsv2 = svm.SVM(
        label='hsv 2 bin SVM',
        preprocessing=[],
        features=[hsv_histogram(bins=2)]
    )

    svm_model_hsv4 = svm.SVM(
        label='hsv 4 bin SVM',
        preprocessing=[],
        features=[hsv_histogram(bins=4)]
    )

    svm_model_hsv8 = svm.SVM(
        label='hsv 8 bin SVM',
        preprocessing=[],
        features=[hsv_histogram(bins=8)]
    )

    svm_model_hsv16 = svm.SVM(
        label='hsv 16 bin SVM',
        preprocessing=[],
        features=[hsv_histogram(bins=16)]
    )

    run_tests(
        'HSV Histogram Bin Evaluations',
        'kaggle',
        [
            rf_model_hsv2,
            rf_model_hsv4,
            rf_model_hsv8,
            rf_model_hsv16,
            rf_model_hsv32,
            svm_model_hsv2,
            svm_model_hsv4,
            svm_model_hsv8,
            svm_model_hsv16
        ],
        n_iterations=5, n_images=10000, training_split=0.5
    )


def hu_moments_haralick_evaluation():
    rf_model_haralick = rf.RF(
        label='RF with haralick',
        preprocessing=[],
        features=[haralick]
    )

    rf_model_hu = rf.RF(
        label='RF with hu moments',
        preprocessing=[],
        features=[hu_moments]
    )

    rf_model_haralick_hu = rf.RF(
        label='RF with haralick and hu moments',
        preprocessing=[],
        features=[haralick, hu_moments]
    )

    svm_model_haralick = svm.SVM(
        label='SVM with haralick',
        preprocessing=[],
        features=[haralick]
    )

    svm_model_hu = svm.SVM(
        label='SVM with hu moments',
        preprocessing=[],
        features=[hu_moments]
    )

    svm_model_haralick_hu = svm.SVM(
        label='SVM with haralick and hu moments',
        preprocessing=[],
        features=[haralick, hu_moments]
    )

    run_tests(
        'Haralick + Hu Moment Initial Evaluation',
        'kaggle',
        [
            rf_model_haralick,
            rf_model_hu,
            rf_model_haralick_hu
        ],
        n_iterations=5, n_images=10000, training_split=0.5)

if __name__ == '__main__':
    # initial_hsv_histogram_bins_evaluation()
    # initial_grey_histogram_bins_evaluation()
    hu_moments_haralick_evaluation()
