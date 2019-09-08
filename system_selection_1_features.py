"""System selection first stage, to determine best feature extraction approaches"""

from testing_framework.test_runner import run_training_and_tests
from preprocessing.feature_extraction import hu_moments, haralick, colour_histogram, greyscale_histogram
from models import rf, svm

def initial_grey_histogram_bins_evaluation():
    """Run evaluation of grey histogram bin sizes"""

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

    run_training_and_tests(
        'system_selection_1_hist_grey',
        'kaggle',
        [
            rf_model_grey2,
            rf_model_grey4,
            rf_model_grey8,
            rf_model_grey16,
            rf_model_grey32,
            rf_model_grey64,
            svm_model_grey2,
            svm_model_grey4,
            svm_model_grey8,
            svm_model_grey16,
            svm_model_grey32,
            svm_model_grey64
        ],
        n_iterations=5, n_images=10000, training_split=0.5
    )


def initial_colour_histogram_bins_evaluation():
    """Run evaluation of colour histogram bin sizes"""

    rf_model_colour2 = rf.RF(
        label='colour 2 bin RF',
        preprocessing=[],
        features=[colour_histogram(bins=2)]
    )

    rf_model_colour4 = rf.RF(
        label='colour 4 bin RF',
        preprocessing=[],
        features=[colour_histogram(bins=4)]
    )

    rf_model_colour8 = rf.RF(
        label='colour 8 bin RF',
        preprocessing=[],
        features=[colour_histogram(bins=8)]
    )

    rf_model_colour16 = rf.RF(
        label='colour 16 bin RF',
        preprocessing=[],
        features=[colour_histogram(bins=16)]
    )

    rf_model_colour32 = rf.RF(
        label='colour 32 bin RF',
        preprocessing=[],
        features=[colour_histogram(bins=32)]
    )

    svm_model_colour2 = svm.SVM(
        label='colour 2 bin SVM',
        preprocessing=[],
        features=[colour_histogram(bins=2)]
    )

    svm_model_colour4 = svm.SVM(
        label='colour 4 bin SVM',
        preprocessing=[],
        features=[colour_histogram(bins=4)]
    )

    svm_model_colour8 = svm.SVM(
        label='colour 8 bin SVM',
        preprocessing=[],
        features=[colour_histogram(bins=8)]
    )

    svm_model_colour16 = svm.SVM(
        label='colour 16 bin SVM',
        preprocessing=[],
        features=[colour_histogram(bins=16)]
    )

    run_training_and_tests(
        'system_selection_1_hist_colour',
        'kaggle',
        [
            rf_model_colour2,
            rf_model_colour4,
            rf_model_colour8,
            rf_model_colour16,
            rf_model_colour32,
            svm_model_colour2,
            svm_model_colour4,
            svm_model_colour8,
            svm_model_colour16
        ],
        n_iterations=5, n_images=10000, training_split=0.5
    )


def hist_hu_moments_haralick_evaluation():
    """Run comparative evaluation of hu moments, haralick texture
    attributes, colour histograms and greyscale histograms
    """

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

    rf_model_grey_hist = rf.RF(
        label='RF with grey hist',
        preprocessing=[],
        features=[greyscale_histogram(bins=64)]
    )

    rf_model_colour_hist = rf.RF(
        label='RF with colour hist',
        preprocessing=[],
        features=[colour_histogram(bins=16)]
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

    svm_model_grey_hist = svm.SVM(
        label='SVM with grey hist',
        preprocessing=[],
        features=[greyscale_histogram(bins=64)]
    )

    svm_model_colour_hist = svm.SVM(
        label='SVM with colour hist',
        preprocessing=[],
        features=[colour_histogram(bins=16)]
    )

    run_training_and_tests(
        'system_selection_1_hist_hu_haralick',
        'kaggle',
        [
            rf_model_haralick,
            rf_model_hu,
            rf_model_grey_hist,
            rf_model_colour_hist,
            svm_model_haralick,
            svm_model_hu,
            svm_model_grey_hist,
            svm_model_colour_hist
        ],
        n_iterations=5, n_images=10000, training_split=0.5)


# Run evaluations
if __name__ == '__main__':
    initial_colour_histogram_bins_evaluation()
    initial_grey_histogram_bins_evaluation()
    hist_hu_moments_haralick_evaluation()
