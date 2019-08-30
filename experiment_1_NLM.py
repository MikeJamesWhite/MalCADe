# Final evaluation to test performance of best models on unseen pathcare data
# Author: Michael White (mike.james.white@icloud.com)

from testing_framework.test_runner import run_training_and_tests
from preprocessing.feature_extraction import hu_moments, haralick, colour_histogram, greyscale_histogram
from preprocessing.image_filters import contrast, isolate_saturation, hsv_model, threshold
from models import rf, svm

# Pre-defined filter combinations
hsv = [hsv_model]
hsv_saturation = [hsv_model, isolate_saturation]
hsv_saturation_threshold = [hsv_model, isolate_saturation, lambda x: contrast(x, 2.0, 10), lambda x: threshold(x, thresh=200)]
hsv_saturation_contrast = [hsv_model, isolate_saturation, lambda x: contrast(x, 2.5, 10)]

def run_experiment():
    optimal_svm = svm.SVM(
        label='Optimised SVM model',
        preprocessing=hsv_saturation_threshold,
        features=[haralick],
        kernel='poly',
        degree=3
    )

    optimal_rf = rf.RF(
        label='Optimised RF model',
        preprocessing=hsv_saturation,
        features=[greyscale_histogram(bins=64)],
        n_estimators=100,
        max_depth=100
    )

    run_training_and_tests(
        'experiment_1_NLM_performance',
        'kaggle',
        [
            optimal_rf,
            optimal_svm
        ], 
        n_iterations=10,
        n_training_images=5000,
        n_test_images=20000
    )

if __name__ == '__main__':
    run_experiment()
