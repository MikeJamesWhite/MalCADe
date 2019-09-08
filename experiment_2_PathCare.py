"""Experiment to test performance of best models on unseen PathCare data"""

from testing_framework.test_runner import run_tests
from preprocessing.feature_extraction import hu_moments, haralick, colour_histogram, greyscale_histogram
from preprocessing.image_filters import contrast, isolate_saturation, hsv_model, threshold
from models import rf, svm
from joblib import load

# Pre-defined filter combinations
hsv = [hsv_model]
hsv_saturation = [hsv_model, isolate_saturation]
hsv_saturation_threshold = [hsv_model, isolate_saturation, lambda x: contrast(x, 2.0, 10), lambda x: threshold(x, thresh=200)]
hsv_saturation_contrast = [hsv_model, isolate_saturation, lambda x: contrast(x, 2.5, 10)]

def run_experiment():
    """Run experiment two, testing performance on PathCare data"""

    optimal_svm = load("./optimal_models/svm.joblib")
    optimal_rf = load("./optimal_models/rf.joblib")

    run_tests(
        'experiment_2_PathCare_performance',
        'pathcare',
        [
            optimal_rf,
            optimal_svm
        ]
    )

if __name__ == '__main__':
    run_experiment()
