# MalCADe
This is a UCT honours project, investigating the use of computer vision and supervised learning techniques for computer aided detection of malaria in blood cell images. All code is written in Python, version 3.6.7, with the use of various libraries, most notably scikit-learn for ML models, OpenCV for image handling and computer vision techniques and joblib for saving and loading of models.

## Software Outline
Various system selection and experiment scripts are provided in the root directory, which serve as the entry point to the software. These can be run using Python 3, assuming all dependencies are installed. (e.g. 'python3 experiment_1_NLM.py')

### Models
Code for the ML systems are contained in the 'models' subpackage. An abstract model class is implemented when concrete models are developed. RF and SVM models can be imported and used in experimentation scripts, such as those in the root directory, or run directly to perform assertion testing.

### Pre-processing
All pre-processing code is stored in the 'preprocessing' subpackage. This includes image utility functions, image filters, and feature extraction functions. Running 'image_utils.py' directly will perform assertion testing on functions for loading of images and datasets. Running 'image_filters.py' provides a CLI to allow users to select an image in the 'images' folder, which will then be displayed, along with all the filtered versions which are experimented with. No assertion tests are run, as this code only provides a wrapper for OpenCV functions. Running 'feature_extraction.py' has no effect, as there is no assertion testing (this code is also only a wrapper for imported library functions).

### Testing framework
Code for the testing framework is stored in the 'testing_framework' subpackage. This includes a test runner script and a metrics class. Running 'test_runner.py' directly has no effect, as this simply provides functions to be imported and used in experimentation scripts such as those in the root directory. These functions handle the iterative data loading, training and testing of models or, in cases where the model is pre-trained, just the testing. The functions also dump the trained models and metric data to file. Running 'metrics.py' directly performs assertion testing which ensures that the metrics class works as expected, and that the 'combine_metrics' function correctly aggregates a list of metrics, which is important to get a final result from iterative tests.

## Contributions
The project was developed by Mike White.