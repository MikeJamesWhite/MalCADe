import sys
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from .model import Model

# Constants
TRAIN_SIZE = 1000
TEST_SIZE = 2000
N_BINS = 16


class SVM(Model):
    ''' Support Vector Machine model for use with MalSeg testing framework'''

    def __init__(self, label='Unlabeled SVM model', kernel='rbf',
                 random_state=42, C=1, degree=3, gamma=0.0005, shrinking=True,
                 preprocessing=[], features=[], *args, **kwargs):
        print('Initialising model:', label)
        self.classifier = svm.SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            shrinking=shrinking
        )

        self.label = label
        self.preprocessing = preprocessing
        self.features = features

        return super().__init__(*args, **kwargs)

    def train(self, train_data, train_labels):
        # Apply preprocessing
        if len(self.preprocessing) > 0:
            print('Applying', len(self.preprocessing), 'filter(s) to training data')
            for filter in self.preprocessing:
                for i in range(len(train_data)):
                    train_data[i] = filter(train_data[i])
                    if len(train_data[i].shape) != 3:
                        print("Image has", len(train_data[i].shape), "channels")

        # Apply feature extraction
        if len(self.features) > 0:
            print('Extracting', len(self.features), 'feature(s) from training data')
            scaler = MinMaxScaler(feature_range=(0, 1))
            for i in range(len(train_data)):
                features = []
                for feature in self.features:
                    features.append(feature(train_data[i]))
                train_data[i] = np.hstack(features)
            train_data = scaler.fit_transform(train_data)
        else:
            # Flatten images (not necessary when using feature extraction)
            train_data = np.array(train_data).reshape((len(train_data), -1))

        # Fit model
        print('Fitting SVM model on', len(train_labels), 'images')
        self.classifier.fit(train_data, train_labels)

    def run(self, images):
        # Apply preprocessing
        if len(self.preprocessing) > 0:
            print(
                'Applying',
                len(self.preprocessing),
                'filter(s) to input images'
            )
            for filter in self.preprocessing:
                for i in range(len(images)):
                    images[i] = filter(images[i])

        # Apply feature extraction
        if len(self.features) > 0:
            print(
                'Extracting',
                len(self.features),
                'feature(s) from input images'
            )
            scaler = MinMaxScaler(feature_range=(0, 1))
            for i in range(len(images)):
                features = []
                for feature in self.features:
                    features.append(feature(images[i]))
                images[i] = np.hstack(features)
            images = scaler.fit_transform(images)
        else:
            # Flatten images (not necessary when using feature extraction)
            images = np.array(images).reshape((len(images), -1))

        # Run predictions
        print('Predicting presence of parasites in', len(images), 'images\n')
        return self.classifier.predict(images)

    def save(self, path, name):
        pass

if __name__ == '__main__':
    import preprocessing.image_utils as image_utils
    import preprocessing.image_filters as img_filters
    from preprocessing.feature_extraction import hsv_histogram, greyscale_histogram, haralick, hu_moments

    train_data, train_labels, test_data, test_labels = image_utils.read_dataset(
        TRAIN_SIZE,
        TEST_SIZE,
        './datasets/kaggle'
    )

    # Build and train SVM model
    svm_model = SVM(
        label='Contrast-edge SVM',
        preprocessing=[img_filters.contrast, img_filters.edge],
        features=[hu_moments, haralick, hsv_histogram(bins=16)]  
    )
    svm_model.train(train_data, train_labels)

    # Run predictions on test set
    expected = test_labels
    predicted = svm_model.run(test_data)

    # Print results
    print("Classification report:\n%s\n" % (
        metrics.classification_report(expected, predicted)
    ))
    print("Confusion matrix:\n%s" % (
        metrics.confusion_matrix(expected, predicted)
    ))
