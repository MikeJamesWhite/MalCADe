import sys
import numpy as np
from sklearn import ensemble, metrics
from sklearn.preprocessing import MinMaxScaler
from .model import Model
from joblib import dump, load

# Constants
TRAIN_SIZE = 700
TEST_SIZE = 500
N_BINS = 16


class RF(Model):
    ''' Random Forest model for use with MalSeg testing framework'''

    def __init__(self, label='Unlabeled RF model', preprocessing=[],
                 features=[], n_estimators=10, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features='auto',
                 bootstrap=True, *args, **kwargs):
        print('Initialising model:', label)
        self.classifier = ensemble.RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap
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
        print('Fitting RF model on', len(train_labels), 'images')
        self.classifier.fit(train_data, train_labels)

    def run(self, images):
        if len(self.preprocessing) > 0:
            # Apply preprocessing
            print('Applying', len(self.preprocessing), 'filter(s) to input images')
            for filter in self.preprocessing:
                for i in range(len(images)):
                    images[i] = filter(images[i])

        # Apply feature extraction
        if len(self.features) > 0:
            print('Extracting', len(self.features), 'feature(s) from input images')
            scaler = MinMaxScaler(feature_range=(0, 1))
            for i in range(len(images)):
                features = []
                for feature in self.features:
                    features.append(feature(images[i]))
                images[i] = np.hstack(features)
            images = scaler.fit_transform(images)
        else:
            # Flatten images (not necessary when using feature extraction)
            train_data = np.array(train_data).reshape((len(train_data), -1))

        # Run predictions
        print('Predicting presence of parasites in', len(images), 'images\n')
        return self.classifier.predict(images)

    def save(self, path, name):
        pass

if __name__ == '__main__':
    import preprocessing.image_utils as image_utils
    import preprocessing.image_filters as img_filters
    from preprocessing.feature_extraction import hsv_histogram, greyscale_histogram, haralick, hu_moments

    train_data, train_labels, test_data, test_labels = image_utils.read_dataset(TRAIN_SIZE, TEST_SIZE, './datasets/kaggle')

    # Build and train RF model
    rf_model = RF(
        label='Comtrast-edge RF',
        preprocessing=[img_filters.contrast, img_filters.edge],
        features=[hu_moments, haralick, lambda x: hsv_histogram(bins=16)]
    )
    rf_model.train(train_data, train_labels)

    # Run predictions on test set
    expected = test_labels
    predicted = rf_model.run(test_data)

    # Print results
    print("Classification report:\n%s\n" % (metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
