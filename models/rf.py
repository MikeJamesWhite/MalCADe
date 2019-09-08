import sys
import numpy as np
from sklearn import ensemble, metrics
from sklearn.preprocessing import MinMaxScaler
from .model import Model
from joblib import dump, load

class RF(Model):
    ''' Random Forest model for use with MalSeg testing framework'''

    def __init__(self, label='Unlabeled RF model', preprocessing=[],
                 features=[], n_estimators=10, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features='auto',
                 bootstrap=True, *args, **kwargs):
        """Initialise the RF system"""

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
        """Apply filtering and feature extraction, before training model"""

        # Apply filtering
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
        """Apply filtering and feature extraction, before running model on new data"""

        # Apply filtering
        if len(self.preprocessing) > 0:           
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


# If module is run directly, run assertion tests
if __name__ == '__main__':
    print('Running assertion tests...')

    rf_model_1 = RF(
        label='Test rf model 1',
        n_estimators=10,
        max_depth=10
    )
    assert(rf_model_1.classifier.n_estimators == 10)
    assert(rf_model_1.classifier.max_depth == 10)

    rf_model_2 = RF(
        label='Test rf model 2',
        n_estimators=1000,
        max_depth=None
    )
    assert(rf_model_2.classifier.n_estimators == 1000)
    assert(rf_model_2.classifier.max_depth == None)

    print('All tests successful!')