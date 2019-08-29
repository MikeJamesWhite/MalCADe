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
                 random_state=42, C=1, degree=3, gamma='scale', shrinking=True,
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

if __name__ == '__main__':
    print('Running assertion tests...')

    poly_model = SVM(
        label='Test poly model',
        kernel='poly',
        degree=5,
        gamma=100
    )
    assert(poly_model.classifier.gamma == 100)
    assert(poly_model.classifier.C == 300)
    assert(poly_model.classifier.kernel == 'poly')
    assert(poly_model.classifier.degree == 5)

    rbf_model = SVM(
        label='Test rbf model',
        kernel='rbf',
        gamma=1
    )
    assert(rbf_model.classifier.gamma == 1)
    assert(rbf_model.classifier.C == 500)
    assert(rbf_model.classifier.kernel == 'rbf')

    print('All tests successful!')