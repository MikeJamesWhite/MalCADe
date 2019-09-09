"""Test runner module of the MalCADe evaulation framework"""

import sys
import os
import time
import copy
import dill
import numpy as np
from sklearn import metrics
from joblib import dump
from .metrics import Metrics, combine_metrics
import preprocessing.image_utils as image_utils

def run_training_and_tests(test_name, dataset, models, n_images = 1000, training_split = 0.7, 
              n_training_images = None, n_test_images = None, 
              n_iterations = 1, dimensions = (50, 50)):
    """Iteratively train and test models, re-loading data each iteration"""

    aggregate_metrics = {}

    # Run specified number of iterations
    for i in range(n_iterations):
        print("\nTest iteration", i+1)

        # Handle if specific training and test set size isn't given
        if (n_training_images is None):
            n_training_images = n_images * training_split
        if (n_test_images is None):
            n_test_images = n_images * (1-training_split)

        # Load training and test sets from single dataset
        train_data, train_labels, test_data, test_labels = image_utils.read_dataset(
            n_training_images, 
            n_test_images, 
            './datasets/' + dataset, 
            dimensions[0], 
            dimensions[1]
        )

        # Train and run tests for each model
        for model in models:
            print("Working with model '" + model.label + "'")

            # Train model
            start = time.time()
            model.train(copy.deepcopy(train_data), train_labels)
            end = time.time()
            training_time = round(end - start, 3)

            # Run predictions on test set
            start = time.time()
            predicted = model.run(copy.deepcopy(test_data))
            end = time.time()
            test_time = round(end - start, 3)

            # Calculate metrics and store for aggregate calculations
            metrics = Metrics(test_labels, predicted, training_time, test_time)
            if model.label in aggregate_metrics:
                aggregate_metrics[model.label].append(metrics)
            else:
                aggregate_metrics[model.label] = [metrics]

            # Print results
            print("Results\n" + "------")
            print(str(metrics))

            # Save model
            filepath = "./test/" + test_name + "/" + model.label + "/iteration" + str(i+1) + "/"
            print("Saving model to '" + filepath + model.label + ".joblib'")
            os.makedirs(os.path.dirname(filepath), exist_ok = True)
            with open(filepath + model.label + '.joblib', 'wb') as file:
                dump(model, file)

            # Save results
            print("Saving results to '" + filepath + "results.txt'\n")
            with open(filepath + "results.txt", 'w') as file:
                file.write(str(metrics))

    # Calculate, print and write aggregate metrics
    print(
        'Aggregate Results' + '\n' +
        '-----------------'
    )
    for model in models:
        aggregate = combine_metrics(aggregate_metrics[model.label])
        print(model.label)
        print(aggregate)
        filepath = "./test/" + test_name + "/" + model.label + "/"
        print("Saving results to '" + filepath + "aggregate_results.txt'" + "\n -- -\n")
        with open(filepath + "aggregate_results.txt", 'w') as file:
            file.write(str(aggregate))


def run_tests(test_name, dataset, models, n_test_images = 120, dimensions = (50, 50)):
    """Run a single iteration of tests, without training models. For use with
    pre-trained models, such as those loaded using joblib
    """

    # Load test data
    train_data, train_labels, test_data, test_labels = image_utils.read_dataset(
        0, 
        n_test_images, 
        './datasets/' + dataset, 
        dimensions[0], 
        dimensions[1]
    )

    # Run test for each model
    for model in models:
        print("Working with model '" + model.label + "'")

        # Run predictions on test set
        start = time.time()
        predicted = model.run(copy.deepcopy(test_data))
        end = time.time()
        test_time = round(end - start, 3)

        # Calculate metrics
        metrics = Metrics(test_labels, predicted, 0.0, test_time)

        # Print results
        print("Results\n" + "------")
        print(str(metrics))

        # Save results
        filepath = "./test/" + test_name + "/" + model.label + "/"
        os.makedirs(os.path.dirname(filepath), exist_ok = True)

        print("Saving results to '" + filepath + "results.txt'\n")
        with open(filepath + "results.txt", 'w') as file:
            file.write(str(metrics))

