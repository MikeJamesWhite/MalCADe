import sys
import os
import time
import copy
import dill
import numpy as np
from sklearn import metrics
from joblib import dump
from .metrics import Metrics
import preprocessing.image_utils as image_utils

# Combine a list of metrics objects, giving the mean timings, accuracy, precision and recall.
# Also provides total tp, tn, fp, fn for all runs.
def combine_metrics(metrics_list):
    accuracy = 0
    precision = 0
    recall = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    training_time = 0
    test_time = 0

    for m in metrics_list:
        accuracy += m.accuracy
        precision += m.precision
        recall += m.recall
        fp += m.fp
        fn += m.fn
        tp += m.tp
        tn += m.tn
        training_time += m.training_time
        test_time += m.test_time

    return Metrics(pre_calculated_values={
        'accuracy': accuracy/len(metrics_list),
        'precision': precision/len(metrics_list),
        'recall': recall/len(metrics_list),
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'tn': tn,
        'training_time': training_time/len(metrics_list),
        'test_time': test_time/len(metrics_list)
    })


# Training split is only used if no separate test set is specified
def run_tests(test_name, dataset, models, n_images=1000, training_split=0.7,
              test_set=None, n_training_images=None, n_test_images=None,
              n_iterations=1, dimensions=(50, 50)):
    aggregate_metrics = {}

    # Run specified number of iterations
    for i in range(n_iterations):
        print("\nTest iteration", i+1)

        # Handle if specific training and test set size isn't given
        if (n_training_images is None):
            n_training_images = n_images * training_split
        if (n_test_images is None):
            n_test_images = n_images * (1-training_split)

        if (test_set is None):
            # Load training and test sets from single dataset
            train_data, train_labels, test_data, test_labels = image_utils.read_dataset(
                n_training_images,
                n_test_images,
                './datasets/' + dataset,
                dimensions[0],
                dimensions[1]
            )
        else:
            # Load train set from specified dataset
            train_data, train_labels = image_utils.read_dataset(
                n_training_images, 0, './datasets/' + dataset, dimensions[0], dimensions[1]
            )

            # Load test set from specified dataset
            test_data, test_labels = image_utils.read_dataset(
                n_test_images, 0, './datasets/' + test_set, dimensions[0], dimensions[1]
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
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
        print("Saving results to '" + filepath + "aggregate_results.txt'" + "\n---\n")
        with open(filepath + "aggregate_results.txt", 'w') as file:
            file.write(str(aggregate))

# If module is run directly, run assertion tests
if __name__ == '__main__':
    import math
    print('Running tests...')

    metrics1 = Metrics(['Infected', 'Infected', 'Infected'], ['Uninfected', 'Infected', 'Infected'])
    metrics2 = Metrics(['Uninfected', 'Uninfected'], ['Uninfected', 'Infected'])
    metrics3 = combine_metrics([metrics1, metrics2])

    assert (metrics3.tp == 2)
    assert (metrics3.tn == 1)
    assert (metrics3.fp == 1)
    assert (metrics3.fn == 1)
    assert (math.isclose(metrics3.recall, (metrics1.recall + metrics2.recall)/2))
    assert (math.isclose(metrics3.precision, (metrics1.precision + metrics2.precision)/2))

    print('Passed all tests!')
