from sklearn import metrics

class Metrics:
    def __init__(self, expected=[], predicted=[], training_time=0, test_time=0, pre_calculated_values=None):
        if not (pre_calculated_values is None):
            self.accuracy = pre_calculated_values['accuracy']
            self.precision = pre_calculated_values['precision']
            self.recall = pre_calculated_values['recall']
            self.fp = pre_calculated_values['fp']
            self.fn = pre_calculated_values['fn']
            self.tp = pre_calculated_values['tp']
            self.tn = pre_calculated_values['tn']
            self.training_time = pre_calculated_values['training_time']
            self.test_time = pre_calculated_values['test_time']
        else:
            self.accuracy = metrics.accuracy_score(expected, predicted)
            self.precision = metrics.precision_score(expected, predicted, pos_label='Infected')
            self.recall = metrics.recall_score(expected, predicted, pos_label='Infected')
            confusion_matrix = metrics.confusion_matrix(expected, predicted)
            self.fp = confusion_matrix[1][0]
            self.fn = confusion_matrix[0][1]
            self.tp = confusion_matrix[0][0]
            self.tn = confusion_matrix[1][1]
            self.training_time = training_time
            self.test_time = test_time

    def __str__(self):
        return ("Accuracy: " + str(self.accuracy) + "\n" +
                "Precision: " + str(self.precision) + "\n" +
                "Recall: " + str(self.recall) + "\n" +
                "TP: " + str(self.tp) + "\n" +
                "FP: " + str(self.fp) + "\n" +
                "TN: " + str(self.tn) + "\n" +
                "FN: " + str(self.fn) + "\n" +
                "Training time: " + str(self.training_time) + "s\n" +
                "Test time: " + str(self.test_time) + "s\n")

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

# If module is run directly, run assertion tests
if __name__ == '__main__':    
    print('Running tests...')
    metric = Metrics(
        ['Infected',
        'Infected',
        'Infected',
        'Uninfected',
        'Uninfected',
        'Infected'],
        ['Infected',
        'Infected',
        'Infected',
        'Uninfected',
        'Uninfected',
        'Uninfected'])

    assert(metric.tp == 3)
    assert(metric.tn == 2)
    assert(metric.fp == 0)
    assert(metric.fn == 1)

    import math
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