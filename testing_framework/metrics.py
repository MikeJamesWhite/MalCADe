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
            self.fp = confusion_matrix[0][1]
            self.fn = confusion_matrix[1][0]
            self.tp = confusion_matrix[1][1]
            self.tn = confusion_matrix[0][0]
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
