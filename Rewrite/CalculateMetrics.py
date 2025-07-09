from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def labels_from_preds(y_preds):
    return [np.argmax(y_pred) for y_pred in y_preds]

def calculate_accuracy(y, y_preds):
    return accuracy_score(y, labels_from_preds(y_preds))


def calculate_all_metrics(y, y_preds):

    accuracy = accuracy_score(y, labels_from_preds(y_preds))
    f1 = f1_score(y, labels_from_preds(y_preds))
    mean_absolute_err = None

    return {'accuracy': accuracy, 'f1': f1, 'mean_absolute_error': mean_absolute_err}
