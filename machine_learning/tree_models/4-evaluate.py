#!/usr/bin/env python3
"""
Classification report for a trained classifier's predictions.
"""
from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """
    Builds a classification report summarizing performance per class.

    The report gives the precision, recall and F1-score of each class,
    together with the number of true samples it has, and the overall
    accuracy and averages.

    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
        class_names: List of class names matching the label indices

    Returns:
        str: The classification report.
    """
    return metrics.classification_report(true_labels,
                                         predicted_labels,
                                         target_names=class_names)
