#!/usr/bin/env python3
"""
Predictions from a trained tree-based classifier.
"""


def generate_predictions(clf, X):
    """
    Predicts the class label of each sample in X.

    Args:
        clf: A trained scikit-learn classifier instance
        X: Feature matrix, a numpy.ndarray or pandas DataFrame

    Returns:
        numpy.ndarray: The predicted class label of each input sample.
    """
    return clf.predict(X)
