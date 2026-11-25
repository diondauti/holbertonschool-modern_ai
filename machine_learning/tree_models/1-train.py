#!/usr/bin/env python3
"""
Training of a tree-based classifier.
"""


def train_tree(clf, X, y):
    """
    Trains a tree-based classifier on the given data.

    The classifier is fitted in place, so the caller keeps using the
    same instance it passed in.

    Args:
        clf: A scikit-learn classifier instance
        X: Input features
        y: Target labels

    Returns:
        None
    """
    clf.fit(X, y)
