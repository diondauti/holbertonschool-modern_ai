#!/usr/bin/env python3
"""
Random forest classifier.
"""
from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """
    Creates a random forest classifier.

    A random forest fits many decision trees, each on a bootstrap
    sample of the data and a random subset of the features, and has
    them vote. Averaging over the trees cancels out the overfitting a
    single unpruned tree suffers from.

    Args:
        n_estimators: Number of trees in the forest.
        random_state: Seed used by the random number generator for
            reproducibility.

    Returns:
        model: A scikit-learn RandomForestClassifier instance.
    """
    model = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                            random_state=random_state)

    return model
