#!/usr/bin/env python3
"""
Decision tree classifier.
"""
from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """
    Creates a decision tree classifier.

    The tree uses the Gini impurity measure to score the quality of a
    split, and sets no maximum depth, so it grows until its leaves are
    pure or another stopping criterion stops it.

    Args:
        min_samples_leaf: Minimum number of samples required to be at
            a leaf node.
        min_samples_split: Minimum number of samples required to split
            an internal node.
        random_state: Seed used by the random number generator for
            reproducibility.

    Returns:
        model: A scikit-learn DecisionTreeClassifier instance.
    """
    model = tree.DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state)

    return model
