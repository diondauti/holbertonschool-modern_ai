#!/usr/bin/env python3
"""
Cost-complexity pruning path of a decision tree.
"""


def get_pruning_path(clf, X, y):
    """
    Retrieves the cost-complexity pruning path of a decision tree.

    The path lists the effective alpha at which each branch of the tree
    would be pruned away, paired with the total leaf impurity that
    pruning leaves behind. Post-pruning picks one of these alphas and
    cuts the fully grown tree back to it.

    Args:
        clf: A DecisionTreeClassifier instance
        X: Input features
        y: Target labels

    Returns:
        tuple: (ccp_alphas, impurities) where
            ccp_alphas: numpy.ndarray of the effective alpha values
            impurities: numpy.ndarray of the total leaf impurity at
                each corresponding alpha
    """
    path = clf.cost_complexity_pruning_path(X, y)

    return path.ccp_alphas, path.impurities
