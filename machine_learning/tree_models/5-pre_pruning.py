#!/usr/bin/env python3
"""
Pre-pruning of a decision tree by grid search.
"""
from sklearn import model_selection


def prepruning(X, y, clf):
    """
    Searches for the best pre-pruning hyperparameters of a decision
    tree with an exhaustive grid search.

    Pre-pruning stops the tree growing in the first place, rather than
    cutting branches off afterwards, so each combination below caps the
    tree in a different way and is scored by cross-validation.

    Args:
        X: Input features
        y: Target labels
        clf: An untrained DecisionTreeClassifier instance

    Returns:
        dict: The best combination of hyperparameters found.
    """
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": range(2, 5),
        "min_samples_leaf": range(2, 5),
        "min_samples_split": range(2, 5),
    }

    grid_search = model_selection.GridSearchCV(clf, param_grid)
    grid_search.fit(X, y)

    return grid_search.best_params_
