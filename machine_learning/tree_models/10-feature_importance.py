#!/usr/bin/env python3
"""
Feature importances of a trained random forest.
"""
import numpy as np


def feature_importance(rf):
    """
    Computes the feature importances of a trained random forest.

    Each score is how much that feature reduced impurity across the
    trees of the forest, so the scores rank the features by how much
    the model relies on them.

    Args:
        rf: A trained scikit-learn RandomForestClassifier instance

    Returns:
        tuple: (importances, indices) where
            importances: numpy.ndarray of the importance score of each
                feature, in the order the features were given
            indices: numpy.ndarray of the feature indices sorted from
                least to most important
    """
    importances = rf.feature_importances_
    indices = np.argsort(importances)

    return importances, indices
