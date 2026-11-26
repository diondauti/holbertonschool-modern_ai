#!/usr/bin/env python3
"""
Boosting classifiers from several gradient boosting libraries.
"""
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """
    Creates a boosting classifier of the named algorithm.

    Boosting fits its trees one after another, each correcting the
    errors the ones before it made, so the four algorithms below differ
    mainly in how they choose and grow that next tree.

    Args:
        name: str naming the algorithm, one of
            'adaboost': an AdaBoostClassifier
            'gradientboosting': a GradientBoostingClassifier
            'xgboost': an XGBClassifier
            'lightgbm': an LGBMClassifier
        n_estimators: int number of boosting iterations (trees)
        random_state: int random seed for reproducibility

    Returns:
        An untrained instance of the selected boosting classifier.

    Raises:
        ValueError: If name is not one of the four known algorithms.
    """
    if name == "adaboost":
        return ensemble.AdaBoostClassifier(n_estimators=n_estimators,
                                           random_state=random_state)

    if name == "gradientboosting":
        return ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_state)

    if name == "xgboost":
        return xgb.XGBClassifier(n_estimators=n_estimators,
                                 random_state=random_state)

    if name == "lightgbm":
        return lgb.LGBMClassifier(n_estimators=n_estimators,
                                  random_state=random_state,
                                  verbose=-1)

    raise ValueError(f"Unknown model name '{name}'")
