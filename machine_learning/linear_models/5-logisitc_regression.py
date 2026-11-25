#!/usr/bin/env python3
"""
Logistic regression model for binary classification.
"""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """
    Creates a logistic regression model, which performs binary
    classification by fitting a logistic function to the data.

    Args:
        random_state: int seed used for reproducibility

    Returns:
        model: An untrained LogisticRegression instance.
    """
    model = linear_model.LogisticRegression(random_state=random_state)

    return model
