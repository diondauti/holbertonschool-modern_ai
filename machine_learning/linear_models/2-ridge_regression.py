#!/usr/bin/env python3
"""
Ridge regression model with L2 regularization.
"""
from sklearn import linear_model


def ridge_regression(random_state):
    """
    Creates a Ridge regression model, a linear model whose L2 penalty
    shrinks large coefficients and so stabilizes the fit when features
    are correlated.

    Args:
        random_state: int seed used for reproducibility

    Returns:
        model: An untrained Ridge regression model instance.
    """
    model = linear_model.Ridge(random_state=random_state)

    return model
