#!/usr/bin/env python3
"""
Lasso regression model with L1 regularization.
"""
from sklearn import linear_model


def lasso_regression(random_state):
    """
    Creates a Lasso regression model, a linear model whose L1 penalty
    drives some coefficients to exactly zero and so selects features
    automatically.

    Args:
        random_state: int seed used for reproducibility

    Returns:
        model: An untrained Lasso regression model instance.
    """
    model = linear_model.Lasso(random_state=random_state)

    return model
