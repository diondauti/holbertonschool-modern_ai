#!/usr/bin/env python3
"""
Ordinary least squares linear regression model.
"""
from sklearn import linear_model


def Linear_Regression():
    """
    Creates an ordinary least squares linear regression model.

    Args:
        None

    Returns:
        model: An untrained LinearRegression instance.
    """
    model = linear_model.LinearRegression()

    return model
