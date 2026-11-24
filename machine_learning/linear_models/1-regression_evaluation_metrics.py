#!/usr/bin/env python3
"""
Evaluation metrics for regression models.
"""
from sklearn import metrics
import numpy as np


def evaluation_metrics_for_regression(y_true, y_pred):
    """
    Computes common evaluation metrics for a regression task.

    Args:
        y_true: 1D numpy.ndarray of the true target values
        y_pred: 1D numpy.ndarray of the predicted target values

    Returns:
        tuple: (mse, rmse, mae, r2) where
            mse: Mean Squared Error
            rmse: Root Mean Squared Error, in the units of the target
            mae: Mean Absolute Error
            r2: R squared score, the share of the variance explained
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    return mse, rmse, mae, r2
