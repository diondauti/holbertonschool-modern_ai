#!/usr/bin/env python3
"""
SHAP explainer and SHAP values for a trained model.
"""
import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """
    Builds a SHAP explainer for a trained model and explains predictions.

    X_train serves as the background dataset, the reference the
    explainer compares each prediction against, and the SHAP values
    are then computed for the samples in X_test.

    Args:
        model: A trained regression model
        X_train: Input data used to initialize the explainer
        X_test: Input data to explain

    Returns:
        tuple: (explainer, shap_values) where
            explainer: the SHAP explainer object
            shap_values: SHAP values for the predictions on X_test
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    return explainer, shap_values
