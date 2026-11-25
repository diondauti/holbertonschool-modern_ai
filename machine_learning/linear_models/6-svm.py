#!/usr/bin/env python3
"""
Support vector machine classifiers with selectable kernels.
"""
from sklearn import svm


def get_SVM_model(name, random_state):
    """
    Creates a support vector machine classifier using the named kernel.

    Args:
        name: str naming the kernel to use, one of
            'linear': a linear kernel
            'poly': a polynomial kernel
            'rbf': a radial basis function kernel
        random_state: int seed used by the random number generator
            for reproducibility

    Returns:
        model: An untrained SVC instance using the named kernel.
    """
    model = svm.SVC(kernel=name, random_state=random_state)

    return model
