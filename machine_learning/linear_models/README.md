# Linear Models

Python linear models project for the Holberton School Modern AI curriculum.

## Description

This project covers the linear family of supervised learning models with
`scikit-learn`: fitting an ordinary least squares regression, measuring
regression error, controlling multicollinearity and overfitting with the
Ridge and Lasso penalties, explaining a model's predictions with SHAP
values, and moving from regression to classification with logistic
regression and support vector machines.

## Requirements

- Ubuntu 20.04 LTS, Python 3.11
- numpy, scikit-learn, shap
- Code follows `pycodestyle` 2.11.1
- All files are executable and start with `#!/usr/bin/env python3`
- All modules, classes, and functions are documented
- Each task may only import the module its specification allows

## Tasks

| File                                 | Description                                                         |
| ------------------------------------ | ------------------------------------------------------------------- |
| `0-linear_regression.py`             | Create an untrained ordinary least squares `LinearRegression` model |
| `1-regression_evaluation_metrics.py` | Compute MSE, RMSE, MAE and R² for a regression model's predictions  |
| `2-ridge_regression.py`              | Create a Ridge regression model, whose L2 penalty resists collinearity |
| `3-Lasso_regression.py`              | Create a Lasso regression model, whose L1 penalty selects features  |
| `4-shap.py`                          | Explain a model's predictions with SHAP values                      |
| `5-logisitc_regression.py`           | Create a logistic regression model for classification               |
| `6-svm.py`                           | Create a support vector machine model                               |
