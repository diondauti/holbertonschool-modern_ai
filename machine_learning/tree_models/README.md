# Tree Models

Python tree models project for the Holberton School Modern AI curriculum.

## Description

This project builds tree-based classifiers with `scikit-learn`, one
factory function per model. Each task returns an **untrained**
estimator, so the test files control the fitting and the effect of a
single hyperparameter can be read straight off the results.

## Requirements

- Ubuntu 20.04 LTS, Python 3.11
- `numpy`, `scikit-learn`, `matplotlib`, `xgboost`, `lightgbm`
- Code follows `pycodestyle` 2.11.1
- All files are executable and start with `#!/usr/bin/env python3`
- All modules and functions are documented
- Each task may only import the module its specification allows

## Tasks

| File                        | Function                                                                                                                    | Description                                                              |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `0-build.py`                | `build_decision_tree(min_samples_leaf, min_samples_split, random_state)`                                                    | Decision tree classifier scored by Gini impurity, grown to full depth    |
| `1-train.py`                | `train_tree(clf, X, y)`                                                                                                     | Fit a tree-based classifier in place on the given features and labels    |
| `2-draw.py`                 | `draw(clf, feature_names, class_names)`                                                                                     | Print the decision rules of a trained tree as indented text              |
| `3-generate_predictions.py` | `generate_predictions(clf, X)`                                                                                              | Predict the class label of each sample with a trained classifier         |
| `4-evaluate.py`             | `evaluate(true_labels, predicted_labels, class_names)`                                                                      | Build a per-class precision, recall and F1 classification report         |
| `5-pre_pruning.py`          | `prepruning(X, y, clf)`                                                                                                     | Grid-search the best pre-pruning hyperparameters for a decision tree     |
| `6-pruning_path.py`         | `get_pruning_path(clf, X, y)`                                                                                               | Retrieve the cost-complexity pruning path: alphas and leaf impurities    |
| `7-prune_decision_tree.py`  | `prune_and_evaluate_trees(X_train, y_train, X_test, y_test, ccp_alphas, random_state, min_samples_leaf, min_samples_split)` | Train one tree per pruning alpha and score each on train and test        |
| `8-best_ccp_alpha.py`       | `get_best_alpha(clfs, train_scores, test_scores, ccp_alphas)`                                                               | Pick the pruning alpha with the best test accuracy and generalization    |
| `9-random_forest.py`        | `random_forest(n_estimators, random_state)`                                                                                 | Random forest classifier: many trees on bootstrap samples, voting        |
| `10-feature_importance.py`  | `feature_importance(rf)`                                                                                                    | Feature importance scores of a trained forest, with an ascending ranking |
| `11-boosting.py`            | `compare_boosting_classifiers(name, n_estimators, random_state)`                                                            | Boosting classifier from AdaBoost, GradientBoosting, XGBoost or LightGBM |

## Usage

Each task has a matching test file. Run one from this directory:

```bash
./0-main.py
```
