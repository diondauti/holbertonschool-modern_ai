# Tree Models

Python tree models project for the Holberton School Modern AI curriculum.

## Description

This project builds tree-based classifiers with `scikit-learn`, one
factory function per model. Each task returns an **untrained**
estimator, so the test files control the fitting and the effect of a
single hyperparameter can be read straight off the results.

## Requirements

- Ubuntu 20.04 LTS, Python 3.11
- `numpy`, `scikit-learn`, `matplotlib`
- Code follows `pycodestyle` 2.11.1
- All files are executable and start with `#!/usr/bin/env python3`
- All modules and functions are documented
- Each task may only import the module its specification allows

## Tasks

| File                        | Function                                                                 | Description                                                           |
| --------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| `0-build.py`                | `build_decision_tree(min_samples_leaf, min_samples_split, random_state)` | Decision tree classifier scored by Gini impurity, grown to full depth |
| `1-train.py`                | `train_tree(clf, X, y)`                                                  | Fit a tree-based classifier in place on the given features and labels |
| `2-draw.py`                 | `draw(clf, feature_names, class_names)`                                  | Print the decision rules of a trained tree as indented text           |
| `3-generate_predictions.py` | `generate_predictions(clf, X)`                                           | Predict the class label of each sample with a trained classifier      |
| `4-evaluate.py`             | `evaluate(true_labels, predicted_labels, class_names)`                   | Build a per-class precision, recall and F1 classification report      |

## Usage

Each task has a matching test file. Run one from this directory:

```bash
./0-main.py
```
