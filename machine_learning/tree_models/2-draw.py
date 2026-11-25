#!/usr/bin/env python3
"""
Textual view of the decision rules of a trained tree.
"""
from sklearn import tree


def draw(clf, feature_names, class_names):
    """
    Prints the structure of a trained decision tree as readable text.

    Each line is one node: the feature and threshold a split tests, or
    the class a leaf predicts, indented by its depth in the tree.

    Args:
        clf: A trained DecisionTreeClassifier instance
        feature_names: A list of the input feature names
        class_names: A list of the target class names

    Returns:
        None
    """
    print(tree.export_text(clf,
                           feature_names=feature_names,
                           class_names=class_names))
