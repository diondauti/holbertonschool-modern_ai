#!/usr/bin/env python3
"""
Selection of the best cost-complexity pruning value.
"""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """
    Selects the best pruning value among a set of pruned trees.

    The highest test accuracy wins. Trees tied on it are separated by
    the gap between their training and test accuracy, the smallest gap
    generalizing best, and any tree still tied after that is settled by
    the largest alpha, which is the simplest tree of the remaining.

    Args:
        clfs: List of trained DecisionTreeClassifier instances, each
            trained with a different ccp_alpha
        train_scores: List of the training accuracy of each classifier
        test_scores: List of the test accuracy of each classifier
        ccp_alphas: List or array of the ccp_alpha values used

    Returns:
        tuple: (best_alpha, best_clf) where
            best_alpha: the most appropriate ccp_alpha value
            best_clf: the trained classifier trained with it
    """
    best_score = max(test_scores)
    candidates = [i for i, score in enumerate(test_scores)
                  if score == best_score]

    best = min(candidates,
               key=lambda i: (abs(train_scores[i] - test_scores[i]),
                              -ccp_alphas[i]))

    return ccp_alphas[best], clfs[best]
