import numpy as np
import sklearn.metrics as skm
import torch

import sys
sys.path.append('../metal')
from metal.utils import arraylike_to_numpy, pred_to_prob


def accuracy_score(gold, pred, ignore_in_gold=[], ignore_in_pred=[]):
    """
    Calculate (micro) accuracy.
    Args:
        gold: A 1d array-like of gold labels
        pred: A 1d array-like of predicted labels (assuming abstain = 0)
        ignore_in_gold: A list of labels for which elements having that gold
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that pred
            label will be ignored.

    Returns:
        A float, the (micro) accuracy score
    """
    gold, pred = _preprocess(gold, pred, ignore_in_gold, ignore_in_pred)

    if len(gold) and len(pred):
        acc = np.sum(gold == pred) / len(gold)
    else:
        acc = 0

    return acc


def ndcg_score(gold, probs, ignore_in_gold=[], ignore_in_pred=[]):
    gold = arraylike_to_numpy(gold)
    gold = [1 if y == 1 else 0 for y in gold]
    return ndcg(gold, probs[...,0]) 
   
def dcg(y_true, y_score, k=None):
    """
    Function for Discounted Cumulative Gain
    """
    k = len(y_true) if k is None else k
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def ndcg(y_true, y_score, k=None):
    """
    Function for Normalized Discounted Cumulative Gain
    """
    y_true, y_score = np.squeeze(y_true), np.squeeze(y_score)
    k = len(y_true) if k is None else k

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different shapes.")

    IDCG    = dcg(y_true, y_true)
    DCG     = dcg(y_true, y_score)

    return DCG/IDCG

def _preprocess(gold, pred, ignore_in_gold, ignore_in_pred):
    gold = arraylike_to_numpy(gold)
    pred = arraylike_to_numpy(pred)
    if ignore_in_gold or ignore_in_pred:
        gold, pred = _drop_ignored(gold, pred, ignore_in_gold, ignore_in_pred)
    return gold, pred

if __name__ == "__main__":
    import ipdb; ipdb.set_trace()

