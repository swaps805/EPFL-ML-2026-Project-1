# -*- coding: utf-8 -*-
"""Model evaluation: stratified k-fold cross-validation and classification metrics.

Every model is evaluated the same way, so that the scores of different
preprocessing choices and methods can be compared (baselines, ablation study):

1. The training set is split into k folds that keep the positive rate (8.8%)
   of the whole dataset (stratification).
2. For each fold, the model is trained on the k - 1 other folds and scores the
   held-out fold. Each sample is therefore scored exactly once, by a model that
   never saw it ("out-of-fold" scores).
3. The metrics (F1, precision, recall, accuracy) are computed on these scores,
   for a given decision threshold or for the threshold that maximizes F1.

The model is given as a function ``fit_predict(x_train, y_train, x_val)`` that
returns one real-valued score per row of ``x_val`` (higher = more likely to be
positive). Any preprocessing statistic (mean, median, standard deviation, ...)
must be computed inside this function on ``x_train`` only, otherwise
information from the held-out fold leaks into training and the scores are
too optimistic.

Running the module as a script evaluates two trivial baselines and a ridge
regression sanity check:

    python cross_validation.py            # 5 folds on the full training set
    python cross_validation.py --k 10     # 10 folds
    python cross_validation.py --sub-sample
"""

import argparse

import numpy as np

DEFAULT_K = 5
DEFAULT_SEED = 1


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def stratified_k_fold(y, k=DEFAULT_K, seed=DEFAULT_SEED):
    """Split the sample indices into k folds with the same positive rate.

    The positive and the negative samples are shuffled and split into k parts
    separately, and fold i gathers part i of each class.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        k: int, the number of folds.
        seed: int, the seed of the random shuffling (for reproducibility).

    Returns:
        list of k tuples (train_indices, val_indices) of numpy int arrays.
    """
    rng = np.random.default_rng(seed)
    parts_per_class = [
        np.array_split(rng.permutation(np.flatnonzero(y == label)), k)
        for label in (0, 1)
    ]
    folds = []
    for i in range(k):
        val_indices = np.sort(np.concatenate([parts[i] for parts in parts_per_class]))
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_indices] = False
        folds.append((np.flatnonzero(train_mask), val_indices))
    return folds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def classification_metrics(y_true, y_pred):
    """Compute the classification metrics of binary predictions.

    Args:
        y_true: numpy array of shape (N,), true labels in {0, 1}.
        y_pred: numpy array of shape (N,), predicted labels in {0, 1}.

    Returns:
        dict with tp, fp, fn, tn (counts), accuracy, precision, recall and f1.
        Precision (resp. F1) is 0 when no sample is predicted positive.
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    # F1 = harmonic mean of precision and recall = 2 TP / (2 TP + FP + FN)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def best_f1_threshold(y_true, scores):
    """Find the decision threshold that maximizes the F1 score.

    All the thresholds are tested at once: once the samples are sorted by
    decreasing score, predicting the first m samples as positive gives
    TP = (number of positives among them) and FP = m - TP, and
    F1 = 2 TP / (TP + FP + P) where P is the total number of positives.

    Args:
        y_true: numpy array of shape (N,), true labels in {0, 1}.
        scores: numpy array of shape (N,), real-valued scores.

    Returns:
        threshold: float, predict positive when ``score >= threshold``.
        f1: float, the F1 score obtained with this threshold.
    """
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_y = scores[order], y_true[order]
    tp = np.cumsum(sorted_y)
    fp = np.arange(1, len(sorted_y) + 1) - tp
    f1 = 2 * tp / (tp + fp + sorted_y.sum())
    # a threshold can only separate samples with different scores: keep the
    # last position of each group of tied scores
    is_cut = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    best = np.flatnonzero(is_cut)[np.argmax(f1[is_cut])]
    return float(sorted_scores[best]), float(f1[best])


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate(
    fit_predict, x, y, k=DEFAULT_K, seed=DEFAULT_SEED, threshold=None, verbose=True
):
    """Evaluate a model with stratified k-fold cross-validation.

    Args:
        fit_predict: function (x_train, y_train, x_val) -> scores of shape
            (len(x_val),). It trains the model (including any preprocessing
            fitted on x_train only) and scores x_val.
        x: numpy array of shape (N, D), the features.
        y: numpy array of shape (N,), labels in {0, 1}.
        k: int, the number of folds.
        seed: int, the seed of the fold split.
        threshold: float or None. If given, samples with ``score >= threshold``
            are predicted positive. If None, the threshold maximizing F1 on the
            out-of-fold scores is used (a single parameter chosen on 300k
            samples, so the optimistic bias is negligible; the same rule is
            applied to every model, so comparisons remain fair).
        verbose: bool, if True print the F1 of each fold.

    Returns:
        dict with
            threshold: the decision threshold used,
            metrics: metrics on all out-of-fold predictions,
            fold_f1: numpy array of shape (k,), the F1 score of each fold,
            f1_mean, f1_std: mean and standard deviation of fold_f1,
            oof_scores: numpy array of shape (N,), the out-of-fold scores.
    """
    folds = stratified_k_fold(y, k, seed)
    oof_scores = np.empty(len(y))
    for i, (train_idx, val_idx) in enumerate(folds):
        oof_scores[val_idx] = fit_predict(x[train_idx], y[train_idx], x[val_idx])
        if verbose:
            print(f"  fold {i + 1}/{k} done")

    if threshold is None:
        threshold, _ = best_f1_threshold(y, oof_scores)

    y_pred = (oof_scores >= threshold).astype(int)
    fold_f1 = np.array(
        [classification_metrics(y[val], y_pred[val])["f1"] for _, val in folds]
    )
    return {
        "threshold": threshold,
        "metrics": classification_metrics(y, y_pred),
        "fold_f1": fold_f1,
        "f1_mean": fold_f1.mean(),
        "f1_std": fold_f1.std(),
        "oof_scores": oof_scores,
    }


def format_result(name, result):
    """One line summarizing a cross-validation result."""
    m = result["metrics"]
    return (
        f"{name:<28} F1 {result['f1_mean']:.3f} ± {result['f1_std']:.3f}  "
        f"precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
        f"accuracy {m['accuracy']:.3f}  threshold {result['threshold']:.4g}"
    )


# ---------------------------------------------------------------------------
# Sanity check: trivial baselines and ridge regression
# ---------------------------------------------------------------------------


def always_positive(x_train, y_train, x_val):
    """Trivial baseline: the same score for everyone (all predicted positive)."""
    return np.ones(len(x_val))


def random_scores(x_train, y_train, x_val, seed=DEFAULT_SEED):
    """Trivial baseline: random scores, unrelated to the features."""
    return np.random.default_rng(seed).random(len(x_val))


def ridge_naive(x_train, y_train, x_val, lambda_=1e-6):
    """Ridge regression on minimally processed features (sanity check only).

    Missing values are replaced by the training mean and every feature is
    standardized with training statistics. Special codes, categorical
    features, etc. are NOT handled: this is not the project preprocessing,
    only a check that the evaluation pipeline gives sensible numbers.

    Plain least squares fails here (singular matrix) because of constant and
    duplicated features (e.g. SEQNO = _PSU); a tiny ridge penalty makes the
    system always solvable.
    """
    from implementations import ridge_regression

    mean = np.nanmean(x_train, axis=0)
    mean = np.where(np.isnan(mean), 0, mean)  # features missing on the whole fold
    x_train = np.where(np.isnan(x_train), mean, x_train)
    x_val = np.where(np.isnan(x_val), mean, x_val)
    std = x_train.std(axis=0)
    std[std == 0] = 1  # constant features
    x_train, x_val = (x_train - mean) / std, (x_val - mean) / std

    tx_train = np.c_[np.ones(len(x_train)), x_train]
    tx_val = np.c_[np.ones(len(x_val)), x_val]
    w, _ = ridge_regression(y_train, tx_train, lambda_)
    return tx_val.dot(w)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data", help="folder with the csv files")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="number of folds")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="fold seed")
    parser.add_argument(
        "--sub-sample", action="store_true", help="use 1/50 of the training rows"
    )
    args = parser.parse_args()

    from data_exploration import load_data

    data = load_data(args.data, sub_sample=args.sub_sample)
    x, y = data["x_train"], data["y_train"]
    print(f"{len(y)} samples, positive rate {y.mean():.2%}, {args.k} folds\n")

    models = [
        ("always positive", always_positive),
        ("random scores", random_scores),
        ("ridge (naive prep.)", ridge_naive),
    ]
    lines = []
    for name, fit_predict in models:
        print(name)
        result = cross_validate(fit_predict, x, y, args.k, args.seed)
        lines.append(format_result(name, result))

    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
