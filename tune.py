# -*- coding: utf-8 -*-
"""Step-size, iteration and regularization sweep for the iterative methods.

Uses the same protocol as ``run.py --evaluate-only`` (default preprocessing,
stratified 5-fold cross-validation, seed 1, F1-maximizing threshold), but
preprocesses each fold once and trains every setting of a group on it.

For each setting, the weights are evaluated at several iteration counts
("checkpoints") by continuing gradient descent from the previous weights. This
is exactly equivalent to separate runs with ``--max-iters`` set to each
checkpoint; for SGD the seeded random stream also continues, as in
``run.train_model``.

The step-size ranges follow from the data: the largest eigenvalue of
tx^T tx / N is about 42, so full-batch GD on the MSE needs gamma < 2/42 and
logistic regression gamma < 2/(42/4). For single-sample SGD the bound is
gamma < 2/||x_n||^2 for every row; the largest ||x_n||^2 is about 330000, so
gamma above about 6e-6 eventually diverges.

    python tune.py                      # all groups (several hours)
    python tune.py --groups logistic    # one group
"""

import argparse

import numpy as np

from cross_validation import (
    best_f1_threshold,
    classification_metrics,
    stratified_k_fold,
)
from data_exploration import load_data
from implementations import (
    compute_logistic_loss,
    compute_mse,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
)
from preprocessing import DEFAULT_OPTIONS, Preprocessor, add_bias
from run import predict_scores

# group -> list of (model, lambda, gamma, iteration checkpoints)
GROUPS = {
    "logistic": [
        ("logistic", 0.0, gamma, (500, 1000, 2000, 4000)) for gamma in (0.05, 0.15)
    ],
    "reg_logistic": [
        ("reg_logistic", lambda_, 0.15, (1000, 2000)) for lambda_ in (1e-3, 1e-2, 1e-1)
    ],
    "mse_gd": [("mse_gd", 0.0, gamma, (1000, 4000)) for gamma in (0.02, 0.04)],
    "mse_sgd": [
        ("mse_sgd", 0.0, gamma, (262_500, 1_000_000, 3_000_000))
        for gamma in (2e-6, 5e-6, 1e-5, 1e-4)
    ],
}


def continue_training(model, y, tx, w, lambda_, gamma, iters):
    """Run ``iters`` more steps of the method, starting from ``w``."""
    if model == "logistic":
        return logistic_regression(y, tx, w, iters, gamma)[0]
    if model == "reg_logistic":
        return reg_logistic_regression(y, tx, lambda_, w, iters, gamma)[0]
    if model == "mse_gd":
        return mean_squared_error_gd(y, tx, w, iters, gamma)[0]
    return mean_squared_error_sgd(y, tx, w, iters, gamma)[0]


def training_loss(model, y, tx, w):
    """Unregularized training loss, to check convergence."""
    if model in ("logistic", "reg_logistic"):
        return compute_logistic_loss(y, tx, w)
    return compute_mse(y, tx, w)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data", help="folder with the csv files")
    parser.add_argument(
        "--groups", nargs="+", choices=GROUPS, default=list(GROUPS), metavar="GROUP"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    data = load_data(args.data, use_cache=False)
    x, y = data["x_train"], data["y_train"]
    folds = stratified_k_fold(y, 5, args.seed)
    settings = [s for group in args.groups for s in GROUPS[group]]
    keys = [(s, it) for s in settings for it in s[3]]
    oof_scores = {key: np.empty(len(y)) for key in keys}
    losses = {key: [] for key in keys}
    diverged = set()

    for i, (train_idx, val_idx) in enumerate(folds):
        # fitted on the training folds only, as in run.py
        prep = Preprocessor(data["feature_names"], **DEFAULT_OPTIONS)
        tx_train = add_bias(prep.fit_transform(x[train_idx]))
        tx_val = add_bias(prep.transform(x[val_idx]))
        y_train = y[train_idx]
        for setting in settings:
            model, lambda_, gamma, checkpoints = setting
            w = np.zeros(tx_train.shape[1])
            initial_loss = training_loss(model, y_train, tx_train, w)
            if model == "mse_sgd":
                np.random.seed(args.seed)  # same seeding as run.train_model
            done = 0
            # diverging SGD settings overflow: flag them instead of scoring
            with np.errstate(over="ignore", invalid="ignore"):
                for iters in checkpoints:
                    w = continue_training(
                        model, y_train, tx_train, w, lambda_, gamma, iters - done
                    )
                    done = iters
                    loss = training_loss(model, y_train, tx_train, w)
                    # worse than the starting point w = 0: the method diverged
                    if not loss <= initial_loss:
                        diverged.add((setting, iters))
                        continue
                    scores = predict_scores(tx_val, w, model)
                    oof_scores[(setting, iters)][val_idx] = scores
                    losses[(setting, iters)].append(loss)
        print(f"fold {i + 1}/5 done", flush=True)

    print(
        f"\n{'model':<13} {'lambda':>7} {'gamma':>7} {'iters':>9}  "
        "F1 (5 folds)     precision  recall  threshold  train loss"
    )
    for setting, iters in keys:
        model, lambda_, gamma, _ = setting
        prefix = f"{model:<13} {lambda_:>7g} {gamma:>7g} {iters:>9}"
        if (setting, iters) in diverged:
            print(f"{prefix}  diverged")
            continue
        scores = oof_scores[(setting, iters)]
        loss = np.mean(losses[(setting, iters)])
        threshold, _ = best_f1_threshold(y, scores)
        y_pred = (scores >= threshold).astype(int)
        fold_f1 = [classification_metrics(y[v], y_pred[v])["f1"] for _, v in folds]
        m = classification_metrics(y, y_pred)
        print(
            f"{prefix}  {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}  "
            f"{m['precision']:>9.3f}  {m['recall']:>6.3f}  {threshold:>9.4g}  "
            f"{loss:>10.4g}"
        )


if __name__ == "__main__":
    main()
