"""Evaluate a model, train on all training rows, and write an AIcrowd CSV.

Defaults are the documented ridge baseline, not a confirmed competition winner.
Only the Python standard library and NumPy are used.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex

import numpy as np

from cross_validation import cross_validate, format_result
from data_exploration import load_data
from helpers import create_csv_submission
from implementations import (
    logistic_regression,
    reg_logistic_regression,
    ridge_regression,
    sigmoid,
)
from preprocessing import DEFAULT_OPTIONS, Preprocessor, add_bias


def train_model(tx, y, model, lambda_, gamma, max_iters):
    """Train with 0/1 labels and return weights and unregularized loss."""
    if model == "ridge":
        w, loss = ridge_regression(y, tx, lambda_)
    else:
        initial_w = np.zeros(tx.shape[1])
        if model == "logistic":
            w, loss = logistic_regression(y, tx, initial_w, max_iters, gamma)
        elif model == "reg_logistic":
            w, loss = reg_logistic_regression(
                y, tx, lambda_, initial_w, max_iters, gamma
            )
        else:
            raise ValueError(f"Unknown model: {model}")
    if not np.all(np.isfinite(w)) or not np.isfinite(loss):
        raise ValueError("Training produced non-finite values; try a smaller gamma.")
    return w, float(loss)


def predict_scores(tx, w, model):
    """Ridge returns raw scores; logistic models return probabilities."""
    scores = tx.dot(w)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Prediction produced non-finite scores.")
    return scores if model == "ridge" else sigmoid(scores)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument(
        "--model", choices=("ridge", "logistic", "reg_logistic"), default="ridge"
    )
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--keep-sparse", action="store_true", help="disable sparse-feature removal"
    )
    parser.add_argument(
        "--threshold", type=float,
        help="reuse a previously selected threshold and skip cross-validation",
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="print cross-validation metrics without training a final model or writing CSV",
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.lambda_) or args.lambda_ < 0:
        parser.error("--lambda must be finite and non-negative")
    if args.model == "ridge" and args.lambda_ == 0:
        parser.error("ridge requires --lambda > 0 because one-hot columns are dependent")
    if not np.isfinite(args.gamma) or args.gamma <= 0 or args.max_iters < 1:
        parser.error("--gamma must be finite and positive; --max-iters must be positive")
    if args.k < 2 or args.seed < 0:
        parser.error("--k must be at least 2 and --seed must be non-negative")
    if args.threshold is not None and not np.isfinite(args.threshold):
        parser.error("--threshold must be finite")
    if args.evaluate_only and args.threshold is not None:
        parser.error("--evaluate-only cannot be combined with --threshold")
    for name in ("x_train.csv", "y_train.csv", "x_test.csv"):
        if not (args.data / name).is_file():
            parser.error(f"Missing {args.data / name}; put the competition CSVs in --data")

    # Read the CSVs directly so a stale exploration cache cannot change the run.
    data = load_data(args.data, use_cache=False)
    x, y = data["x_train"], data["y_train"]
    options = dict(DEFAULT_OPTIONS)
    if args.keep_sparse:
        options["max_missing"] = None
    print(f"{len(y)} training rows, {len(data['x_test'])} test rows; model={args.model}")

    def fit_model(tx, labels):
        return train_model(
            tx, labels, args.model, args.lambda_, args.gamma, args.max_iters
        )

    def fit_predict(x_train, y_train, x_val):
        # A fresh preprocessor per fold prevents validation-data leakage.
        prep = Preprocessor(data["feature_names"], **options)
        w, _ = fit_model(add_bias(prep.fit_transform(x_train)), y_train)
        return predict_scores(add_bias(prep.transform(x_val)), w, args.model)

    threshold = args.threshold
    cv_summary = None
    if threshold is None:
        if np.bincount(y, minlength=2).min() < args.k:
            parser.error("Each class needs at least --k training rows")
        result = cross_validate(fit_predict, x, y, k=args.k, seed=args.seed)
        print(format_result(args.model, result))
        threshold = result["threshold"]
        cv_summary = {
            "metrics": result["metrics"],
            "fold_f1": result["fold_f1"].tolist(),
            "f1_mean": float(result["f1_mean"]),
            "f1_std": float(result["f1_std"]),
        }
        print("F1 is a tuning score: the threshold was selected on these OOF labels.")
    if args.evaluate_only:
        return

    print(f"Training on all rows; decision threshold={threshold:.17g}")
    prep = Preprocessor(data["feature_names"], **options)
    w, loss = fit_model(add_bias(prep.fit_transform(x)), y)
    print(prep.describe())
    scores = predict_scores(add_bias(prep.transform(data["x_test"])), w, args.model)
    predictions = np.where(scores >= threshold, 1, -1)
    if len(predictions) != len(data["test_ids"]):
        raise ValueError("Prediction count does not match test IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(data["test_ids"], predictions, args.output)

    # The replay command fixes the selected threshold and skips CV next time.
    command = [
        "python", "run.py", "--data", str(args.data), "--output", str(args.output),
        "--model", args.model, "--lambda", repr(args.lambda_),
        "--gamma", repr(args.gamma), "--max-iters", str(args.max_iters),
        "--k", str(args.k), "--seed", str(args.seed),
        "--threshold", repr(threshold),
    ]
    if args.keep_sparse:
        command.append("--keep-sparse")
    metadata = {
        "model": args.model,
        "lambda": args.lambda_,
        "gamma": args.gamma,
        "max_iters": args.max_iters,
        "k": args.k,
        "seed": args.seed,
        "preprocessing": options,
        "threshold": threshold,
        "training_loss": loss,
        "validation": cv_summary,
        "training_rows": len(y),
        "prediction_rows": len(predictions),
        "numpy_version": np.__version__,
        "csv_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "reproduce_command": shlex.join(command),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output} ({len(predictions)} predictions) and {metadata_path}")


if __name__ == "__main__":
    main()
