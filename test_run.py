"""Regression checks for the six-method training and submission pipeline."""

import contextlib
import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest

import numpy as np

from implementations import least_squares
from run import MODELS, main, predict_scores, train_model


class ModelPipelineTests(unittest.TestCase):
    def test_least_squares_with_dependent_columns(self):
        x = np.arange(12, dtype=float)
        tx = np.column_stack((np.ones(12), x, x))
        y = 2 + 3 * x
        w, loss = least_squares(y, tx)
        np.testing.assert_allclose(tx @ w, y, atol=1e-12)
        self.assertLess(loss, 1e-24)

    def test_linear_scores_are_not_probabilities(self):
        tx = np.array([[1, -3], [1, 3]])
        w = np.array([0, 1])
        for model in MODELS:
            scores = predict_scores(tx, w, model)
            if model in ("logistic", "reg_logistic"):
                self.assertTrue(np.all((scores > 0) & (scores < 1)))
            else:
                np.testing.assert_array_equal(scores, [-3, 3])

    def test_sgd_seed_is_reproducible_and_isolated(self):
        tx = np.column_stack((np.ones(20), np.linspace(-1, 1, 20)))
        y = np.arange(20) % 2
        state = np.random.get_state()
        first, _ = train_model(tx, y, "mse_sgd", 0, 0.01, 50, seed=42)
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        second, _ = train_model(tx, y, "mse_sgd", 0, 0.01, 50, seed=42)
        other, _ = train_model(tx, y, "mse_sgd", 0, 0.01, 50, seed=43)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))

    def test_all_models_cv_submission_and_replay(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            x = rng.normal(size=(90, 4))
            x[:, 0] = np.arange(90) % 3  # dependent one-hot columns
            x[::7, 1] = np.nan
            ids = np.arange(90) + 100
            test_ids = np.array([900, 700, 800])
            for name, values in (
                ("x_train.csv", np.column_stack((ids, x))),
                ("x_test.csv", np.column_stack((test_ids, x[:3]))),
            ):
                np.savetxt(
                    root / name, values, delimiter=",", header="Id,A,B,C,D", comments=""
                )
            np.savetxt(
                root / "y_train.csv",
                np.column_stack((ids, np.where(x[:, 2] > 0, 1, -1))),
                delimiter=",",
                fmt="%d",
                header="Id,Prediction",
                comments="",
            )
            for model in MODELS:
                with self.subTest(model=model), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    output = root / f"{model}.csv"
                    args = [
                        "--data",
                        str(root),
                        "--model",
                        model,
                        "--k",
                        "3",
                        "--max-iters",
                        "40",
                        "--gamma",
                        "0.001",
                        "--output",
                        str(output),
                    ]
                    main(args + ["--evaluate-only"])
                    self.assertFalse(output.exists())
                    main(args)
                    original = output.read_bytes()
                    metadata = json.loads(output.with_suffix(".csv.json").read_text())
                    self.assertEqual(len(metadata["validation"]["fold_f1"]), 3)
                    predictions = np.loadtxt(output, delimiter=",", skiprows=1)
                    np.testing.assert_array_equal(predictions[:, 0], test_ids)
                    self.assertTrue(np.all(np.isin(predictions[:, 1], [-1, 1])))
                    main(shlex.split(metadata["reproduce_command"])[2:])
                    self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
