# CS-433 Machine Learning — Project 1

Prediction of coronary heart disease (MICHD) from the BRFSS health survey
(AIcrowd competition). Deadline: October 29th, 2026, 16:00.

Allowed libraries: Python standard library, NumPy, and matplotlib/seaborn for
visualization only.

## Repository structure

- `implementations.py` — the 6 required methods (Table 1 of the project
  description): `mean_squared_error_gd`, `mean_squared_error_sgd`,
  `least_squares`, `ridge_regression`, `logistic_regression`,
  `reg_logistic_regression`. Each returns `(w, loss)`, the last weight vector
  and its loss (without the penalty term for regularized methods).
- `helpers.py` — provided by the course: `load_csv_data` and
  `create_csv_submission`. Note: `load_csv_data` returns labels in {-1, 1};
  map them to {0, 1} before using the logistic regression methods.
- `data_exploration.py` — exploratory data analysis tools (see below).
- `DATA_RECAP.md` — summary of what we learned about the data and its
  implications for preprocessing.
- `cross_validation.py` — model evaluation: stratified k-fold
  cross-validation, metrics (F1, precision, recall, accuracy) and decision
  threshold (see below).
- `run.py` — produces the final submission `.csv` (to be written).
- `environment.yml` — conda environment.

## Setup

```
conda env create --file=environment.yml   # creates env MachineLearningProject
conda activate MachineLearningProject
```

Python 3.9, NumPy 1.23.1, matplotlib, seaborn, pytest, black — versions aligned
with the course grading environment.

Data: download `x_train.csv`, `y_train.csv` and `x_test.csv` from AIcrowd and
put them in a `data/` folder at the root of the repository (do not rename
them). `data/` is ignored by git.

## Data exploration

```
python data_exploration.py                    # full report on the dataset
python data_exploration.py --feature GENHLTH  # deep dive into one feature
python data_exploration.py --sub-sample       # faster run on 1/50 of the rows
```

The first run parses the csv files (about one minute) and caches them in
`data/cache_full.npz`; later runs take a few seconds. Outputs go to
`exploration_output/` (ignored by git).

The full report prints six sections: overview (sizes, class balance, feature
types, missing values, correlation ranking), data quality (constant, nearly
empty, administrative and mixed-unit features), informative missingness,
redundant features, positive rate per answer for known risk factors, and the
scores of trivial baselines. It also writes:

- `feature_summary.csv` — one row per feature: inferred type, missing rate
  (train and test), range, candidate BRFSS special codes (7/9/77/99/...,
  "don't know" / "refused" / "none"), positive rate when the feature is missing,
  correlation with the target and spread of the positive rate across categories.
- `redundant_pairs.csv` — pairs of features with |correlation| > 0.95.
- `figures/` — missing values per feature, top correlated features, and one
  figure per feature analysed with `--feature`.

The functions can also be imported, e.g. in a personal notebook:

```python
from data_exploration import load_data, summarize_features, print_feature_report

data = load_data("data")
print_feature_report(data, "PHYSHLTH")
```

Special codes are detected heuristically and must be checked in the
[BRFSS 2015 codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf)
(e.g. 88 means "none", i.e. 0 days, for PHYSHLTH).

## Model evaluation

All models are compared with the same protocol, implemented in
`cross_validation.py`:

- **Stratified 5-fold cross-validation** (seed 1): every fold keeps the 8.8%
  positive rate. Each sample is scored once by a model trained on the other
  folds.
- **Metric: F1 score** (accuracy is misleading with 8.8% positives: predicting
  "always negative" gives 91% accuracy). We report the mean ± standard
  deviation of the F1 over the folds, with precision, recall and accuracy.
- **Decision threshold:** the one maximizing F1 on the out-of-fold scores.

A model is a function `fit_predict(x_train, y_train, x_val)` returning one
score per row of `x_val`. All preprocessing statistics (means, medians,
standard deviations, ...) must be computed inside it on `x_train` only, to
avoid leaking information from the held-out fold.

```python
from cross_validation import cross_validate, format_result

result = cross_validate(my_fit_predict, x, y)  # y in {0, 1}
print(format_result("my model", result))
```

`python cross_validation.py` runs a sanity check on the full training set:

| Model | F1 (5 folds) | Precision | Recall |
|---|---|---|---|
| Always positive | 0.162 ± 0.000 | 0.088 | 1.000 |
| Random scores | 0.162 ± 0.000 | 0.088 | 1.000 |
| Ridge, naive preprocessing (λ = 1e-6) | 0.411 ± 0.004 | 0.327 | 0.553 |

"Naive preprocessing" only imputes the mean and standardizes; it is not the
project preprocessing.

## Tests

The public tests are in the course repository
(`ML_course/projects/project1/grading_tests`). From that folder:

```
pytest --github_link <path-to-this-repo-or-github-url> .
```

## Team workflow

- Never work directly on `main`: one branch per task, merged into `main`
  through a pull request.
- `git pull` on `main` before starting a new branch.
- Format code with `black .` before committing.
