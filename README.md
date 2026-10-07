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
- `preprocessing.py` — feature preprocessing fitted on training data only
  (see below).
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

## Preprocessing

`preprocessing.py` defines a `Preprocessor` that learns all its statistics
with `fit` on the training data only and applies them with `transform`:

```python
from preprocessing import Preprocessor, add_bias

preprocessor = Preprocessor(feature_names).fit(x_train)
tx_train = add_bias(preprocessor.transform(x_train))
tx_test = add_bias(preprocessor.transform(x_test))
```

Steps (each one can be disabled through an option, for the ablation study):

1. Drop survey administration features and raw mixed-unit features (WEIGHT2,
   HEIGHT3; the CDC versions WTKG3, HTM4 and _BMI5 are kept).
2. Map BRFSS special codes: "don't know" / "refused" → missing, "none" (88,
   ...) → 0.
3. Add missing-value indicators (identical indicators merged).
4. Drop constant features and features missing for > 90% of the samples.
5. Drop redundant features (|correlation| > 0.95).
6. Impute the remaining missing values with the training median.
7. One-hot encode categorical features (3 to 15 integer values).
8. Standardize every column.

With the default options, the 321 raw features become 619 columns.

`python preprocessing.py` (about 6 minutes) compares variants where one step
is removed at a time, with ridge regression (λ = 1e-6), stratified 5-fold
cross-validation and the F1-maximizing threshold:

| Variant | F1 (5 folds) | Precision | Recall |
|---|---|---|---|
| All steps | 0.420 ± 0.008 | 0.338 | 0.555 |
| Without admin drop | 0.420 ± 0.008 | 0.341 | 0.547 |
| Without special codes | 0.422 ± 0.005 | 0.355 | 0.520 |
| Without missing indicators | 0.418 ± 0.006 | 0.334 | 0.558 |
| Without sparse drop | 0.423 ± 0.007 | 0.338 | 0.564 |
| Without redundancy drop | 0.421 ± 0.005 | 0.351 | 0.526 |
| Without one-hot | 0.416 ± 0.007 | 0.334 | 0.552 |
| Minimal (impute + scale only) | 0.412 ± 0.004 | 0.326 | 0.560 |

The full preprocessing improves F1 from 0.412 to 0.420, but removing any
single step changes F1 by less than the fold-to-fold variation: with a linear
model, no single step dominates. Mapping "don't know" to missing may lose
information ("don't know" answers are themselves associated with the target,
see `DATA_RECAP.md`).

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
