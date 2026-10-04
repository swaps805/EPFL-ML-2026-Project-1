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

## Tests

The public tests are in the course repository
(`ML_course/projects/project1/grading_tests`). From that folder:

```
pytest --github_link <path-to-this-repo-or-github-url> .
```

## Team workflow

- Never work directly on `main`: one branch per task, merged through a pull
  request reviewed by a teammate.
- `git pull` on `main` before starting a new branch.
- Format code with `black .` before committing.
