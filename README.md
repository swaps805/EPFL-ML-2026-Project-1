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
