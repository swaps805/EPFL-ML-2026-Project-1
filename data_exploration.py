# -*- coding: utf-8 -*-
"""Tools to investigate the BRFSS dataset (exploratory data analysis).

The module can be imported (e.g. from a personal notebook) or run as a script:

    python data_exploration.py                    # full report on the dataset
    python data_exploration.py --feature GENHLTH  # deep dive into one feature
    python data_exploration.py --sub-sample       # faster run on 1/50 of the rows

The full report covers: overview (sizes, class balance, feature types, missing
values, correlation ranking), data quality (constant, nearly empty,
administrative and mixed-unit features), informative missingness, redundant
features, positive rate per answer for known risk factors, and the scores of
trivial baselines.

Outputs (tables and figures) are written to ``exploration_output/``, which is
ignored by git.

Only NumPy (computations) and matplotlib (figures) are used, as required by the
project rules. The meaning of each feature and of its special codes is given
in the BRFSS 2015 codebook:
https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
"""

import argparse
import csv
import os

import numpy as np

from helpers import load_csv_data

# A feature with at most this many distinct integer values is considered
# categorical (ordinal or nominal); above, it is considered continuous.
CATEGORICAL_MAX_UNIQUE = 15

# Codes used by BRFSS for "don't know / not sure" (7, 77, ...), "refused"
# (9, 99, ...) and "none" (8, 88, ...). A candidate is only flagged when it is
# isolated, i.e. the value just below it never occurs: 77 is a code for
# PHYSHLTH (days, 1-30) but a valid value for an age ranging from 18 to 80.
# Each flagged code must be checked in the codebook: "none" often means 0.
SPECIAL_CODE_CANDIDATES = (7, 8, 9, 77, 88, 99, 777, 888, 999, 7777, 8888, 9999)

# Categories with fewer samples are ignored when comparing positive rates,
# to avoid noisy rates computed on a handful of people.
MIN_CATEGORY_COUNT = 100

# Features observed for fewer samples are left out of the correlation ranking:
# a correlation computed on a few hundred people is mostly noise.
MIN_OBSERVED_FOR_RANKING = 1000

# Survey administration features (location, interview date, record numbers,
# sampling weights): they describe how the interview was run, not the person.
ADMIN_FEATURES = (
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_LLCPWT",
)

# Known cardiovascular risk factors, shown with their positive rate per answer.
RISK_FACTORS = (
    "_AGEG5YR",
    "SEX",
    "GENHLTH",
    "CVDSTRK3",
    "CHCKIDNY",
    "CHCCOPD1",
    "DIFFWALK",
    "DIABETE3",
    "BPHIGH4",
    "TOLDHI2",
    "INCOME2",
    "EDUCA",
    "SMOKE100",
    "EXERANY2",
    "_BMI5CAT",
)

# Features whose raw answer mixes imperial and metric units: BRFSS stores the
# metric answers as 9000 + value (e.g. WEIGHT2 = 9070 means 70 kg).
MIXED_UNIT_FEATURES = ("WEIGHT2", "HEIGHT3")
METRIC_CODE_RANGE = (9000, 9998)

# Two features with a correlation above this threshold carry the same
# information (e.g. age in years and age in 5-year groups).
REDUNDANCY_THRESHOLD = 0.95

CACHE_FILE = "cache_full.npz"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_feature_names(data_dir):
    """Read the feature names from the header of x_train.csv.

    Args:
        data_dir: str, folder containing the AIcrowd csv files.

    Returns:
        list of str, the D feature names (the "Id" column is excluded).
    """
    with open(os.path.join(data_dir, "x_train.csv"), "r") as f:
        header = f.readline().strip().split(",")
    return header[1:]


def load_data(data_dir="data", sub_sample=False, use_cache=True):
    """Load the dataset, with labels mapped to {0, 1}.

    Parsing the csv files takes about a minute, so the arrays are cached in a
    ``.npz`` file inside ``data_dir`` after the first call.

    Args:
        data_dir: str, folder containing the AIcrowd csv files.
        sub_sample: bool, if True keep one training sample out of 50.
        use_cache: bool, if True read / write the ``.npz`` cache.

    Returns:
        dict with keys x_train (N, D), y_train (N,) in {0, 1}, x_test (M, D),
        train_ids (N,), test_ids (M,) and feature_names (list of D str).
    """
    cache_path = os.path.join(data_dir, CACHE_FILE)
    if use_cache and os.path.exists(cache_path):
        cached = np.load(cache_path)
        x_train, x_test = cached["x_train"], cached["x_test"]
        y_train = cached["y_train"]
        train_ids, test_ids = cached["train_ids"], cached["test_ids"]
    else:
        x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_dir)
        if use_cache:
            np.savez(
                cache_path,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                train_ids=train_ids,
                test_ids=test_ids,
            )

    if sub_sample:
        # same sub-sampling as helpers.load_csv_data
        x_train, y_train, train_ids = x_train[::50], y_train[::50], train_ids[::50]

    return {
        "x_train": x_train,
        # helpers returns labels in {-1, 1}; {0, 1} is what logistic regression
        # expects and makes "mean of y" directly the positive rate
        "y_train": (y_train == 1).astype(int),
        "x_test": x_test,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "feature_names": load_feature_names(data_dir),
    }


# ---------------------------------------------------------------------------
# Per-feature statistics
# ---------------------------------------------------------------------------


def infer_feature_type(values):
    """Guess the type of a feature from its non-missing values.

    Args:
        values: numpy array of shape (n,), the non-missing values of a feature.

    Returns:
        str, one of "empty", "constant", "binary", "categorical", "continuous".
    """
    if len(values) == 0:
        return "empty"
    unique = np.unique(values)
    if len(unique) == 1:
        return "constant"
    if len(unique) == 2:
        return "binary"
    is_integer = np.all(unique == np.round(unique))
    if is_integer and len(unique) <= CATEGORICAL_MAX_UNIQUE:
        return "categorical"
    return "continuous"


def find_special_codes(values):
    """Find the isolated BRFSS special codes present in a feature.

    Args:
        values: numpy array of shape (n,), the non-missing values of a feature.

    Returns:
        list of int, the candidate special codes present in the feature whose
        predecessor (code - 1) is absent.
    """
    unique = np.unique(values)
    return [
        code
        for code in SPECIAL_CODE_CANDIDATES
        if code in unique and code - 1 not in unique
    ]


def positive_rate_by_value(values, y):
    """Compute the fraction of positive labels for each distinct value.

    Args:
        values: numpy array of shape (n,), the non-missing values of a feature.
        y: numpy array of shape (n,), the corresponding labels in {0, 1}.

    Returns:
        unique: numpy array of shape (k,), the distinct values.
        counts: numpy array of shape (k,), the number of samples per value.
        rates: numpy array of shape (k,), the positive rate per value.
    """
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse, weights=y, minlength=len(unique))
    return unique, counts, positives / counts


def correlation(a, b):
    """Pearson correlation between two vectors (0 if one of them is constant).

    Args:
        a: numpy array of shape (n,).
        b: numpy array of shape (n,).

    Returns:
        float, the correlation coefficient.
    """
    a_centered, b_centered = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a_centered**2).sum() * (b_centered**2).sum())
    if denominator == 0:
        return 0.0
    return float((a_centered * b_centered).sum() / denominator)


def summarize_feature(name, column, y, test_column):
    """Compute the summary statistics of one feature.

    Args:
        name: str, the feature name.
        column: numpy array of shape (N,), the feature on the training set.
        y: numpy array of shape (N,), the training labels in {0, 1}.
        test_column: numpy array of shape (M,), the feature on the test set.

    Returns:
        dict of statistics (see ``SUMMARY_FIELDS``).
    """
    observed = ~np.isnan(column)
    values, y_observed = column[observed], y[observed]
    stats = {
        "name": name,
        "type": infer_feature_type(values),
        "missing_frac": 1 - observed.mean(),
        "missing_frac_test": np.isnan(test_column).mean(),
        "n_observed": int(observed.sum()),
        "n_unique": len(np.unique(values)),
    }
    if len(values) == 0:
        return stats

    special_codes = find_special_codes(values)
    stats.update(
        {
            "min": values.min(),
            "median": np.median(values),
            "max": values.max(),
            "special_codes": " ".join(str(c) for c in special_codes),
            "special_frac": np.isin(values, special_codes).mean(),
            # positive rate when the feature is missing vs observed: missingness
            # itself can be informative (e.g. a question skipped for healthy people)
            "pos_rate_missing": y[~observed].mean() if (~observed).any() else np.nan,
            "pos_rate_observed": y_observed.mean(),
        }
    )

    # association with the target, computed without the special codes, which
    # would otherwise dominate the correlation of ordinal features
    clean = ~np.isin(values, special_codes)
    stats["corr_target"] = correlation(values[clean], y_observed[clean])

    if stats["type"] in ("binary", "categorical"):
        _, counts, rates = positive_rate_by_value(values, y_observed)
        rates = rates[counts >= MIN_CATEGORY_COUNT]
        if len(rates) >= 2:
            stats["pos_rate_range"] = rates.max() - rates.min()
    return stats


SUMMARY_FIELDS = [
    "name",
    "type",
    "missing_frac",
    "missing_frac_test",
    "n_observed",
    "n_unique",
    "min",
    "median",
    "max",
    "special_codes",
    "special_frac",
    "pos_rate_missing",
    "pos_rate_observed",
    "corr_target",
    "pos_rate_range",
]


def summarize_features(x, y, x_test, feature_names):
    """Compute the summary statistics of every feature.

    Args:
        x: numpy array of shape (N, D), the training features.
        y: numpy array of shape (N,), the training labels in {0, 1}.
        x_test: numpy array of shape (M, D), the test features.
        feature_names: list of D str.

    Returns:
        list of D dicts, one per feature (see ``summarize_feature``).
    """
    return [
        summarize_feature(name, x[:, d], y, x_test[:, d])
        for d, name in enumerate(feature_names)
    ]


def save_summary_csv(summary, path):
    """Write the feature summary to a csv file (one row per feature).

    Args:
        summary: list of dicts returned by ``summarize_features``.
        path: str, the output csv path.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for stats in summary:
            row = {}
            for key, value in stats.items():
                is_float = isinstance(value, (float, np.floating))
                row[key] = f"{value:.4g}" if is_float else value
            writer.writerow(row)


def rank_by_correlation(summary):
    """Sort the features by decreasing |correlation| with the target.

    Features observed for fewer than ``MIN_OBSERVED_FOR_RANKING`` samples are
    excluded.

    Args:
        summary: list of dicts returned by ``summarize_features``.

    Returns:
        list of dicts, the sorted subset of ``summary``.
    """
    reliable = [
        s
        for s in summary
        if s["n_observed"] >= MIN_OBSERVED_FOR_RANKING and "corr_target" in s
    ]
    return sorted(reliable, key=lambda s: -abs(s["corr_target"]))


# ---------------------------------------------------------------------------
# Dataset-level analyses
# ---------------------------------------------------------------------------


def rank_informative_missingness(summary, min_missing=0.05, max_missing=0.95):
    """Sort the features by how much their missingness tells about the target.

    BRFSS skips some questions depending on previous answers (e.g. the
    diabetes module is only asked to diabetics), so the fact that a value is
    missing can itself be predictive.

    Args:
        summary: list of dicts returned by ``summarize_features``.
        min_missing: float, features with a lower missing rate are ignored.
        max_missing: float, features with a higher missing rate are ignored.

    Returns:
        list of dicts, sorted by decreasing |pos_rate_missing - pos_rate_observed|.
    """
    candidates = [
        s
        for s in summary
        if min_missing < s["missing_frac"] < max_missing and "pos_rate_observed" in s
    ]
    return sorted(
        candidates,
        key=lambda s: -abs(s["pos_rate_missing"] - s["pos_rate_observed"]),
    )


def find_redundant_pairs(
    x, summary, feature_names, threshold=REDUNDANCY_THRESHOLD, max_missing=0.05
):
    """Find pairs of features that are almost perfectly correlated.

    The correlation is computed on the rows where all the considered features
    are observed, so only features with few missing values are considered.

    Args:
        x: numpy array of shape (N, D), the training features.
        summary: list of D dicts returned by ``summarize_features``.
        feature_names: list of D str.
        threshold: float, minimum |correlation| to report a pair.
        max_missing: float, features with a higher missing rate are ignored.

    Returns:
        list of tuples (name_a, name_b, correlation), sorted by decreasing
        |correlation|.
    """
    columns = [
        d
        for d, s in enumerate(summary)
        if s["missing_frac"] < max_missing and s["type"] not in ("constant", "empty")
    ]
    sub = x[:, columns]
    sub = sub[~np.isnan(sub).any(axis=1)]
    corr = np.corrcoef(sub.T)
    # upper triangle only, so that each pair is reported once
    i, j = np.nonzero(np.triu(np.abs(corr) > threshold, k=1))
    pairs = [
        (feature_names[columns[a]], feature_names[columns[b]], corr[a, b])
        for a, b in zip(i, j)
    ]
    return sorted(pairs, key=lambda p: -abs(p[2]))


def count_metric_entries(column):
    """Count the answers given in metric units in a mixed-unit feature.

    Args:
        column: numpy array of shape (N,), e.g. WEIGHT2 or HEIGHT3.

    Returns:
        int, the number of values in ``METRIC_CODE_RANGE``.
    """
    low, high = METRIC_CODE_RANGE
    return int(((column >= low) & (column <= high)).sum())


def trivial_baselines(y):
    """Scores of the two constant classifiers, the floor any model must beat.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.

    Returns:
        dict with the accuracy of "always negative" and the F1 score of
        "always positive" (the F1 of "always negative" is 0).
    """
    p = y.mean()
    return {
        "accuracy_always_negative": 1 - p,
        # precision = p and recall = 1, so F1 = 2p / (1 + p)
        "f1_always_positive": 2 * p / (1 + p),
    }


def save_pairs_csv(pairs, path):
    """Write the redundant feature pairs to a csv file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_a", "feature_b", "correlation"])
        for a, b, r in pairs:
            writer.writerow([a, b, f"{r:.4f}"])


# ---------------------------------------------------------------------------
# Text reports
# ---------------------------------------------------------------------------


def print_overview(data, summary, top_k=20):
    """Print the main facts about the dataset and its features.

    Args:
        data: dict returned by ``load_data``.
        summary: list of dicts returned by ``summarize_features``.
        top_k: int, number of most associated features to list.
    """
    x, y = data["x_train"], data["y_train"]
    print(f"Training set: {x.shape[0]} samples, {x.shape[1]} features")
    print(f"Test set:     {data['x_test'].shape[0]} samples")
    print(f"Positive rate (MICHD): {y.mean():.2%}")

    types = [s["type"] for s in summary]
    print("\nFeature types (inferred):")
    for t in ("binary", "categorical", "continuous", "constant", "empty"):
        print(f"  {t:<12} {types.count(t)}")

    missing = np.array([s["missing_frac"] for s in summary])
    print("\nMissing values:")
    print(f"  overall fraction of missing entries: {np.isnan(x).mean():.1%}")
    for threshold in (0.1, 0.5, 0.9):
        n = (missing > threshold).sum()
        print(f"  features with > {threshold:.0%} missing: {n}")

    n_special = sum(1 for s in summary if s.get("special_codes"))
    print(f"\nFeatures with candidate special codes: {n_special}")

    shift = [
        s["name"]
        for s in summary
        if abs(s["missing_frac"] - s["missing_frac_test"]) > 0.05
    ]
    print(f"Features whose missing rate differs train vs test (> 5 pts): {shift}")

    ranked = rank_by_correlation(summary)
    print(
        f"\nTop {top_k} features by |correlation| with the target "
        f"(observed for >= {MIN_OBSERVED_FOR_RANKING} samples):"
    )
    print(f"  {'name':<12} {'type':<12} {'corr':>7} {'missing':>8}")
    for s in ranked[:top_k]:
        print(
            f"  {s['name']:<12} {s['type']:<12} "
            f"{s['corr_target']:>7.3f} {s['missing_frac']:>8.1%}"
        )


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def print_data_quality(data, summary):
    """Print the features that are useless or need cleaning.

    Args:
        data: dict returned by ``load_data``.
        summary: list of dicts returned by ``summarize_features``.
    """
    names = data["feature_names"]
    by_name = {s["name"]: s for s in summary}

    constant = [s["name"] for s in summary if s["type"] == "constant"]
    print(f"Constant features ({len(constant)}): {constant}")

    nearly_empty = [s["name"] for s in summary if s["missing_frac"] > 0.99]
    print(f"\nFeatures with > 99% missing ({len(nearly_empty)}): {nearly_empty}")

    missing_per_sample = np.isnan(data["x_train"]).mean(axis=1)
    p5, p50, p95 = np.percentile(missing_per_sample, [5, 50, 95])
    print(
        f"\nMissing answers per person: median {p50:.0%} "
        f"(5th percentile {p5:.0%}, 95th percentile {p95:.0%})"
    )

    print("\nSurvey administration features (expected to be uninformative):")
    print(f"  {'name':<10} {'type':<12} {'n_unique':>8} {'corr':>7}")
    for name in ADMIN_FEATURES:
        if name in by_name:
            s = by_name[name]
            print(
                f"  {name:<10} {s['type']:<12} {s['n_unique']:>8} "
                f"{s.get('corr_target', np.nan):>7.3f}"
            )

    print("\nMixed imperial / metric units (metric stored as 9000 + value):")
    for name in MIXED_UNIT_FEATURES:
        if name in names:
            column = data["x_train"][:, names.index(name)]
            codes = by_name[name]["special_codes"] or "none"
            print(
                f"  {name}: {count_metric_entries(column)} metric answers, "
                f"special codes {codes}"
            )


def print_informative_missingness(summary, top_k=10):
    """Print the features whose missingness is most related to the target."""
    ranked = rank_informative_missingness(summary)
    print(
        "Positive rate when the feature is missing vs observed "
        "(features 5-95% missing):"
    )
    print(f"  {'name':<10} {'missing':>8} {'pos|missing':>12} {'pos|observed':>13}")
    for s in ranked[:top_k]:
        print(
            f"  {s['name']:<10} {s['missing_frac']:>8.0%} "
            f"{s['pos_rate_missing']:>12.1%} {s['pos_rate_observed']:>13.1%}"
        )


def print_redundant_pairs(pairs):
    """Print the pairs of almost identical features."""
    print(
        f"Pairs of features with |correlation| > {REDUNDANCY_THRESHOLD} "
        f"(features < 5% missing): {len(pairs)}"
    )
    for a, b, r in pairs:
        print(f"  {a:<10} {b:<10} {r:>7.3f}")


def print_risk_factors(data, names=RISK_FACTORS):
    """Print the positive rate per answer for known risk factors.

    Each line reads ``value: positive rate (count)``.
    """
    feature_names = data["feature_names"]
    x, y = data["x_train"], data["y_train"]
    print(f"Positive rate per answer (overall rate {y.mean():.1%}):")
    for name in names:
        if name not in feature_names:
            continue
        column = x[:, feature_names.index(name)]
        observed = ~np.isnan(column)
        unique, counts, rates = positive_rate_by_value(column[observed], y[observed])
        cells = [f"{v:g}: {r:.1%} ({c})" for v, c, r in zip(unique, counts, rates)]
        print(f"  {name:<9} " + "  ".join(cells))


def print_baselines(y):
    """Print the scores of the trivial constant classifiers."""
    scores = trivial_baselines(y)
    print(
        "Accuracy of 'always negative': "
        f"{scores['accuracy_always_negative']:.3f} (useless model, high accuracy)"
    )
    print(
        "F1 score of 'always positive': "
        f"{scores['f1_always_positive']:.3f} (floor to beat with F1)"
    )


def print_feature_report(data, name, max_values=25):
    """Print a deep dive into one feature: value counts and positive rates.

    Args:
        data: dict returned by ``load_data``.
        name: str, the feature name.
        max_values: int, maximum number of distinct values to list.
    """
    d = data["feature_names"].index(name)
    column, y = data["x_train"][:, d], data["y_train"]
    stats = summarize_feature(name, column, y, data["x_test"][:, d])

    print(f"Feature {name} (column {d}), inferred type: {stats['type']}")
    print(
        f"  missing: {stats['missing_frac']:.1%} (train), "
        f"{stats['missing_frac_test']:.1%} (test)"
    )
    print(f"  overall positive rate: {y.mean():.2%}")
    if stats["n_unique"] == 0:
        return
    print(f"  positive rate when missing: {stats['pos_rate_missing']:.2%}")
    print(f"  candidate special codes: {stats['special_codes'] or 'none'}")
    print(
        f"  correlation with target (without special codes): {stats['corr_target']:.3f}"
    )

    observed = ~np.isnan(column)
    unique, counts, rates = positive_rate_by_value(column[observed], y[observed])
    if len(unique) > max_values:
        values = column[observed]
        print(f"  {len(unique)} distinct values, quantiles:")
        for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
            print(f"    {q:>3}%: {np.percentile(values, q):.4g}")
        # only the special codes are listed, with their positive rate
        order = np.flatnonzero(np.isin(unique, find_special_codes(column[observed])))
    else:
        order = np.arange(len(unique))
    print(f"  {'value':>10} {'count':>8} {'share':>7} {'pos. rate':>10}")
    for i in order:
        print(
            f"  {unique[i]:>10.4g} {counts[i]:>8} "
            f"{counts[i] / len(column):>7.1%} {rates[i]:>10.2%}"
        )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_missingness(summary, path):
    """Histogram of the fraction of missing values per feature."""
    import matplotlib.pyplot as plt

    missing = [s["missing_frac"] for s in summary]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(missing, bins=20, range=(0, 1), color="tab:blue", edgecolor="white")
    ax.set_xlabel("Fraction of missing values")
    ax.set_ylabel("Number of features")
    ax.set_title("Missing values per feature (training set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_correlations(summary, path, top_k=25):
    """Bar chart of the features most correlated with the target."""
    import matplotlib.pyplot as plt

    ranked = rank_by_correlation(summary)[:top_k]
    names = [s["name"] for s in ranked][::-1]
    corrs = [s["corr_target"] for s in ranked][::-1]
    colors = ["tab:red" if c > 0 else "tab:blue" for c in corrs]
    fig, ax = plt.subplots(figsize=(6, 0.25 * top_k + 1))
    ax.barh(names, corrs, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with MICHD (special codes excluded)")
    ax.set_title(f"Top {top_k} features by |correlation|")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature(data, name, path, max_categories=CATEGORICAL_MAX_UNIQUE):
    """Distribution of one feature, split by class.

    Categorical features are shown as a bar chart of the positive rate per
    value; continuous features as two normalized histograms.
    """
    import matplotlib.pyplot as plt

    d = data["feature_names"].index(name)
    column, y = data["x_train"][:, d], data["y_train"]
    observed = ~np.isnan(column)
    values, y_observed = column[observed], y[observed]

    fig, ax = plt.subplots(figsize=(6, 4))
    unique = np.unique(values)
    if len(unique) <= max_categories:
        unique, counts, rates = positive_rate_by_value(values, y_observed)
        bars = ax.bar([f"{v:g}" for v in unique], rates, color="tab:red")
        # sample size of each category, written above its bar
        ax.bar_label(bars, labels=[f"n={c}" for c in counts], fontsize=7)
        ax.axhline(y.mean(), color="black", linestyle="--", label="overall rate")
        ax.set_ylabel("Positive rate (MICHD)")
        ax.legend()
    else:
        # clip at the 1st/99th percentiles so that special codes such as 9999
        # do not squash the histogram
        low, high = np.percentile(values, [1, 99])
        if np.all(values == np.round(values)) and high - low <= 100:
            # one bin per integer, otherwise integer data gives a comb pattern
            bins = np.arange(low, high + 2) - 0.5
        else:
            bins = np.linspace(low, high, 40)
        for label, color in ((0, "tab:blue"), (1, "tab:red")):
            ax.hist(
                values[y_observed == label],
                bins=bins,
                density=True,
                alpha=0.5,
                color=color,
                label=f"MICHD = {label}",
            )
        ax.set_ylabel("Density")
        ax.legend()
    ax.set_xlabel(name)
    ax.set_title(f"{name} ({observed.mean():.0%} observed)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data", help="folder with the csv files")
    parser.add_argument("--out", default="exploration_output", help="output folder")
    parser.add_argument("--feature", help="deep dive into one feature (by name)")
    parser.add_argument(
        "--sub-sample", action="store_true", help="use 1/50 of the training rows"
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")  # write figures to files, no window

    figures_dir = os.path.join(args.out, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    data = load_data(args.data, sub_sample=args.sub_sample)

    if args.feature:
        print_feature_report(data, args.feature)
        path = os.path.join(figures_dir, f"feature_{args.feature}.png")
        plot_feature(data, args.feature, path)
        print(f"\nFigure written to {path}")
        return

    summary = summarize_features(
        data["x_train"], data["y_train"], data["x_test"], data["feature_names"]
    )
    print_section("1. Overview")
    print_overview(data, summary)

    print_section("2. Data quality")
    print_data_quality(data, summary)

    print_section("3. Informative missingness")
    print_informative_missingness(summary)

    print_section("4. Redundant features")
    pairs = find_redundant_pairs(data["x_train"], summary, data["feature_names"])
    print_redundant_pairs(pairs)

    print_section("5. Known risk factors")
    print_risk_factors(data)

    print_section("6. Trivial baselines")
    print_baselines(data["y_train"])

    summary_path = os.path.join(args.out, "feature_summary.csv")
    pairs_path = os.path.join(args.out, "redundant_pairs.csv")
    save_summary_csv(summary, summary_path)
    save_pairs_csv(pairs, pairs_path)
    plot_missingness(summary, os.path.join(figures_dir, "missingness.png"))
    plot_top_correlations(summary, os.path.join(figures_dir, "top_correlations.png"))
    print(f"\nTables written to {summary_path} and {pairs_path}")
    print(f"Figures written to {figures_dir}/")


if __name__ == "__main__":
    main()
