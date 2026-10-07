# -*- coding: utf-8 -*-
"""Feature preprocessing for the BRFSS dataset.

The ``Preprocessor`` learns every statistic it needs (special codes, medians,
categories, means, standard deviations, ...) with ``fit`` on the training data
only, and applies them with ``transform`` to any data (training, validation or
test). Inside cross-validation it must be fitted on the training folds only
(see ``make_fit_predict``).

Steps, in order (each can be switched on or off for the ablation study):

1. Drop the survey administration features (interview date, record numbers,
   sampling weights, ...) and the raw features with mixed units (WEIGHT2,
   HEIGHT3), which have clean CDC-computed versions (WTKG3, HTM4, _BMI5).
2. Map the BRFSS special codes: "don't know" / "refused" (7, 9, 77, 99, ...)
   become missing and "none" (88, 888, ...) becomes 0. A code is only mapped
   for the features where it is isolated (see ``find_special_codes``).
3. Add missing-value indicators: a 0/1 column telling whether a feature was
   missing (some questions are only asked to some people, e.g. the diabetes
   module, so missingness is informative). Identical indicators are merged.
4. Drop the features that are constant or missing for more than
   ``max_missing`` of the training samples.
5. Drop redundant features: of each pair with |correlation| above
   ``redundancy_threshold``, the second one is dropped.
6. Impute the remaining missing values with the training median.
7. One-hot encode the categorical features (3 to ``max_categories`` integer
   values); binary and continuous features stay numerical.
8. Standardize every column with the training mean and standard deviation.

Running the module as a script compares preprocessing variants with ridge
regression under the common cross-validation protocol:

    python preprocessing.py
    python preprocessing.py --sub-sample
"""

import argparse

import numpy as np

from data_exploration import ADMIN_FEATURES, find_special_codes

# Raw answers mixing imperial and metric units; the CDC computed clean versions
# of them (WTKG3 = weight in kg, HTM4 = height in m, _BMI5 = body mass index).
RAW_MIXED_UNIT_FEATURES = ("WEIGHT2", "HEIGHT3")

# Special codes meaning "none" (e.g. PHYSHLTH = 88: no day of poor health),
# mapped to 0. The other special codes ("don't know", "refused") become
# missing, except 8, whose meaning depends on the question (e.g. "never" for
# CHECKUP1, a valid category elsewhere): it is left unchanged.
NONE_CODES = (88, 888, 8888)
UNCHANGED_CODES = (8,)

DEFAULT_OPTIONS = {
    "drop_admin": True,
    "map_special_codes": True,
    "missing_indicators": True,
    "max_missing": 0.9,
    "redundancy_threshold": 0.95,
    "one_hot": True,
    "max_categories": 15,
}


class Preprocessor:
    """Fit preprocessing statistics on training data and apply them.

    Args:
        feature_names: list of D str, the names of the raw features.
        **options: overrides of ``DEFAULT_OPTIONS``. ``redundancy_threshold``
            and ``max_missing`` can be set to None to disable the step.
    """

    def __init__(self, feature_names, **options):
        unknown = set(options) - set(DEFAULT_OPTIONS)
        if unknown:
            raise ValueError(f"unknown options: {sorted(unknown)}")
        self.feature_names = list(feature_names)
        self.options = {**DEFAULT_OPTIONS, **options}

    # -- fitting ------------------------------------------------------------

    def fit(self, x):
        """Learn the preprocessing statistics from the training features.

        Args:
            x: numpy array of shape (N, D), the raw training features.

        Returns:
            self
        """
        opts = self.options

        # 1. features dropped by name
        dropped = set()
        if opts["drop_admin"]:
            dropped |= set(ADMIN_FEATURES) | set(RAW_MIXED_UNIT_FEATURES)
        self.columns = [
            d for d, name in enumerate(self.feature_names) if name not in dropped
        ]
        x = x[:, self.columns]

        # 2. special codes, found on the training data of each feature
        self.special_codes = []
        if opts["map_special_codes"]:
            for d in range(x.shape[1]):
                values = x[~np.isnan(x[:, d]), d]
                codes = find_special_codes(values) if len(values) else []
                self.special_codes.append(
                    [c for c in codes if c not in UNCHANGED_CODES]
                )
            x = self._map_special_codes(x)

        # 3. missing-value indicators, for features neither always nor never
        # missing; identical indicator columns are kept once
        self.indicator_columns = []
        if opts["missing_indicators"]:
            missing = np.isnan(x)
            frac = missing.mean(axis=0)
            candidates = np.flatnonzero((frac > 0.01) & (frac < 0.99))
            _, first = np.unique(missing[:, candidates], axis=1, return_index=True)
            self.indicator_columns = list(candidates[np.sort(first)])
        indicators = np.isnan(x[:, self.indicator_columns]).astype(float)

        # 4. constant or too sparse features
        observed = ~np.isnan(x)
        keep = []
        for d in range(x.shape[1]):
            values = x[observed[:, d], d]
            too_sparse = (
                opts["max_missing"] is not None
                and 1 - observed[:, d].mean() > opts["max_missing"]
            )
            if len(np.unique(values)) > 1 and not too_sparse:
                keep.append(d)
        self.kept = keep
        x = x[:, keep]

        # 6. (computed before 5, which needs complete data) training medians
        self.medians = np.nanmedian(x, axis=0)
        x = np.where(np.isnan(x), self.medians, x)

        # 5. redundant features
        if opts["redundancy_threshold"] is not None:
            corr = np.abs(np.corrcoef(x.T))
            redundant = set()
            for i in range(corr.shape[0]):
                if i in redundant:
                    continue
                partners = np.flatnonzero(
                    corr[i, i + 1 :] > opts["redundancy_threshold"]
                )
                redundant |= set((partners + i + 1).tolist())
            not_redundant = [i for i in range(x.shape[1]) if i not in redundant]
            self.kept = [self.kept[i] for i in not_redundant]
            self.medians = self.medians[not_redundant]
            x = x[:, not_redundant]

        # 7. categorical features and their categories
        self.categories = {}
        if opts["one_hot"]:
            for d in range(x.shape[1]):
                unique = np.unique(x[:, d])
                is_integer = np.all(unique == np.round(unique))
                if is_integer and 3 <= len(unique) <= opts["max_categories"]:
                    self.categories[d] = unique

        # 8. standardization statistics of the final columns
        x = self._encode(x, indicators)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1
        return self

    # -- transforming -------------------------------------------------------

    def transform(self, x):
        """Apply the fitted preprocessing to raw features.

        Args:
            x: numpy array of shape (M, D), raw features (same columns as fit).

        Returns:
            numpy array of shape (M, D_out), the preprocessed features
            (without bias column, see ``add_bias``).
        """
        x = x[:, self.columns]
        if self.options["map_special_codes"]:
            x = self._map_special_codes(x)
        indicators = np.isnan(x[:, self.indicator_columns]).astype(float)
        x = x[:, self.kept]
        x = np.where(np.isnan(x), self.medians, x)
        x = self._encode(x, indicators)
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        """Fit on x, then transform x."""
        return self.fit(x).transform(x)

    def _map_special_codes(self, x):
        """Map special codes: "none" -> 0, the others -> missing."""
        x = x.copy()
        for d, codes in enumerate(self.special_codes):
            for code in codes:
                is_code = x[:, d] == code
                x[is_code, d] = 0 if code in NONE_CODES else np.nan
        return x

    def _encode(self, x, indicators):
        """One-hot encode the categorical columns and append the indicators.

        Values unseen during fit get all-zero one-hot columns.
        """
        blocks = []
        for d in range(x.shape[1]):
            if d in self.categories:
                blocks.append((x[:, [d]] == self.categories[d]).astype(float))
            else:
                blocks.append(x[:, [d]])
        blocks.append(indicators)
        return np.hstack(blocks)

    def describe(self):
        """Short text summary of the fitted preprocessing."""
        n_codes = sum(1 for codes in self.special_codes if codes)
        n_one_hot = sum(len(c) for c in self.categories.values())
        n_numeric = len(self.kept) - len(self.categories)
        return (
            f"{len(self.feature_names)} raw features -> {len(self.mean)} columns: "
            f"{n_numeric} numerical, {len(self.categories)} categorical "
            f"({n_one_hot} one-hot columns), "
            f"{len(self.indicator_columns)} missing indicators; "
            f"special codes mapped in {n_codes} features"
        )


def add_bias(x):
    """Prepend a column of ones (intercept) to the features."""
    return np.c_[np.ones(len(x)), x]


# ---------------------------------------------------------------------------
# Comparison of preprocessing variants
# ---------------------------------------------------------------------------


def make_fit_predict(feature_names, lambda_=1e-6, **options):
    """Build a ridge regression ``fit_predict`` for ``cross_validate``.

    The preprocessing is fitted on the training folds only.

    Args:
        feature_names: list of D str.
        lambda_: float, the ridge regularization parameter.
        **options: preprocessing options (see ``DEFAULT_OPTIONS``).

    Returns:
        function (x_train, y_train, x_val) -> scores of shape (len(x_val),).
    """
    from implementations import ridge_regression

    def fit_predict(x_train, y_train, x_val):
        preprocessor = Preprocessor(feature_names, **options).fit(x_train)
        tx_train = add_bias(preprocessor.transform(x_train))
        w, _ = ridge_regression(y_train, tx_train, lambda_)
        return add_bias(preprocessor.transform(x_val)).dot(w)

    return fit_predict


# Each variant removes one step from the full preprocessing ("all steps"), so
# that the drop in F1 measures the contribution of that step.
VARIANTS = [
    ("all steps", {}),
    ("without admin drop", {"drop_admin": False}),
    ("without special codes", {"map_special_codes": False}),
    ("without missing indicators", {"missing_indicators": False}),
    ("without sparse drop", {"max_missing": None}),
    ("without redundancy drop", {"redundancy_threshold": None}),
    ("without one-hot", {"one_hot": False}),
    (
        "minimal (impute + scale)",
        {
            "drop_admin": False,
            "map_special_codes": False,
            "missing_indicators": False,
            "max_missing": None,
            "redundancy_threshold": None,
            "one_hot": False,
        },
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data", help="folder with the csv files")
    parser.add_argument(
        "--sub-sample", action="store_true", help="use 1/50 of the training rows"
    )
    args = parser.parse_args()

    from cross_validation import cross_validate, format_result
    from data_exploration import load_data

    data = load_data(args.data, sub_sample=args.sub_sample)
    x, y, names = data["x_train"], data["y_train"], data["feature_names"]
    print(Preprocessor(names).fit(x).describe())
    print()

    lines = []
    for name, options in VARIANTS:
        print(name)
        fit_predict = make_fit_predict(names, **options)
        result = cross_validate(fit_predict, x, y, verbose=False)
        lines.append(format_result(name, result))
        print("  " + lines[-1])

    print("\nRidge regression (lambda = 1e-6), stratified 5-fold cross-validation:")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
