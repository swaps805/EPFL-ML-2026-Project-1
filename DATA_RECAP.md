# Data recap

Key facts about the BRFSS dataset and what they imply for preprocessing and
modelling. All numbers are reproduced by `python data_exploration.py` (see the
README); feature meanings and codes are in the
[BRFSS 2015 codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf).

## At a glance

| | |
|---|---|
| Training set | 328,135 people × 321 features |
| Test set | 109,379 people |
| Target | `_MICHD` (coronary heart disease or heart attack), labels in {-1, 1} |
| Positive rate | **8.8%** (strong class imbalance) |
| Feature types (inferred) | 14 binary, 209 categorical, 92 continuous, 6 constant |
| Missing entries | **44.8%** of the whole table |
| Train/test shift | none detected on missing rates |

## 1. Class imbalance

- Only 8.8% positives: accuracy is misleading. Predicting "always negative"
  gives **91.2% accuracy** and is useless.
- Predicting "always positive" gives **F1 = 0.162**: the floor any model must
  beat if we evaluate with F1.
- → Use F1 (with precision and recall) as the main metric, and tune the
  decision threshold or weight the classes.

## 2. Missing values

- 182 features have > 10% missing, 147 have > 50%, **99 have > 90%** and 30
  have > 99% (mostly optional modules: vision `VI*`, asthma `AS*`, prostate
  cancer `PC*`).
- Only 115 features have < 5% missing.
- Each person has a median of 43% missing answers.
- **Missingness is informative.** Some modules are only asked depending on a
  previous answer. Example: the diabetes module (`DIABAGE2`, `INSULIN`,
  `BLDSUGAR`, ...) is only asked to diabetics, and the positive rate is
  **22% when it is filled vs 8% when it is missing**.
- → Do not blindly drop sparse features; consider missing-value indicators.

## 3. Special codes and units

- 241 features contain BRFSS special codes that are **not real values**:
  - `7`, `77`, `777`, ... = "don't know / not sure";
  - `9`, `99`, `999`, ... = "refused";
  - `8`, `88`, ... = "none", i.e. **0** (e.g. `PHYSHLTH = 88` means 0 days of
    poor physical health, not 88 days).
- Left as is, a linear model reads "refused" (9) as worse than "poor" (5).
- **Mixed units:** `WEIGHT2` mixes pounds (50-999) and kilograms stored as
  9000 + kg (677 answers); `HEIGHT3` has the same issue (1,357 metric answers).
  Raw `WEIGHT2` has almost no correlation with the target. The CDC computed
  versions are cleaner (`_BMI5`: BMI from 12 to 98, median 26.9).
- **Answer coding:** for most yes/no questions, `1 = yes` and `2 = no`, so a
  negative correlation usually means that answering "yes" increases the risk.
- → Special codes must be checked against the codebook; the automatic
  detection has a few false positives (e.g. `_STATE = 8` is Colorado).

## 4. Useless or redundant features

- **6 constant features:** `CTELENUM`, `COLGHOUS`, `STATERES`, `CTELNUM1`,
  `CELLFON2`, `CCLGHOUS`.
- **Survey administration features** (state, interview date, record numbers
  `SEQNO`/`_PSU`, sampling weights `_LLCPWT`, ...): all have |correlation|
  < 0.05 with the target. They describe the interview, not the person.
- **26 pairs of near-duplicate features** (|correlation| > 0.95, listed in
  `exploration_output/redundant_pairs.csv`). The CDC added computed versions
  (prefix `_`) of raw answers, e.g. age in three versions (`_AGEG5YR`,
  `_AGE80`, `_AGE_G`), height in inches and meters (`HTIN4`, `HTM4`), several
  fruit and vegetable variants.
- → Drop constants and administrative features, keep one feature per
  redundant group (duplicates make least squares and ridge unstable).

## 5. What predicts the target

Positive rate per answer for known cardiovascular risk factors (overall 8.8%):

| Risk factor | Positive rate |
|---|---|
| Age (`_AGEG5YR`), 18-24 → 80+ | 0.5% → **22%**, steadily increasing |
| General health (`GENHLTH`), excellent → poor | 2% → **32%** |
| Previous stroke (`CVDSTRK3`) | **37%** vs 8% |
| Kidney disease (`CHCKIDNY`) | 29% vs 8% |
| COPD (`CHCCOPD1`) | 27% vs 7% |
| Difficulty walking (`DIFFWALK`) | 22% vs 6% |
| Diabetes (`DIABETE3`) | 22% vs 7% |
| High blood pressure (`BPHIGH4`) | 16% vs 4% |
| High cholesterol (`TOLDHI2`) | 16% vs 5% |
| Income (`INCOME2`), lowest → highest | 13-16% → 5% |
| Smoked ≥ 100 cigarettes (`SMOKE100`) | 12% vs 6% |
| Sex (`SEX`), men vs women | 11% vs 7% |
| Physical activity (`EXERANY2`), none vs some | 13% vs 7.5% |

- These are the known cardiovascular risk factors: the data is consistent.
- The strongest |correlation| is only about **0.25**: no single feature is
  enough, the model has to combine many weak signals.
- Many features are **ordinal** (age group, general health, income, education)
  and can stay numerical once special codes are removed; nominal features
  (state, race, ...) need one-hot encoding.

## Implications for preprocessing

1. Drop constant, administrative and redundant features.
2. Map special codes: "don't know" / "refused" → missing, "none" → 0; fix
   units (or use the CDC computed features such as `_BMI5`).
3. Handle missing values: drop features that are too sparse unless their
   missingness is informative, impute (median / mode), add missing indicators.
4. One-hot encode nominal features, then standardize.
5. Evaluate with F1 under stratified cross-validation and tune the decision
   threshold.
