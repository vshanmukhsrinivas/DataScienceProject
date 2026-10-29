# Test 4: Text-Based Prediction of Review Helpfulness

## 1. Objective

This experiment evaluates whether review text provides predictive
information about helpfulness beyond structured review features. It
compares a structured-feature baseline with a model that adds TF-IDF
representations of the review summary and review text.

The outcome is the helpfulness rate:

`helpfulness rate = helpfulness numerator / helpfulness denominator`

Only reviews with a valid helpfulness rate are eligible. Reviews with a
zero denominator or an invalid numerator/denominator relationship are
excluded because their outcome is not valid.

## 2. Data split and leakage prevention

The experiment uses the pre-created exploration and confirmation splits.

  Dataset          Eligible reviews                 Products
  -------------- ------------------ ------------------------
  Exploration               239,569   Used for model fitting
  Confirmation               58,726                    9,794

The TF-IDF vectorizer is fitted on exploration data only and then
applied to the confirmation data. The confirmation set is held out for
final evaluation, rather than being used to fit the vocabulary or model.

Uncertainty is estimated by paired product-level bootstrap resampling,
rather than resampling individual reviews, to account for dependence
among reviews of the same product.

## 3. Features and models

### Structured baseline

The baseline model uses these structured features:

-   Rating (`Score`)
-   Review length in characters
-   Summary length in characters
-   Review year

### Text-enhanced model

The text-enhanced model uses the same structured features and adds
TF-IDF features derived from the review summary and review text.

TF-IDF configuration:

-   Maximum features: 20,000
-   Minimum document frequency: 5
-   N-gram range: unigrams and bigrams `(1, 2)`
-   English stop-word removal
-   Feature data type: `float32`

A Ridge regression model with `alpha = 1.0` is used. Predictions are
clipped to the valid helpfulness-rate range `[0, 1]`.

## 4. Evaluation metric

Performance is evaluated on the confirmation set using mean absolute
error (MAE):

`MAE = (1/n) * sum(abs(y_i - predicted_y_i))`

Lower MAE means predictions are, on average, closer to observed
helpfulness rates.

The primary comparison is the paired MAE reduction:

`MAE reduction = baseline MAE - text-model MAE`

A positive value means the text-enhanced model performs better. The
practical-improvement threshold used in the analysis is an absolute MAE
reduction of `0.01`.

## 5. Results

  Metric                                                           Result
  -------------------------------------------- --------------------------
  Baseline MAE                                                  0.2487049
  Text-enhanced model MAE                                       0.2186708
  Absolute MAE reduction                                        0.0300341
  Relative MAE reduction                                           12.08%
  95% product-bootstrap CI for MAE reduction     \[0.0262011, 0.0340414\]
  Practical-improvement threshold                                    0.01

The 95% product-bootstrap confidence interval for the MAE reduction lies
entirely above both zero and the practical-improvement threshold of
`0.01`. This supports the conclusion that adding TF-IDF text features
improves held-out predictive performance by a practically meaningful
amount under this evaluation setup.

The analysis used 2,000 paired product-bootstrap replicates with random
seed `42`. The reported empirical one-sided bootstrap p-value was
approximately `0.00049975`. Because this value is estimated using a
finite number of bootstrap replicates, the confidence interval is the
primary uncertainty summary.

## 6. Interpretation

On the held-out confirmation data, the text-enhanced model reduced MAE
from approximately `0.249` to `0.219`, an absolute reduction of
approximately `0.030` and a relative reduction of `12.08%`.

This result indicates that review summary and review body text contain
predictive signal for helpfulness beyond the included structured
features. It is a predictive comparison, not a causal claim: it does not
establish that text length or any particular word causes a review to
receive helpful votes.

## 7. Limitations

-   The outcome is defined only for reviews with valid helpfulness
    numerator and denominator values; findings may not generalize to
    reviews without votes or with invalid vote metadata.
-   The result depends on the chosen split, feature configuration, Ridge
    regularization value, and evaluation metric.
-   Product-level bootstrap intervals address clustering by product but
    do not remove all possible sources of dataset bias.
-   The comparison demonstrates predictive value from the combined text
    representation; it does not identify which individual words or
    phrases drive the improvement.
-   The empirical bootstrap p-value has limited resolution because it
    uses a finite number of replicates. The confidence interval is
    emphasized for interpretation.

## 8. Reproducibility

Experiment implementation:

-   `scripts/tests/test4_text_prediction.py`

Machine-readable results:

-   `reports/test4_text_prediction.json`

The TF-IDF vectorizer and feature matrix are generated artifacts and are
excluded from Git. On a fresh checkout, rerun the data preparation,
split, and feature-generation pipeline before running the experiment.

## 9. Conclusion

The text-enhanced model achieved a confirmation-set MAE of `0.2187`,
compared with `0.2487` for the structured baseline. The estimated MAE
reduction was `0.0300` (12.08%), with a 95% product-bootstrap confidence
interval of `[0.0262, 0.0340]`. Because the interval is entirely above
the practical threshold of `0.01`, the results support a meaningful
predictive improvement from adding TF-IDF review text features.
