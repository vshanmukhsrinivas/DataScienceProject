# Test 4 — Does Review Text Improve Helpfulness Prediction?

## 1. Research Question

Does adding TF-IDF features extracted from review summaries and bodies improve out-of-sample prediction of review helpfulness beyond structured review metadata?

## 2. Hypotheses and Decision Criteria

- **Null hypothesis (H₀):** Adding TF-IDF features does not reduce expected mean absolute error (MAE).
- **Alternative hypothesis (H₁):** Adding TF-IDF features reduces expected MAE.
- **Significance level:** α = 0.05.
- **Primary metric:** MAE of the predicted review-level helpfulness rate.
- **Minimum practically meaningful improvement:** An absolute MAE reduction of 0.01.
- **Uncertainty estimation:** Product-cluster bootstrap with 2,000 replicates and random seed 42.

## 3. Models and Methodology

The baseline model uses four structured features: star rating, review length in characters, summary length in characters, and review year. The comparison model uses these same features plus TF-IDF representations of review summaries and bodies, using unigrams and bigrams with a vocabulary limit of 20,000 features.

Both models use Ridge regression with α = 1.0. Predictions are clipped to the interval [0, 1], consistent with the helpfulness-rate target.

The TF-IDF vocabulary and inverse-document-frequency weights are fitted on the exploration data only and then applied unchanged to the confirmation data. The confirmation set contains products absent from the exploration set.

The target for each eligible review is:

helpfulness rate = helpful votes / total votes.

Only reviews with valid, positive-denominator vote counts are included. The confirmation evaluation contains 58,726 reviews across 9,794 products.

## 4. Results

| Metric | Result |
|---|---:|
| Baseline MAE | 0.248705 |
| TF-IDF model MAE | 0.218671 |
| Absolute MAE reduction | 0.030034 |
| Relative MAE reduction | 12.08% |
| Product-bootstrap 95% CI for reduction | [0.026201, 0.034041] |
| One-sided bootstrap p-value | 0.000500 |
| Bootstrap replicates | 2,000 |
| Practical threshold | 0.010000 |

The absolute MAE reduction is approximately 0.0300, exceeding the practical threshold of 0.01. The entire 95% confidence interval lies above zero and above the practical threshold.

## 5. Decision and Interpretation

The results support rejecting the null hypothesis at the 0.05 significance level. Adding TF-IDF features improved predictive performance on the held-out confirmation set, reducing MAE by approximately 12.08% relative to the structured-feature baseline.

This is evidence that review text contains useful predictive information beyond the four structured features considered here. It does not establish that text causes reviews to receive more helpfulness votes.

## 6. Limitations

The helpfulness rate is an observed proportion and can be noisy, especially for reviews with few votes. Product-level bootstrap resampling accounts for clustering at the product level, but the reported interval remains conditional on the evaluation design and the chosen models.

The bootstrap p-value is an empirical tail estimate based on 2,000 replicates, not an exact probability. Ridge regularization was fixed at α = 1.0 rather than tuned on the confirmation set. No claim is made that this is the optimal model or feature configuration.

## 7. Reproducibility

Run the analysis from the repository root:

`python3 scripts/tests/test4_text_prediction.py`

Results are saved to `reports/test4_text_prediction.json`. The script uses random seed 42 for bootstrap resampling and does not fit the TF-IDF vectorizer on confirmation text.