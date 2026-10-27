# Pre-Analysis Plan: Amazon Fine Food Review Helpfulness

## 1. Research problem

**Primary problem statement:** Are rating extremity and review length associated with the proportion of helpfulness votes a review receives, after accounting for other observable review characteristics?

**Secondary problem statement:** Does review text improve the prediction of review helpfulness beyond structured features such as rating, review length, summary length, and review year?

These questions go beyond describing which ratings or reviews have higher average helpfulness. They investigate adjusted associations and whether textual information provides additional predictive value.

This is an observational dataset. Results will describe associations and predictive performance, not causal effects.

## 2. Data and analysis policy

- Use the cleaned, product-grouped exploration split for exploratory analysis, model development, and the planned inferential analyses.
- Keep the product-grouped confirmation split separate until model choices and tuning are finalized.
- Use a fixed random seed of 42 where resampling is required.
- Define helpful votes as `HelpfulnessNumerator` and total votes as `HelpfulnessDenominator`.
- A review is eligible for the primary vote-based analysis when the vote counts are nonnegative, the denominator is positive, and helpful votes do not exceed total votes.
- Reviews with zero votes have undefined helpfulness rates and will not be assigned a rate of zero.
- Invalid vote records will be documented and excluded from analyses requiring valid vote counts.
- Report sample sizes and the number of products represented in every analysis.
- Use product-aware resampling or uncertainty estimates where feasible, because reviews from the same product may not be independent.

## 3. Significance level and multiplicity

The nominal significance level is alpha = 0.05.

Tests 1–3 form the confirmatory family of explanatory hypotheses. Their p-values will be adjusted using the Holm procedure to control the family-wise error rate at 0.05. Test 4 is a predictive comparison and will be reported separately with a 95% confidence interval for the change in predictive error.

All hypotheses, effect thresholds, and analysis decisions will be documented before examining the corresponding inferential results.

## 4. Hypothesis tests

### Test 1 — Primary: Rating extremity

**Question:** Is rating extremity positively associated with helpfulness after adjusting for review length, summary length, and review year?

- Null hypothesis (H0): The adjusted rating-extremity coefficient equals zero.
- Alternative hypothesis (H1): The adjusted rating-extremity coefficient is greater than zero.
- Direction: One-sided.
- Significance level: 0.05 before multiplicity adjustment.
- Primary method: Binomial regression using helpful and unhelpful vote counts, with a logit link, subject to model diagnostics.
- Predictor: `RatingExtremity = abs(Score - 3)`.
- Adjustment variables: review length, summary length, and review year.
- Practical-effect threshold: An adjusted odds ratio of 1.05 per one-unit increase in rating extremity is the proposed minimum meaningful effect. This threshold must be justified and finalized before testing.

The primary estimate will be the adjusted odds ratio with a 95% confidence interval. The report will also translate the estimate into predicted helpfulness probabilities for interpretable review profiles.

Alternatives considered: categorical star-rating indicators, endpoint versus middle ratings, and a two-sided coefficient test. The absolute-distance measure is the primary specification because it represents the pre-specified hypothesis that ratings farther from the midpoint are associated with greater helpfulness. Categorical and endpoint specifications will be sensitivity analyses, not replacements chosen based on significance.

### Test 2 — Review length

**Question:** Is review length positively associated with helpfulness after adjusting for rating extremity, summary length, and review year?

- H0: The adjusted review-length coefficient equals zero.
- H1: The adjusted review-length coefficient is greater than zero.
- Direction: One-sided.
- Significance level: 0.05 before multiplicity adjustment.
- Method: Binomial regression using helpful and unhelpful vote counts.
- Predictor: log-transformed review length, using `log1p(ReviewLengthChars)`.
- Adjustment variables: rating extremity, summary length, and review year.
- Practical-effect threshold: A 5% increase in the odds of a helpful vote across an interquartile-range increase in review length is the proposed minimum meaningful effect; finalize this threshold before testing.

Report the coefficient, odds ratio for the specified length contrast, 95% confidence interval, test statistic, degrees of freedom where applicable, p-value, and adjusted decision.

Alternatives considered: raw length in a linear term, length quantile groups, and nonlinear splines. The log transformation is proposed to reduce the influence of the long right tail while preserving information from genuine long reviews. A nonlinear specification will be considered as a sensitivity analysis.

### Test 3 — Additional information from summary length

**Question:** Does summary length improve the explanatory model beyond rating extremity, review length, and review year?

- H0: The summary-length term contributes no additional explanatory information; its coefficient is zero in the pre-specified nested comparison.
- H1: The summary-length term contributes additional explanatory information.
- Direction: Two-sided / omnibus.
- Significance level: 0.05 before multiplicity adjustment.
- Method: Likelihood-ratio test comparing nested binomial regression models, subject to appropriate model assumptions and diagnostics.
- Practical-effect threshold: An improvement corresponding to at least a 1% reduction in the model's held-out binomial deviance is proposed as a practical benchmark; finalize the benchmark before testing.

The estimate will include the change in deviance, the likelihood-ratio statistic, degrees of freedom, p-value, and a confidence interval for a pre-specified interpretable summary-length contrast where estimable.

Alternatives considered: a Wald test for one coefficient and a test using grouped summary lengths. The nested-model comparison is proposed because it directly asks whether adding the feature improves fit. Grouping a continuous variable discards information.

### Test 4 — Does text improve prediction?

**Question:** Does adding TF-IDF text features improve out-of-sample prediction compared with structured metadata alone?

- Null hypothesis (H0): The expected reduction in mean absolute error from adding TF-IDF features is zero.
- Alternative hypothesis (H1): Adding TF-IDF features reduces mean absolute error.
- Direction: One-sided.
- Significance level: 0.05.
- Method: Paired comparison of prediction errors on the same held-out confirmation reviews, with a product-aware bootstrap confidence interval and p-value where feasible.
- Baseline: Structured-feature model.
- Comparison model: The same structured features plus TF-IDF features.
- Primary metric: Mean absolute error (MAE) on review-level helpfulness rates, with a clearly defined eligible evaluation population.
- Practical-effect threshold: An absolute MAE reduction of 0.01 (one percentage point in helpfulness rate) is the proposed minimum meaningful improvement; finalize before evaluating the confirmation set.

Report both models' MAE, the paired MAE difference, a 95% confidence interval, the test statistic and p-value for the selected bootstrap procedure, and whether the improvement reaches the practical threshold.

Alternatives considered: RMSE and R-squared. MAE is selected as the primary metric because it is interpretable in helpfulness-rate units and less sensitive to large individual errors than RMSE. RMSE and R-squared may be reported as secondary descriptive metrics.

## 5. Assumptions and diagnostics

Before interpreting inferential results:

1. Check convergence, separation, influential observations, and whether the chosen binomial model is appropriate.
2. Examine overdispersion and dependence between reviews from the same product.
3. Use robust or product-aware uncertainty estimates where justified.
4. Document how missing or invalid values are handled.
5. Check whether results are sensitive to vote-count thresholds and alternative rating specifications.
6. Do not interpret statistical significance alone as practical importance.

If the binomial model's assumptions are not adequate, document the problem and use a justified alternative rather than silently changing methods.

## 6. Confirmation-set policy

The confirmation split will be used only after the feature set, model specification, and tuning decisions have been finalized using the exploration data. Text vectorization must be fitted on exploration data only and then applied unchanged to confirmation data.

No model or hypothesis will be selected because it performs best on the confirmation set. Report the final confirmation performance and its uncertainty once the comparison is specified.

## 7. Interpretation

All conclusions are associational or predictive. This dataset is observational and cannot establish that changing a rating, review length, summary length, or text causes helpfulness to change. Causal claims would require a defensible causal design, suitable assumptions, and additional evidence.