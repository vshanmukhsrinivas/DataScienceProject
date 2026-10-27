# Test 3 Documentation: Additional Explanatory Value of Summary Length

## 1. Research Question

Does review summary length provide additional explanatory information about review helpfulness after accounting for rating extremity, review length, and review year?

The analysis uses the Amazon Fine Food Reviews dataset and investigates adjusted associations rather than causal effects.

## 2. Hypotheses and Significance Level

- **Null hypothesis (\(H_0\)):** The summary-length coefficient equals zero after adjusting for the other predictors.
- **Alternative hypothesis (\(H_1\)):** The summary-length coefficient differs from zero.
- **Test direction:** Two-sided.
- **Nominal significance level:** \(\alpha=0.05\).
- **Multiplicity adjustment:** Holm adjustment across explanatory Tests 1–3.

The practical-effect benchmark specified in the pre-analysis plan is a relative reduction of at least 1% in model deviance when summary length is added.

## 3. Data and Eligibility

The analysis used the cleaned exploration split, `data/splits/exploration.csv.gz`.

Reviews were eligible when:
- Helpful-vote counts were nonnegative.
- The total-vote denominator was positive.
- Helpful votes did not exceed total votes.
- The required predictors and product identifier were available.

The final analysis included:

- **Reviews:** 239,593
- **Products:** 39,522
- **Helpful votes:** 802,496
- **Unhelpful votes:** 223,816

Reviews with zero total votes were excluded because their helpfulness proportion is undefined. The confirmation split was not used for this explanatory analysis.

## 4. Variables and Model Specification

The response was represented as grouped binomial counts:

- Helpful votes: `HelpfulnessNumerator`
- Unhelpful votes: `HelpfulnessDenominator - HelpfulnessNumerator`

The baseline model included:
- `RatingExtremity = abs(Score - 3)`
- `LogReviewLength = log1p(ReviewLengthChars)`
- `ReviewYear`

The extended model added:
- `SummaryLengthChars`

Continuous predictors were centered before model fitting. Both models used the same observations and grouped-binomial response, with a logit link.

## 5. Primary Planned Analysis: Likelihood-Ratio Test

The pre-analysis plan specified a likelihood-ratio test comparing the nested baseline and extended models.

| Quantity | Result |
|---|---:|
| Baseline deviance | 532,733.789 |
| Extended deviance | 532,637.365 |
| Likelihood-ratio statistic | 96.424 |
| Degrees of freedom | 1 |
| Likelihood-ratio p-value | \(9.27\times10^{-23}\) |
| Relative deviance reduction | 0.0181% |
| Practical-effect threshold | 1% |
| Threshold reached | No |

The conventional likelihood-ratio test rejects its null hypothesis at the nominal 0.05 level. However, it relies on the standard binomial likelihood assumptions and does not account for product clustering in its reference distribution. Given the observed overdispersion diagnostic, this result should not be treated as the sole basis for inference.

## 6. Product-Clustered Robust Sensitivity Analysis

A separate product-clustered Wald test was conducted using the extended model's robust covariance matrix.

| Quantity | Result |
|---|---:|
| Summary-length coefficient per character | −0.001671 |
| Product-clustered standard error | 0.000511 |
| Wald chi-square statistic | 10.676 |
| Degrees of freedom | 1 |
| Two-sided p-value | 0.001085 |
| 95% coefficient confidence interval | [−0.002673, −0.000669] |
| Odds ratio per additional 10 characters | 0.9834 |
| 95% odds-ratio confidence interval | [0.9736, 0.9933] |

The robust Wald test rejects the null hypothesis at the nominal significance level. The negative coefficient indicates that, conditional on the included predictors, greater summary length is associated with slightly lower odds of a helpful vote.

The reported odds ratio of 0.9834 corresponds to approximately 1.66% lower odds per additional 10 summary characters. This is an adjusted association, not a causal effect.

The Wald statistic uses an asymptotic chi-square reference distribution with one degree of freedom.

## 7. Holm Multiplicity Adjustment

Test 3 was included in the planned family of three explanatory tests. Its product-clustered Wald p-value was used for the multiplicity adjustment.

- **Raw p-value:** 0.001085
- **Holm-adjusted p-value:** 0.001085
- **Decision at family-wise \(\alpha=0.05\):** Reject \(H_0\).

The adjusted result remains statistically significant. The multiplicity adjustment does not eliminate concerns about model misspecification or overdispersion.

## 8. Practical Significance and Interpretation

Although summary length is statistically significant, adding it reduces deviance by only 0.0181%, which is substantially below the pre-specified 1% benchmark.

Therefore, the results distinguish between two conclusions:

1. **Statistical significance:** Summary length has a detectable adjusted association with helpfulness in this dataset.
2. **Practical explanatory improvement:** Its additional explanatory contribution does not reach the pre-specified deviance-reduction threshold.

A statistically significant coefficient does not necessarily imply a substantial improvement in explanatory power or prediction.

## 9. Assumptions and Limitations

- The data are observational, so causal claims are not justified.
- Reviews from the same product may be dependent; product-clustered covariance was used for the robust sensitivity test.
- The earlier Pearson dispersion diagnostic was approximately 2.12, indicating possible extra-binomial variability.
- The likelihood-ratio test is a conventional binomial comparison and is not product-cluster robust.
- The clustered Wald test uses asymptotic inference.
- The 1% deviance-reduction threshold is a proposed project-specific practical benchmark, not a universal standard.
- The analysis examines summary length as a linear term; it does not rule out every possible nonlinear relationship.

## 10. Reproducibility

Run the following command from the project root:

```bash
python3 scripts/tests/test3_summary_length.py
```

The results are saved to:

`reports/test3_summary_length.json`

The multiplicity-adjusted results are saved to:

`reports/holm_adjustment.json`

The separate product-clustered spline sensitivity analysis for Test 2 is saved to:

`reports/test2_clustered_spline_wald.json`

## 11. Final Conclusion

Summary length has a statistically significant negative adjusted association with review helpfulness. This finding remains significant after the planned Holm adjustment. Nevertheless, the relative deviance reduction is only 0.0181%, far below the pre-specified 1% practical benchmark. The evidence therefore supports a statistically detectable association, but not a substantial improvement in explanatory fit from adding summary length alone.


### Assessment against the hypothesis-test requirements

Test 3 reports the pre-specified null and alternative hypotheses, two-sided significance level, planned nested-model likelihood-ratio test, and the alternative methods considered. The product-clustered Wald test provides a robust sensitivity analysis, with its test statistic, one degree of freedom, raw p-value, Holm-adjusted p-value, and 95% confidence interval reported.

The robust Wald test rejects the null hypothesis. The estimated odds ratio is 0.9834 per additional 10 summary characters, with a 95% confidence interval of [0.9736, 0.9933]. However, adding summary length reduces model deviance by only 0.0181%, below the pre-specified practical benchmark of 1%. Therefore, the evidence supports a statistically detectable adjusted association, but not a practically substantial improvement in explanatory fit under the selected benchmark.

The likelihood-ratio comparison uses conventional binomial likelihood assumptions and is not product-cluster robust. The clustered Wald result is therefore emphasized for inference, while the likelihood-ratio result is reported as the planned comparison with this limitation clearly stated.