# Inference Audit — Tests 1–3

**Project:** Amazon Fine Food Reviews  
**Status:** Tests 1–3 completed with Holm adjustment; Test 4 completed on the confirmation partition. Statistical and design limitations remain documented below.  
**Significance level:** \(\alpha = 0.05\)

## 1. Test 1 — Rating Extremity

**Research question:** Are more extreme ratings associated with higher helpfulness, after adjusting for review length, summary length, and review year?

- **Null hypothesis:** \(H_0: \beta_{\text{extremity}}=0\)
- **Alternative hypothesis:** \(H_1: \beta_{\text{extremity}}>0\)
- **Primary method:** Grouped-binomial GLM with a logit link and product-clustered standard errors.
- **Sample:** 239,593 reviews across 39,522 products.
- **Coefficient:** 0.3494.
- **Cluster-robust standard error:** 0.01309.
- **Asymptotic z-statistic:** 26.697.
- **One-sided p-value:** \(2.57\times10^{-157}\).
- **Odds ratio per extremity unit:** 1.4182.
- **95% odds-ratio confidence interval:** [1.3823, 1.4550].
- **Pearson dispersion ratio:** 2.1196.

**Provisional conclusion:** The estimated association is positive and statistically significant under the specified asymptotic inference. The point estimate exceeds the pre-specified practical threshold of an odds ratio of 1.05 per extremity unit.

**Outstanding checks:** Document the asymptotic reference distribution and confidence-interval method; investigate possible model misspecification and overdispersion; apply the planned Holm correction across Tests 1–3; and avoid interpreting the association causally.

## 2. Test 2 — Review Length

**Research question:** Is review length positively associated with helpfulness after adjusting for rating extremity, summary length, and review year?

- **Null hypothesis:** \(H_0: \beta_{\text{length}}=0\)
- **Alternative hypothesis:** \(H_1: \beta_{\text{length}}>0\)
- **Primary method:** Grouped-binomial GLM with a logit link and product-clustered standard errors.
- **Sample:** 239,593 reviews across 39,522 products.
- **Coefficient on log review length:** 0.4508.
- **Cluster-robust standard error:** 0.01468.
- **Asymptotic z-statistic:** 30.702.
- **One-sided p-value:** \(2.70\times10^{-207}\).
- **Odds ratio across the observed review-length IQR:** 1.6409.
- **95% confidence interval for that odds ratio:** [1.5898, 1.6936].
- **Observed review-length IQR:** 196–590 characters.

**Provisional conclusion:** The estimated association is positive and statistically significant under the specified asymptotic inference. The point estimate exceeds the pre-specified practical threshold of an odds ratio of 1.05 across the observed IQR.

### Nonlinearity analysis

The earlier likelihood-ratio comparison favored the spline model, with an LR statistic of 1661.56 and an AIC reduction of approximately 1655.56. However, the ordinary likelihood-ratio p-value relies on standard binomial likelihood assumptions and does not account for product clustering.

A separate product-clustered Wald test produced:

- Wald statistic: 1707.77.
- Degrees of freedom: 4.
- Reported p-value: 0.0 due to numerical underflow; this is not an exact zero probability.

The Wald test rejects the null that all four spline basis coefficients are zero. This supports a nonzero spline contribution, but it does not directly establish that the spline model outperforms the linear model. The two models use different representations of review length.

**Outstanding checks:** Document the asymptotic reference distributions, apply Holm adjustment to the planned explanatory tests, and interpret the spline analysis as evidence about functional form rather than proof of superior prediction.

## 3. Overdispersion diagnostic

The Pearson dispersion ratio from the primary grouped-binomial model was approximately 2.12. A separate diagnostic multiplied the primary cluster-robust standard errors by \(\sqrt{2.1196}\), approximately 1.456.

This stress test retained very large z-statistics for Tests 1 and 2. However, it is **not a formal correction**: product-clustered standard errors already account for certain forms of within-product dependence, and an additional dispersion multiplier could double-count variability.

The inflated p-values must not be reported as corrected inference. The diagnostic is retained as a sensitivity exercise only.

## 4. GEE sensitivity analyses

Product-grouped GEE analyses returned estimates and standard errors almost identical to the primary GLM analyses. These analyses use essentially the same observations, predictors, response information, and product grouping.

They should be described as sensitivity analyses, not independent replications or definitive proof that overdispersion and model misspecification have been resolved.

## 5. Test 3 — Summary Length

**Status:** Completed.

- **Null hypothesis:** The summary-length coefficient equals zero.
- **Alternative hypothesis:** The summary-length coefficient differs from zero.
- **Significance level:** 0.05, two-sided.
- **Primary planned method:** Likelihood-ratio comparison of nested grouped-binomial GLMs.
- **Likelihood-ratio statistic:** 96.424, with 1 degree of freedom.
- **Conventional likelihood-ratio p-value:** 9.27 × 10⁻²³.
- **Product-clustered Wald sensitivity:** χ²(1) = 10.676, p = 0.001085.
- **Coefficient:** −0.001671 per summary character.
- **95% coefficient confidence interval:** [−0.002673, −0.000669].
- **Odds ratio per 10 additional characters:** 0.9834; 95% CI [0.9736, 0.9933].
- **Relative deviance reduction:** 0.0181%, below the proposed 1% practical benchmark.

The product-clustered Wald p-value was used for the Holm adjustment. The ordinary likelihood-ratio test is reported separately because it does not account for product clustering in its reference distribution. The association is statistically detectable but the deviance reduction is small. It does not establish causation.

## 6. Inference audit and limitations

- [x] Report the hypotheses, significance levels, methods, statistics, and confidence intervals for Tests 1–3.
- [x] Apply Holm adjustment to the three explanatory tests.
- [x] Conduct a product-clustered GEE sensitivity analysis for Tests 1 and 2.
- [x] Investigate overdispersion and document why the dispersion multiplier is only a stress test.
- [x] Conduct a spline-based review-length analysis and a product-clustered joint Wald test of the spline terms.
- [x] Complete the confirmation-set predictive comparison and product-level bootstrap interval for Test 4.
- [x] Document missingness, dependence, practical significance, and observational limitations.
- [ ] Verify that practical-effect thresholds were finalized before the corresponding results were examined.
- [ ] Complete the final report and verify the eight-page PDF limit and all course requirements.

The first three tests use product-clustered asymptotic inference, with one-sided alternatives for Tests 1–2 and a two-sided alternative for Test 3. The product-level bootstrap interval is the primary uncertainty summary for the Test 4 paired MAE improvement; its empirical tail p-value is approximate, not an exact null-test p-value.

The confirmation-set evaluation used 58,726 eligible reviews across 9,794 products. Because model settings and the timing of all practical thresholds have not been independently established here, do not claim that every choice was preregistered or that the confirmation set was untouched by all prior decisions.

## 7. Final Holm-adjusted results for Tests 1–3

The three explanatory tests were adjusted using the Holm step-down procedure at a family-wise significance level of 0.05. The adjustment used the product-clustered asymptotic p-values: one-sided tests for Tests 1 and 2, and a two-sided Wald test for Test 3.

| Test | Raw p-value | Holm-adjusted p-value | Decision |
|---|---:|---:|---|
| Test 1 — Rating extremity | \(2.57\times10^{-157}\) | \(5.13\times10^{-157}\) | Reject H0 |
| Test 2 — Review length | \(2.70\times10^{-207}\) | \(8.11\times10^{-207}\) | Reject H0 |
| Test 3 — Summary length | 0.001085 | 0.001085 | Reject H0 |

### Interpretation

- **Rating extremity:** More extreme ratings are positively associated with helpful-vote odds after adjustment for the included covariates.
- **Review length:** Longer reviews are positively associated with helpful-vote odds; the spline analysis indicates that the relationship may be nonlinear.
- **Summary length:** Summary length has a statistically detectable negative adjusted association, but its addition reduces deviance by only 0.0181%, below the proposed 1% practical benchmark.

The findings remain observational associations rather than causal effects. The Holm adjustment addresses multiplicity, but it does not resolve model misspecification, overdispersion, or dependence not adequately captured by the inference procedure.

The Test 3 likelihood-ratio p-value is reported separately as a conventional, non-cluster-robust comparison. The Holm-adjusted Test 3 result above uses the product-clustered Wald p-value.

## 8. Remaining work

- [ ] Verify whether practical-effect thresholds were finalized before examining the relevant results.
- [ ] Update the EDA and novelty document to remove stale completion notes and include final plot takeaways.
- [ ] Finalize the report with reproducible figures, methods, limitations, and code references.
- [ ] Check the report against the eight-page PDF limit and all project requirements.

