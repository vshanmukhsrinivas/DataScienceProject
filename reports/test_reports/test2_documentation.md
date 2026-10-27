# Test 2: Relationship Between Review Length and Review Helpfulness

## 1. Overview

### Research question

**Are longer Amazon food reviews associated with higher helpfulness, after accounting for rating extremity, summary length, and review year?**

The broader project investigates what makes an Amazon review helpful and whether review characteristics can explain or predict helpfulness.

Review length is an interesting characteristic because a longer review may provide more detail, evidence, context, and explanation. However, additional text may not always provide additional value. Very long reviews could contain unnecessary information, while short reviews may communicate useful information efficiently.

Therefore, this test investigates whether review length is positively associated with helpful-vote odds and whether a simple linear relationship adequately describes the association.

### Why investigate review length?

Review length is easy to measure and has a plausible relationship with perceived usefulness. A longer review may help readers understand the product, its advantages, and its limitations.

However, length alone does not guarantee quality. The analysis must determine whether length is associated with helpfulness after accounting for other observable characteristics.

---

## 2. Dataset and sample

### Data source

The analysis uses the Amazon Fine Food Reviews dataset from the Stanford Network Analysis Project (SNAP).

**Input file:** `data/splits/exploration.csv.gz`

The exploration split is used for analysis development. The confirmation split should remain untouched until the analysis choices are finalized.

### Sample used

| Quantity | Value |
|---|---:|
| Reviews in the exploration split | 454,951 |
| Reviews included in Test 2 | 239,593 |
| Unique products represented | 39,522 |
| Total helpful votes | 802,496 |
| Total unhelpful votes | 223,816 |
| Total valid votes | 1,026,312 |

The test includes reviews with valid helpfulness vote counts:

- The helpful-vote count is non-negative.
- The total-vote denominator is greater than zero.
- The helpful-vote count does not exceed the denominator.
- The required predictors are available.

Reviews without votes cannot provide an observed helpfulness proportion. Invalid vote-count records are excluded rather than assigning them artificial values.

Consequently, the results apply directly to reviews with usable vote counts, not necessarily to every review in the original dataset.

---

## 3. Variable definitions

### 3.1 Response: helpfulness

For review \(i\), define:

- \(h_i\): number of helpful votes.
- \(u_i\): number of unhelpful votes.
- \(v_i=h_i+u_i\): total votes.

The observed helpfulness proportion is:

\[
p_i=\frac{h_i}{h_i+u_i}
\]

For example, a review receiving 8 helpful votes and 2 unhelpful votes has an observed helpfulness proportion of 0.8.

The model uses the helpful and unhelpful vote counts as a grouped-binomial response. This preserves information about the number of votes supporting each review's observed proportion.

### 3.2 Main predictor: review length

Review length is measured by the number of characters in the review text.

The predictor is transformed as:

\[
L_i=\log(1+\text{ReviewLengthChars}_i)
\]

The logarithmic transformation is used because review lengths are strongly right-skewed. It also allows the relationship to reflect relative increases in length rather than assuming that every additional character has the same effect.

The transformed variable is centered by subtracting its sample mean:

\[
L_{i,c}=L_i-\bar L
\]

Centering improves the interpretation of the intercept without changing the slope coefficient.

### 3.3 Control variables

The primary model includes the following controls.

| Variable | Definition | Reason for inclusion |
|---|---|---|
| Rating extremity | \(|\text{Score}-3|\) | Accounts for how far the rating is from the midpoint. |
| Summary length | Number of characters in the review summary | Accounts for additional information conveyed by the summary. |
| Review year | Calendar year derived from the review date | Accounts for temporal changes in reviews and voting practices. |

These controls allow us to examine the association between review length and helpfulness conditional on the included predictors. They do not eliminate every possible source of confounding.

---

## 4. Hypotheses

The hypotheses were specified in the pre-analysis plan.

\[
H_0:\beta_L=0
\]

\[
H_1:\beta_L>0
\]

Where \(\beta_L\) is the coefficient of centered log review length in the primary model.

- **Null hypothesis (\(H_0\)):** Review length has no association with the log-odds of a helpful vote, conditional on the included predictors.
- **Alternative hypothesis (\(H_1\)):** Greater review length is associated with higher log-odds of a helpful vote, conditional on the included predictors.
- **Significance level:** \(\alpha=0.05\)
- **Test direction:** One-sided, positive effect.

A one-sided test is used because the directional hypothesis was specified in advance. The direction must not be selected after inspecting the results.

The nominal decision rule is to reject \(H_0\) if the one-sided p-value is below 0.05. Tests 1–3 are also subject to the planned Holm multiple-testing adjustment, so the final adjusted decision must be determined after all three tests have been completed.

---

## 5. Why grouped-binomial logistic regression?

### 5.1 Model specification

The primary model is a grouped-binomial generalized linear model (GLM) with a logit link:

\[
h_i\sim\operatorname{Binomial}(v_i,p_i)
\]

The linear predictor is:

\[
\begin{aligned}
\operatorname{logit}(p_i)={}&
\beta_0+\beta_1 E_i+\beta_2 L_{i,c}\\
&+\beta_3 S_{i,c}+\beta_4 Y_{i,c}
\end{aligned}
\]

Where:

- \(p_i\) is the modeled probability that a vote is helpful.
- \(E_i\) is rating extremity.
- \(L_{i,c}\) is centered log review length.
- \(S_{i,c}\) is centered summary length.
- \(Y_{i,c}\) is centered review year.

The logit function is:

\[
\operatorname{logit}(p_i)
=\log\left(\frac{p_i}{1-p_i}\right)
\]

The coefficient of log review length describes the change in log-odds associated with a one-unit increase in the transformed predictor. Exponentiating the coefficient gives an odds ratio.

### 5.2 Why not ordinary linear regression?

A helpfulness proportion is bounded between 0 and 1. Ordinary linear regression does not naturally respect this bound and does not directly model the vote counts underlying the proportion.

The grouped-binomial model is a more natural starting point for helpful and unhelpful vote counts.

### 5.3 Why not binary classification?

A classification approach would require defining a threshold to label reviews as helpful or unhelpful. This would discard information about the degree of helpfulness and the number of votes behind the observed proportion.

The grouped-binomial model retains the vote-count information and provides interpretable odds ratios.

### 5.4 Why cluster standard errors by product?

Reviews of the same product may share unobserved characteristics or voting influences. Product-clustered robust standard errors allow for potential dependence among reviews associated with the same product.

Clustering does not remove every possible dependence or correct every form of model misspecification.

---

## 6. Practical significance

Before testing, we proposed a minimum practically meaningful odds ratio of 1.05 across the interquartile range (IQR) of raw review length.

The IQR represents the difference between the 75th and 25th percentiles of review length.

This threshold means that an odds ratio of at least 1.05 across that range is considered potentially meaningful for the project.

The threshold is a project-specific benchmark, not a universal scientific standard. It should be justified in the final report.

An increase of 5% in odds is not the same as an increase of five percentage points in probability.

---

## 7. Primary test results

### 7.1 Numerical results

| Statistic | Result |
|---|---:|
| Reviews | 239,593 |
| Products | 39,522 |
| Review-length coefficient | 0.450796 |
| Product-clustered standard error | 0.014683 |
| Test statistic | 30.7017 |
| One-sided p-value | \(2.70\times10^{-207}\) |
| Coefficient 95% confidence interval | [0.422017, 0.479574] |
| Odds ratio per log unit | 1.56956 |
| 25th percentile of raw review length | 196 characters |
| 75th percentile of raw review length | 590 characters |
| Log-length IQR | 1.098612 |
| Odds ratio across the length IQR | 1.64091 |
| 95% CI for IQR odds ratio | [1.58984, 1.69362] |
| Nominal decision | Reject \(H_0\) |
| Model convergence | Yes |

The reported test uses a large-sample normal approximation with product-clustered robust standard errors. The coefficient confidence interval is a two-sided 95% interval.

### 7.2 Interpretation of the coefficient

The estimated coefficient is positive:

\[
\hat\beta_L=0.450796
\]

This indicates that, in the primary model, higher log review length is associated with higher log-odds of a helpful vote after adjusting for the other included variables.

The odds ratio for a one-unit increase in log review length is:

\[
OR=e^{0.450796}\approx1.56956
\]

However, a one-unit increase in log length does not correspond to adding one character. For interpretation, the IQR-based odds ratio is more directly useful.

### 7.3 Interpretation across the IQR

The 25th percentile of review length is 196 characters, and the 75th percentile is 590 characters.

The estimated odds ratio across this range is:

\[
OR_{\mathrm{IQR}}\approx1.64091
\]

Thus, the primary linear model estimates approximately **64.1% higher odds of a helpful vote** at the 75th percentile than at the 25th percentile, holding the other included predictors constant.

The 95% confidence interval is approximately 1.590 to 1.694.

This is an odds comparison, not a 64.1-percentage-point increase in helpfulness probability.

### 7.4 Statistical conclusion

The one-sided p-value is far below the nominal significance level of 0.05. The nominal test therefore rejects \(H_0\) and provides strong evidence of a positive association under the primary model.

The estimated IQR odds ratio also exceeds the pre-specified practical-significance threshold of 1.05.

These findings must be interpreted alongside the subsequent sensitivity analysis, the nonlinearity diagnostic, the model assumptions, and the planned Holm correction.

---

## 8. Sensitivity analysis: product-grouped GEE

We fitted a product-grouped generalized estimating equations (GEE) model to examine whether the primary result was stable under an alternative estimating approach.

The sensitivity analysis used:

- The same exploration split.
- The same 239,593 reviews and 39,522 products.
- The same grouped helpfulness response and predictors.
- Vote-count weights.
- An independence working correlation structure.
- Product-grouped robust covariance estimation.

### Results comparison

| Statistic | Primary GLM | GEE sensitivity |
|---|---:|---:|
| Review-length coefficient | 0.450796 | 0.450796 |
| Robust standard error | 0.014683 | 0.014683 |
| Odds ratio across the IQR | 1.640908 | 1.640908 |
| 95% IQR odds-ratio CI | 1.589840–1.693616 | 1.589841–1.693615 |
| One-sided p-value | \(2.70\times10^{-207}\) | \(2.65\times10^{-207}\) |
| Convergence | Yes | Yes |

The estimates are essentially identical. This supports numerical stability across these two model specifications.

However, the models use the same observations, predictors, weighting scheme, and product grouping. They are not independent replications, and the agreement does not prove that every model assumption is satisfied.

The GEE sensitivity analysis does not eliminate all concerns about overdispersion or misspecification.

---

## 9. Nonlinearity analysis

### 9.1 Why investigate nonlinearity?

The primary model assumes a linear relationship between log review length and the log-odds of helpfulness.

This may be too restrictive. For example, increasing review length might be associated with a large increase in helpfulness among short reviews, followed by diminishing returns among longer reviews.

To investigate this possibility, we compared the primary model against a model using a cubic regression spline for log review length.

### 9.2 Models compared

**Linear model:** log review length enters as one predictor.

**Spline model:** log review length is represented using a cubic spline with four basis degrees of freedom.

Both models use the same observations, grouped-binomial response, and other predictors.

### 9.3 Numerical comparison

| Statistic | Linear model | Spline model |
|---|---:|---:|
| Log-likelihood | −330,327.03 | −329,496.25 |
| AIC | 660,664.05 | 659,008.49 |
| Model degrees of freedom | 4 | 7 |
| Model convergence | Yes | Yes |

The likelihood-ratio comparison produced:

- Likelihood-ratio statistic: 1661.56
- Difference in model degrees of freedom: 3
- Printed likelihood-ratio p-value: 0.0
- AIC improvement in favor of spline: 1655.56

The printed p-value of 0.0 reflects numerical underflow; it should not be interpreted as a literally zero probability.

The much lower AIC and improved likelihood favor the spline model under these likelihood-based comparisons.

### 9.4 Important qualification

The ordinary likelihood-ratio test relies on the binomial likelihood assumptions and does not use a product-cluster-robust likelihood-ratio test.

The earlier Test 1 analysis produced a Pearson dispersion ratio of approximately 2.12. This raises concerns about the simple binomial variance assumptions.

Therefore, the likelihood-ratio result should be treated as a diagnostic comparison rather than definitive evidence from a fully robust nonlinearity test.

### 9.5 Interpretation of the fitted curve

The adjusted spline plot is saved as:

`reports/figures/test2_review_length_spline.png`

The plotted curve shows:

1. A steep increase in predicted helpfulness among shorter reviews.
2. Continued increases as reviews become longer, but at a decreasing rate.
3. A flattening pattern among longer reviews, consistent with diminishing returns in the fitted relationship.

For example, the displayed curve is approximately 0.81 near 500 characters and approximately 0.88 near 2,000 characters, holding the other predictors at the reference values used for prediction.

These are adjusted model predictions, not raw observed proportions.

The plot does not include a confidence band, so it does not display uncertainty around the curve. Predictions near the edges of the displayed range should also be interpreted cautiously.

The pattern is consistent with the spline model capturing a relationship that the single linear log-length term cannot fully represent. It does not establish that increasing review length causes more helpful votes.

---

## 10. Alternative approaches

| Alternative | What it would tell us | Why it is not the sole primary approach |
|---|---|---|
| Raw review length | Whether each additional character has a constant effect on log-odds | Review length is strongly right-skewed, and the relationship may not be linear on this scale. |
| Binary classification | Whether reviews cross a chosen helpfulness threshold | Requires an arbitrary threshold and discards vote-count information. |
| Ordinary linear regression | A simple relationship with observed helpfulness proportions | Does not naturally represent grouped vote counts or bounded proportions. |
| Polynomial terms | Whether a quadratic or cubic relationship improves fit | Can behave poorly near the data boundaries and may be difficult to interpret. |
| Regression splines | A flexible relationship that can bend across the length range | Requires choosing spline complexity and checking the fitted curve. |
| GAM or other smooth models | A more flexible smooth relationship | Adds complexity and requires careful uncertainty estimation and validation. |
| Binomial model without product clustering | A simpler variance estimate | May understate uncertainty if reviews of the same product are dependent. |
| GEE with another working correlation | A different representation of within-product dependence | Requires additional assumptions and diagnostic comparisons. |

The grouped-binomial model was selected as the primary approach because it represents the helpful and unhelpful vote counts directly and provides interpretable odds ratios.

The spline analysis is a diagnostic extension that examines whether the primary model's functional-form assumption is reasonable.

---

## 11. Limitations

### 11.1 Observational data

The dataset is observational. Review length was not randomly assigned.

Consequently, the analysis establishes an association, not a causal effect. Longer reviews may differ from shorter reviews in other unobserved ways.

### 11.2 Sample selection

Only reviews with valid, nonzero vote denominators can be included. These reviews may differ from reviews that receive no votes.

The conclusions may not generalize to every review in the dataset.

### 11.3 Overdispersion

The primary binomial analysis showed a Pearson dispersion ratio of approximately 2.12.

This indicates more variability than expected under a simple binomial variance model and motivates caution in inference.

The GEE comparison provides a useful sensitivity check but does not establish that all overdispersion concerns have been resolved.

### 11.4 Dependence

Product-clustered robust standard errors account for potential within-product dependence, but other sources of dependence may remain.

### 11.5 Functional form

The spline comparison indicates that the linear log-length specification may be too restrictive. The primary coefficient should therefore be interpreted as a summary of the association under that particular functional form, not as a constant effect at every review length.

### 11.6 Multiple testing

Tests 1–3 form the planned explanatory hypothesis family. The Holm correction must be applied before finalizing the inferential conclusions.

### 11.7 Prediction versus explanation

A statistically significant association does not automatically establish strong predictive performance. Predictive performance must be evaluated separately on the appropriate held-out confirmation data.

---

## 12. Reproducibility

### Code files

- Primary test: `scripts/tests/test2_review_length.py`
- GEE sensitivity analysis: `scripts/tests/test2_sensitivity.py`
- Nonlinearity check: `scripts/tests/test2_nonlinearity.py`
- Spline visualization: `scripts/tests/plot_test2_length_spline.py`

### Machine-readable outputs

- Primary results: `reports/test2_review_length.json`
- GEE sensitivity results: `reports/test2_sensitivity.json`
- Nonlinearity results: `reports/test2_nonlinearity.json`
- Figure: `reports/figures/test2_review_length_spline.png`

### Reproduce the analysis

From the project root:

```bash
python3 scripts/tests/test2_review_length.py
python3 scripts/tests/test2_sensitivity.py
python3 scripts/tests/test2_nonlinearity.py
python3 scripts/tests/plot_test2_length_spline.py
```

The scripts use the exploration split and save their results under `reports/`.

The confirmation split should remain untouched until the analysis choices are finalized. For complete reproducibility, retain the split-generation procedure, recorded random seed, dependency versions, and pre-analysis plan.

---

## 13. Final inference

The primary grouped-binomial GLM found a positive association between log review length and helpful-vote odds after adjusting for rating extremity, summary length, and review year.

The estimated odds ratio comparing the 75th and 25th percentiles of raw review length was approximately 1.641, with a 95% confidence interval of approximately 1.590 to 1.694. The nominal one-sided test strongly rejected the null hypothesis, and the point estimate exceeded the proposed practical-significance threshold.

The product-grouped GEE sensitivity analysis produced virtually identical estimates, supporting numerical stability across the two fitted approaches.

The nonlinearity analysis favored a cubic spline over the single linear log-length term according to the likelihood and AIC comparisons. The adjusted spline curve showed a steep increase in predicted helpfulness among shorter reviews followed by a flattening pattern among longer reviews.

However, the ordinary likelihood-ratio p-value is not product-cluster robust, and the earlier dispersion diagnostic raises concerns about the binomial variance assumptions. The GEE comparison does not eliminate every possible model limitation. The final inference must also account for the planned Holm adjustment.

**Overall conclusion:** Among the analyzed Amazon Fine Food Reviews with valid helpfulness votes, longer reviews were positively associated with helpful-vote odds after adjusting for the included review characteristics. The relationship appears nonlinear, with the fitted association becoming weaker as review length increases. These findings are observational and should not be interpreted as evidence that increasing review length causes a review to become more helpful.
