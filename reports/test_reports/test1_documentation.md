# Test 1: Relationship Between Rating Extremity and Review Helpfulness

## 1. Overview

### Research question

**Are reviews with more extreme ratings more likely to receive helpful votes, after accounting for review length, summary length, and review year?**

An Amazon review can receive helpful and unhelpful votes from other users. These votes provide an observable measure of how useful readers consider a review.

Our objective is to investigate whether the extremity of a review's star rating is associated with its helpfulness, while controlling for other observable characteristics.

This analysis is part of our broader research question:

> What makes an Amazon review helpful? Can review characteristics such as rating, length, summary length, and textual content explain or predict helpfulness?

### Why investigate rating extremity?

A review with a rating of 1 star expresses a strongly negative evaluation, while a review with 5 stars expresses a strongly positive evaluation. Both ratings are extreme relative to a neutral 3-star rating.

Extreme ratings might attract attention because they communicate strong opinions. However, they could also reflect polarized opinions that are not necessarily useful to readers.

Therefore, we investigate whether greater rating extremity is positively associated with the probability that a vote is helpful.

This is an empirical question: the direction and magnitude of the association must be established using data.

---

## 2. Dataset and observations

### Data source

The analysis uses the Amazon Fine Food Reviews dataset from the Stanford Network Analysis Project (SNAP).

**Input file:** `data/splits/exploration.csv.gz`

We use the exploration split because the analysis decisions are being developed on that split. The confirmation split must remain untouched until the planned analysis decisions are finalized.

### Sample used in Test 1

| Quantity | Value |
|---|---:|
| Reviews in the exploration split | 454,951 |
| Reviews included in Test 1 | 239,593 |
| Unique products represented | 39,522 |
| Total helpful votes | 802,496 |
| Total unhelpful votes | 223,816 |
| Total valid votes | 1,026,312 |

The test uses only reviews with valid helpfulness vote counts.

A review is included when:

- The helpfulness numerator is non-negative.
- The helpfulness denominator is greater than zero.
- The helpfulness numerator does not exceed the denominator.
- The required predictors are available.

Reviews with zero total votes cannot provide a meaningful observed helpfulness proportion. Records with invalid vote counts are excluded from this test rather than assigning them an artificial helpfulness value.

Consequently, the test sample is smaller than the original exploration split. The results apply directly to the included reviews, not automatically to every review in the dataset.

---

## 3. Defining the variables

### 3.1 Response: helpfulness

Let:

- \(h_i\) = number of helpful votes for review \(i\)
- \(u_i\) = number of unhelpful votes for review \(i\)
- \(v_i = h_i + u_i\) = total votes for review \(i\)

The observed helpfulness proportion is:

\[
p_i = \frac{h_i}{h_i+u_i}
\]

For example, if a review receives 8 helpful votes and 2 unhelpful votes:

\[
p_i = \frac{8}{8+2}=0.8
\]

Its observed helpfulness proportion is 0.8, or 80%.

However, a proportion alone does not convey how much evidence supports it. An 8/10 review has more votes behind its proportion than a 1/1 review.

For this reason, the primary model uses the helpful and unhelpful vote counts directly as a grouped-binomial response instead of treating all review-level proportions as equally precise.

### 3.2 Main predictor: rating extremity

We define rating extremity as:

\[
E_i = |\text{Score}_i-3|
\]

The resulting values are:

| Star rating | Extremity |
|---:|---:|
| 1 | 2 |
| 2 | 1 |
| 3 | 0 |
| 4 | 1 |
| 5 | 2 |

A larger value represents a rating farther from the midpoint.

#### Why use absolute distance from 3?

Our research question concerns the *strength* of the rating rather than whether it is positive or negative.

The absolute-distance measure provides a single numerical predictor and allows us to test whether helpfulness tends to increase with rating extremity.

It also avoids treating 1-star and 5-star ratings as having opposite values on the predictor.

#### Assumption introduced by this choice

Using absolute distance imposes a simplifying assumption: a one-unit increase in extremity has the same effect on log-odds at either side of the midpoint.

For example, moving from 3 to 4 stars and moving from 2 to 1 star both increase extremity by one unit.

This does not establish that positive and negative extreme reviews behave identically. We will consider alternative rating encodings as sensitivity analyses.

### 3.3 Control variables

The model includes three additional predictors.

| Variable | Definition | Reason for inclusion |
|---|---|---|
| Review length | Natural logarithm of \(1+\) review character count | Longer reviews may provide more explanation, evidence, and context. |
| Summary length | Number of characters in the review summary | A summary may convey useful information independently of the full review length. |
| Review year | Calendar year derived from the review date | Voting practices, products, and the composition of reviews may change over time. |

The continuous predictors are centered by subtracting their sample means. Centering makes the intercept easier to interpret and does not change the fitted slopes.

Review length is log-transformed because review lengths are highly right-skewed. The logarithm also allows the effect to vary with the relative rather than only the absolute increase in length.

The inclusion of these controls helps us examine the association between rating extremity and helpfulness conditional on the included variables. It does not eliminate all possible confounding.

---

## 4. Hypotheses and statistical decision

The primary hypotheses were specified in the pre-analysis plan.

\[
H_0:\beta_E=0
\]

\[
H_1:\beta_E>0
\]

Where \(\beta_E\) is the coefficient of rating extremity in the model.

- **Null hypothesis (\(H_0\)):** Rating extremity has no association with the log-odds of a helpful vote, conditional on the included predictors.
- **Alternative hypothesis (\(H_1\)):** Higher rating extremity is associated with higher log-odds of a helpful vote, conditional on the included predictors.
- **Significance level:** \(\alpha=0.05\)
- **Test direction:** One-sided, positive effect.

We use a one-sided test because the pre-analysis plan specifies a directional hypothesis: the expected association is positive.

The direction must be specified before examining the test result. A one-sided test should not be chosen after seeing which direction the coefficient takes.

The nominal decision rule is:

- Reject \(H_0\) if the one-sided p-value is below 0.05.
- Otherwise, fail to reject \(H_0\).

The final project must also account for the planned Holm adjustment across Tests 1–3. Therefore, the nominal decision from Test 1 is not necessarily the final multiple-testing-adjusted decision.

---

## 5. Why did we choose grouped-binomial logistic regression?

### 5.1 Model specification

The primary model is a grouped-binomial generalized linear model (GLM) with a logit link.

For review \(i\):

\[
h_i \sim \operatorname{Binomial}(v_i,p_i)
\]

The model is:

\[
\begin{aligned}
\operatorname{logit}(p_i) ={}&
\beta_0+\beta_1 E_i\\
&+\beta_2 L_i+\beta_3 S_i+\beta_4 Y_i
\end{aligned}
\]

Where:

- \(p_i\) is the modeled probability that an individual vote is helpful.
- \(E_i\) is rating extremity.
- \(L_i\) is centered log review length.
- \(S_i\) is centered summary length.
- \(Y_i\) is centered review year.

The logit transformation is:

\[
\operatorname{logit}(p_i)
=\log\left(\frac{p_i}{1-p_i}\right)
\]

The coefficients describe changes in log-odds. Exponentiating a coefficient gives an odds ratio.

### 5.2 Why not ordinary linear regression?

Ordinary linear regression is not the natural first choice for a response representing a proportion bounded between 0 and 1.

A linear model can predict values outside that range and does not naturally represent the different amounts of voting evidence behind different proportions.

The grouped-binomial model directly represents helpful versus unhelpful votes.

### 5.3 Why not classification?

We could classify each review as either helpful or not helpful by applying a threshold to its helpfulness proportion.

However, doing so would discard information. For example, proportions of 0.6 and 0.9 would become identical if both were assigned the same class.

Our objective is to understand how review characteristics relate to helpfulness and to quantify the magnitude of the association. Modeling vote counts retains more information than imposing an arbitrary binary threshold.

### 5.4 Why not model helpfulness rate without vote counts?

Treating each review's observed helpfulness proportion as equally precise would ignore differences in total votes.

A proportion based on one vote can be much less informative than a proportion based on twenty votes.

The grouped-binomial model incorporates the total vote count into the response likelihood.

### 5.5 Why cluster by product?

Multiple reviews can refer to the same product. Reviews of the same product may share characteristics or unobserved influences.

We therefore calculate product-clustered robust standard errors. This allows uncertainty estimates to account for potential dependence among reviews associated with the same product.

Clustering does not remove all dependence or guarantee that the model is correctly specified.

---

## 6. Practical significance

Statistical significance alone is insufficient. With a large dataset, even small associations can produce very small p-values.

Before testing, we proposed a minimum practically meaningful odds ratio of 1.05 per one-unit increase in rating extremity.

This corresponds to a coefficient of:

\[
\log(1.05)\approx0.04879
\]

The threshold means that we would regard an increase of at least 5% in the odds per extremity unit as a potentially meaningful effect, subject to substantive justification.

This threshold is a project decision rather than a universal scientific standard. Its justification should be documented.

The threshold must also be interpreted correctly: a 5% increase in odds is not the same as a five-percentage-point increase in probability.

---

## 7. Primary model results

### 7.1 Numerical results

The primary grouped-binomial GLM produced the following results.

| Statistic | Result |
|---|---:|
| Rating-extremity coefficient | 0.349377 |
| Product-clustered standard error | 0.013087 |
| Test statistic | 26.6968 |
| One-sided p-value | \(2.57\times10^{-157}\) |
| Coefficient 95% confidence interval | [0.323727, 0.375027] |
| Odds ratio | 1.418184 |
| Odds-ratio 95% confidence interval | [1.382270, 1.455031] |
| Nominal decision at \(\alpha=0.05\) | Reject \(H_0\) |
| Convergence | Yes |

The reported test statistic uses a large-sample normal approximation. The confidence interval is a two-sided 95% interval.

### 7.2 Interpretation of the coefficient

The coefficient is positive:

\[
\hat{\beta}_E=0.349377
\]

This indicates a positive conditional association between rating extremity and the log-odds of a helpful vote.

The odds ratio is:

\[
OR=e^{0.349377}\approx1.4182
\]

Thus, each one-unit increase in rating extremity is associated with approximately **41.8% higher odds of a helpful vote**, holding the other included predictors constant.

The 95% confidence interval for the odds ratio is approximately 1.382 to 1.455.

This interval is entirely above 1, which is consistent with a positive association under the fitted model.

### 7.3 Statistical conclusion

The one-sided p-value is far below the nominal significance level of 0.05.

Therefore, the nominal test rejects \(H_0\) and provides strong evidence of a positive association under the model and its inferential assumptions.

The final conclusion must be revisited after the planned Holm adjustment and the model-diagnostic review.

### 7.4 Practical conclusion

The estimated odds ratio of 1.418 is above the proposed practical threshold of 1.05.

The point estimate therefore exceeds the pre-specified practical-significance benchmark.

This is evidence of a potentially meaningful association, but it does not establish causation or prove that the effect will generalize to all reviews.

---

## 8. Sensitivity analysis

We fitted a product-grouped generalized estimating equations (GEE) model as a sensitivity analysis.

The model used:

- The same exploration data.
- The same 239,593 reviews.
- The same 39,522 products.
- The same outcome and predictors.
- A binomial family with an independence working correlation structure.
- Product-grouped robust covariance estimation.

### Results comparison

| Statistic | Primary GLM | GEE sensitivity model |
|---|---:|---:|
| Extremity coefficient | 0.349377 | 0.349377 |
| Robust standard error | 0.013087 | 0.013087 |
| Odds ratio | 1.418184 | 1.418184 |
| One-sided p-value | \(2.57\times10^{-157}\) | \(2.53\times10^{-157}\) |
| Convergence | Yes | Yes |

The estimated coefficient and uncertainty are essentially identical across the two fitted models.

This supports the numerical stability of the estimated association under these specifications.

However, the two models use the same data and closely related modeling assumptions. They should not be treated as independent replications.

The GEE analysis also does not, by itself, resolve every concern about overdispersion or misspecification.

---

## 9. Model diagnostics and limitations

### 9.1 Overdispersion

The primary model's Pearson dispersion ratio was:

\[
\frac{\text{Pearson chi-square}}
{\text{Residual degrees of freedom}}
\approx2.12
\]

The observed ratio is greater than 1 and suggests more variability than expected under the simple binomial variance assumptions.

This matters because an inadequately specified variance model can affect uncertainty estimates and statistical conclusions.

Product-clustered standard errors account for potential dependence within products, but they do not automatically resolve every form of overdispersion.

The sensitivity analysis produced similar estimates, but that similarity does not prove the absence of overdispersion.

Further diagnostics and sensitivity checks are required before treating the inferential result as final.

### 9.2 Observational data

The dataset is observational. We did not randomly assign star ratings or review lengths.

Therefore, the results establish an association rather than a causal effect.

### 9.3 Selection into the analyzed sample

Only reviews with valid, nonzero vote denominators can contribute to this analysis.

Reviews with no votes are excluded, and such reviews may differ systematically from reviews that receive votes.

The findings may not generalize to reviews without voting activity.

### 9.4 Product-level dependence

Product-clustered standard errors address one plausible source of dependence. Other sources may remain, including reviewer-level dependence, temporal changes, or unobserved product characteristics.

### 9.5 Rating symmetry

The extremity predictor assumes a common effect for movements away from the midpoint, regardless of direction.

This could hide differences between strongly negative and strongly positive reviews.

### 9.6 Multiple hypothesis testing

Tests 1–3 belong to the planned explanatory hypothesis family. We will apply the pre-specified Holm adjustment across that family.

The nominal p-value reported here must not be presented as the final adjusted result before that step is completed.

### 9.7 Statistical versus practical significance

A very small p-value does not automatically imply a useful or important effect.

The odds ratio and confidence interval should be considered alongside the pre-specified practical threshold, model assumptions, and real-world interpretation.

---

## 10. Alternative approaches we considered

| Alternative | What it would tell us | Why it is not the primary choice |
|---|---|---|
| Treat rating as a five-level categorical variable | Allows each rating to have a separate association | Does not impose a single linear effect of extremity; useful as a sensitivity analysis |
| Use an endpoint indicator | Compares 1- and 5-star reviews against 2-, 3-, and 4-star reviews | Discards distinctions between different levels of extremity |
| Use squared distance from 3 | Allows a nonlinear relationship with rating distance | Gives a different functional form and interpretation |
| Separate positive and negative extremity | Tests whether 1–2-star and 4–5-star ratings behave differently | More flexible, but requires additional coefficients and tests |
| Ordinary linear regression on helpfulness rate | Provides a simple association model | Does not naturally respect the binomial response structure |
| Binary classification of helpfulness | Predicts a chosen helpful/not-helpful label | Requires an arbitrary threshold and loses vote-count information |
| Binomial regression without clustering | Models vote counts with simpler uncertainty estimation | May understate uncertainty if reviews of the same product are dependent |
| GEE with a different working correlation | Allows a different representation of within-product dependence | Requires additional assumptions and diagnostic comparison |

The primary model was selected because it matches the research question, uses the helpful and unhelpful vote counts, provides interpretable odds ratios, and adjusts for relevant review characteristics.

The alternatives are not inherently wrong. They answer slightly different questions or impose different assumptions. Some should be used as sensitivity analyses rather than discarded entirely.

---

## 11. Reproducibility

### Code files

- Primary analysis: `scripts/tests/test1_rating_extremity.py`
- Sensitivity analysis: `scripts/tests/test1_sensitivity.py`

### Machine-readable outputs

- Primary results: `reports/test1_rating_extremity.json`
- Sensitivity results: `reports/test1_sensitivity.json`

### Reproduce the analyses

From the project root:

```bash
python3 scripts/tests/test1_rating_extremity.py
python3 scripts/tests/test1_sensitivity.py
```

The scripts read the exploration split and write their results to JSON files under `reports/`.

The exploration split should remain fixed during the analysis, and the confirmation split should not be used to choose predictors or revise the model.

For full reproducibility, retain the recorded random seed and split-generation procedure, dependency versions, and the pre-analysis plan alongside the project.

---

## 12. Final inference

The primary model found a positive association between rating extremity and helpful-vote odds after adjusting for log review length, summary length, and review year.

The estimated odds ratio was approximately 1.418 per one-unit increase in extremity, with a 95% confidence interval of approximately 1.382 to 1.455.

The nominal one-sided test strongly rejected the null hypothesis, and the point estimate exceeded the proposed practical-significance threshold. The sensitivity model produced nearly identical estimates.

Nevertheless, the Pearson dispersion ratio of approximately 2.12 raises concerns about the binomial variance assumptions. Product clustering and the GEE comparison provide useful checks but do not eliminate every possible model limitation. The final report must account for the planned Holm correction and the remaining diagnostics.

**Overall conclusion:** In the analyzed subset of Amazon Fine Food Reviews, more extreme star ratings were positively associated with helpful-vote odds, conditional on the included review characteristics. This is a strong preliminary statistical finding, not evidence that rating extremity causes reviews to become more helpful.

The finding is consistent with the idea that reviews expressing stronger opinions may attract helpful votes, but further analyses are needed to assess rating asymmetry, model robustness, and generalizability.
