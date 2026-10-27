# Research Hypothesis: Rating Extremity and Review Helpfulness

## 1. Primary Research Question

**Are reviews with more extreme star ratings associated with higher helpfulness rates, after accounting for review length?**

This project studies the relationship between review characteristics and helpfulness in the Amazon Fine Food Reviews dataset. The primary focus is whether the strength of a review's rating is associated with the proportion of helpful votes it receives.

The analysis is associational. A relationship between rating extremity and helpfulness does not, by itself, show that extreme ratings cause reviews to become more helpful.

## 2. Defining Rating Extremity

Amazon ratings range from 1 to 5 stars. A 3-star rating is the midpoint of this scale, while 1-star and 5-star ratings are equally far from it.

We define rating extremity as:

\[
\text{Rating Extremity} = |\text{Score} - 3|
\]

The absolute value measures distance from the midpoint regardless of whether the rating is positive or negative.

### How the formula works

| Star rating | General interpretation | Calculation | Extremity |
|---:|---|---|---:|
| 1 | Very negative | \(|1-3|=2\) | 2 |
| 2 | Somewhat negative | \(|2-3|=1\) | 1 |
| 3 | Midpoint of the scale | \(|3-3|=0\) | 0 |
| 4 | Somewhat positive | \(|4-3|=1\) | 1 |
| 5 | Very positive | \(|5-3|=2\) | 2 |

This definition captures how far a rating is from the midpoint, rather than whether it is positive or negative. For example, a 1-star review and a 5-star review both have extremity 2, while a 3-star review has extremity 0.

This allows us to test whether more extreme opinions tend to receive higher helpfulness rates.

## 3. Alternative Ways to Measure Rating Extremity

Several reasonable definitions are available. Each represents a slightly different research question.

| Approach | Formula or definition | What it captures | Limitation |
|---|---|---|---|
| **Absolute distance (primary choice)** | \(|s-3|\) | Degree of extremity | Assumes equal distance between adjacent ratings |
| Squared distance | \((s-3)^2\) | Gives greater numerical emphasis to endpoints | On a 1–5 scale, has the same ordering as absolute distance, though the numerical scale differs |
| Binary extremity | 1 if rating is 1 or 5; otherwise 0 | Whether a rating is at either endpoint | Treats ratings 2, 3, and 4 as equally non-extreme |
| Categorical rating | Separate indicators for 1-, 2-, 3-, 4-, and 5-star ratings | The separate association for each rating | Requires more coefficients and does not impose a single extremity trend |
| Signed deviation | \(s-3\) | Direction of the rating relative to the midpoint | Measures direction, not extremity; 1-star and 5-star ratings have opposite signs |

## 4. Why Use Absolute Distance as the Primary Definition?

We use absolute distance for three main reasons.

1. **It matches the conceptual question.** We want to study whether the strength of an opinion is associated with helpfulness, regardless of whether the opinion is positive or negative.
2. **It is simple and interpretable.** The variable takes only three possible values—0, 1, and 2. In a regression model, its coefficient represents the modeled change in helpfulness associated with a one-unit increase in extremity, holding the included control variables constant.
3. **It supports a directional hypothesis.** We can test whether increasing rating extremity is positively associated with helpfulness rate while controlling for review length.

### Important assumption

This formula assumes a symmetric relationship: a 1-star rating and a 5-star rating are treated as equally extreme. This is a simplifying research assumption, not an established fact. Very negative reviews and very positive reviews may have different relationships with helpfulness.

We will therefore compare the primary definition with alternative specifications rather than assume that the relationship is symmetric.

## 5. Primary Hypothesis and Statistical Test

Let:

- \(Y_i\) be the helpfulness rate for review \(i\);
- \(E_i = |\text{Score}_i - 3|\) be its rating extremity;
- \(L_i\) be review length.

For reviews with valid vote counts and a positive helpfulness denominator, define:

\[
Y_i = \frac{\text{HelpfulnessNumerator}_i}
{\text{HelpfulnessDenominator}_i}
\]

The rate is undefined when the denominator is zero. Records with a numerator greater than the denominator have invalid counts and must be flagged rather than treated as valid rates.

Our primary regression can be represented conceptually as:

\[
Y_i = \beta_0 + \beta_1 E_i + \beta_2 L_i + \epsilon_i
\]

The exact model form and uncertainty estimates will be selected with the vote-count structure in mind.

### Hypotheses

- **Null hypothesis (\(H_0\)):** \(\beta_1 \leq 0\). After controlling for review length, rating extremity has no positive association with helpfulness rate.
- **Alternative hypothesis (\(H_1\)):** \(\beta_1 > 0\). After controlling for review length, greater rating extremity is positively associated with helpfulness rate.

We will use a pre-specified significance level of 0.05 and report the estimated effect size, confidence interval, and p-value. Statistical significance will not be interpreted as proof of causation or practical importance.

## 6. Alternative Specifications and Sensitivity Analysis

To evaluate whether the conclusion depends on how extremity is defined, we will compare three specifications.

### Model A — Primary: Absolute extremity

Use:

\[
E_i = |\text{Score}_i - 3|
\]

This tests for a single increasing trend in helpfulness as ratings move farther from the midpoint.

### Model B — Categorical star rating

Use separate indicator variables for the star ratings, with 3 stars as the reference category. This does not force the 1-star and 5-star ratings to have equal effects. It allows us to inspect whether the association is stronger for negative or positive ratings.

### Model C — Endpoint indicator

Define a binary variable equal to 1 for 1-star or 5-star reviews and 0 for ratings 2, 3, or 4. This tests the simpler question of whether endpoint ratings differ from all other ratings combined.

We will compare estimates and model fit where appropriate, and examine rating-specific coefficients. This will help determine whether a general extremity pattern is plausible or whether the result is driven primarily by one end of the rating scale.

## 7. Modeling the Helpfulness Rate Carefully

The helpfulness rate is a proportion derived from vote counts. Rates based on different numbers of votes have different levels of uncertainty: for example, 1 helpful vote out of 1 total vote is less stable evidence than 80 helpful votes out of 100.

Therefore:

- Reviews with a zero denominator have an undefined helpfulness rate and will not be assigned a rate of zero.
- Reviews with invalid vote counts will be flagged and excluded from analyses that require a valid rate, while remaining in the raw dataset.
- The main analysis will use reviews with valid counts and a positive denominator.
- A sensitivity analysis will examine whether results change when restricting to reviews with more votes. The threshold will be chosen and documented before inspecting the corresponding results.
- Alongside ordinary regression of the rate, we will consider a binomial or quasi-binomial model using helpful and non-helpful vote counts, where appropriate. The model choice will be documented, including any assumptions and limitations.

## 8. Interpretation

If the estimated extremity coefficient is positive and the evidence is sufficiently strong, the results will support an association between more extreme ratings and higher helpfulness rates in the analyzed sample, conditional on the included controls.

If the estimate is not positive or is too uncertain, the analysis will not support the proposed directional hypothesis. We will report that result transparently rather than changing the hypothesis after seeing the data.

Even a positive association does not imply that changing a review's rating would cause its helpfulness rate to increase. Other factors—such as review content, product category, reviewer behavior, and the number of votes—may also be relevant.
