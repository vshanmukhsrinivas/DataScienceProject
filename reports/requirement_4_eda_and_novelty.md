# Requirement 4: Research Questions, Novelty, and Exploratory Data Analysis (EDA)

## 1. Objective

The objective of this stage is to identify meaningful questions about the Amazon Fine Food Reviews dataset, explore initial patterns using visualizations, and establish how the project's problem statement goes beyond basic descriptive analysis.

The analysis focuses on review ratings, review length, helpfulness votes, and textual information. These exploratory observations will guide the formulation of hypotheses and the subsequent statistical and predictive modeling stages.

**Dataset:** Stanford SNAP Amazon Fine Food Reviews dataset  
**Dataset used for EDA:** Cleaned exploration split  
**Split strategy:** Product-grouped random split  
**Random seed:** 42

> Important: All exploratory plots and descriptive statistics reported in the final version must be generated from the current cleaned exploration split. Earlier EDA results obtained from the pre-cleaning split are preliminary and must not be copied as final results without revalidation.

---

## 2. Basic Research Questions

The following questions motivate the exploratory analysis.

### RQ1: How are review ratings distributed?

This question investigates whether the dataset contains a balanced distribution of ratings or is dominated by a particular rating category.

**Proposed visualization:** Bar chart showing the number of reviews for each rating from 1 to 5.

**Purpose:** To understand rating imbalance and determine whether descriptive comparisons across rating categories require special care.

### RQ2: Is review helpfulness associated with the rating assigned by the reviewer?

This question investigates whether the helpfulness rate differs across rating categories.

For reviews with valid helpfulness counts, the helpfulness rate is defined as:

\[
\text{Helpfulness Rate} =
\frac{\text{Helpful Votes}}{\text{Total Helpfulness Votes}}
\]

A rate is defined only when the denominator is greater than zero and the helpful-vote numerator does not exceed the denominator.

**Proposed visualization:** Plot the mean helpfulness rate by rating, with 95% confidence intervals.

**Purpose:** To explore whether rating categories differ in their observed helpfulness rates. This is an association, not evidence that a rating causes helpfulness.

### RQ3: Is review length associated with helpfulness?

Longer reviews may provide more detail, but length alone may not indicate whether a review is useful. The relationship could also be nonlinear.

**Proposed visualization:** Plot mean helpfulness rate across review-length bins, accompanied by uncertainty intervals where appropriate.

**Purpose:** To explore whether review length contains useful information about helpfulness and whether transformations such as `log1p(review length)` merit consideration in modeling.

### RQ4: How reliable are the observed helpfulness rates?

A review receiving one helpful vote out of one total vote has a rate of 100%, but this estimate is based on very little evidence. A review receiving 100 helpful votes out of 100 total votes has the same observed rate but substantially more supporting votes.

**Proposed visualization:** Histogram or distribution plot of the total helpfulness-vote count, preferably using a logarithmic horizontal axis when the distribution is highly skewed.

**Purpose:** To understand how much the reliability of observed helpfulness rates varies across reviews and to motivate sensitivity analyses that account for vote counts.

Reviews with zero total votes have an undefined helpfulness rate; they must not be assigned a rate of zero.

### RQ5: Does review text provide predictive information beyond structured attributes?

Review summaries and bodies contain information that may not be fully captured by rating, review length, summary length, or review year.

**Proposed visualization:** A descriptive comparison of review-length distributions or selected text characteristics across helpfulness groups, provided the grouping and its limitations are clearly defined.

**Purpose:** To motivate the formal comparison of a structured-feature model against a model that additionally incorporates TF-IDF text features.

The primary evidence for this question will come from held-out predictive evaluation, rather than an EDA plot alone.

---

## 3. EDA Figures and Evidence-Based Captions

The following observations were calculated from the cleaned exploration dataset. Retain figures only after verifying that the corresponding image exists and matches the stated statistic.

| Figure | Suggested caption / observed takeaway |
|---|---|
| Figure 1: Rating distribution | Five-star reviews dominate the dataset: 290,934 of 454,951 exploration reviews (64.0%). The rating distribution is strongly imbalanced. |
| Figure 2: Helpfulness by rating | Among reviews with valid helpfulness rates, mean helpfulness rises from 0.540 for one-star reviews to 0.872 for five-star reviews. These are descriptive, unadjusted differences. |
| Figure 3: Helpfulness versus review length | The adjusted spline analysis suggests that predicted helpfulness increases with review length, with a steeper rise among shorter reviews and a more gradual increase among longer reviews. The relationship is not well summarized by a simple linear effect on raw length. |
| Figure 4: Total helpfulness-vote counts | The median total vote count among reviews with valid rates is 2, and 37.7% have only one vote. Observed rates can therefore be based on very limited evidence. |
| Figure 5: Optional text comparison | The text-enhanced model reduced confirmation-set MAE from 0.2487 to 0.2187, an absolute reduction of 0.0300 (12.1%). This predictive result belongs in the held-out evaluation section, not as an EDA-only conclusion. |

**Interpretation:** Helpfulness rates are undefined when the total vote count is zero. Figure 2 includes only valid rates; sample sizes should be shown where practical. The spline result describes an adjusted association, not a causal effect. Use the exact generated plots and their labels in the final report.

---

## 4. Problem Statement and Novelty

### 4.1 Basic descriptive problem

Basic descriptive analysis can establish how ratings are distributed, how review lengths vary, and whether observed helpfulness rates differ across ratings or review-length groups.

These questions characterize the dataset, but they do not establish whether the available features can predict helpfulness for reviews associated with previously unseen products.

### 4.2 Project problem statement

This project investigates whether review characteristics and textual content can explain and predict the helpfulness of Amazon Fine Food Reviews.

The primary predictive question is:

**Does incorporating review text improve held-out helpfulness prediction beyond structured review characteristics?**

The structured baseline will use selected review attributes, such as rating, review length, summary length, and review year. The text-enhanced model will additionally use TF-IDF features extracted from review summaries and bodies.

Both models will be evaluated using the same held-out confirmation set to make their predictive performance comparable.

### 4.3 Project contribution

The project extends beyond basic EDA in three ways:

1. **Descriptive analysis:** Characterizes rating distributions, review lengths, and helpfulness-vote patterns.
2. **Statistical analysis:** Tests pre-specified hypotheses about associations between review characteristics and helpfulness.
3. **Predictive comparison:** Evaluates whether textual information improves prediction beyond structured metadata on products excluded from model development.

The product-grouped split makes the predictive evaluation specifically relevant to generalization across products not represented in the exploration set.

This is the project's defined analytical contribution. The project does not claim that helpfulness prediction or TF-IDF-based review analysis is itself a novel research idea.

---

## 5. Relationship Between EDA and Hypothesis Testing

EDA is used to understand the data and motivate questions; it is not sufficient to establish statistically significant relationships.

The following hypotheses are planned for formal evaluation:

- **H1 — Rating extremity:** Reviews with ratings farther from the rating midpoint may have different helpfulness rates. The primary directional hypothesis is that helpfulness increases with absolute distance from rating 3.
- **H2 — Review length:** Longer reviews may be positively associated with helpfulness.
- **H3 — Summary length:** Summary length may provide additional explanatory information beyond review length and rating.
- **H4 — Textual information:** Adding TF-IDF features may improve held-out predictive performance compared with a structured-only model.

The precise null and alternative hypotheses, significance levels, test procedures, practical-effect thresholds, and confidence intervals will be documented before examining the corresponding confirmatory results.

EDA findings will not be presented as confirmation of these hypotheses.

---

## 6. Methodological Considerations

### 6.1 Exploration and confirmation separation

The cleaned dataset is divided by product using a fixed random seed of 42. The exploration set is used for EDA, feature development, and model selection. The confirmation set is reserved for evaluating the final, pre-specified analysis.

### 6.2 Undefined and invalid helpfulness rates

- Reviews with zero helpfulness votes have undefined helpfulness rates.
- Records with helpful-vote counts greater than total-vote counts are flagged as invalid for rate-based analysis.
- These records remain in the cleaned dataset but are excluded from analyses requiring a valid rate.
- The number of records included in each analysis must be reported.

### 6.3 Interpretation of confidence intervals

Confidence intervals quantify uncertainty under the assumptions of the method used. Since reviews may share users, products, or other characteristics, independence assumptions must be considered when selecting inferential procedures.

### 6.4 Association versus causation

The analysis is observational. Differences in helpfulness rates or predictive associations do not establish that changing a review characteristic would cause a change in helpfulness.

### 6.5 Multiple comparisons

The formal hypothesis-testing stage will distinguish pre-specified primary tests from exploratory comparisons. Any adjustments for multiple testing will be documented where applicable.

---

## 7. Completion Checklist

- [x] Generate descriptive EDA statistics from the cleaned exploration split.
- [x] Examine rating distribution, helpfulness patterns, review length, vote counts, and summary characteristics.
- [x] Record sample sizes and relevant descriptive statistics.
- [x] Document the project problem statement and its distinction from descriptive EDA.
- [x] Complete the planned explanatory tests and multiple-testing adjustment.
- [x] Complete the held-out structured-versus-text predictive comparison.
- [ ] Verify every retained figure against its actual image, axis labels, and sample size.
- [ ] Verify whether practical-effect thresholds were finalized before the corresponding results were examined.
- [ ] Select and place the final figures in the report.
- [ ] Complete the report and check the eight-page limit.

## 8. Expected Outcome

This stage will produce a set of reproducible EDA figures and a clearly motivated research question. The resulting observations will guide the hypothesis-testing and regression stages, while the held-out evaluation will determine whether textual features provide meaningful predictive improvements beyond structured review attributes.

**Status:** Research questions, descriptive findings, hypothesis tests, and held-out predictive comparison have been documented. The final report still requires a figure-by-figure verification, consistency checks, and a page-limit review.

### Additional EDA: Empty Review Summaries

We investigated reviews with zero-length summaries after observing an unusually low mean helpfulness rate for this group.

Out of 454,951 reviews in the exploration dataset, 25 (0.005%) had zero-length summaries. Of these, 24 had valid helpfulness rates. Their mean helpfulness rate was 0.0816 and their median was 0.0417. In comparison, the 239,569 reviews with non-empty summaries and valid helpfulness rates had a mean rate of 0.7774 and a median of 1.0000.

The zero-summary group is extremely small, so this difference should be treated as a descriptive observation rather than evidence of a general relationship. We retain these reviews rather than removing them based on this observation. Empty summaries are represented as empty strings during text processing, and helpfulness rates remain undefined when vote counts do not permit a valid rate.

**Takeaway:** Empty summaries are rare in the dataset, but their observed helpfulness rates differ substantially from those of non-empty summaries. The small sample size prevents a reliable general conclusion.