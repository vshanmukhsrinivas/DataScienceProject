
# Amazon Fine Food Reviews: Data Science and Statistical Analysis

A reproducible analysis of the Stanford SNAP Amazon Fine Food Reviews dataset, studying review helpfulness, rating extremity, review length, and the predictive value of review text.

## 1. Research Questions

1. Are more extreme ratings associated with higher helpfulness, after adjusting for observable review characteristics?
2. Is review length associated with helpfulness after adjustment, and is the relationship nonlinear?
3. Does summary length contribute additional explanatory information?
4. Does review text improve the prediction of helpfulness beyond structured metadata?

These are observational questions. Associations do not establish causation.

## 2. Dataset

**Source:** Stanford Network Analysis Project (SNAP), Amazon Fine Food Reviews.

The dataset contains product reviews, user and product identifiers, ratings, review text and summaries, timestamps, and helpfulness vote counts.

The parser preserves raw values and derives numeric fields. The original source file is retained unchanged.

### Data preparation

The preparation pipeline:

- Parses the original text format into a compressed CSV.
- Trims selected text fields and converts empty text fields to missing values.
- Removes 281 exact duplicate rows, leaving 568,173 reviews.
- Fills missing summaries with empty strings for text processing.
- Recalculates review and summary lengths after trimming.
- Flags invalid helpfulness counts instead of silently correcting them.
- Leaves helpfulness rate undefined when the denominator is zero or the counts are invalid.
- Retains long reviews; length outliers are investigated rather than automatically deleted.

There are 269,850 reviews with zero helpfulness-vote denominator in the cleaned dataset. These reviews remain in the prepared dataset but cannot contribute to analyses requiring a valid helpfulness rate.

See `reports/data_preparation_documentation.md`, `reports/data_dictionary.md`, and `reports/preparation_report.json`.

## 3. Exploration and Confirmation Split

The cleaned data is split by product ID using seed `42`. Products do not overlap between the exploration and confirmation partitions.

| Partition | Reviews | Purpose |
|---|---:|---|
| Exploration | 454,951 | Exploratory analysis and model development |
| Confirmation | 113,222 | Held-out evaluation |

The predictive analysis uses only reviews with valid helpfulness rates, so its eligible sample sizes are smaller than the full partition sizes. The exact sample sizes are recorded in the corresponding test reports.

See `data/splits/split_report.json`.

## 4. Exploratory Data Analysis

Key observations from the exploration partition:

- Five-star reviews account for approximately 63.9% of reviews.
- The median review length is about 302 characters; the 95th percentile is about 1,203 characters.
- Among reviews with valid helpfulness rates, mean helpfulness rate increases from approximately 0.540 for one-star reviews to 0.872 for five-star reviews.
- The median helpfulness-vote denominator is two, so many review-level rates are based on few votes.
- Empty summaries are rare (25 reviews in the exploration partition) and are retained.

The plots are stored in `reports/figures/`. See `reports/requirement_4_eda_and_novelty.md` and `reports/eda_deep_dive.json` for detailed findings and interpretations.

## 5. Feature-Scale Requirement

The exploration feature matrix combines five structured features with 20,000 TF-IDF text features.

- Samples: 454,951
- Features: 20,005
- Arithmetic: 454,951 × 20,005 = **9,101,294,755 sample-feature positions**

This exceeds the 3,000,000 requirement. The matrix is sparse; the arithmetic counts all sample-feature positions, including zeros. TF-IDF represents text using weighted word and phrase features, not semantic embeddings.

The vectorizer is fitted on the exploration partition and then used to transform the confirmation partition.

See `reports/feature_scale.json`.

## 6. Hypothesis Tests

The first three tests concern explanatory associations. Their product-clustered Wald p-values are adjusted together using the Holm procedure. Test 4 is a separate held-out predictive comparison.

### Test 1 — Rating extremity

- **H0:** The adjusted rating-extremity coefficient is zero.
- **H1:** The coefficient is positive.
- **Method:** Grouped-binomial regression with product-clustered standard errors.
- **Result:** Coefficient ≈ 0.349; odds ratio ≈ 1.418 per one-unit increase in rating extremity.
- **Holm-adjusted p-value:** Approximately 5.13 × 10⁻¹⁵⁷.

The estimated association is positive and statistically significant. The model's overdispersion and observational design remain important limitations.

### Test 2 — Review length

- **H0:** The adjusted log-review-length coefficient is zero.
- **H1:** The coefficient is positive.
- **Method:** Grouped-binomial regression with product-clustered standard errors.
- **Result:** Odds ratio ≈ 1.641 across the interquartile range of review length.
- **Holm-adjusted p-value:** Approximately 8.11 × 10⁻²⁰⁷.

A spline analysis indicates that the relationship is nonlinear. Its ordinary likelihood-ratio comparison is treated as diagnostic because overdispersion affects standard likelihood-based inference.

### Test 3 — Summary length

- **H0:** Summary length adds no explanatory information under the specified model.
- **H1:** Summary length contributes additional information.
- **Method:** Nested grouped-binomial models; product-clustered Wald inference is used for the reported sensitivity result.
- **Result:** The estimated coefficient is approximately −0.00167 per character; odds ratio ≈ 0.983 per 10 characters.
- **Holm-adjusted product-clustered Wald p-value:** Approximately 0.00109.
- **Relative deviance reduction:** Approximately 0.0181%.

The association is statistically significant, but the observed deviance reduction is below the proposed 1% practical benchmark.

### Test 4 — Predictive value of review text

- **H0:** Adding TF-IDF features does not improve expected mean absolute error (MAE).
- **H1:** Adding TF-IDF features reduces expected MAE.
- **Method:** Compare a structured-feature Ridge model with a structured-plus-TF-IDF Ridge model on the confirmation partition; estimate uncertainty by resampling products.
- **Structured-only MAE:** ≈ 0.248705.
- **Text-model MAE:** ≈ 0.218671.
- **MAE reduction:** ≈ 0.030034, or 12.08% relative to baseline.
- **95% product-bootstrap interval for reduction:** Approximately [0.02620, 0.03404].

The interval is above zero and above the proposed 0.01 practical threshold. The bootstrap-tail p-value is approximate and should not be interpreted as an exact null-test p-value.

Detailed results and assumptions are documented in `reports/test_reports/` and `reports/test4_documentation.md`.

**Threshold timing:** The practical-effect benchmarks in the analysis documentation should be described as prespecified only if their values were finalized before the relevant results were examined.

## 7. Reproducibility

Run commands from the repository root. Use Python 3 and install the required packages in your environment.

The main workflow is:

```bash
python3 scripts/parse_finefoods.py
python3 scripts/inspect_data.py
python3 scripts/investigate_quality.py
python3 scripts/save_quality_baseline.py
python3 scripts/analyze_duplicates.py
python3 scripts/create_split.py
python3 scripts/eda_exploration.py
python3 scripts/check_feature_scale.py
python3 scripts/prepare_data.py
python3 scripts/analyze_prepared_data.py
python3 scripts/audit_normalized_duplicates.py
python3 scripts/eda_deep_dive.py
