# Data Preparation and Justification

> **Status:** Working draft based on the current preparation scripts and observed outputs. This document distinguishes decisions already implemented from alternatives that have only been considered and still need empirical comparison. Do not describe an alternative as “tested” until its comparison has been run.

## 1. Dataset and preparation objective

The project uses the Stanford SNAP Amazon Fine Food Reviews dataset. The parsed raw dataset contains **568,454 review records and 14 columns**. The preparation goal is to retain useful review information, correct clear record-level issues, construct a defensible helpfulness outcome, and produce reproducible features for exploratory analysis and modeling.

The original source and parsed dataset are preserved. Preparation is performed on a separate analytical copy.

## 2. Before-and-after summary

| Measure | Before preparation | After first cleaning pass | Interpretation |
|---|---:|---:|---|
| Review rows | 568,454 | 568,173 | 281 exact duplicate rows removed |
| Columns | 14 | 16 | Added `InvalidHelpfulness` and `HelpfulnessRate` |
| Missing `ProfileName` | 26 | 26 | Retained; profile name is not required for the main research question |
| Missing `Summary` | 27 | 0 | Represented as empty strings for text processing; no summary text was invented |
| Missing `Text` | 0 | 0 | No missing review bodies observed |
| Invalid helpfulness counts (`numerator > denominator`) | 2 | 2 flagged | Records retained but excluded from analyses requiring valid rates |
| Zero vote denominator | 270,052 in the earlier raw inspection | 269,850 | Count is after exact-row deduplication; rate remains undefined |
| Valid helpfulness rates | Not yet added | 298,321 | Rate calculated only when denominator > 0 and numerator ≤ denominator |
| Reviews with undefined helpfulness rate | Not yet added | 269,852 | Includes zero-denominator and invalid-count records |
| Review length: median | 302 characters in initial inspection | 301 characters | Small change after duplicate removal and whitespace normalization |
| Review length: 95th percentile | 1,202 characters in initial inspection | 1,201 characters | Long tail remains |
| Review length: 99th percentile | 2,166 characters in initial inspection | 2,164 characters | Long tail remains |
| Maximum review length | 21,409 characters | 21,409 characters | Retained; maximum alone is not proof of an error |
| Rows above review-length upper IQR threshold | Not previously calculated | 38,773 (threshold 1,049 characters) | Diagnostic flag, not automatic deletion |
| Rows above summary-length upper IQR threshold | Not previously calculated | 18,433 (threshold 55.5 characters) | Diagnostic flag, not automatic deletion |

**Count clarification:** The initial inspection counted 270,052 zero-denominator records before deduplication. The cleaned dataset has 269,850. The valid rate count is 298,321. These categories reconcile: 269,850 zero-denominator + 2 invalid helpfulness = 269,852 undefined rates; 568,173 − 269,852 = 298,321 valid rates.

## 3. Preparation decisions, alternatives, and evidence

### 3.1 Exact duplicate rows

**Observed evidence:** 568,454 rows before cleaning; `DataFrame.duplicated()` identified 281 rows beyond the first occurrence. After dropping exact duplicate rows, 568,173 remain.

**Chosen method:** Remove exact duplicate rows from the analytical copy and preserve the source/parsed input unchanged.

**Why:** Exact copies can give repeated observations extra weight in summaries and models without adding new information.

**Alternatives considered:**
- Keep every row: preserves every occurrence but risks giving exact copies additional influence.
- Remove all records participating in duplicate groups: more aggressive and can discard the first valid copy as well.
- Deduplicate on selected columns such as user, text, date, and score: may merge distinct reviews or remove legitimate cross-product reviews.

**Evidence-based choice:** The current measured effect is 281 rows removed, about 0.0495% of the original dataset. We have not yet run a model sensitivity comparison with and without duplicates, so we should not claim that removal empirically improves prediction. It is a conservative data-integrity decision. A follow-up audit should check for rows that become duplicates only after text normalization.

### 3.2 Missing values

**Observed evidence:** 26 missing profile names, 27 missing summaries, and no missing review bodies in the initial inspection. After preparation, 26 profile names remain missing; missing summaries are represented as empty strings.

**Chosen method:**
- Retain missing `ProfileName` values because profile names are not required for the primary research question.
- Convert missing `Summary` values to empty strings when constructing text input for TF-IDF.
- Do not fabricate review text or profile names.

**Alternatives considered:**
- Drop rows with missing profile name or summary: would remove up to 53 rows if the missing sets did not overlap; the actual overlap should be checked before reporting a combined row-removal total.
- Impute a placeholder such as `"unknown"`: could create an artificial token and mislead text analysis.
- For summaries, use an empty string: preserves the review and avoids inventing content.

**Evidence-based choice:** The observed missingness is small and confined to fields that are not essential to the primary target. We have not yet compared model results under row deletion versus empty-summary handling; describe the current choice as a low-loss, semantically conservative approach rather than a proven predictive improvement.

### 3.3 Invalid helpfulness counts and undefined rates

**Observed evidence:** Two records have helpful votes greater than total votes. The cleaned dataset contains 269,850 records with a zero vote denominator.

**Chosen method:** Preserve records, add `InvalidHelpfulness`, and calculate
`HelpfulnessRate = HelpfulnessNumerator / HelpfulnessDenominator`
only when the denominator is positive and the counts are valid. Leave the rate undefined otherwise.

**Alternatives considered:**
- Set a zero-denominator rate to zero: rejected because no votes does not mean zero helpfulness.
- Drop all zero-denominator reviews from the dataset: rejected for general analysis because they still contain useful text and rating information.
- Remove the two invalid records entirely: unnecessary for analyses that do not use helpfulness counts; flag and exclude only when a valid rate is required.
- Use a vote-count threshold (e.g., at least 3, 5, or 10 votes): possible sensitivity analysis, but it reduces sample size and changes the population being analyzed.

**Evidence-based choice:** The target is mathematically undefined when the denominator is zero, and two records violate the count constraint. The rule is based on the definition and validity of the measure. The threshold alternatives have not yet been compared inferentially; any minimum-vote threshold should be specified before hypothesis testing and reported as a sensitivity analysis.

### 3.4 Outliers and long reviews

**Observed evidence on the cleaned dataset:**
- Review length: Q1 = 179, Q3 = 527, IQR = 348; upper IQR threshold = 1,049 characters; 38,773 reviews above threshold.
- Summary length: Q1 = 13, Q3 = 30, IQR = 17; upper IQR threshold = 55.5 characters; 18,433 summaries above threshold.
- Review length median = 301; 95th percentile = 1,201; 99th percentile = 2,164; maximum = 21,409.
- Summary length maximum = 128.
- Helpfulness denominator maximum = 923; helpful numerator maximum = 866.

**Chosen method:** Do not automatically remove IQR-flagged values. Retain them for now and investigate distributional effects. For modeling, consider a `log1p` transformation of review length and compare it with the untransformed version using exploration-only validation.

**Alternatives considered:**
- Delete IQR outliers: simple, but risks deleting legitimate detailed reviews and high-vote reviews.
- Winsorize/cap lengths: limits extreme influence but changes observed values.
- Apply `log1p(length)`: preserves row order and reduces skew, while changing the scale.
- Keep raw length: easiest to interpret, but may be sensitive to extreme values.

**Evidence-based choice:** IQR flags identify 38,773 long reviews, but this rule alone does not prove errors. Because length is a meaningful review characteristic, retaining records is the least destructive initial decision. We have not yet run the transformation comparison; no transformation should be described as empirically superior until its results are available.

### 3.5 Encoding and text representation

**Implemented feature-count demonstration:** On the earlier exploration split (455,172 rows), a TF-IDF vectorizer produced 20,000 text features, combined with five structured features, for 20,005 columns. The matrix dimensions were:

`455,172 × 20,005 = 9,105,715,860`

This exceeded the guideline threshold of 3,000,000. The sparse matrix had 20,440,498 stored nonzero entries and density approximately 0.224%.

**Important caveat:** This feature matrix was created before duplicate removal. It must be regenerated from the cleaned data and the new exploration split before it is treated as the final prepared feature matrix or reported as the final arithmetic.

**Chosen approach:** Use TF-IDF with unigrams and bigrams for an interpretable text baseline, alongside meaningful structured features such as rating, review length, summary length, and review year. Fit the vectorizer on exploration data only; transform confirmation data with that fitted vectorizer.

**Alternatives:**
- Bag-of-words counts: simple and interpretable, but ignores how common terms are across documents.
- TF-IDF: downweights words common across many reviews and is efficient for sparse text data.
- Learned embeddings: can represent semantic relationships, but require a model and additional computation; not necessary just to satisfy the feature-count rule.
- Drop text and use structured features only: useful as a baseline, but cannot test whether text improves prediction.

**Evidence-based choice:** TF-IDF is selected as a transparent baseline that satisfies the dimensionality requirement and supports a structured-only versus structured-plus-text comparison. We have demonstrated the feature count, but have not yet established that TF-IDF predicts helpfulness better than other text representations. That claim must wait for held-out model comparisons.

## 4. What we have statistically established—and what remains to test

### Established by current outputs
1. Exact duplicate removal reduced the dataset by 281 rows (about 0.0495%).
2. Missingness is limited to profile names and summaries; the review body had no missing values in the initial inspection.
3. There are two invalid helpfulness-count records.
4. A zero denominator occurs frequently, so the helpfulness rate is undefined for those reviews.
5. Review and summary lengths have long right tails; IQR flags many observations, but these are not proven errors.
6. The earlier TF-IDF matrix met the dimensionality requirement, but it must be regenerated after cleaning.

### Not yet established
- That dropping duplicates improves model accuracy or hypothesis-test validity in this particular dataset.
- That deleting missing-summary rows or imputing a placeholder improves results.
- That removing, capping, or log-transforming long reviews is better than retaining raw lengths.
- That a minimum helpfulness-vote threshold is optimal.
- That TF-IDF outperforms count vectors or learned embeddings.

We should not invent comparisons or p-values. To support a stronger “we explored alternatives and selected this method statistically” statement, run the following on exploration data only, with the same product-grouped validation folds and fixed random seed:
1. Raw length vs. `log1p` length vs. capped length, comparing validation MAE (and RMSE as a secondary metric).
2. Structured-only features vs. structured + TF-IDF, comparing MAE and a paired bootstrap confidence interval for the difference in MAE.
3. Count-vector text vs. TF-IDF, using the same folds and vocabulary limits.
4. If useful, valid-rate models using all positive denominators vs. a pre-specified minimum-vote subset (e.g. denominator ≥ 3 or ≥ 5), explicitly reporting the sample-size change and interpreting the different target population.

Use the confirmation set only after choices are fixed. Do not use confirmation results to choose preprocessing settings.

## 5. Suggested report wording

“We preserved the original parsed dataset and performed preparation on a separate analytical copy. We removed 281 exact duplicate rows, retaining 568,173 records. Missing profile names were retained because they were not required for the primary analysis; missing summaries were represented as empty text for vectorization rather than imputed with invented words. Helpfulness rates were calculated only for records with a positive denominator and valid counts. Two records with helpful votes greater than total votes were flagged and excluded only from analyses requiring valid helpfulness rates. We retained long reviews and high-vote observations because IQR thresholds are diagnostic flags, not proof of erroneous data. The cleaned review-length distribution had a median of 301 characters, a 99th percentile of 2,164, and a maximum of 21,409. We will compare alternative length transformations and text representations using exploration-only validation before making predictive-performance claims. TF-IDF is used as an interpretable sparse text representation; the final feature-count calculation will be regenerated after the cleaned data are split.”

## 6. Reproducibility files

Current scripts and reports:
- `scripts/prepare_data.py`
- `scripts/analyze_prepared_data.py`
- `reports/preparation_report.json`
- `reports/outlier_analysis.json`

Next actions:
1. Audit duplicates after normalization.
2. Recreate the product-grouped split from `finefoods_cleaned.csv.gz`.
3. Refit and save TF-IDF from the new exploration set.
4. Recalculate final before/after and feature-count reports.
5. Run the specified sensitivity/model comparisons before claiming a statistically superior alternative.
