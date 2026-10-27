
import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.stats import norm

DATA_PATH = "data/splits/exploration.csv.gz"
REPORT_PATH = Path("reports/test2_review_length.json")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
MIN_MEANINGFUL_OR_IQR = 1.05

# 1. Load exploration data and validate columns.
df = pd.read_csv(DATA_PATH, compression="gzip")

required = [
    "ProductId", "Score", "ReviewLengthChars",
    "SummaryLengthChars", "ReviewDateUTC",
    "HelpfulnessNumerator", "HelpfulnessDenominator",
]
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing columns: {missing}")

numeric_cols = [
    "Score", "ReviewLengthChars", "SummaryLengthChars",
    "HelpfulnessNumerator", "HelpfulnessDenominator",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 2. Keep only reviews with valid helpfulness vote counts.
num = df["HelpfulnessNumerator"]
den = df["HelpfulnessDenominator"]

valid = (
    num.ge(0) & den.gt(0) & num.le(den)
    & num.notna() & den.notna()
)
df = df.loc[valid].copy()

df["HelpfulVotes"] = df["HelpfulnessNumerator"]
df["UnhelpfulVotes"] = (
    df["HelpfulnessDenominator"] - df["HelpfulnessNumerator"]
)
df["RatingExtremity"] = (df["Score"] - 3).abs()
df["LogReviewLength"] = np.log1p(df["ReviewLengthChars"])
df["ReviewDateUTC"] = pd.to_datetime(
    df["ReviewDateUTC"], errors="coerce", utc=True
)
df["ReviewYear"] = df["ReviewDateUTC"].dt.year

columns = [
    "ProductId", "HelpfulVotes", "UnhelpfulVotes",
    "ReviewLengthChars", "LogReviewLength",
    "RatingExtremity", "SummaryLengthChars", "ReviewYear",
]
model_df = (
    df[columns]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .copy()
)

if len(model_df) == 0:
    raise ValueError("No usable observations remain.")

# 3. Center continuous predictors.
for col in ["LogReviewLength", "SummaryLengthChars", "ReviewYear"]:
    model_df[col + "_c"] = model_df[col] - model_df[col].mean()

# 4. Fit grouped-binomial GLM.
endog = np.column_stack([
    model_df["HelpfulVotes"],
    model_df["UnhelpfulVotes"],
])

formula = (
    "RatingExtremity + LogReviewLength_c + "
    "SummaryLengthChars_c + ReviewYear_c"
)
X = patsy.dmatrix(formula, data=model_df, return_type="dataframe")

result = sm.GLM(
    endog,
    X,
    family=sm.families.Binomial(),
).fit(
    cov_type="cluster",
    cov_kwds={"groups": model_df["ProductId"]},
)

# 5. Pre-specified one-sided test:
# H0: beta_length = 0
# H1: beta_length > 0
term = "LogReviewLength_c"
beta = float(result.params[term])
se = float(result.bse[term])
z_stat = beta / se
p_one_sided = float(norm.sf(z_stat))

ci = result.conf_int().loc[term]
ci_low, ci_high = float(ci.iloc[0]), float(ci.iloc[1])

# Translate the coefficient into an odds ratio across the
# observed IQR of raw review length.
q1, q3 = model_df["ReviewLengthChars"].quantile([0.25, 0.75])
log_length_iqr = float(np.log1p(q3) - np.log1p(q1))

or_iqr = float(np.exp(beta * log_length_iqr))
or_iqr_ci = [
    float(np.exp(ci_low * log_length_iqr)),
    float(np.exp(ci_high * log_length_iqr)),
]

report = {
    "analysis": "Test 2: review length",
    "dataset": DATA_PATH,
    "n_reviews": int(len(model_df)),
    "n_products": int(model_df["ProductId"].nunique()),
    "total_helpful_votes": int(model_df["HelpfulVotes"].sum()),
    "total_unhelpful_votes": int(model_df["UnhelpfulVotes"].sum()),
    "hypotheses": {
        "H0": "beta_LogReviewLength = 0",
        "H1": "beta_LogReviewLength > 0",
    },
    "alpha": ALPHA,
    "test": "Grouped-binomial GLM, product-clustered standard errors",
    "coefficient": beta,
    "cluster_robust_standard_error": se,
    "z_statistic": float(z_stat),
    "degrees_of_freedom": None,
    "p_value_one_sided": p_one_sided,
    "coefficient_95_ci": [ci_low, ci_high],
    "review_length_q1_chars": float(q1),
    "review_length_q3_chars": float(q3),
    "log_length_iqr": log_length_iqr,
    "odds_ratio_per_log_unit": float(np.exp(beta)),
    "odds_ratio_across_length_iqr": or_iqr,
    "odds_ratio_across_length_iqr_95_ci": or_iqr_ci,
    "practical_threshold_or_iqr": MIN_MEANINGFUL_OR_IQR,
    "nominal_decision": (
        "Reject H0" if p_one_sided < ALPHA else "Fail to reject H0"
    ),
    "practical_threshold_reached_by_point_estimate": (
        or_iqr >= MIN_MEANINGFUL_OR_IQR
    ),
    "convergence": bool(result.converged),
    "caution": (
        "The test uses product-clustered asymptotic inference. "
        "The practical threshold is assessed using the point estimate; "
        "statistical significance does not establish causation. "
        "Tests 1-3 are subject to the pre-specified Holm adjustment."
    ),
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, allow_nan=False)

print("=" * 65)
print("TEST 2: REVIEW LENGTH")
print("=" * 65)
print(json.dumps(report, indent=2))
print("\nModel coefficients:")
print(result.summary())
print(f"\nSaved report: {REPORT_PATH}")
