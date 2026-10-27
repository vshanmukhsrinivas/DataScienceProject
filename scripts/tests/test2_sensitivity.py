
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

DATA_PATH = "data/splits/exploration.csv.gz"
REPORT_PATH = Path("reports/test2_sensitivity.json")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05

# 1. Load the same exploration data.
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

# 2. Apply the same valid-vote filtering as Test 2.
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
df["TotalVotes"] = df["HelpfulVotes"] + df["UnhelpfulVotes"]
df["HelpfulnessFraction"] = df["HelpfulVotes"] / df["TotalVotes"]
df["RatingExtremity"] = (df["Score"] - 3).abs()
df["LogReviewLength"] = np.log1p(df["ReviewLengthChars"])
df["ReviewDateUTC"] = pd.to_datetime(
    df["ReviewDateUTC"], errors="coerce", utc=True
)
df["ReviewYear"] = df["ReviewDateUTC"].dt.year

cols = [
    "ProductId", "HelpfulVotes", "UnhelpfulVotes",
    "TotalVotes", "HelpfulnessFraction", "RatingExtremity",
    "ReviewLengthChars", "LogReviewLength",
    "SummaryLengthChars", "ReviewYear",
]
model_df = (
    df[cols]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .copy()
)

# Center the continuous predictors.
for col in ["LogReviewLength", "SummaryLengthChars", "ReviewYear"]:
    model_df[col + "_c"] = model_df[col] - model_df[col].mean()

# 3. Fit product-grouped GEE.
# Same response, predictors, observations and vote-count weights
# as the primary Test 2 analysis.
formula = (
    "HelpfulnessFraction ~ RatingExtremity + "
    "LogReviewLength_c + SummaryLengthChars_c + ReviewYear_c"
)

gee = smf.gee(
    formula=formula,
    groups="ProductId",
    data=model_df,
    family=sm.families.Binomial(),
    cov_struct=sm.cov_struct.Independence(),
    weights=model_df["TotalVotes"],
)

result = gee.fit()

# 4. Extract the pre-specified one-sided review-length test.
term = "LogReviewLength_c"
beta = float(result.params[term])
se = float(result.bse[term])
z_stat = beta / se
p_one_sided = float(norm.sf(z_stat))

ci = result.conf_int().loc[term]
ci_low = float(ci.iloc[0])
ci_high = float(ci.iloc[1])

# 5. Express the effect across the same raw-length IQR.
q1, q3 = model_df["ReviewLengthChars"].quantile([0.25, 0.75])
log_iqr = float(np.log1p(q3) - np.log1p(q1))

or_iqr = float(np.exp(beta * log_iqr))
or_iqr_ci = [
    float(np.exp(ci_low * log_iqr)),
    float(np.exp(ci_high * log_iqr)),
]

report = {
    "analysis": "Test 2 sensitivity: product-grouped GEE",
    "dataset": DATA_PATH,
    "n_reviews": int(len(model_df)),
    "n_products": int(model_df["ProductId"].nunique()),
    "method": (
        "Binomial GEE, independence working correlation, "
        "vote-count weights, product-grouped robust covariance"
    ),
    "hypotheses": {
        "H0": "beta_LogReviewLength = 0",
        "H1": "beta_LogReviewLength > 0",
    },
    "alpha": ALPHA,
    "coefficient": beta,
    "robust_standard_error": se,
    "z_statistic": float(z_stat),
    "p_value_one_sided": p_one_sided,
    "coefficient_95_ci": [ci_low, ci_high],
    "review_length_q1_chars": float(q1),
    "review_length_q3_chars": float(q3),
    "log_length_iqr": log_iqr,
    "odds_ratio_per_log_unit": float(np.exp(beta)),
    "odds_ratio_across_length_iqr": or_iqr,
    "odds_ratio_across_length_iqr_95_ci": or_iqr_ci,
    "decision": (
        "Reject H0" if p_one_sided < ALPHA else "Fail to reject H0"
    ),
    "converged": bool(result.converged),
    "caution": (
        "This sensitivity model uses the same observations and "
        "predictors as the primary analysis. Similar estimates support "
        "stability across specifications but do not independently "
        "validate the model or eliminate all overdispersion concerns."
    ),
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, allow_nan=False)

print("=" * 65)
print("TEST 2 SENSITIVITY: PRODUCT-GROUPED GEE")
print("=" * 65)
print(json.dumps(report, indent=2))
print("\nModel summary:")
print(result.summary())
print(f"\nSaved report: {REPORT_PATH}")
