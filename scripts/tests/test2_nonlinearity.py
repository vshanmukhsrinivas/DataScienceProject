
import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.stats import chi2

DATA_PATH = "data/splits/exploration.csv.gz"
REPORT_PATH = Path("reports/test2_nonlinearity.json")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 1. Load exploration data and validate the required columns.
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

# 2. Apply the same valid-vote filters as Test 2.
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
df["LogReviewLength"] = np.log1p(df["ReviewLengthChars"])
df["RatingExtremity"] = (df["Score"] - 3).abs()
df["ReviewDateUTC"] = pd.to_datetime(
    df["ReviewDateUTC"], errors="coerce", utc=True
)
df["ReviewYear"] = df["ReviewDateUTC"].dt.year

cols = [
    "ProductId", "HelpfulVotes", "UnhelpfulVotes",
    "LogReviewLength", "RatingExtremity",
    "SummaryLengthChars", "ReviewYear",
]
model_df = (
    df[cols]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .copy()
)

if len(model_df) == 0:
    raise ValueError("No usable observations remain.")

# Center the continuous predictors.
for col in ["LogReviewLength", "SummaryLengthChars", "ReviewYear"]:
    model_df[col + "_c"] = model_df[col] - model_df[col].mean()

# 3. Grouped-binomial response: helpful and unhelpful votes.
endog = np.column_stack([
    model_df["HelpfulVotes"],
    model_df["UnhelpfulVotes"],
])

# Model A: linear effect of log review length.
formula_linear = (
    "RatingExtremity + LogReviewLength_c + "
    "SummaryLengthChars_c + ReviewYear_c"
)

X_linear = patsy.dmatrix(
    formula_linear,
    data=model_df,
    return_type="dataframe",
)

linear_result = sm.GLM(
    endog,
    X_linear,
    family=sm.families.Binomial(),
).fit(
    cov_type="cluster",
    cov_kwds={"groups": model_df["ProductId"]},
)

# Model B: flexible cubic regression spline for log review length.
formula_spline = (
    "RatingExtremity + "
    "bs(LogReviewLength_c, df=4, degree=3, "
    "include_intercept=False) + "
    "SummaryLengthChars_c + ReviewYear_c"
)

X_spline = patsy.dmatrix(
    formula_spline,
    data=model_df,
    return_type="dataframe",
)

spline_result = sm.GLM(
    endog,
    X_spline,
    family=sm.families.Binomial(),
).fit(
    cov_type="cluster",
    cov_kwds={"groups": model_df["ProductId"]},
)

# 4. Compare model fit.
# The likelihood-ratio comparison assumes the binomial likelihood
# is adequately specified. Given the dispersion diagnostic from
# Test 1, interpret this p-value cautiously.
lr_stat = max(
    0.0,
    float(2 * (spline_result.llf - linear_result.llf))
)
df_difference = int(
    spline_result.df_model - linear_result.df_model
)
lr_p_value = (
    float(chi2.sf(lr_stat, df_difference))
    if df_difference > 0 else None
)

# AIC comparison is descriptive; smaller is preferred.
aic_linear = float(linear_result.aic)
aic_spline = float(spline_result.aic)

# 5. Record estimates and diagnostics.
report = {
    "analysis": "Test 2 sensitivity: review-length nonlinearity",
    "dataset": DATA_PATH,
    "n_reviews": int(len(model_df)),
    "n_products": int(model_df["ProductId"].nunique()),
    "linear_model": {
        "description": "Linear effect of log review length",
        "log_likelihood": float(linear_result.llf),
        "aic": aic_linear,
        "review_length_coefficient": float(
            linear_result.params["LogReviewLength_c"]
        ),
        "review_length_cluster_robust_se": float(
            linear_result.bse["LogReviewLength_c"]
        ),
    },
    "spline_model": {
        "description": "Cubic regression spline with 4 basis degrees of freedom",
        "log_likelihood": float(spline_result.llf),
        "aic": aic_spline,
        "converged": bool(spline_result.converged),
    },
    "model_comparison": {
        "likelihood_ratio_statistic": lr_stat,
        "degrees_of_freedom_difference": df_difference,
        "likelihood_ratio_p_value": lr_p_value,
        "aic_difference_linear_minus_spline": aic_linear - aic_spline,
        "interpretation": (
            "Positive AIC difference favors the spline model. "
            "The likelihood-ratio test compares the nested likelihoods "
            "under standard binomial assumptions."
        ),
    },
    "caution": (
        "The likelihood-ratio p-value is not product-cluster robust and "
        "relies on binomial likelihood assumptions. Earlier diagnostics "
        "showed a Pearson dispersion ratio around 2.12. Treat this as a "
        "diagnostic comparison, not definitive evidence of nonlinearity. "
        "If the spline is favored, inspect its predicted curve and "
        "consider a product-aware or robust follow-up comparison."
    ),
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, allow_nan=False)

print("=" * 65)
print("TEST 2: REVIEW-LENGTH NONLINEARITY CHECK")
print("=" * 65)
print(json.dumps(report, indent=2))
print("\nLinear model:")
print(linear_result.summary())
print("\nSpline model:")
print(spline_result.summary())
print(f"\nSaved report: {REPORT_PATH}")
