
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

DATA_PATH = "data/splits/exploration.csv.gz"
REPORT_PATH = Path("reports/test1_sensitivity.json")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05

# 1. Load the same exploration data as Test 1.
df = pd.read_csv(DATA_PATH, compression="gzip")

numeric_cols = [
    "Score",
    "ReviewLengthChars",
    "SummaryLengthChars",
    "HelpfulnessNumerator",
    "HelpfulnessDenominator",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

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
    "ProductId", "HelpfulnessFraction", "TotalVotes",
    "RatingExtremity", "LogReviewLength",
    "SummaryLengthChars", "ReviewYear",
]
model_df = (
    df[cols]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .copy()
)

# Center predictors consistently with Test 1.
for col in ["LogReviewLength", "SummaryLengthChars", "ReviewYear"]:
    model_df[col + "_c"] = model_df[col] - model_df[col].mean()

formula = (
    "HelpfulnessFraction ~ RatingExtremity + "
    "LogReviewLength_c + SummaryLengthChars_c + ReviewYear_c"
)

# 2. GEE: reviews are grouped by product.
# Independence is the working correlation structure;
# the robust covariance estimates uncertainty at product level.
gee = smf.gee(
    formula=formula,
    groups="ProductId",
    data=model_df,
    family=sm.families.Binomial(),
    cov_struct=sm.cov_struct.Independence(),
    weights=model_df["TotalVotes"],
)

result = gee.fit()

term = "RatingExtremity"
beta = float(result.params[term])
se = float(result.bse[term])
z_stat = beta / se
p_one_sided = float(norm.sf(z_stat))

ci = result.conf_int().loc[term]
ci_low, ci_high = float(ci.iloc[0]), float(ci.iloc[1])

report = {
    "analysis": "Test 1 sensitivity: product-grouped GEE",
    "dataset": DATA_PATH,
    "n_reviews": int(len(model_df)),
    "n_products": int(model_df["ProductId"].nunique()),
    "method": "Binomial GEE, independence working correlation, vote-count weights, robust covariance",
    "hypotheses": {
        "H0": "beta_RatingExtremity = 0",
        "H1": "beta_RatingExtremity > 0",
    },
    "alpha": ALPHA,
    "coefficient": beta,
    "robust_standard_error": se,
    "z_statistic": z_stat,
    "p_value_one_sided": p_one_sided,
    "coefficient_95_ci": [ci_low, ci_high],
    "odds_ratio_per_extremity_unit": float(np.exp(beta)),
    "odds_ratio_95_ci": [
        float(np.exp(ci_low)),
        float(np.exp(ci_high)),
    ],
    "decision": "Reject H0" if p_one_sided < ALPHA else "Fail to reject H0",
    "converged": bool(result.converged),
    "caution": (
        "GEE is a sensitivity analysis, not a complete solution to "
        "all possible overdispersion or model misspecification. "
        "Vote-count weighting and the working correlation assumption "
        "should be documented."
    ),
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, allow_nan=False)

print("=" * 65)
print("TEST 1 SENSITIVITY: PRODUCT-GROUPED GEE")
print("=" * 65)
print(json.dumps(report, indent=2))
print("\nModel summary:")
print(result.summary())
print(f"\nSaved report: {REPORT_PATH}")
