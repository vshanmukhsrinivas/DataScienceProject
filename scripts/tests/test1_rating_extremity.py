
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import norm

DATA_PATH = "data/splits/exploration.csv.gz"
REPORT_PATH = Path("reports/test1_rating_extremity.json")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
MIN_MEANINGFUL_OR = 1.05

# ---------------------------------------------------------
# 1. Load exploration data only
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH, compression="gzip")

required = [
    "ProductId",
    "Score",
    "ReviewLengthChars",
    "SummaryLengthChars",
    "ReviewDateUTC",
    "HelpfulnessNumerator",
    "HelpfulnessDenominator",
]
missing_columns = sorted(set(required) - set(df.columns))
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

for col in [
    "Score",
    "ReviewLengthChars",
    "SummaryLengthChars",
    "HelpfulnessNumerator",
    "HelpfulnessDenominator",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------------------------
# 2. Define valid vote counts and response
# ---------------------------------------------------------
num = df["HelpfulnessNumerator"]
den = df["HelpfulnessDenominator"]

valid = (
    num.ge(0)
    & den.gt(0)
    & num.le(den)
    & num.notna()
    & den.notna()
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

# Missing predictor values are excluded in this initial model.
# We report how many records remain.
model_columns = [
    "ProductId",
    "HelpfulVotes",
    "UnhelpfulVotes",
    "RatingExtremity",
    "LogReviewLength",
    "SummaryLengthChars",
    "ReviewYear",
]
model_df = df[model_columns].replace([np.inf, -np.inf], np.nan).dropna().copy()

if len(model_df) == 0:
    raise ValueError("No usable rows remain after filtering.")

if model_df["RatingExtremity"].nunique() < 2:
    raise ValueError("Rating extremity has insufficient variation.")

# Center continuous predictors to make the intercept more interpretable.
model_df["LogReviewLength_c"] = (
    model_df["LogReviewLength"] - model_df["LogReviewLength"].mean()
)
model_df["SummaryLength_c"] = (
    model_df["SummaryLengthChars"] - model_df["SummaryLengthChars"].mean()
)
model_df["ReviewYear_c"] = (
    model_df["ReviewYear"] - model_df["ReviewYear"].mean()
)

# ---------------------------------------------------------
# 3. Fit grouped-binomial GLM
# Each review contributes helpful and unhelpful vote counts.
# ---------------------------------------------------------
endog = np.column_stack([
    model_df["HelpfulVotes"],
    model_df["UnhelpfulVotes"],
])

formula = (
    "RatingExtremity + "
    "LogReviewLength_c + SummaryLength_c + ReviewYear_c"
)

import patsy

X = patsy.dmatrix(
    formula,
    data=model_df,
    return_type="dataframe",
)

glm = sm.GLM(
    endog,
    X,
    family=sm.families.Binomial(),
)

result = glm.fit(
    cov_type="cluster",
    cov_kwds={"groups": model_df["ProductId"]},
)

# ---------------------------------------------------------
# 4. Extract the pre-specified one-sided test
# H0: beta_extremity = 0
# H1: beta_extremity > 0
# ---------------------------------------------------------
term = "RatingExtremity"
beta = float(result.params[term])
se = float(result.bse[term])
z_stat = beta / se

# One-sided p-value for a positive effect.
p_one_sided = float(norm.sf(z_stat))

ci = result.conf_int().loc[term]
ci_low = float(ci.iloc[0])
ci_high = float(ci.iloc[1])

odds_ratio = float(np.exp(beta))
or_ci_low = float(np.exp(ci_low))
or_ci_high = float(np.exp(ci_high))

# Proposed practical threshold: OR = 1.05 per extremity unit.
meaningful_log_or = float(np.log(MIN_MEANINGFUL_OR))

# Pearson dispersion diagnostic; interpretation needs care with
# grouped/binomial data and clustered observations.
pearson_chi2 = float(
    np.sum(result.resid_pearson ** 2)
)
residual_df = float(result.df_resid)
dispersion = (
    pearson_chi2 / residual_df if residual_df > 0 else None
)

report = {
    "dataset": DATA_PATH,
    "analysis": "Test 1: rating extremity",
    "n_reviews": int(len(model_df)),
    "n_products": int(model_df["ProductId"].nunique()),
    "total_helpful_votes": int(model_df["HelpfulVotes"].sum()),
    "total_unhelpful_votes": int(model_df["UnhelpfulVotes"].sum()),
    "alpha": ALPHA,
    "alternative": "beta_RatingExtremity > 0",
    "test": "Binomial GLM, logit link, product-clustered standard errors",
    "coefficient": beta,
    "cluster_robust_standard_error": se,
    "z_statistic": float(z_stat),
    "degrees_of_freedom": None,
    "p_value_one_sided": p_one_sided,
    "p_value_note": "Asymptotic normal approximation for cluster-robust z statistic.",
    "odds_ratio_per_extremity_unit": odds_ratio,
    "odds_ratio_95_ci": [or_ci_low, or_ci_high],
    "coefficient_95_ci": [ci_low, ci_high],
    "practical_threshold_odds_ratio": MIN_MEANINGFUL_OR,
    "practical_threshold_log_coefficient": meaningful_log_or,
    "nominal_decision": (
        "Reject H0" if p_one_sided < ALPHA else "Fail to reject H0"
    ),
    "practical_threshold_reached_by_point_estimate": (
        odds_ratio >= MIN_MEANINGFUL_OR
    ),
    "pearson_chi_square": pearson_chi2,
    "residual_degrees_of_freedom": residual_df,
    "pearson_dispersion_ratio": dispersion,
    "convergence": bool(result.converged),
    "caution": (
        "Review-level vote proportions may be overdispersed. "
        "Cluster-robust standard errors account for product clustering "
        "but do not by themselves solve all model misspecification. "
        "Review diagnostics before treating inference as final."
    ),
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, allow_nan=False)

print("=" * 65)
print("TEST 1: RATING EXTREMITY")
print("=" * 65)
print(json.dumps(report, indent=2))
print("\nModel coefficients:")
print(result.summary())
print(f"\nSaved report: {REPORT_PATH}")
