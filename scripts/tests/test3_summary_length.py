
import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/splits/exploration.csv.gz"
REPORT_PATH = ROOT / "reports/test3_summary_length.json"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
PRACTICAL_DEVIANCE_REDUCTION = 0.01

# 1. Load data.
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
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing columns: {missing}")

numeric_cols = [
    "Score",
    "ReviewLengthChars",
    "SummaryLengthChars",
    "HelpfulnessNumerator",
    "HelpfulnessDenominator",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 2. Apply the same valid-vote eligibility rules as Tests 1 and 2.
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
    df["HelpfulnessDenominator"]
    - df["HelpfulnessNumerator"]
)
df["LogReviewLength"] = np.log1p(df["ReviewLengthChars"])
df["RatingExtremity"] = (df["Score"] - 3).abs()
df["ReviewDateUTC"] = pd.to_datetime(
    df["ReviewDateUTC"], errors="coerce", utc=True
)
df["ReviewYear"] = df["ReviewDateUTC"].dt.year

cols = [
    "ProductId",
    "HelpfulVotes",
    "UnhelpfulVotes",
    "LogReviewLength",
    "RatingExtremity",
    "SummaryLengthChars",
    "ReviewYear",
]
model_df = (
    df[cols]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .copy()
)

if model_df.empty:
    raise ValueError("No usable observations remain.")

# Center continuous predictors consistently.
for col in [
    "LogReviewLength",
    "SummaryLengthChars",
    "ReviewYear",
]:
    model_df[col + "_c"] = (
        model_df[col] - model_df[col].mean()
    )

endog = np.column_stack([
    model_df["HelpfulVotes"],
    model_df["UnhelpfulVotes"],
])

# Model A excludes summary length.
formula_a = (
    "RatingExtremity + LogReviewLength_c + ReviewYear_c"
)

# Model B adds summary length.
formula_b = (
    "RatingExtremity + LogReviewLength_c + "
    "SummaryLengthChars_c + ReviewYear_c"
)

X_a = patsy.dmatrix(
    formula_a, data=model_df, return_type="dataframe"
)
X_b = patsy.dmatrix(
    formula_b, data=model_df, return_type="dataframe"
)

# Fit both models with the same observations and response.
model_a = sm.GLM(
    endog, X_a, family=sm.families.Binomial()
).fit(
    cov_type="cluster",
    cov_kwds={"groups": model_df["ProductId"]},
)

model_b = sm.GLM(
    endog, X_b, family=sm.families.Binomial()
).fit(
    cov_type="cluster",
    cov_kwds={"groups": model_df["ProductId"]},
)

if not model_a.converged or not model_b.converged:
    raise RuntimeError("At least one model failed to converge.")

# 3. Planned likelihood-ratio comparison.
# This conventional LR test assumes the binomial likelihood
# is appropriately specified; interpret it cautiously given
# the earlier dispersion diagnostic.
lr_stat = max(
    0.0,
    float(2 * (model_b.llf - model_a.llf))
)
df_difference = int(model_b.df_model - model_a.df_model)

if df_difference != 1:
    raise ValueError(
        f"Expected one added predictor; got df difference "
        f"{df_difference}."
    )

lr_p = float(chi2.sf(lr_stat, df_difference))

# Deviance reduction, using Model A's deviance as the baseline.
deviance_a = float(model_a.deviance)
deviance_b = float(model_b.deviance)

deviance_reduction = (
    (deviance_a - deviance_b) / deviance_a
    if deviance_a > 0 else None
)

# 4. Product-clustered robust Wald test for the added coefficient.
term = "SummaryLengthChars_c"
beta = float(model_b.params[term])
se = float(model_b.bse[term])
z_stat = beta / se
wald_stat = z_stat ** 2
wald_p = float(chi2.sf(wald_stat, df=1))

ci = model_b.conf_int().loc[term]
ci_low = float(ci.iloc[0])
ci_high = float(ci.iloc[1])

# 5. Report the effect for a 10-character increase.
# Since the predictor is centered but not rescaled, this
# contrast is beta * 10 on the log-odds scale.
or_10_chars = float(np.exp(10 * beta))
or_10_chars_ci = [
    float(np.exp(10 * ci_low)),
    float(np.exp(10 * ci_high)),
]

report = {
    "analysis": "Test 3: additional explanatory value of summary length",
    "dataset": str(DATA_PATH.relative_to(ROOT)),
    "n_reviews": int(len(model_df)),
    "n_products": int(model_df["ProductId"].nunique()),
    "total_helpful_votes": int(model_df["HelpfulVotes"].sum()),
    "total_unhelpful_votes": int(model_df["UnhelpfulVotes"].sum()),
    "alpha": ALPHA,
    "hypotheses": {
        "H0": "Summary-length coefficient equals zero",
        "H1": "Summary-length coefficient differs from zero",
    },
    "primary_planned_method": (
        "Likelihood-ratio test comparing nested grouped-binomial GLMs"
    ),
    "likelihood_ratio_statistic": lr_stat,
    "likelihood_ratio_degrees_of_freedom": df_difference,
    "likelihood_ratio_p_value": lr_p,
    "baseline_deviance": deviance_a,
    "extended_deviance": deviance_b,
    "relative_deviance_reduction": deviance_reduction,
    "practical_deviance_reduction_threshold": (
        PRACTICAL_DEVIANCE_REDUCTION
    ),
    "practical_threshold_reached": (
        deviance_reduction is not None
        and deviance_reduction >= PRACTICAL_DEVIANCE_REDUCTION
    ),
    "robust_sensitivity": {
        "method": "Product-clustered Wald test using extended-model covariance",
        "coefficient_per_character": beta,
        "cluster_robust_standard_error": se,
        "z_statistic": z_stat,
        "wald_chi_square_statistic": wald_stat,
        "degrees_of_freedom": 1,
        "p_value": wald_p,
        "coefficient_95_ci": [ci_low, ci_high],
        "odds_ratio_per_10_character_increase": or_10_chars,
        "odds_ratio_per_10_character_increase_95_ci": (
            or_10_chars_ci
        ),
        "reference_distribution": "Asymptotic chi-square for Wald statistic",
    },
    "convergence": {
        "baseline": bool(model_a.converged),
        "extended": bool(model_b.converged),
    },
    "caution": (
        "The likelihood-ratio test uses the conventional binomial "
        "likelihood and is not cluster robust. The product-clustered "
        "Wald test is the robust sensitivity analysis. Both use "
        "asymptotic reference distributions. The 1% deviance reduction "
        "is a proposed practical benchmark, not a universal standard. "
        "A nonzero coefficient does not establish causation."
    ),
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, allow_nan=False)

print("=" * 68)
print("TEST 3: SUMMARY LENGTH")
print("=" * 68)
print(json.dumps(report, indent=2))
print(f"\nSaved report: {REPORT_PATH}")
