
import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/splits/exploration.csv.gz"
REPORT_PATH = ROOT / "reports/test2_clustered_spline_wald.json"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 1. Load the same exploration data and apply the same filters
# as the original Test 2 nonlinearity script.
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

num = df["HelpfulnessNumerator"]
den = df["HelpfulnessDenominator"]

valid = (
    num.ge(0) & den.gt(0) & num.le(den)
    & num.notna() & den.notna()
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

if model_df.empty:
    raise ValueError("No usable observations remain.")

for col in ["LogReviewLength", "SummaryLengthChars", "ReviewYear"]:
    model_df[col + "_c"] = (
        model_df[col] - model_df[col].mean()
    )

endog = np.column_stack([
    model_df["HelpfulVotes"],
    model_df["UnhelpfulVotes"],
])

# Fit the same cubic spline model as the original script.
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

result = sm.GLM(
    endog,
    X_spline,
    family=sm.families.Binomial(),
).fit(
    cov_type="cluster",
    cov_kwds={"groups": model_df["ProductId"]},
)

# Identify all spline basis coefficients.
spline_terms = [
    name for name in X_spline.columns
    if name.startswith("bs(")
]

if len(spline_terms) != 4:
    raise ValueError(
        f"Expected 4 spline basis columns; found "
        f"{len(spline_terms)}: {spline_terms}"
    )

# Joint Wald test: all three spline coefficients equal zero.
R = np.zeros((len(spline_terms), len(result.params)))
parameter_names = list(result.params.index)

for row, term in enumerate(spline_terms):
    R[row, parameter_names.index(term)] = 1.0

wald_result = result.wald_test(R, scalar=True)

statistic = float(np.asarray(wald_result.statistic).item())
p_value = float(chi2.sf(statistic, df=len(spline_terms)))

report = {
    "analysis": "Test 2 sensitivity: product-clustered spline Wald test",
    "dataset": str(DATA_PATH.relative_to(ROOT)),
    "n_reviews": int(len(model_df)),
    "n_products": int(model_df["ProductId"].nunique()),
    "hypotheses": {
        "H0": "All four spline basis coefficients equal zero",
"H1": "At least one spline basis coefficient is nonzero",
    },
    "alpha": 0.05,
    "method": (
        "Joint Wald test using product-clustered covariance "
        "from a grouped-binomial GLM"
    ),
    "spline_terms_tested": spline_terms,
    "test_statistic": statistic,
    "degrees_of_freedom": len(spline_terms),
    "p_value": p_value,
    "reference_distribution": "Asymptotic chi-square",
    "decision": (
        "Reject H0" if p_value < 0.05
        else "Fail to reject H0"
    ),
    "converged": bool(result.converged),
    "caution": (
        "This robust Wald test assesses whether the spline terms "
        "jointly contribute beyond the linear effect. It does not "
        "prove superior out-of-sample prediction. Cluster-robust "
        "inference remains asymptotic, and the model may still have "
        "misspecification or overdispersion concerns."
    ),
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, allow_nan=False)

print("=" * 65)
print("TEST 2: PRODUCT-CLUSTERED SPLINE WALD TEST")
print("=" * 65)
print(json.dumps(report, indent=2))
print(f"\nSaved report: {REPORT_PATH}")
