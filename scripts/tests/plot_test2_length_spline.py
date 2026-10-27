
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = "data/splits/exploration.csv.gz"
FIGURE_PATH = Path("reports/figures/test2_review_length_spline.png")
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

# 1. Load and filter the same observations as Test 2.
df = pd.read_csv(DATA_PATH, compression="gzip")

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
df["UnhelpfulVotes"] = den.loc[valid] - num.loc[valid]
df["LogReviewLength"] = np.log1p(df["ReviewLengthChars"])
df["RatingExtremity"] = (df["Score"] - 3).abs()
df["ReviewDateUTC"] = pd.to_datetime(
    df["ReviewDateUTC"], errors="coerce", utc=True
)
df["ReviewYear"] = df["ReviewDateUTC"].dt.year

cols = [
    "ProductId", "HelpfulVotes", "UnhelpfulVotes",
    "ReviewLengthChars", "LogReviewLength",
    "RatingExtremity", "SummaryLengthChars", "ReviewYear",
]
data = (
    df[cols]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .copy()
)

# 2. Center predictors as in the nonlinearity check.
for col in ["LogReviewLength", "SummaryLengthChars", "ReviewYear"]:
    data[col + "_c"] = data[col] - data[col].mean()

# 3. Fit the spline model.
endog = np.column_stack([
    data["HelpfulVotes"],
    data["UnhelpfulVotes"],
])

formula = (
    "RatingExtremity + "
    "bs(LogReviewLength_c, df=4, degree=3, "
    "include_intercept=False) + "
    "SummaryLengthChars_c + ReviewYear_c"
)
X = patsy.dmatrix(formula, data=data, return_type="dataframe")

model = sm.GLM(
    endog,
    X,
    family=sm.families.Binomial(),
).fit(
    cov_type="cluster",
    cov_kwds={"groups": data["ProductId"]},
)

# 4. Predict over the observed review-length range.
length_grid = np.linspace(
    data["ReviewLengthChars"].quantile(0.01),
    data["ReviewLengthChars"].quantile(0.99),
    150,
)

reference = pd.DataFrame({
    "RatingExtremity": data["RatingExtremity"].median(),
    "LogReviewLength_c": (
        np.log1p(length_grid) - data["LogReviewLength"].mean()
    ),
    "SummaryLengthChars_c": 0.0,
    "ReviewYear_c": 0.0,
})

X_grid = patsy.build_design_matrices(
    [X.design_info],
    reference,
    return_type="dataframe",
)[0]

pred = model.predict(X_grid)

# 5. Plot adjusted predicted helpful-vote probability.
plt.figure(figsize=(9, 5.5))
plt.plot(length_grid, pred)
plt.xlabel("Review length (characters)")
plt.ylabel("Predicted probability a vote is helpful")
plt.title("Adjusted relationship between review length and helpfulness")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(FIGURE_PATH, dpi=180)
plt.close()

print(f"Reviews used: {len(data):,}")
print(f"Products represented: {data['ProductId'].nunique():,}")
print(f"Saved plot: {FIGURE_PATH}")

