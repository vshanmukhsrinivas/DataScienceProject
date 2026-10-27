
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/splits/exploration.csv.gz"
FIGURE_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/eda_summary.json")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1. Load the cleaned exploration split
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH, compression="gzip")

df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
df["ReviewLengthChars"] = pd.to_numeric(
    df["ReviewLengthChars"], errors="coerce"
)
df["SummaryLengthChars"] = pd.to_numeric(
    df["SummaryLengthChars"], errors="coerce"
)
df["HelpfulnessNumerator"] = pd.to_numeric(
    df["HelpfulnessNumerator"], errors="coerce"
)
df["HelpfulnessDenominator"] = pd.to_numeric(
    df["HelpfulnessDenominator"], errors="coerce"
)

num = df["HelpfulnessNumerator"]
den = df["HelpfulnessDenominator"]

valid_votes = (
    num.ge(0)
    & den.gt(0)
    & num.le(den)
    & num.notna()
    & den.notna()
)

df["HelpfulnessRate"] = np.nan
df.loc[valid_votes, "HelpfulnessRate"] = (
    num[valid_votes] / den[valid_votes]
)
df["RatingExtremity"] = (df["Score"] - 3).abs()

valid_df = df.loc[valid_votes].copy()

print("=" * 65)
print("EXPLORATION DATASET")
print("=" * 65)
print("Rows:", len(df))
print("Unique products:", df["ProductId"].nunique())
print("Valid helpfulness rates:", int(valid_votes.sum()))
print("Zero-vote reviews:", int(den.eq(0).sum()))
print("Invalid helpfulness records:", int((num > den).sum()))

# ---------------------------------------------------------
# 2. Descriptive statistics
# ---------------------------------------------------------
rating_counts = (
    df["Score"].value_counts().sort_index().dropna()
)
rating_counts.index = rating_counts.index.astype(int)

rating_summary = (
    valid_df.groupby("Score")
    .agg(
        review_count=("HelpfulnessRate", "size"),
        mean_rate=("HelpfulnessRate", "mean"),
        median_rate=("HelpfulnessRate", "median"),
        mean_votes=("HelpfulnessDenominator", "mean"),
        median_votes=("HelpfulnessDenominator", "median"),
    )
)

length_summary = df[
    ["ReviewLengthChars", "SummaryLengthChars"]
].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])

print("\nRating counts:")
print(rating_counts)

print("\nHelpfulness by rating:")
print(rating_summary.round(4))

print("\nLength summary:")
print(length_summary.round(2))

vote_thresholds = {}
for threshold in [1, 3, 5, 10, 20]:
    count = int(
        valid_df["HelpfulnessDenominator"].ge(threshold).sum()
    )
    vote_thresholds[str(threshold)] = {
        "reviews": count,
        "fraction_of_all_reviews": count / len(df),
    }

print("\nVote thresholds:")
for threshold, result in vote_thresholds.items():
    print(
        f"At least {threshold} votes: "
        f"{result['reviews']:,} reviews "
        f"({result['fraction_of_all_reviews']:.1%} of all reviews)"
    )

# ---------------------------------------------------------
# Helpers: plot saving and approximate descriptive intervals
# ---------------------------------------------------------
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def mean_ci(values):
    """Approximate 95% CI for a mean; descriptive, not cluster-adjusted."""
    values = pd.Series(values).dropna()
    n = len(values)

    if n < 2:
        return np.nan, np.nan

    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(n)
    return mean - 1.96 * se, mean + 1.96 * se


# ---------------------------------------------------------
# Figure 1: Star rating distribution
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
rating_counts.reindex([1, 2, 3, 4, 5], fill_value=0).plot(
    kind="bar"
)
plt.xlabel("Star rating")
plt.ylabel("Number of reviews")
plt.title("Distribution of Amazon Fine Food Review Ratings")
plt.xticks(rotation=0)

dominant_rating = int(rating_counts.idxmax())
dominant_share = rating_counts.max() / len(df)
caption_1 = (
    f"Rating {dominant_rating} is the most common, representing "
    f"{dominant_share:.1%} of reviews; rating imbalance should be "
    "considered when comparing groups."
)
plt.figtext(0.5, -0.02, caption_1, ha="center", wrap=True)
save_plot("eda_01_rating_distribution.png")

# ---------------------------------------------------------
# Figure 2: Helpfulness rate by star rating
# ---------------------------------------------------------
by_score = valid_df.groupby("Score")["HelpfulnessRate"]
score_stats = by_score.agg(["mean", "count", "std"]).reindex(
    [1, 2, 3, 4, 5]
)

score_stats["se"] = (
    score_stats["std"] / np.sqrt(score_stats["count"])
)
score_stats["ci_low"] = score_stats["mean"] - 1.96 * score_stats["se"]
score_stats["ci_high"] = score_stats["mean"] + 1.96 * score_stats["se"]

plt.figure(figsize=(8, 5))
x = score_stats.index.to_numpy(dtype=float)
means = score_stats["mean"].to_numpy(dtype=float)
lower = score_stats["ci_low"].to_numpy(dtype=float)
upper = score_stats["ci_high"].to_numpy(dtype=float)

plt.errorbar(
    x,
    means,
    yerr=np.vstack([
        np.maximum(means - lower, 0),
        np.maximum(upper - means, 0),
    ]),
    marker="o",
    capsize=4,
)
plt.xlabel("Star rating")
plt.ylabel("Mean helpfulness rate")
plt.ylim(0, 1.05)
plt.xticks([1, 2, 3, 4, 5])
plt.title("Observed Helpfulness Rate by Star Rating")

if score_stats["mean"].notna().sum() >= 2:
    lowest = score_stats["mean"].idxmin()
    highest = score_stats["mean"].idxmax()
    caption_2 = (
        f"The lowest observed mean rate is for rating {lowest} "
        f"({score_stats.loc[lowest, 'mean']:.3f}) and the highest "
        f"is for rating {highest} "
        f"({score_stats.loc[highest, 'mean']:.3f}); this is "
        "descriptive and does not establish causation."
    )
else:
    caption_2 = "Insufficient data to compare rating groups."

plt.figtext(0.5, -0.02, caption_2, ha="center", wrap=True)
save_plot("eda_02_helpfulness_by_rating.png")

# ---------------------------------------------------------
# Figure 3: Review length distribution
# ---------------------------------------------------------
lengths = df["ReviewLengthChars"].dropna()
lengths = lengths[lengths > 0]

plt.figure(figsize=(8, 5))
plt.hist(lengths, bins=80)
plt.xscale("log")
plt.xlabel("Review length in characters (log scale)")
plt.ylabel("Number of reviews")
plt.title("Distribution of Review Lengths")

median_length = lengths.median()
p95_length = lengths.quantile(0.95)
caption_3 = (
    f"Review length has a median of {median_length:.0f} characters "
    f"and a 95th percentile of {p95_length:.0f}, illustrating "
    "the spread between typical and long reviews."
)
plt.figtext(0.5, -0.02, caption_3, ha="center", wrap=True)
save_plot("eda_03_review_length_distribution.png")

# ---------------------------------------------------------
# Figure 4: Helpfulness rate by review-length quantile
# ---------------------------------------------------------
length_data = valid_df[
    ["ReviewLengthChars", "HelpfulnessRate"]
].dropna()

length_data = length_data[length_data["ReviewLengthChars"] > 0].copy()

length_data["LengthBin"] = pd.qcut(
    length_data["ReviewLengthChars"],
    q=10,
    duplicates="drop",
)

length_group = length_data.groupby(
    "LengthBin", observed=True
)["HelpfulnessRate"]

length_stats = length_group.agg(["mean", "count", "std"])
length_stats["se"] = (
    length_stats["std"] / np.sqrt(length_stats["count"])
)
length_stats["ci_low"] = (
    length_stats["mean"] - 1.96 * length_stats["se"]
)
length_stats["ci_high"] = (
    length_stats["mean"] + 1.96 * length_stats["se"]
)

plt.figure(figsize=(10, 5))
positions = np.arange(len(length_stats))
means = length_stats["mean"].to_numpy()
lower = length_stats["ci_low"].to_numpy()
upper = length_stats["ci_high"].to_numpy()

plt.errorbar(
    positions,
    means,
    yerr=np.vstack([
        np.maximum(means - lower, 0),
        np.maximum(upper - means, 0),
    ]),
    marker="o",
    capsize=3,
)
plt.xlabel("Review-length group (shortest to longest)")
plt.ylabel("Mean helpfulness rate")
plt.ylim(0, 1.05)
plt.xticks(positions, [str(i + 1) for i in positions])
plt.title("Helpfulness Rate Across Review-Length Groups")

if len(length_stats) >= 2:
    first_mean = length_stats["mean"].iloc[0]
    last_mean = length_stats["mean"].iloc[-1]
    difference = last_mean - first_mean
    direction = "higher" if difference > 0 else "lower"
    caption_4 = (
        f"The longest-review group has a mean helpfulness rate "
        f"{abs(difference):.3f} {direction} than the shortest-review "
        "group; intermediate groups should be examined for "
        "nonlinear patterns."
    )
else:
    caption_4 = "Insufficient variation in review length to compare groups."

plt.figtext(0.5, -0.02, caption_4, ha="center", wrap=True)
save_plot("eda_04_helpfulness_by_review_length.png")

# ---------------------------------------------------------
# Figure 5: Distribution of helpfulness-vote counts
# ---------------------------------------------------------
positive_votes = valid_df["HelpfulnessDenominator"]
positive_votes = positive_votes[positive_votes > 0]

plt.figure(figsize=(8, 5))
plt.hist(positive_votes, bins=60)
plt.xscale("log")
plt.xlabel("Total helpfulness votes (log scale)")
plt.ylabel("Number of reviews")
plt.title("Distribution of Vote Counts for Reviews with Valid Rates")

median_votes = positive_votes.median()
one_vote_share = (
    positive_votes.eq(1).sum() / len(valid_df)
    if len(valid_df) else np.nan
)

caption_5 = (
    f"The median vote count is {median_votes:.0f}; "
    f"{one_vote_share:.1%} of valid-rate reviews have only one vote. "
    "Rates based on different vote counts have different precision."
)
plt.figtext(0.5, -0.02, caption_5, ha="center", wrap=True)
save_plot("eda_05_helpfulness_vote_counts.png")

# ---------------------------------------------------------
# Figure 6: Helpfulness rate by rating extremity
# ---------------------------------------------------------
extremity_stats = valid_df.groupby(
    "RatingExtremity"
)["HelpfulnessRate"].agg(["mean", "count", "std"])

extremity_stats["se"] = (
    extremity_stats["std"] / np.sqrt(extremity_stats["count"])
)
extremity_stats["ci_low"] = (
    extremity_stats["mean"] - 1.96 * extremity_stats["se"]
)
extremity_stats["ci_high"] = (
    extremity_stats["mean"] + 1.96 * extremity_stats["se"]
)

plt.figure(figsize=(8, 5))
x = extremity_stats.index.to_numpy(dtype=float)
means = extremity_stats["mean"].to_numpy()
lower = extremity_stats["ci_low"].to_numpy()
upper = extremity_stats["ci_high"].to_numpy()

plt.errorbar(
    x,
    means,
    yerr=np.vstack([
        np.maximum(means - lower, 0),
        np.maximum(upper - means, 0),
    ]),
    marker="o",
    capsize=4,
)
plt.xlabel("Rating extremity: absolute distance from 3")
plt.ylabel("Mean helpfulness rate")
plt.ylim(0, 1.05)
plt.xticks([0, 1, 2])
plt.title("Observed Helpfulness by Rating Extremity")

if len(extremity_stats) >= 2:
    first_mean = extremity_stats["mean"].iloc[0]
    last_mean = extremity_stats["mean"].iloc[-1]
    difference = last_mean - first_mean
    direction = "higher" if difference > 0 else "lower"
    caption_6 = (
        f"Reviews at extremity 2 have a mean helpfulness rate "
        f"{abs(difference):.3f} {direction} than reviews at "
        "extremity 0; a formal test is needed to assess the "
        "pre-specified directional hypothesis."
    )
else:
    caption_6 = "Insufficient data to compare rating-extremity groups."

plt.figtext(0.5, -0.02, caption_6, ha="center", wrap=True)
save_plot("eda_06_helpfulness_by_rating_extremity.png")

# ---------------------------------------------------------
# 7. Save reproducible summary and figure captions
# ---------------------------------------------------------
report = {
    "dataset": DATA_PATH,
    "rows": int(len(df)),
    "unique_products": int(df["ProductId"].nunique()),
    "valid_helpfulness_rate_reviews": int(valid_votes.sum()),
    "zero_vote_reviews": int(den.eq(0).sum()),
    "invalid_helpfulness_records": int((num > den).sum()),
    "rating_counts": {
        str(int(k)): int(v) for k, v in rating_counts.items()
    },
    "helpfulness_by_rating": {
        str(int(k)): {
            "review_count": int(v["count"]),
            "mean_rate": float(v["mean"]),
            "median_rate": float(
                valid_df.loc[
                    valid_df["Score"].eq(k), "HelpfulnessRate"
                ].median()
            ),
            "ci_low_approx": float(v["ci_low"]),
            "ci_high_approx": float(v["ci_high"]),
        }
        for k, v in score_stats.iterrows()
        if pd.notna(v["mean"])
    },
    "review_length": {
        "median": float(lengths.median()),
        "p95": float(lengths.quantile(0.95)),
        "p99": float(lengths.quantile(0.99)),
        "max": int(lengths.max()),
    },
    "vote_thresholds": vote_thresholds,
    "helpfulness_by_extremity": {
        str(int(k)): {
            "review_count": int(v["count"]),
            "mean_rate": float(v["mean"]),
            "median_rate": float(
                valid_df.loc[
                    valid_df["RatingExtremity"].eq(k),
                    "HelpfulnessRate",
                ].median()
            ),
        }
        for k, v in extremity_stats.iterrows()
        if pd.notna(v["mean"])
    },
    "figure_captions": {
        "eda_01_rating_distribution.png": caption_1,
        "eda_02_helpfulness_by_rating.png": caption_2,
        "eda_03_review_length_distribution.png": caption_3,
        "eda_04_helpfulness_by_review_length.png": caption_4,
        "eda_05_helpfulness_vote_counts.png": caption_5,
        "eda_06_helpfulness_by_rating_extremity.png": caption_6,
    },
    "interval_note": (
        "Plotted intervals are approximate row-level 95% intervals. "
        "They are descriptive, not cluster-adjusted confirmatory "
        "inference. Formal tests must account for the analysis design."
    ),
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, allow_nan=False)

print("\n" + "=" * 65)
print("EDA FIGURES AND REPORT SAVED")
print("=" * 65)
print("Figures:", FIGURE_DIR)
print("Summary:", REPORT_PATH)

for filename, caption in report["figure_captions"].items():
    print(f"\n{filename}\nTakeaway: {caption}")
