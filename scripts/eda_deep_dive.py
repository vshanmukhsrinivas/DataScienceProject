
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/splits/exploration.csv.gz"
FIGURE_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/eda_deep_dive.json")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, compression="gzip")

# Convert relevant columns to numeric values.
numeric_cols = [
    "Score", "ReviewLengthChars", "SummaryLengthChars",
    "HelpfulnessNumerator", "HelpfulnessDenominator",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

num = df["HelpfulnessNumerator"]
den = df["HelpfulnessDenominator"]

valid = num.ge(0) & den.gt(0) & num.le(den)
valid &= num.notna() & den.notna()

df["HelpfulnessRate"] = np.nan
df.loc[valid, "HelpfulnessRate"] = num[valid] / den[valid]

df["ReviewDateUTC"] = pd.to_datetime(
    df["ReviewDateUTC"], errors="coerce", utc=True
)
df["ReviewYear"] = df["ReviewDateUTC"].dt.year

valid_df = df.loc[valid].copy()
captions = {}
stats = {}


def save_figure(number, title, filename, caption):
    plt.title(title)
    plt.figtext(0.5, 0.005, caption, ha="center", wrap=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(FIGURE_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()
    captions[filename] = caption
    print(f"{filename}: {caption}")


# ---------------------------------------------------------
# 7. Summary-length distribution
# ---------------------------------------------------------
summary_lengths = df["SummaryLengthChars"].dropna()
summary_lengths = summary_lengths[summary_lengths >= 0]

plt.figure(figsize=(8, 5))
plt.hist(summary_lengths, bins=50)
plt.xlabel("Summary length (characters)")
plt.ylabel("Number of reviews")

stats["summary_length"] = {
    "median": float(summary_lengths.median()),
    "p95": float(summary_lengths.quantile(0.95)),
    "p99": float(summary_lengths.quantile(0.99)),
    "max": float(summary_lengths.max()),
}

caption = (
    f"Summary length has a median of {summary_lengths.median():.0f} "
    f"characters and a 95th percentile of "
    f"{summary_lengths.quantile(0.95):.0f}; summaries are generally "
    "much shorter than the review bodies."
)
save_figure(
    7, "Distribution of Review Summary Lengths",
    "eda_07_summary_length_distribution.png", caption
)


# ---------------------------------------------------------
# 8. Review length by rating
# ---------------------------------------------------------
box_data = [
    df.loc[df["Score"] == rating, "ReviewLengthChars"].dropna()
    for rating in [1, 2, 3, 4, 5]
]

plt.figure(figsize=(8, 5))
plt.boxplot(box_data, tick_labels=[1, 2, 3, 4, 5], showfliers=False)
plt.xlabel("Star rating")
plt.ylabel("Review length (characters)")
plt.title("Review Length by Star Rating")

length_by_rating = df.groupby("Score")["ReviewLengthChars"].agg(
    ["count", "median", "mean"]
)
stats["review_length_by_rating"] = (
    length_by_rating.round(4).reset_index().to_dict("records")
)

medians = length_by_rating["median"].dropna()
caption = (
    f"Median review length ranges from {medians.min():.0f} to "
    f"{medians.max():.0f} characters across ratings; differences "
    "suggest review length may vary with rating."
)
save_figure(
    8, "Review Length by Star Rating",
    "eda_08_review_length_by_rating.png", caption
)


# ---------------------------------------------------------
# 9. Review activity over time
# ---------------------------------------------------------
year_counts = df["ReviewYear"].dropna().astype(int).value_counts().sort_index()

# Exclude years with fewer than 100 records from this descriptive
# plot to avoid emphasizing extremely sparse years.
year_counts = year_counts[year_counts >= 100]

plt.figure(figsize=(10, 5))
plt.plot(year_counts.index, year_counts.values, marker="o")
plt.xlabel("Review year")
plt.ylabel("Number of reviews")
plt.title("Review Counts by Year")

stats["review_counts_by_year"] = {
    str(int(year)): int(count)
    for year, count in year_counts.items()
}

if len(year_counts) >= 2:
    caption = (
        f"Review volume varies across the observed years, from "
        f"{year_counts.min():,} to {year_counts.max():,} reviews "
        "among years with at least 100 records; sparse years are omitted."
    )
else:
    caption = "Too few years have sufficient records to compare activity."

save_figure(
    9, "Review Activity by Year",
    "eda_09_review_activity_by_year.png", caption
)


# ---------------------------------------------------------
# 10. Total votes versus helpfulness rate
# ---------------------------------------------------------
vote_data = valid_df.copy()
vote_data = vote_data[vote_data["HelpfulnessDenominator"] > 0]

vote_data["VoteBin"] = pd.cut(
    vote_data["HelpfulnessDenominator"],
    bins=[0, 1, 2, 4, 9, 19, np.inf],
    labels=["1", "2", "3–4", "5–9", "10–19", "20+"],
)

vote_stats = vote_data.groupby(
    "VoteBin", observed=True
).agg(
    reviews=("HelpfulnessRate", "size"),
    mean_rate=("HelpfulnessRate", "mean"),
    median_rate=("HelpfulnessRate", "median"),
)

plt.figure(figsize=(8, 5))
plt.bar(vote_stats.index.astype(str), vote_stats["mean_rate"])
plt.xlabel("Total helpfulness votes")
plt.ylabel("Mean helpfulness rate")
plt.ylim(0, 1)
plt.title("Helpfulness Rate by Vote-Count Group")

stats["helpfulness_by_vote_count"] = (
    vote_stats.reset_index().to_dict("records")
)

caption = (
    "Observed mean helpfulness rates differ across vote-count groups; "
    "vote volume may affect estimate precision, so group sizes and "
    "vote counts must be considered in interpretation."
)
save_figure(
    10, "Helpfulness by Total Vote Count",
    "eda_10_helpfulness_by_vote_count.png", caption
)


# ---------------------------------------------------------
# 11. Helpful-vote proportion by rating
# ---------------------------------------------------------
vote_valid = df.loc[valid].copy()
vote_valid["UnhelpfulVotes"] = (
    vote_valid["HelpfulnessDenominator"]
    - vote_valid["HelpfulnessNumerator"]
)

vote_totals = vote_valid.groupby("Score").agg(
    helpful=("HelpfulnessNumerator", "sum"),
    unhelpful=("UnhelpfulVotes", "sum"),
).reindex([1, 2, 3, 4, 5])

vote_totals["total_votes"] = (
    vote_totals["helpful"] + vote_totals["unhelpful"]
)
vote_totals["helpful_vote_fraction"] = (
    vote_totals["helpful"] / vote_totals["total_votes"]
)

plt.figure(figsize=(8, 5))
plt.bar(
    vote_totals.index.astype(int),
    vote_totals["helpful_vote_fraction"],
)
plt.xlabel("Star rating")
plt.ylabel("Fraction of recorded votes that are helpful")
plt.ylim(0, 1)
plt.xticks([1, 2, 3, 4, 5])
plt.title("Vote-Weighted Helpfulness by Rating")

stats["vote_weighted_helpfulness_by_rating"] = (
    vote_totals.reset_index().to_dict("records")
)

caption = (
    "Vote-weighted helpfulness summarizes individual votes, whereas "
    "the earlier plot averages review-level rates; these answer "
    "different questions because reviews receive different vote counts."
)
save_figure(
    11, "Vote-Weighted Helpfulness by Rating",
    "eda_11_vote_weighted_helpfulness.png", caption
)


# ---------------------------------------------------------
# 12. Summary length versus helpfulness
# ---------------------------------------------------------
summary_data = valid_df[
    ["SummaryLengthChars", "HelpfulnessRate"]
].dropna()
summary_data = summary_data[summary_data["SummaryLengthChars"] >= 0].copy()

summary_data["LengthBin"] = pd.cut(
    summary_data["SummaryLengthChars"],
    bins=[-1, 0, 10, 20, 30, 50, 75, np.inf],
    labels=[
        "0", "1–10", "11–20", "21–30",
        "31–50", "51–75", "76+",
    ],
)

summary_stats = summary_data.groupby(
    "LengthBin", observed=True
).agg(
    reviews=("HelpfulnessRate", "size"),
    mean_rate=("HelpfulnessRate", "mean"),
    median_rate=("HelpfulnessRate", "median"),
)

plt.figure(figsize=(9, 5))
plt.bar(summary_stats.index.astype(str), summary_stats["mean_rate"])
plt.xlabel("Summary length (characters)")
plt.ylabel("Mean helpfulness rate")
plt.ylim(0, 1)
plt.title("Helpfulness by Summary Length")

stats["helpfulness_by_summary_length"] = (
    summary_stats.reset_index().to_dict("records")
)

caption = (
    "Mean helpfulness varies across summary-length groups; this "
    "descriptive pattern motivates testing whether summary length "
    "adds information beyond review length and rating."
)
save_figure(
    12, "Helpfulness by Summary Length",
    "eda_12_helpfulness_by_summary_length.png", caption
)


# ---------------------------------------------------------
# 13. Reviews per product
# ---------------------------------------------------------
product_counts = df.groupby("ProductId").size()

plt.figure(figsize=(8, 5))
plt.hist(product_counts, bins=60)
plt.yscale("log")
plt.xlabel("Number of reviews per product")
plt.ylabel("Number of products (log scale)")
plt.title("Distribution of Reviews per Product")

stats["reviews_per_product"] = {
    "products": int(product_counts.size),
    "median": float(product_counts.median()),
    "p95": float(product_counts.quantile(0.95)),
    "p99": float(product_counts.quantile(0.99)),
    "max": int(product_counts.max()),
}

caption = (
    f"The median product has {product_counts.median():.0f} reviews, "
    f"while the 99th percentile is {product_counts.quantile(0.99):.0f}; "
    "review counts are concentrated unevenly across products."
)
save_figure(
    13, "Distribution of Reviews per Product",
    "eda_13_reviews_per_product.png", caption
)


# ---------------------------------------------------------
# 14. Correlations among structured features
# ---------------------------------------------------------
features = df[
    [
        "Score", "ReviewLengthChars",
        "SummaryLengthChars", "ReviewYear",
    ]
].copy()

# Spearman correlation is useful for monotonic relationships
# and is less dependent on linearity than Pearson correlation.
corr = features.corr(method="spearman")

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=35, ha="right")
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index)
fig.colorbar(im, ax=ax, label="Spearman correlation")

for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        value = corr.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center")

stats["spearman_correlations"] = corr.to_dict()

caption = (
    "Structured features show varying degrees of monotonic association; "
    "correlation helps diagnose redundancy but does not imply causation."
)
save_figure(
    14, "Spearman Correlations of Structured Features",
    "eda_14_structured_feature_correlations.png", caption
)


# ---------------------------------------------------------
# Save all results
# ---------------------------------------------------------
report = {
    "dataset": DATA_PATH,
    "rows": int(len(df)),
    "unique_products": int(df["ProductId"].nunique()),
    "valid_helpfulness_reviews": int(valid.sum()),
    "invalid_helpfulness_records": int((num > den).sum()),
    "zero_vote_reviews": int(den.eq(0).sum()),
    "statistics": stats,
    "figure_captions": captions,
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=str, allow_nan=False)

print("\nDeep-dive EDA completed.")
print("Figures saved in:", FIGURE_DIR)
print("Summary saved in:", REPORT_PATH)
print("Figures generated:", len(captions))
