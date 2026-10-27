
import json
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/finefoods_parsed.csv.gz")
OUTPUT_PATH = Path("data/processed/finefoods_cleaned.csv.gz")
REPORT_PATH = Path("reports/preparation_report.json")

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH, compression="gzip")
before_rows = len(df)

# Standardize whitespace-only strings as missing values.
text_columns = ["ProfileName", "Summary", "Text"]
for col in text_columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

# Record the original quality issues before treatment.
missing_before = {
    col: int(df[col].isna().sum())
    for col in df.columns
}

exact_duplicates_before = int(df.duplicated().sum())

invalid_helpfulness_before = int(
    (df["HelpfulnessNumerator"] >
     df["HelpfulnessDenominator"]).sum()
)

zero_denominator_before = int(
    (df["HelpfulnessDenominator"] == 0).sum()
)

# Remove exact duplicate rows from the analytical copy.
# Keep the first occurrence and preserve the raw dataset.
df = df.drop_duplicates().copy()

# Flag invalid helpfulness counts; do not delete the reviews.
df["InvalidHelpfulness"] = (
    df["HelpfulnessNumerator"] >
    df["HelpfulnessDenominator"]
)

# Missing summaries become empty text for later TF-IDF processing.
# Keep ProfileName missing because it is not required by our main model.
df["Summary"] = df["Summary"].fillna("")

# Calculate helpfulness rate only when the denominator is positive
# and the vote counts are valid. Otherwise, leave the rate undefined.
valid_votes = (
    (df["HelpfulnessDenominator"] > 0)
    & (~df["InvalidHelpfulness"])
)

df["HelpfulnessRate"] = pd.NA
df.loc[valid_votes, "HelpfulnessRate"] = (
    df.loc[valid_votes, "HelpfulnessNumerator"]
    / df.loc[valid_votes, "HelpfulnessDenominator"]
)

# Recalculate text lengths after standardizing whitespace.
df["ReviewLengthChars"] = (
    df["Text"].fillna("").str.len()
)
df["SummaryLengthChars"] = df["Summary"].str.len()

# Save the cleaned analytical dataset.
df.to_csv(OUTPUT_PATH, index=False, compression="gzip")

report = {
    "input": str(INPUT_PATH),
    "output": str(OUTPUT_PATH),
    "rows_before": int(before_rows),
    "rows_after": int(len(df)),
    "rows_removed_as_exact_duplicates": int(
        before_rows - len(df)
    ),
    "columns_after": int(len(df.columns)),
    "missing_values_before": missing_before,
    "missing_values_after": {
        col: int(df[col].isna().sum())
        for col in df.columns
    },
    "invalid_helpfulness_records_before": (
        invalid_helpfulness_before
    ),
    "invalid_helpfulness_records_after": int(
        df["InvalidHelpfulness"].sum()
    ),
    "zero_denominator_records_after": int(
        (df["HelpfulnessDenominator"] == 0).sum()
    ),
    "valid_helpfulness_rate_records": int(
        df["HelpfulnessRate"].notna().sum()
    ),
    "reviews_retained_without_valid_rate": int(
        df["HelpfulnessRate"].isna().sum()
    ),
    "outlier_policy": (
        "Retain genuine long reviews; investigate and report "
        "the distribution before deciding on any transformation."
    ),
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("DATA PREPARATION REPORT")
print(json.dumps(report, indent=2))
print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")
print(f"Report saved to: {REPORT_PATH}")
