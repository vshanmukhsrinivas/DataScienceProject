
import json
import re
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/finefoods_cleaned.csv.gz")
REPORT_PATH = Path("reports/duplicate_audit.json")

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH, compression="gzip")
rows_before = len(df)

# Text fields that may differ only because of whitespace.
text_columns = ["ProfileName", "Summary", "Text"]

# Count exact duplicates in the current cleaned dataset.
exact_duplicate_rows = int(df.duplicated().sum())

# Make an audit-only copy. Do not modify the saved cleaned dataset.
audit_df = df.copy()

def normalize_whitespace(value):
    if pd.isna(value):
        return value
    return re.sub(r"\s+", " ", str(value)).strip()

for col in text_columns:
    audit_df[col] = audit_df[col].map(normalize_whitespace)

# Count rows that become duplicates after whitespace normalization.
normalized_duplicate_mask = audit_df.duplicated(keep="first")
normalized_duplicate_rows = int(normalized_duplicate_mask.sum())

# Find groups that contain normalized duplicates.
duplicate_groups = audit_df.loc[
    audit_df.duplicated(keep=False)
].copy()

# Keep a small sample for manual inspection.
sample_columns = [
    col for col in [
        "ProductId", "UserId", "Score",
        "ReviewDateUTC", "Summary", "Text"
    ]
    if col in audit_df.columns
]

examples = (
    duplicate_groups[sample_columns]
    .head(10)
    .fillna("")
    .to_dict(orient="records")
)

# Distinguish cross-product repetition from exact normalized duplicates.
# This is diagnostic only; it does not label these as confirmed duplicates.
cross_product_text_groups = 0
if {"UserId", "Score", "Summary", "Text", "ProductId"}.issubset(
    audit_df.columns
):
    grouped = audit_df.groupby(
        ["UserId", "Score", "Summary", "Text"],
        dropna=False
    )["ProductId"].nunique()

    cross_product_text_groups = int((grouped > 1).sum())

report = {
    "input": str(INPUT_PATH),
    "rows_checked": int(rows_before),
    "exact_duplicate_rows_before_normalization": exact_duplicate_rows,
    "duplicate_rows_after_whitespace_normalization": (
        normalized_duplicate_rows
    ),
    "additional_duplicates_revealed_by_normalization": max(
        0, normalized_duplicate_rows - exact_duplicate_rows
    ),
    "rows_participating_in_normalized_duplicate_groups": int(
        audit_df.duplicated(keep=False).sum()
    ),
    "cross_product_repeated_text_groups": cross_product_text_groups,
    "sample_normalized_duplicate_rows": examples,
    "policy": (
        "Audit only. No rows were removed or overwritten by this script. "
        "Repeated text across products is not automatically considered "
        "a duplicate."
    ),
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("NORMALIZED DUPLICATE AUDIT")
print(json.dumps(report, indent=2, ensure_ascii=False))
print(f"\nSaved report to: {REPORT_PATH}")
print("The cleaned dataset was not modified.")
