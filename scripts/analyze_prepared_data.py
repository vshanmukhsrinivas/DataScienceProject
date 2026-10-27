
import json
from pathlib import Path

import pandas as pd

DATA_PATH = "data/processed/finefoods_cleaned.csv.gz"
REPORT_PATH = Path("reports/outlier_analysis.json")

df = pd.read_csv(DATA_PATH, compression="gzip")

numeric_columns = [
    "ReviewLengthChars",
    "SummaryLengthChars",
    "HelpfulnessDenominator",
    "HelpfulnessNumerator",
]

report = {
    "rows": int(len(df)),
    "numeric_summaries": {},
    "review_length_outliers": {},
    "valid_vote_checks": {},
}

for col in numeric_columns:
    s = df[col].dropna()
    report["numeric_summaries"][col] = {
        "count": int(s.count()),
        "min": float(s.min()),
        "median": float(s.median()),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }

# IQR rule: use as a diagnostic flag, not an automatic deletion rule.
for col in ["ReviewLengthChars", "SummaryLengthChars"]:
    s = df[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr

    report["review_length_outliers"][col] = {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "upper_iqr_threshold": float(upper),
        "above_upper_threshold": int((s > upper).sum()),
        "policy": "Flag for investigation; do not automatically remove.",
    }

report["valid_vote_checks"] = {
    "scores_outside_1_to_5": int(
        (~df["Score"].between(1, 5)).sum()
    ),
    "negative_helpful_votes": int(
        (df["HelpfulnessNumerator"] < 0).sum()
    ),
    "negative_total_votes": int(
        (df["HelpfulnessDenominator"] < 0).sum()
    ),
    "numerator_exceeds_denominator": int(
        (df["HelpfulnessNumerator"] >
         df["HelpfulnessDenominator"]).sum()
    ),
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
print(f"\nSaved report to {REPORT_PATH}")
