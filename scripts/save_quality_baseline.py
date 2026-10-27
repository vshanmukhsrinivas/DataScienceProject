
import json
from pathlib import Path
import pandas as pd

data_path = Path("data/processed/finefoods_parsed.csv.gz")
output_path = Path("reports/quality_baseline.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path, compression="gzip")

baseline = {
    "stage": "before_cleaning",
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "missing_values": {
        col: int(count)
        for col, count in df.isna().sum().items()
    },
    "exact_duplicate_rows_beyond_first": int(
        df.duplicated().sum()
    ),
    "rows_participating_in_exact_duplicates": int(
        df.duplicated(keep=False).sum()
    ),
    "invalid_helpfulness_records": int(
        (df["HelpfulnessNumerator"] >
         df["HelpfulnessDenominator"]).sum()
    ),
    "zero_helpfulness_denominator": int(
        (df["HelpfulnessDenominator"] == 0).sum()
    ),
    "empty_or_whitespace_only": {
        col: int(df[col].fillna("").str.strip().eq("").sum())
        for col in ["ProfileName", "Summary", "Text"]
    },
}

with output_path.open("w", encoding="utf-8") as f:
    json.dump(baseline, f, indent=2)

print(json.dumps(baseline, indent=2))
print(f"\nSaved baseline to {output_path}")
