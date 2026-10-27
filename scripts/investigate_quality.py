
import pandas as pd

DATA_PATH = "data/processed/finefoods_parsed.csv.gz"

df = pd.read_csv(DATA_PATH, compression="gzip")

# 1. Investigate invalid helpfulness counts
invalid_helpfulness = df[
    df["HelpfulnessNumerator"] > df["HelpfulnessDenominator"]
]

print("=" * 70)
print("1. INVALID HELPFULNESS RECORDS")
print("=" * 70)

print(
    invalid_helpfulness[
        [
            "ProductId",
            "UserId",
            "HelpfulnessRaw",
            "HelpfulnessNumerator",
            "HelpfulnessDenominator",
            "Score",
            "Summary",
            "Text",
        ]
    ].to_string(index=False)
)

# 2. Inspect exact duplicate rows
print("\n" + "=" * 70)
print("2. EXACT DUPLICATE ROWS")
print("=" * 70)

duplicates = df[df.duplicated(keep=False)].sort_values(
    by=["ProductId", "UserId", "TimeRaw"]
)

print("Rows participating in exact duplicates:", len(duplicates))
print("\nFirst 10 duplicate rows:")

print(
    duplicates[
        [
            "ProductId",
            "UserId",
            "HelpfulnessRaw",
            "Score",
            "TimeRaw",
            "Summary",
            "Text",
        ]
    ].head(10).to_string(index=False)
)

# 3. Check empty and whitespace-only text fields
print("\n" + "=" * 70)
print("3. EMPTY OR WHITESPACE-ONLY FIELDS")
print("=" * 70)

for col in ["ProfileName", "Summary", "Text"]:
    empty_count = df[col].fillna("").str.strip().eq("").sum()
    print(f"{col}: {empty_count} empty/whitespace-only values")
