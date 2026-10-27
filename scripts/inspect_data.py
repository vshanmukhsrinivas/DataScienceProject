
import pandas as pd
import json

DATA_PATH = "data/processed/finefoods_parsed.csv.gz"
REPORT_PATH = "reports/step1_validation.json"

df = pd.read_csv(DATA_PATH, compression="gzip")

print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\n2. DATA TYPES")
print(df.dtypes)

print("\n3. MISSING VALUES")
print(df.isna().sum())

print("\n4. DUPLICATE CHECKS")
print("Exact duplicate rows:", df.duplicated().sum())
print("Rows with repeated ProductId:", df["ProductId"].duplicated().sum())
print("Rows with repeated UserId:", df["UserId"].duplicated().sum())

print("\n5. SCORE DISTRIBUTION")
print(df["Score"].value_counts(dropna=False).sort_index())

print("\n6. HELPFULNESS VALIDATION")
num = df["HelpfulnessNumerator"]
den = df["HelpfulnessDenominator"]
print("Missing numerator:", num.isna().sum())
print("Missing denominator:", den.isna().sum())
print("Numerator > denominator:", (num > den).sum())
print("Zero denominator:", (den == 0).sum())

print("\n7. REVIEW LENGTH SUMMARY")
print(df[["ReviewLengthChars", "SummaryLengthChars"]].describe(
    percentiles=[0.25, 0.50, 0.75, 0.95, 0.99]
))

print("\n8. PARSER VALIDATION REPORT")
with open(REPORT_PATH, "r", encoding="utf-8") as f:
    report = json.load(f)

print("Records written:", report["records_written"])
print("Continuation lines preserved:",
      report["continuation_lines_attached"])
print("Orphan lines:", report["orphan_lines"])
print("Repeated known fields:", report["repeated_fields"])
print("Missing required fields:",
      report["missing_required_fields"])
print("Missing optional fields:",
      report["missing_optional_fields"])
print("Helpfulness parse failures:",
      report["helpfulness_parse_failures"])
print("Numerator > denominator:",
      report["numerator_gt_denominator"])
print("Unexpected scores:", report["unexpected_scores"])
print("Score parse failures:", report["score_parse_failures"])
print("Time parse failures:", report["time_parse_failures"])
