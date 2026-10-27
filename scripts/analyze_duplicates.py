
import pandas as pd

DATA_PATH = "data/processed/finefoods_parsed.csv.gz"

df = pd.read_csv(DATA_PATH, compression="gzip")

# Count rows beyond the first occurrence of each exact duplicate
print("Rows beyond first occurrence:", df.duplicated().sum())

# Find rows with identical review content and metadata,
# excluding ProductId so cross-product reuse can be examined.
comparison_cols = [
    col for col in df.columns if col != "ProductId"
]

same_except_product = df[
    df.duplicated(subset=comparison_cols, keep=False)
].sort_values(["UserId", "TimeRaw", "Summary"])

print(
    "\nRows identical except potentially ProductId:",
    len(same_except_product)
)

print(
    same_except_product[
        [
            "ProductId",
            "UserId",
            "HelpfulnessRaw",
            "Score",
            "TimeRaw",
            "Summary",
            "Text",
        ]
    ].head(12).to_string(index=False)
)

# Count duplicate groups using all original and derived columns.
exact_duplicate_groups = df.groupby(
    df.columns.tolist(),
    dropna=False
).size()

exact_duplicate_groups = exact_duplicate_groups[
    exact_duplicate_groups > 1
]

print("\nNumber of exact duplicate groups:", len(exact_duplicate_groups))
print("Largest exact duplicate group size:",
      exact_duplicate_groups.max()
      if len(exact_duplicate_groups) else 1)
