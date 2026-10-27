
from pathlib import Path
import numpy as np
import pandas as pd
import json

DATA_PATH = "data/processed/finefoods_cleaned.csv.gz"
OUTPUT_DIR = Path("data/splits")
SEED = 42
CONFIRMATION_FRACTION = 0.20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, compression="gzip")

# Split distinct products, not individual review rows.
# This prevents a product from appearing in both partitions.
products = df["ProductId"].dropna().unique().to_numpy(copy=True)
rng = np.random.default_rng(SEED)
rng.shuffle(products)

n_confirmation = round(len(products) * CONFIRMATION_FRACTION)
confirmation_products = set(products[:n_confirmation])

is_confirmation = df["ProductId"].isin(confirmation_products)

exploration = df.loc[~is_confirmation].copy()
confirmation = df.loc[is_confirmation].copy()

exploration_path = OUTPUT_DIR / "exploration.csv.gz"
confirmation_path = OUTPUT_DIR / "confirmation.csv.gz"

exploration.to_csv(exploration_path, index=False, compression="gzip")
confirmation.to_csv(confirmation_path, index=False, compression="gzip")

# Verify that product IDs do not overlap.
exploration_products = set(exploration["ProductId"].unique())
confirmation_product_set = set(confirmation["ProductId"].unique())
overlap = exploration_products & confirmation_product_set

report = {
    "seed": SEED,
    "split_method": "product-grouped random split",
    "confirmation_fraction_of_products": CONFIRMATION_FRACTION,
    "total_rows": int(len(df)),
    "exploration_rows": int(len(exploration)),
    "confirmation_rows": int(len(confirmation)),
    "total_unique_products": int(len(products)),
    "exploration_unique_products": int(len(exploration_products)),
    "confirmation_unique_products": int(len(confirmation_product_set)),
    "overlapping_product_ids": len(overlap),
}

with open(OUTPUT_DIR / "split_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))

assert len(overlap) == 0, "Product leakage detected!"
assert len(exploration) + len(confirmation) == len(df)
print("\nSplit verified successfully.")
